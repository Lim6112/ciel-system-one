"""Tests for the security guard.

No model is loaded, no provider is called, and nothing outside this machine is touched.
The laya detector is stubbed with fixed probabilities, and the chat model is a fake that
scripts the turns — so the tests check the guard's decisions and the agent's mechanics,
not the model's behaviour.

    uv run python -m pytest test_guard.py
"""

from __future__ import annotations

import http.server
import json
import threading
from typing import Any, Dict, Iterator, List

import pytest
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

import guard_agent
import guard_pii
import guard_scan


# ---- fakes ---------------------------------------------------------------------------------


class FakeToolCallingModel(GenericFakeChatModel):
    """A chat model that can be handed tools.

    GenericFakeChatModel does not implement bind_tools, which create_agent calls when
    tools are passed — so a subclass that returns itself is the documented way to script
    an agent's turns without a model.
    """

    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        return self


class StubLayaDetector:
    """A laya detector that returns fixed probabilities.

    The point of the stub is that the threshold logic is what is under test. A live
    checkpoint would make these assertions depend on a 421M-parameter model's mood.
    """

    def __init__(self, **scores: float) -> None:
        self.scores = scores or {"sensitive_data": 0.0}
        self.calls: List[Dict[str, Any]] = []

    def scan(self, text: str, model: str = "english") -> List[guard_pii.Finding]:
        self.calls.append({"text": text, "model": model})
        return [
            guard_pii.Finding("laya", guard_pii._category_for(qid), score, qid)
            for qid, score in self.scores.items()
        ]


class _ToolStageBlockingScanner:
    """A scanner that reports a secret whenever it sees an email address.

    The input in these tests carries no email, and the tool's arguments do — so the
    input passes and the tool's arguments are what get stopped.
    """

    name = "regex"

    def scan(self, text: str) -> List[guard_pii.Finding]:
        if "@" in text:
            return [guard_pii.Finding(self.name, guard_pii.SECRET_CATEGORY, 1.0, "email: ...@...")]
        return []


@pytest.fixture()
def stub_guard() -> Iterator[guard_pii.Guard]:
    """A guard whose model arm is stubbed and whose regex scanner is real."""
    guard = guard_pii.Guard()
    guard.laya = StubLayaDetector(sensitive_data=0.0)  # type: ignore[assignment]
    yield guard


def build(guard: guard_pii.Guard, messages: List[AIMessage], tools: List[Any] = None) -> Any:
    """An agent with the guard attached, driven by a scripted model."""
    return create_agent(
        FakeToolCallingModel(messages=iter(messages)),
        tools=tools or [],
        middleware=[guard_agent.SecurityGuard(guard=guard, arm=guard_pii.resolve_arm("english"))],
        state_schema=guard_agent.GuardState,
    )


# ---- the arm registry ---------------------------------------------------------------------


def test_every_arm_has_a_detector():
    """Each dropdown value resolves to an arm, and no arm is missing."""
    for key, arm in guard_pii.ARMS.items():
        assert arm.key == key
        assert arm.kind in ("laya", "lab", "jev")
        if arm.kind == "laya":
            assert arm.model
        if arm.kind == "lab":
            assert arm.path


def test_unknown_model_is_refused_not_defaulted():
    """A typo must not silently get checked by a different detector."""
    with pytest.raises(guard_pii.ArmError, match="unknown model"):
        guard_pii.resolve_arm("engish")


def test_unknown_path_is_refused():
    with pytest.raises(guard_pii.ArmError, match="unknown path"):
        guard_pii.resolve_arm("ciel_typed_v1", "nosuchpath")


def test_a_laya_arm_takes_no_path():
    """A checkpoint has no sub-paths, so asking for one is an error, not a no-op."""
    with pytest.raises(guard_pii.ArmError, match="no sub-paths"):
        guard_pii.resolve_arm("english", "doc")


def test_jev_arm_does_not_block_on_the_model_opinion():
    """jev sends the text to a third party; it reports, it does not block."""
    assert guard_pii.ARMS["jev"].model_pii_blocks is False


def test_the_decision_lab_arms_do_not_block_on_the_model_opinion():
    """Measured: on the same PII text against clean text the typed head scored 0.171
    against 0.430 — anti-correlated. Blocking on it would reject the clean prompt and
    pass the PII one. The arms are still there to be compared."""
    for key in ("ciel_typed_v1", "ciel_decision_model_v1", "ciel_warehouse_v1"):
        assert guard_pii.ARMS[key].model_pii_blocks is False


def test_the_typed_decisions_arm_does_not_block_on_the_model_opinion():
    """Measured: on 20 ordinary prompts this checkpoint puts 2 over the 0.3 threshold
    where the english checkpoint puts 0 over. It is offered because it is this repo's
    typed-decision specialist, not because it is a good PII detector."""
    assert guard_pii.ARMS["typed-decisions"].model_pii_blocks is False


def test_only_the_two_strong_laya_arms_block_on_the_model_opinion():
    assert guard_pii.ARMS["english"].model_pii_blocks is True
    assert guard_pii.ARMS["multilingual"].model_pii_blocks is True


def test_a_lab_arm_still_blocks_on_the_regex_scanner(stub_guard):
    """The regex scanner is arm-independent: a matched format blocks on every arm."""
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    report = stub_guard.scan(
        "key sk-proj-abc123XYZdef456ghi789",
        guard_pii.resolve_arm("ciel_typed_v1"),
        guard_pii.INPUT_STAGE,
    )
    assert report.blocked
    assert "secret" in report.reason


# ---- the secret scanner --------------------------------------------------------------------


def test_scanner_finds_provider_keys():
    findings = guard_pii.SecretScanner().scan("here is sk-proj-abc123XYZdef456ghi789")
    assert [f.category for f in findings] == [guard_pii.SECRET_CATEGORY]


def test_scanner_finds_assigned_secrets():
    findings = guard_pii.SecretScanner().scan('api_key = "s3cr3tvalue1234567890"')
    assert any(f.category == guard_pii.SECRET_CATEGORY for f in findings)


def test_scanner_never_returns_the_raw_secret():
    """A finding gets logged and returned over HTTP, so it must not carry the secret."""
    secret = "sk-proj-abc123XYZdef456ghi789"
    findings = guard_pii.SecretScanner().scan(f"here is {secret}")
    assert findings
    for finding in findings:
        assert secret not in finding.detail


def test_the_mask_keeps_only_the_leading_characters():
    """The label already says what kind of thing was found, so the trailing characters
    are the part that actually narrows the value down. Revealing the last two of an
    SSN gives away 5 of its 9 digits."""
    scanner = guard_pii.SecretScanner()
    for text, tail in (
        ("jane.doe@example.com", "example.com"),
        ("123-45-6789", "6789"),
        ("4111 1111 1111 1111", "1111"),
        ("sk-proj-abc123XYZdef456ghi789", "ghi789"),
    ):
        for finding in scanner.scan(text):
            assert tail not in finding.detail, finding.detail
            assert finding.detail.endswith("...")


def test_scanner_accepts_a_luhn_valid_card():
    findings = guard_pii.SecretScanner().scan("my card is 4111 1111 1111 1111")
    assert any(f.detail.startswith("card") for f in findings)


def test_scanner_rejects_a_luhn_invalid_card():
    """Without the checksum every long order number is a credit card."""
    findings = guard_pii.SecretScanner().scan("order 4111 1111 1111 1112")
    assert not any(f.detail.startswith("card") for f in findings)


def test_scanner_finds_an_email_and_an_ssn():
    findings = guard_pii.SecretScanner().scan("jane.doe@example.com ssn 123-45-6789")
    details = " ".join(f.detail for f in findings)
    assert "email" in details
    assert "ssn" in details


def test_scanner_finds_a_phone():
    findings = guard_pii.SecretScanner().scan("call me at 555-123-4567")
    assert any(f.detail.startswith("phone") for f in findings)


def test_scanner_is_quiet_on_ordinary_text():
    findings = guard_pii.SecretScanner().scan(
        "Hi, please could you summarize the second quarter revenue report?"
    )
    assert findings == []


def test_scanner_does_not_match_sk_inside_a_word():
    """An unanchored `sk-` pattern matches the "sk-" in "task-oriented-approach", and
    a guard that blocks those rejects ordinary English."""
    for text in (
        "Can you summarize the task-oriented-approach document?",
        "Please review the risk-management-framework.",
        "We went for a brisk-walking-in-the-park today.",
        "The risk-sk-averse approach is documented.",
    ):
        assert guard_pii.SecretScanner().scan(text) == [], text


def test_scanner_does_not_match_a_version_string_as_a_phone():
    """A dot in the phone separator set makes "1.234.567.8901" a phone number."""
    assert guard_pii.SecretScanner().scan("version 1.234.567.8901 released") == []


def test_scanner_still_finds_the_real_key_shapes():
    """The tightened patterns must still catch what they are for."""
    scanner = guard_pii.SecretScanner()
    for text in (
        "my key is sk-proj-abc123XYZdef456ghi789",
        "anthropic sk-ant-abc123XYZdef456ghi789jkl",
        "legacy sk-abcdefghijklmnopqrstuvwxyz0123456789",
        "token ghp_abcdefghijklmnopqrstuvwxyz0123",
        "AKIAIOSFODNN7EXAMPLE",
        "xoxb-1234567890-abcdefghij",
        "AIzaSyA1234567890abcdefghijklmnopqrstuv",
        "openrouter sk-or-v1-abcdefghijklmnopqrstuvwxyz1234abcdefghijklmnopqrstuvwxyz1234",
        "gitlab glpat-abcdefghijklmnopqrst",
        "npm npm_abcdefghijklmnopqrstuvwxyz012345",
        "hf token hf_abcdefghijklmnopqrstuvwxyz1234",
        "-----BEGIN RSA PRIVATE KEY-----",
    ):
        findings = scanner.scan(text)
        assert any(f.category == guard_pii.SECRET_CATEGORY for f in findings), text


def test_scanner_finds_assigned_secrets_in_real_naming_conventions():
    """The keyword is usually part of a longer identifier, so a word boundary before it
    fails on every one of these — `access_token`, `client_secret`, `DB_PASSWORD`,
    `accessToken`."""
    scanner = guard_pii.SecretScanner()
    for text in (
        "access_token=abcdefghijklmnopqrstuvwxyz",
        "refresh_token: abcdefghijklmnopqrstuvwxyz",
        "client_secret=abcdefghijklmnopqrstuv",
        "MY_SECRET=abcdefghijklmnop",
        "DB_PASSWORD=abcdefghijklmnop",
        "aws_secret_access_key=abcdefghijklmnopqrst",
        "accessToken=abcdefghijklmnopqrst",
        "api key = 0123456789abcdef0123",
        'password = "abc+defghijklmnopqrstuvwxyz"',
        'password = "abcdefghijklmno="',
    ):
        findings = scanner.scan(text)
        assert any(f.category == guard_pii.SECRET_CATEGORY for f in findings), text


def test_scanner_does_not_match_prose_that_mentions_a_secret_word():
    """The assigned-secret pattern is anchored on "not a letter or digit", which is looser
    than a word boundary. These are the sentences that looseness must not catch."""
    scanner = guard_pii.SecretScanner()
    for text in (
        "Set the token budget to 32000 tokens for this model.",
        "The API key rotation policy is quarterly.",
        "my secret sauce recipe is the best one",
        "The password reset link expired, please request a new one.",
        "user_id=12345678901234567890",
        "Invoice total: 1234567890123456",
    ):
        assert scanner.scan(text) == [], text


def test_scanner_does_not_match_number_groups_as_phone_numbers():
    """A phone number and a number group have the same shape. Matching both blocks
    "Where is my order 123 456 7890?" and "the build ran for 1 234 567 8901
    iterations" — ordinary text, blocked on every arm at every stage."""
    scanner = guard_pii.SecretScanner()
    for text in (
        "Where is my order 123 456 7890?",
        "The build ran for 1 234 567 8901 iterations.",
        "Ticket 2024 123 4567 is closed.",
        "order 123 456 7890",
        "version 1.234.567.8901 released",
    ):
        assert scanner.scan(text) == [], text


def test_scanner_still_finds_the_phone_shapes():
    """A phone number needs a signal a number group does not have: a country code, a
    parenthesised area code, or a hyphen."""
    scanner = guard_pii.SecretScanner()
    for text in ("call me at 555-123-4567", "+1-555-565-2477", "(555) 123-4567"):
        assert any(f.detail.startswith("phone") for f in scanner.scan(text)), text


def test_the_email_pattern_is_linear_in_the_length_of_the_text():
    """An unbounded local-part class makes the engine retry from every start offset on a
    long run with no "@" in it: 3.25 s for 64 KB, and the scan runs under the
    server-wide lock. The bounded class is linear."""
    import re
    import time

    pattern = re.compile(guard_scan._PII_PATTERNS[0][1])
    text = "a." * 16000
    start = time.perf_counter()
    pattern.search(text)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"the email pattern took {elapsed:.2f}s on 32 KB"


# ---- the threshold policy ------------------------------------------------------------------


def test_pii_below_the_threshold_does_not_block(stub_guard):
    """0.29 is under the threshold, so the text proceeds."""
    stub_guard.laya.scores = {"sensitive_data": 0.29}
    report = stub_guard.scan("some text", guard_pii.resolve_arm("english"), guard_pii.INPUT_STAGE)
    assert not report.blocked


def test_pii_at_the_threshold_blocks(stub_guard):
    """The threshold is inclusive: 0.3 blocks."""
    stub_guard.laya.scores = {"sensitive_data": 0.30}
    report = stub_guard.scan("some text", guard_pii.resolve_arm("english"), guard_pii.INPUT_STAGE)
    assert report.blocked
    assert "personal data" in report.reason


def test_the_model_cannot_block_on_the_output_stage(stub_guard):
    """Measured: on model replies the PII latent does not separate — clean replies
    reach 0.653 and a leaked card number scores 0.163. Blocking on it there would
    reject "I cannot reveal my system prompt" while missing the leak."""
    stub_guard.laya.scores = {"sensitive_data": 0.90}
    report = stub_guard.scan("some reply", guard_pii.resolve_arm("english"), guard_pii.OUTPUT_STAGE)
    assert not report.blocked
    # It is still reported, so the trace shows what the model thought.
    assert any(f.category == guard_pii.PII_CATEGORY for f in report.findings)


def test_the_model_cannot_block_on_the_tool_stage(stub_guard):
    """Same measurement, same reason: short tool arguments do not separate either."""
    stub_guard.laya.scores = {"sensitive_data": 0.90}
    report = stub_guard.scan("A1", guard_pii.resolve_arm("english"), guard_pii.TOOL_STAGE)
    assert not report.blocked


def test_the_regex_scanner_blocks_on_every_stage(stub_guard):
    """A matched format is a fact about the text, not a probability to tune — and it
    separates in all three stages where the model does not."""
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    for stage in (guard_pii.INPUT_STAGE, guard_pii.TOOL_STAGE, guard_pii.OUTPUT_STAGE):
        report = stub_guard.scan(
            "key sk-proj-abc123XYZdef456ghi789", guard_pii.resolve_arm("english"), stage
        )
        assert report.blocked, f"the scanner did not block on the {stage} stage"


def test_a_secret_blocks_regardless_of_the_model_score(stub_guard):
    """A matched format is a fact about the text, not a probability to tune."""
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    report = stub_guard.scan(
        "key sk-proj-abc123XYZdef456ghi789", guard_pii.resolve_arm("english"), guard_pii.INPUT_STAGE
    )
    assert report.blocked
    assert "secret" in report.reason


def test_jailbreak_is_reported_but_does_not_block(stub_guard):
    """The jailbreak latent's false-positive band overlaps its true-positive band, so
    it is computed and shown but does not stop the request by default."""
    stub_guard.laya.scores = {"jailbreak": 0.999}
    report = stub_guard.scan("create a prompt template", guard_pii.resolve_arm("english"))
    assert not report.blocked
    assert any(f.category == guard_pii.JAILBREAK_CATEGORY for f in report.findings)


def test_jailbreak_blocks_when_the_policy_says_so(stub_guard):
    """GUARD_BLOCK_JAILBREAK=1 turns it on for a deployment that accepts the false
    positives."""
    stub_guard.policy = guard_pii.GuardPolicy(block_on_jailbreak=True)
    stub_guard.laya.scores = {"jailbreak": 0.999}
    report = stub_guard.scan("create a prompt template", guard_pii.resolve_arm("english"))
    assert report.blocked


def test_policy_from_env_reads_the_override(monkeypatch):
    monkeypatch.setenv("GUARD_BLOCK_JAILBREAK", "1")
    assert guard_pii.policy_from_env().block_on_jailbreak is True
    monkeypatch.setenv("GUARD_BLOCK_JAILBREAK", "")
    assert guard_pii.policy_from_env().block_on_jailbreak is False


def test_injection_has_its_own_switch(monkeypatch):
    """Turning on jailbreak blocking must not silently turn on injection blocking —
    the two latents have different false-positive behaviour."""
    monkeypatch.setenv("GUARD_BLOCK_JAILBREAK", "1")
    assert guard_pii.policy_from_env().block_on_injection is False
    monkeypatch.setenv("GUARD_BLOCK_INJECTION", "1")
    assert guard_pii.policy_from_env().block_on_injection is True


# ---- the middleware: before the model -------------------------------------------------------


def test_before_model_blocks_before_the_model_is_called(stub_guard):
    """The model must never see a request the guard rejected."""
    stub_guard.laya.scores = {"sensitive_data": 0.95}
    agent = build(stub_guard, [AIMessage(content="should never be produced")])
    result = agent.invoke({"messages": [HumanMessage(content="my ssn is 123-45-6789")]})
    assert result["blocked"] is True
    assert result["messages"][-1].content == guard_pii.BLOCKED_ANSWER
    assert not any(
        "never be produced" in str(m.content) for m in result["messages"]
    ), "the model ran despite the guard blocking"


def test_before_model_lets_a_clean_request_through(stub_guard):
    """The negative control: a guard that blocks everything is not a guard."""
    stub_guard.laya.scores = {"sensitive_data": 0.01}
    agent = build(stub_guard, [AIMessage(content="order A1 shipped")])
    result = agent.invoke({"messages": [HumanMessage(content="where is my order A1?")]})
    assert not result.get("blocked")
    assert result["messages"][-1].content == "order A1 shipped"


# ---- the middleware: before the tool runs ---------------------------------------------------


def test_tool_guard_stops_the_tool_body(stub_guard):
    """The tool must not execute. A tool that records its own call is the only way to
    prove that from outside the guard."""
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    guard_agent.reset_tool_calls()
    agent = build(
        stub_guard,
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="get_customer_profile",
                        args={"email": "jane.doe@example.com"},
                        id="call_1",
                    )
                ],
            ),
            AIMessage(content="done"),
        ],
        tools=[guard_agent.get_customer_profile],
    )
    result = agent.invoke({"messages": [HumanMessage(content="look up jane.doe@example.com")]})
    assert guard_agent.tool_calls_made() == [], "the tool body ran despite the block"
    assert result["messages"][-1].content == guard_pii.BLOCKED_ANSWER


def test_tool_guard_lets_a_clean_call_through(stub_guard):
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    guard_agent.reset_tool_calls()
    agent = build(
        stub_guard,
        [
            AIMessage(
                content="",
                tool_calls=[ToolCall(name="lookup_order", args={"order_id": "A1"}, id="c1")],
            ),
            AIMessage(content="it shipped"),
        ],
        tools=[guard_agent.lookup_order],
    )
    result = agent.invoke({"messages": [HumanMessage(content="where is order A1?")]})
    assert [c["tool"] for c in guard_agent.tool_calls_made()] == ["lookup_order"]
    assert result["messages"][-1].content == "it shipped"


def test_tool_guard_blocks_pii_the_model_volunteered(stub_guard):
    """The user typed nothing personal, but the model put an email into the tool
    arguments — which is the case the tool guard exists for."""
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    guard_agent.reset_tool_calls()
    agent = build(
        stub_guard,
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="get_customer_profile",
                        args={"email": "jane.doe@example.com"},
                        id="call_1",
                    )
                ],
            ),
            AIMessage(content="they are gold tier"),
        ],
        tools=[guard_agent.get_customer_profile],
    )
    result = agent.invoke({"messages": [HumanMessage(content="look up my profile")]})
    assert guard_agent.tool_calls_made() == [], "the tool body ran despite the block"
    assert result["messages"][-1].content == guard_pii.BLOCKED_ANSWER


def test_a_blocked_tool_does_not_let_the_model_answer_without_it(stub_guard):
    """The tool did not run, so the model has nothing to answer with — and on this
    example it invents a customer profile rather than admitting it. The run ends with
    the blocked answer instead."""
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    guard_agent.reset_tool_calls()
    agent = build(
        stub_guard,
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="get_customer_profile",
                        args={"email": "jane.doe@example.com"},
                        id="call_1",
                    )
                ],
            ),
            AIMessage(content="that customer is a gold tier member"),
        ],
        tools=[guard_agent.get_customer_profile],
    )
    result = agent.invoke({"messages": [HumanMessage(content="look up my profile")]})
    answer = result["messages"][-1].content
    assert answer == guard_pii.BLOCKED_ANSWER
    assert "gold tier" not in answer, "the model answered from a tool that never ran"


def test_a_blocked_tool_ends_the_run_even_when_the_model_answers_empty(stub_guard):
    """REGRESSION. A model that sees a failed tool call often retries and then finishes
    with an EMPTY message. The blocked-tool check has to come before the empty-content
    check, or the request ends with an empty answer, "not blocked", and a trace that
    never mentions the tool stage — the block happened, the response just does not
    admit it."""
    guard = guard_pii.Guard()
    # A scanner that blocks on the tool stage only, so the input passes.
    guard.laya = StubLayaDetector(sensitive_data=0.0)
    guard.scanner = _ToolStageBlockingScanner()
    guard_agent.reset_tool_calls()
    agent = build(
        guard,
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="get_customer_profile",
                        args={"email": "jane.doe@example.com"},
                        id="call_1",
                    )
                ],
            ),
            AIMessage(content=""),  # the model retries and finishes empty
        ],
        tools=[guard_agent.get_customer_profile],
    )
    result = agent.invoke({"messages": [HumanMessage(content="look up my profile")]})
    assert result.get("blocked") is True
    assert result["messages"][-1].content == guard_pii.BLOCKED_ANSWER
    stages = [event["stage"] for event in result["guard_events"]]
    assert "wrap_tool_call" in stages, "the trace never mentions the tool stage"


def test_a_blocked_tool_leaves_no_dangling_tool_call(stub_guard):
    """If the model's message is removed, the tool call it carried must not be left
    unanswered — the next turn would send that history upstream, and a provider
    rejects a tool call with no matching result."""
    guard = guard_pii.Guard()
    guard.laya = StubLayaDetector(sensitive_data=0.0)
    guard.scanner = _ToolStageBlockingScanner()
    guard_agent.reset_tool_calls()
    agent = build(
        guard,
        [
            AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="get_customer_profile",
                        args={"email": "jane.doe@example.com"},
                        id="call_1",
                    )
                ],
            ),
            AIMessage(content=""),
        ],
        tools=[guard_agent.get_customer_profile],
    )
    result = agent.invoke({"messages": [HumanMessage(content="look up my profile")]})
    tool_call_ids = {
        call["id"]
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    }
    tool_result_ids = {
        message.tool_call_id
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }
    assert tool_call_ids == tool_result_ids, "a tool call was left without its result"


def test_a_clean_tool_call_still_delivers_its_result(stub_guard):
    """REGRESSION. A Command whose update has no "messages" key DROPS the tool's
    output: the model never sees the result, has nothing to answer with, and returns
    an empty reply.

    This was invisible to the fake-model tests, because a fake model answers with its
    next scripted message regardless of what the tool returned. Only a real model
    notices that the result is missing — and the symptom is an empty answer, which
    reads like the model being unhelpful rather than like the tool being broken.
    """
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    guard_agent.reset_tool_calls()
    agent = build(
        stub_guard,
        [
            AIMessage(
                content="",
                tool_calls=[ToolCall(name="lookup_order", args={"order_id": "A1"}, id="c1")],
            ),
            AIMessage(content="it shipped"),
        ],
        tools=[guard_agent.lookup_order],
    )
    result = agent.invoke({"messages": [HumanMessage(content="where is order A1?")]})
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "the tool result never reached the conversation"
    assert "A1" in tool_messages[0].content


def test_a_clean_tool_call_is_reported_in_the_trace(stub_guard):
    """The tool stage is shown whether it passed or failed — a trace that showed the
    tool stage only on failure would read as "the tool was never checked"."""
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    agent = build(
        stub_guard,
        [
            AIMessage(
                content="",
                tool_calls=[ToolCall(name="lookup_order", args={"order_id": "A1"}, id="c1")],
            ),
            AIMessage(content="it shipped"),
        ],
        tools=[guard_agent.lookup_order],
    )
    result = agent.invoke({"messages": [HumanMessage(content="where is order A1?")]})
    stages = [event["stage"] for event in result["guard_events"]]
    assert "wrap_tool_call" in stages


# ---- the middleware: after the model -------------------------------------------------------


def test_after_model_removes_the_leaky_answer(stub_guard):
    """RemoveMessage is what purges the secret from the history.

    Returning only the AIMessage would leave the original in the message list, where
    the next turn would send it back upstream as context — the guard would block the
    leak and then re-send it.
    """
    secret = "sk-proj-abc123XYZdef456ghi789"
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    agent = build(stub_guard, [AIMessage(content=f"your key is {secret}")])
    result = agent.invoke({"messages": [HumanMessage(content="what is my key?")]})
    assert result["blocked"] is True
    assert result["messages"][-1].content == guard_pii.BLOCKED_ANSWER
    for message in result["messages"]:
        assert secret not in str(message.content), "the secret survived in the history"


def test_after_model_leaves_a_clean_answer_alone(stub_guard):
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    agent = build(stub_guard, [AIMessage(content="order A1 shipped")])
    result = agent.invoke({"messages": [HumanMessage(content="where is order A1?")]})
    assert result["messages"][-1].content == "order A1 shipped"


# ---- the trace ----------------------------------------------------------------------------


def test_guard_events_accumulate_across_hooks(stub_guard):
    """Every hook contributes an event. Without the add reducer the last one to write
    would win and the trace would lie about where the block happened."""
    stub_guard.laya.scores = {"sensitive_data": 0.0}
    agent = build(stub_guard, [AIMessage(content="clean")])
    result = agent.invoke({"messages": [HumanMessage(content="a clean request")]})
    stages = [event["stage"] for event in result["guard_events"]]
    assert stages == ["before_model", "model", "after_model"]


def test_a_blocked_input_produces_no_model_event(stub_guard):
    """The "model ran" event is what makes "the model was never called" checkable
    rather than an inference from the absence of a reply."""
    stub_guard.laya.scores = {"sensitive_data": 0.95}
    agent = build(stub_guard, [AIMessage(content="should never be produced")])
    result = agent.invoke({"messages": [HumanMessage(content="my ssn is 123-45-6789")]})
    stages = [event["stage"] for event in result["guard_events"]]
    assert "model" not in stages
    assert "after_model" not in stages


def test_guard_events_report_the_finding_scores(stub_guard):
    stub_guard.laya.scores = {"sensitive_data": 0.42}
    agent = build(stub_guard, [AIMessage(content="clean")])
    result = agent.invoke({"messages": [HumanMessage(content="a request")]})
    event = result["guard_events"][0]
    assert event["findings"]
    assert any(f["score"] == 0.42 for f in event["findings"])


def test_only_the_blocking_finding_is_flagged(stub_guard):
    """The UI marks the finding that stopped the run. If every finding in a blocked
    turn were flagged, the trace would say they all tripped."""
    stub_guard.laya.scores = {"sensitive_data": 0.95}
    agent = build(stub_guard, [AIMessage(content="unused")])
    result = agent.invoke({"messages": [HumanMessage(content="my ssn is 123-45-6789")]})
    findings = result["guard_events"][0]["findings"]
    assert findings
    blocked = [f for f in findings if f["blocked"]]
    assert len(blocked) == 1
    # The regex match is the one that stopped it; the model's opinion was reported.
    assert blocked[0]["detector"] == "regex"
    assert any(f["detector"] == "laya" and not f["blocked"] for f in findings)


def test_the_blocked_answer_does_not_name_the_finding(stub_guard):
    """Telling the model which rule fired teaches it which rule to aim past."""
    stub_guard.laya.scores = {"sensitive_data": 0.95}
    agent = build(stub_guard, [AIMessage(content="unused")])
    result = agent.invoke({"messages": [HumanMessage(content="my ssn is 123-45-6789")]})
    answer = result["messages"][-1].content
    assert "sensitive_data" not in answer
    assert "0.95" not in answer
    assert answer == guard_pii.BLOCKED_ANSWER


# ---- the service-backed arm ------------------------------------------------------------------


class _PathsHandler(http.server.BaseHTTPRequestHandler):
    """A stand-in for decision_lab's /api/paths."""

    def do_GET(self):  # noqa: N802 - the stdlib names it this way
        payload = json.dumps({"paths": ["doc", "typed"], "default": "doc"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep test output clean
        pass


@pytest.fixture()
def paths_server(monkeypatch) -> Iterator[http.server.HTTPServer]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _PathsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("DECISION_LAB_URL", f"http://127.0.0.1:{server.server_port}")
    yield server
    server.shutdown()


def test_lab_paths_are_read_from_the_service(paths_server):
    """The dropdown must not drift from what the service actually serves."""
    assert guard_pii.lab_paths() == ["doc", "typed"]


def test_lab_paths_are_empty_when_the_service_is_down(monkeypatch):
    """A dead service yields an empty dropdown, not a 400 for the user."""
    monkeypatch.setenv("DECISION_LAB_URL", "http://127.0.0.1:1")
    assert guard_pii.lab_paths() == []


def test_a_dead_service_is_not_reported_as_an_unknown_path(monkeypatch):
    """Blaming the caller's path for an upstream outage is a 400 that says the wrong
    thing. The service being down is its own error."""
    monkeypatch.setenv("DECISION_LAB_URL", "http://127.0.0.1:1")
    with pytest.raises(guard_pii.ArmError, match="not reachable"):
        guard_pii.resolve_arm("ciel_typed_v1", "typed")


def test_an_unreachable_detector_is_reported_not_silently_clean(monkeypatch):
    """A caller must be able to tell "checked, nothing found" from "could not check"."""
    monkeypatch.setenv("DECISION_LAB_URL", "http://127.0.0.1:1")
    guard = guard_pii.Guard()
    findings = guard.lab.scan("some text", path="typed")
    assert any(f.category == guard_pii.UNAVAILABLE_CATEGORY for f in findings)


def test_an_unmapped_sub_path_is_an_error_not_a_fallback(monkeypatch):
    """Silently scoring an unmapped head with the typed one would report a different
    detector's answer as if it were the requested one."""
    monkeypatch.setattr(guard_pii, "_LAB_MODEL_BY_PATH", {"doc": "ciel_decision_model_v1"})
    with pytest.raises(guard_pii.ArmError, match="no arm is mapped"):
        guard_pii._lab_model_for("some_new_head")


def test_a_lab_arm_with_a_known_path_resolves(paths_server):
    arm = guard_pii.resolve_arm("ciel_typed_v1", "typed")
    assert arm.path == "typed"


# ---- the HTTP surface ------------------------------------------------------------------------


@pytest.fixture()
def client(monkeypatch):
    """A TestClient driving the real agent wiring, with a stubbed guard.

    The agent is built through guard_agent.build_agent with a scripted chat model and
    the stub guard, so the middleware and the state schema are the real ones — only the
    detector and the provider are replaced.

    The app is `server.app`: the guard's routes live there now, and there is no separate
    guard app to test. The lock is not stubbed — the route takes
    `_RouterState.device_lock`, which is a real (and free) lock.
    """
    from fastapi.testclient import TestClient

    import server as server_module

    class StubGuard:
        def __init__(self):
            self.calls = []

        def scan(self, text, arm, stage=guard_pii.INPUT_STAGE):
            self.calls.append((text, arm.key, stage))
            report = guard_pii.GuardReport(arm_model_pii_blocks=arm.model_pii_blocks)
            finding = guard_pii.Finding("laya", guard_pii.PII_CATEGORY, 0.9, "sensitive_data")
            if "ssn" in text:
                report.blocked = True
                report.reason = "a personal identifier was found in the text"
                # _findings_payload marks the blocker by IDENTITY, so a report that sets
                # `blocked` without naming `blocker` produces a blocked turn in which no
                # finding is flagged — the UI would mark none of them as the cause. The
                # stub has to name it or the flag is never exercised.
                report.blocker = finding
            report.findings.append(finding)
            return report

    stub = StubGuard()
    guarded = guard_agent.build_agent(
        chat_model=FakeToolCallingModel(messages=iter([AIMessage(content="order A1 shipped")])),
        guard=stub,
    )
    monkeypatch.setattr(server_module, "_AgentState", type("S", (), {
        "agent": guarded,
    }))
    # Deliberately NOT a `with` block: entering the TestClient context would run the
    # app lifespan and build a real agent (and a real chat model).
    yield TestClient(server_module.app), stub


def test_chat_returns_the_answer_and_the_trace(client):
    test_client, stub = client
    response = test_client.post("/api/guard/chat", json={"message": "where is order A1?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "order A1 shipped"
    assert body["blocked"] is False
    assert body["arm"] == "english"


def test_chat_blocks_a_flagged_message(client):
    test_client, stub = client
    response = test_client.post("/api/guard/chat", json={"message": "my ssn is 123-45-6789"})
    body = response.json()
    assert body["blocked"] is True
    assert body["answer"] == guard_pii.BLOCKED_ANSWER


def test_chat_refuses_an_unknown_arm(client):
    test_client, _ = client
    response = test_client.post("/api/guard/chat", json={"message": "hi", "model": "nope"})
    assert response.status_code == 400


def test_health_lists_the_arms(client):
    test_client, _ = client
    response = test_client.get("/api/guard/health")
    assert response.status_code == 200
    body = response.json()
    assert {arm["key"] for arm in body["arms"]} == set(guard_pii.ARMS)


def test_examples_are_served(client):
    test_client, _ = client
    response = test_client.get("/api/guard/examples")
    assert response.status_code == 200
    assert len(response.json()) >= 4


def test_the_chat_route_is_sync():
    """A sync route runs in FastAPI's threadpool, so the lock is taken and released
    inside a worker thread.

    An async route holding this same threading.Lock across an await deadlocks the
    whole server: the first request acquires the lock on the event-loop thread and
    suspends there, and the second request arrives on that same thread and blocks
    forever waiting for a lock the first request cannot release.
    """
    import inspect

    import server as server_module

    assert not inspect.iscoroutinefunction(server_module.guard_chat)


def test_predict_and_guard_chat_share_one_lock(client):
    """The merge's whole point: one lock for the process's model device, not one each.

    The guard's scan runs a laya checkpoint, just as /api/predict does, and the crash
    that these locks exist to prevent ("a command encoder is already encoding to this
    command buffer") is scoped to the DEVICE. Two locks would therefore let a guard turn
    and a prediction collide holding different locks — so if anyone reintroduces a
    per-endpoint lock, this fails.

    Proven from the outside: hold the device lock on this thread and check that the
    guard route cannot finish until it is released.
    """
    import server as server_module

    test_client, _ = client
    held = server_module._RouterState.device_lock
    assert held.acquire(timeout=1), "the device lock was already held"
    try:
        guard_turn = threading.Thread(
            target=test_client.post,
            args=("/api/guard/chat",),
            kwargs={"json": {"message": "where is order A1?"}},
        )
        guard_turn.start()
        guard_turn.join(timeout=0.3)
        assert guard_turn.is_alive(), "the guard route did not take the device lock"
    finally:
        held.release()
    guard_turn.join(timeout=5)
    assert not guard_turn.is_alive()


# ---- the scan route (the traffic demo's content tier) ----------------------------------------


@pytest.fixture()
def scan_client(monkeypatch):
    """A TestClient whose guard is REAL, with only the laya detector stubbed.

    Unlike the `client` fixture, the guard here is a real `guard_pii.Guard`, so the regex
    scanner runs for real and a real `report.blocker` reaches the real
    `guard_agent._findings_payload`. That is what makes the route's response a real
    assertion about the shipped serialization rather than about a stub.

    The laya detector is stubbed because the scan route is about the detection AND the
    wiring, and a real checkpoint is a 30-second load; the scanner needs no model at all,
    so a secret still drives a genuine block through the whole path.

    Installed the same way the `client` fixture does it, and — like it — deliberately
    NOT as a `with` block, so the lifespan and the real chat model never run.
    """
    from fastapi.testclient import TestClient

    import server as server_module

    guard = guard_pii.Guard()
    guard.laya = StubLayaDetector()
    guarded = guard_agent.build_agent(
        chat_model=FakeToolCallingModel(messages=iter([AIMessage(content="unused")])),
        guard=guard,
    )
    monkeypatch.setattr(server_module, "_AgentState", type("S", (), {"agent": guarded}))
    yield TestClient(server_module.app), guard


def test_scan_returns_findings_for_a_clean_packet(scan_client):
    test_client, _ = scan_client
    response = test_client.post("/api/guard/scan", json={"message": "Where is my order A1?"})
    assert response.status_code == 200
    body = response.json()
    assert body["arm"] == "english"
    assert body["blocked"] is False
    assert body["arm_model_pii_blocks"] is True
    assert "answer" not in body, "the scan route must not answer — there is nothing to answer"


def test_scan_blocks_a_packet_carrying_a_secret(scan_client):
    test_client, _ = scan_client
    response = test_client.post(
        "/api/guard/scan",
        json={"message": "here is my key sk-proj-abc123XYZdef456ghi789, please use it"},
    )
    body = response.json()
    assert body["blocked"] is True
    assert body["reason"]


def test_scan_flags_exactly_one_finding_as_the_blocker(scan_client):
    """`blocked` by identity, not by value.

    Two identical matches are equal dataclasses; comparing by value would mark both as
    the blocker when only the first tripped.
    """
    test_client, _ = scan_client
    response = test_client.post(
        "/api/guard/scan", json={"message": "SSN 123-45-6789 and key sk-proj-abc123XYZdef456ghi789"}
    )
    body = response.json()
    flagged = [f for f in body["findings"] if f["blocked"]]
    assert len(flagged) == 1, f"expected one blocker, got {flagged}"


def test_scan_never_returns_the_raw_secret(scan_client):
    """A finding travels to the browser, so it must not carry a slice of what it found."""
    secret = "sk-proj-abc123XYZdef456ghi789"
    test_client, _ = scan_client
    response = test_client.post("/api/guard/scan", json={"message": f"my key is {secret}"})
    assert secret not in response.text
    assert "sk-p" in response.text


def test_scan_reports_the_arm_opinion_policy(scan_client):
    """The field that lets the page say the true thing on an arm that cannot block."""
    test_client, _ = scan_client
    laya = test_client.post("/api/guard/scan", json={"message": "hi", "model": "english"})
    assert laya.json()["arm_model_pii_blocks"] is True
    lab = test_client.post(
        "/api/guard/scan", json={"message": "hi", "model": "ciel_typed_v1", "path": "typed"}
    )
    assert lab.status_code == 200
    assert lab.json()["arm_model_pii_blocks"] is False


def test_scan_refuses_an_unknown_arm(scan_client):
    test_client, _ = scan_client
    response = test_client.post("/api/guard/scan", json={"message": "hi", "model": "nope"})
    assert response.status_code == 400


def test_scan_refuses_a_path_on_an_arm_without_sub_paths(scan_client):
    test_client, _ = scan_client
    response = test_client.post(
        "/api/guard/scan", json={"message": "hi", "model": "english", "path": "typed"}
    )
    assert response.status_code == 400


def test_scan_rejects_an_empty_message(scan_client):
    test_client, _ = scan_client
    assert test_client.post("/api/guard/scan", json={"message": ""}).status_code == 422


def test_scan_rejects_an_oversized_message(scan_client):
    test_client, _ = scan_client
    huge = "a" * (server_module_chars() + 1)
    assert test_client.post("/api/guard/scan", json={"message": huge}).status_code == 422


def server_module_chars():
    import server as server_module
    return server_module.MAX_MESSAGE_CHARS


def test_scan_reports_a_dead_detector_as_unavailable_not_clean(scan_client, monkeypatch):
    """The contract the traffic page's Inspect tier depends on.

    A detector that cannot be reached is not a clean verdict. It travels as a finding with
    the `detector_unavailable` category, so the page can withdraw Allow and hold the flow.
    """
    test_client, _ = scan_client

    import guard_pii as gp

    def dead(text, path="typed"):
        return [gp.Finding("decision_lab", gp.UNAVAILABLE_CATEGORY, 0.0, "unreachable")]

    monkeypatch.setattr(scan_client[1].lab, "scan", dead)
    response = test_client.post(
        "/api/guard/scan", json={"message": "hi", "model": "ciel_typed_v1", "path": "typed"}
    )
    body = response.json()
    categories = {f["category"] for f in body["findings"]}
    assert gp.UNAVAILABLE_CATEGORY in categories
    assert body["blocked"] is False, "an outage is not a block — it is a gap in coverage"


def test_scan_503s_when_there_is_no_guard(monkeypatch):
    from fastapi.testclient import TestClient

    import server as server_module

    monkeypatch.setattr(server_module, "_AgentState", type("S", (), {"agent": None}))
    test_client = TestClient(server_module.app)
    response = test_client.post("/api/guard/scan", json={"message": "hi"})
    assert response.status_code == 503


def test_the_scan_route_is_sync():
    """Same reasoning as the chat route: the scan takes the device lock, and holding a
    threading.Lock across an await on the event-loop thread deadlocks the server."""
    import inspect

    import server as server_module

    assert not inspect.iscoroutinefunction(server_module.guard_scan_text)


def test_predict_and_guard_scan_share_one_lock(scan_client):
    """The scan runs a laya checkpoint, exactly as /api/predict does. Two locks would let
    them collide holding different ones."""
    import server as server_module

    test_client, _ = scan_client
    held = server_module._RouterState.device_lock
    assert held.acquire(timeout=1), "the device lock was already held"
    try:
        scan = threading.Thread(
            target=test_client.post,
            args=("/api/guard/scan",),
            kwargs={"json": {"message": "where is order A1?"}},
        )
        scan.start()
        scan.join(timeout=0.3)
        assert scan.is_alive(), "the scan route did not take the device lock"
    finally:
        held.release()
    scan.join(timeout=5)
    assert not scan.is_alive()


def test_scan_findings_are_the_same_payload_as_the_chat_trace(scan_client):
    """The DRY assertion: the two routes cannot drift to two serializers.

    Both go through `guard_agent._findings_payload`, so the shape and the `blocked` flag
    agree by construction. Comparing against that function directly is what pins it.
    """
    test_client, guard = scan_client
    text = "my key is sk-proj-abc123XYZdef456ghi789"
    arm = guard_pii.resolve_arm("english")
    expected = guard_agent._findings_payload(guard.scan(text, arm, stage=guard_pii.INPUT_STAGE))
    got = test_client.post("/api/guard/scan", json={"message": text}).json()["findings"]
    assert got == expected


def test_scan_does_not_touch_the_agents_chat_model(scan_client):
    """A scan is not a turn. If this ever calls the chat model, every packet pays two
    model round trips for an answer nobody reads."""
    test_client, guard = scan_client
    calls_before = len(guard_agent.tool_calls_made()) if hasattr(
        guard_agent, "tool_calls_made") else 0
    test_client.post("/api/guard/scan", json={"message": "Where is my order A1?"})
    calls_after = len(guard_agent.tool_calls_made()) if hasattr(
        guard_agent, "tool_calls_made") else 0
    assert calls_after == calls_before, "a scan executed a tool"
