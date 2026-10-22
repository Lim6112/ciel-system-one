"""The LangChain v1 middleware stack that puts a guard between the user and the model.

Three hooks, in the order a request passes through them:

    before_model      the human message is scanned before the model ever sees it
    wrap_tool_call    the tool's arguments are scanned before the tool body runs
    after_model       the model's own answer is scanned before it is returned

Each one short-circuits with a single BLOCKED_ANSWER rather than a finding-by-finding
explanation. That is deliberate: naming the rule that fired teaches a probing model which
rule to aim past, and it tells the browser more than the user needs.

THE OUTPUT GUARD REMOVES THE MESSAGE, AND THAT IS NOT OPTIONAL.

When after_model finds a secret in the model's answer, it returns
`[RemoveMessage(id=...), AIMessage(BLOCKED_ANSWER)]`. The RemoveMessage is what actually
purges the leaky answer from the conversation history. Returning only the AIMessage
leaves the original in the message list, where the next turn would send it back to the
provider as context — the guard would have blocked the leak and then re-sent it. That
was verified against langchain 1.4.7, and it is why the pair is used rather than the
message alone.

THE TOOL GUARD DOES NOT CALL THE HANDLER.

`wrap_tool_call` inspects the tool's arguments and, on a hit, returns an error ToolMessage
without invoking `handler(request)`. The tool body therefore never executes. Returning
the rejection instead of the result also keeps the agent well-formed: the tool call still
gets the ToolMessage its tool_call_id expects, so the graph does not end up with a
dangling call.
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, Callable, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain.tools import tool
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from typing_extensions import NotRequired

import guard_pii
from guard_pii import Arm, BLOCKED_ANSWER, Guard, GuardPolicy, GuardReport

# The chat model that writes the answer. Deliberately separate from the detector arm:
# the dropdown picks which model CHECKS for PII, this picks which model ANSWERS.
#
# OpenRouter, not the local Ollama model this used to default to: gemma4:e4b-mlx took long
# enough per guarded turn to dominate the request. This needs OPENROUTER_API_KEY in the
# environment; a local model is still one GUARD_MODEL away (`GUARD_MODEL=ollama:gemma4:e4b-mlx`).
MODEL_ENV = "GUARD_MODEL"
DEFAULT_MODEL = "openrouter:deepseek/deepseek-v4.1-flash"

SYSTEM_PROMPT = (
    "You are a customer support assistant for an order-tracking service. "
    "Use the tools to look up orders and customer details when the request needs them. "
    "Answer concisely and factually."
)


class GuardState(AgentState):
    """The agent state, plus what the guard saw.

    `guard_events` uses the add reducer so each hook appends rather than overwrites.
    Without it the last hook to write wins, and a turn that produced both a
    before_model event and an after_model event would show only one of them — which
    would make the trace lie about where a block happened.

    `tool_blocked` is written by the tool hook and read by after_model, which is how a
    blocked tool ends the run instead of letting the model answer on without it.
    """

    guard_events: Annotated[List[Dict[str, Any]], operator.add]
    tool_blocked: Annotated[List[Dict[str, Any]], operator.add]
    blocked: NotRequired[bool]


# ---- the tools ---------------------------------------------------------------------------
# Two, so the tool-call guard has something to guard. The second is deliberately
# PII-adjacent: it is the case where a tool argument carries personal data into a
# backend, which is the leak the tool guard exists to stop.

_TOOL_CALLS: List[Dict[str, Any]] = []


@tool
def lookup_order(order_id: str) -> str:
    """Look up the status of an order by its id.

    Args:
        order_id: The order identifier, e.g. "A1".
    """
    _TOOL_CALLS.append({"tool": "lookup_order", "args": {"order_id": order_id}})
    known = {"A1": "shipped", "B2": "out for delivery", "C3": "processing"}
    return f"order {order_id}: {known.get(order_id, 'not found')}"


@tool
def get_customer_profile(email: str) -> str:
    """Look up a customer profile by their email address.

    Args:
        email: The customer's email address.
    """
    _TOOL_CALLS.append({"tool": "get_customer_profile", "args": {"email": email}})
    return "profile: tier=gold, since=2019"


def tool_calls_made() -> List[Dict[str, Any]]:
    """Which tools actually ran, and with what arguments.

    The tests use this to assert that a blocked tool call never reached the tool body.
    A tool that records its own invocation is the only way to prove that from outside.
    """
    return list(_TOOL_CALLS)


def reset_tool_calls() -> None:
    """Forget the recorded calls — between tests, and before a fresh demo run."""
    _TOOL_CALLS.clear()


# ---- the middleware ----------------------------------------------------------------------

class SecurityGuard(AgentMiddleware):
    """Scans input, tool arguments and output, and stops the run when one fires.

    The detector is a per-request choice, not a per-process one: the same agent serves a
    request checked by the english checkpoint and the next one checked by the decision_lab
    doc head, which is what makes the arms comparable in the UI.
    """

    def __init__(self, guard: Optional[Guard] = None, arm: Optional[Arm] = None) -> None:
        self.guard = guard or Guard()
        self.arm = arm or guard_pii.resolve_arm("english")

    # -- before the model ------------------------------------------------------------------

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: GuardState, runtime: Any) -> Optional[Dict[str, Any]]:
        """Scan the newest human message. A hit ends the run before the model is called.

        This hook runs once per model call, and an agent that calls a tool calls the
        model twice — once to decide to use the tool and once to answer with the tool's
        result. The second call would re-scan the same user message and report it
        twice, which makes the trace look like two checks happened when only one text
        was checked.

        So the scan happens only when a fresh user message is the LAST thing in the
        conversation — which is exactly when it just arrived. Mid-tool-loop the last
        message is a tool result, and the user text was checked on the way in.
        """
        text = _latest_human_text(state)
        if not text:
            return None
        if not _human_message_is_last(state):
            return None
        report = self.guard.scan(text, self.arm, stage=guard_pii.INPUT_STAGE)
        event = {
            "stage": "before_model",
            "arm": self.arm.key,
            "blocked": report.blocked,
            "reason": report.reason,
            "findings": _findings_payload(report),
        }
        if not report.blocked:
            return {"guard_events": [event]}
        return {
            "messages": [AIMessage(content=BLOCKED_ANSWER)],
            "guard_events": [event],
            "blocked": True,
            "jump_to": "end",
        }

    # -- before a tool runs ----------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> Any:
        """Scan the tool's arguments. A hit returns a rejection; the tool never runs.

        The rejection is returned as a Command rather than a bare ToolMessage, for two
        reasons. The ToolMessage still carries the tool_call_id, so the agent's message
        list stays well-formed. And the Command carries a state update recording that a
        tool was blocked, which is what lets after_model end the run.

        That second part matters more than it looks. Returning only the error
        ToolMessage lets the agent loop again: the model sees the failure and answers
        anyway, which on this example means it invents a profile for a customer whose
        details it was never allowed to look up. The blocked tool did not run, so the
        model has nothing to answer with — and the honest answer is the blocked one.
        """
        call = request.tool_call
        name = call.get("name") or "tool"
        arguments = call.get("args") or {}
        text = " ".join(str(value) for value in arguments.values())
        report = self.guard.scan(text, self.arm, stage=guard_pii.TOOL_STAGE)
        event = {
            "stage": "wrap_tool_call",
            "tool": name,
            "arm": self.arm.key,
            "blocked": report.blocked,
            "reason": report.reason,
            "findings": _findings_payload(report),
        }
        if report.blocked:
            # The handler is not called: the tool body does not execute.
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=BLOCKED_ANSWER,
                            tool_call_id=call["id"],
                            name=name,
                            status="error",
                        )
                    ],
                    "tool_blocked": [event],
                }
            )
        result = handler(request)
        # A clean tool call still gets an event. before_model and after_model both
        # report a pass, and a trace that showed the tool stage only when it failed
        # would read as "the tool was never checked".
        #
        # The result MUST be in the update. A Command whose update has no "messages"
        # key DROPS the tool's output: the model never sees it, has nothing to answer
        # with, and returns an empty reply. That is not a subtle degradation — it is
        # the tool silently ceasing to work.
        return Command(update={"messages": [result], "guard_events": [event]})

    # -- after the model -------------------------------------------------------------------

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: GuardState, runtime: Any) -> Optional[Dict[str, Any]]:
        """Scan the model's answer. A hit removes it and ends the run.

        A tool that was blocked ends the run here too. The tool did not execute, so
        the model is answering without the data it asked for — and on this example it
        answers by inventing it. The blocked answer is the honest one.

        The blocked-tool check comes BEFORE the empty-content check, and that order is
        load-bearing. A model that sees a failed tool call often retries the tool and
        then finishes with an EMPTY message — so the empty-content early return would
        skip the blocked-tool branch entirely, and the request would end with an empty
        answer, "not blocked", and a trace that never mentions the tool stage. The
        block would still have happened; the response would just not admit it.
        """
        messages = state.get("messages") or []
        if not messages:
            return None

        blocked_tools = state.get("tool_blocked") or []
        if blocked_tools:
            last = messages[-1]
            # Remove the model's own message, not whatever is last. When the model
            # retried the tool, the last message can be the tool result — and removing
            # that would leave the tool call dangling with no answer, which the next
            # turn would then send upstream as a malformed history.
            updates: List[Any] = []
            if getattr(last, "id", None) is not None and not isinstance(last, ToolMessage):
                updates.append(RemoveMessage(id=last.id))
            updates.append(AIMessage(content=BLOCKED_ANSWER))
            return {
                "messages": updates,
                "guard_events": list(blocked_tools),
                "blocked": True,
                "jump_to": "end",
            }

        last = messages[-1]
        content = _message_text(last)
        if not content:
            return None

        report = self.guard.scan(content, self.arm, stage=guard_pii.OUTPUT_STAGE)
        event = {
            "stage": "after_model",
            "arm": self.arm.key,
            "blocked": report.blocked,
            "reason": report.reason,
            "findings": _findings_payload(report),
        }
        if not report.blocked:
            return {"guard_events": [event]}
        # RemoveMessage is what purges the leaky answer from the history. Without it the
        # secret stays in the message list and the next turn sends it back upstream.
        return {
            "messages": [RemoveMessage(id=last.id), AIMessage(content=BLOCKED_ANSWER)],
            "guard_events": [event],
            "blocked": True,
            "jump_to": "end",
        }

    # -- the model call itself -------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> Any:
        """Pass the model call through, and record that it happened.

        The guard's work happens in the hooks around it, not in the call itself. What
        this hook adds is the "model ran" event, which is what makes "the model was
        never called" a checkable claim rather than an inference from the absence of a
        reply. A blocked input produces a before_model event and no model event, and
        that difference is the evidence.
        """
        response = handler(request)
        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"guard_events": [{"stage": "model", "arm": self.arm.key}]}),
        )


def _latest_human_text(state: GuardState) -> str:
    """The text of the newest human message, or "" if there is none."""
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


def _human_message_is_last(state: GuardState) -> bool:
    """True when the newest human message is also the newest message.

    That is the moment a user message has just arrived and nothing has been said in
    response yet — the only moment the input guard needs to fire. Once the model has
    replied and the agent is in a tool loop, the last message is a tool result and the
    user text has already been checked on the way in.
    """
    messages = state.get("messages") or []
    if not messages:
        return False
    return isinstance(messages[-1], HumanMessage)


def _message_text(message: Any) -> str:
    """A message's text, whatever shape its content takes."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return " ".join(parts)
    return ""


def _findings_payload(report: GuardReport) -> List[Dict[str, Any]]:
    """The findings, with the blocking one flagged.

    The flag comes from the backend rather than the browser re-deriving it: only the
    policy knows which finding crossed a line, and a UI that guessed would mark every
    finding in a blocked turn as the cause.

    Identity, not equality: two identical matches (the same email twice) are equal
    dataclasses, and comparing by value would flag both as the blocker when only the
    first one tripped.
    """
    payload = []
    for finding in report.findings:
        entry = {
            "detector": finding.detector,
            "category": finding.category,
            "score": round(finding.score, 4),
            "detail": finding.detail,
            "blocked": finding is report.blocker,
        }
        payload.append(entry)
    return payload


# ---- building the agent ------------------------------------------------------------------

class GuardedAgent:
    """The compiled agent, and the middleware that guards it.

    The compiled graph does not expose its middleware list — `create_agent` compiles the
    middleware into the graph's nodes and the original objects are not reachable from
    the result. So the guard is held here instead, which is what lets the detector arm
    be swapped per request without rebuilding the agent (and without reloading the chat
    model, which is the expensive part).
    """

    def __init__(self, agent: Any, middleware: SecurityGuard) -> None:
        self.agent = agent
        self.middleware = middleware

    def with_arm(self, arm: Arm) -> "GuardedAgent":
        """Point the guard at a different arm. The chat model is unaffected."""
        self.middleware.arm = arm
        return self


def build_agent(
    model: Optional[str] = None,
    arm: Optional[Arm] = None,
    policy: Optional[GuardPolicy] = None,
    chat_model: Optional[Any] = None,
    guard: Optional[Guard] = None,
) -> GuardedAgent:
    """The guarded agent.

    The chat model comes from GUARD_MODEL (default: deepseek/deepseek-v4.1-flash on
    OpenRouter, which needs OPENROUTER_API_KEY; `GUARD_MODEL=ollama:...` runs it against a
    local model with no key). The detector arm is a separate argument, because the two
    choices are independent: one model checks for PII, another writes the answer.

    `chat_model` and `guard` are injectable so a test can drive the real wiring — the
    same middleware, the same state schema — with a scripted model and a stubbed
    detector instead of a provider and a 421M-parameter checkpoint.
    """
    if chat_model is None:
        from langchain.chat_models import init_chat_model

        chat_model = init_chat_model(
            model or os.environ.get(MODEL_ENV, DEFAULT_MODEL), temperature=0
        )
    guard = guard or Guard(policy=policy)
    middleware = SecurityGuard(guard=guard, arm=arm or guard_pii.resolve_arm("english"))
    agent = create_agent(
        model=chat_model,
        tools=[lookup_order, get_customer_profile],
        middleware=[middleware],
        state_schema=GuardState,
        system_prompt=SYSTEM_PROMPT,
    )
    return GuardedAgent(agent, middleware)


def chat(
    message: str,
    model: Optional[str] = None,
    path: Optional[str] = None,
    agent: Optional[GuardedAgent] = None,
    arm: Optional[Arm] = None,
) -> Dict[str, Any]:
    """Run one guarded turn and return the answer plus what the guard saw.

    The agent is built once and reused; only the detector arm changes per request, so
    switching arms in the UI costs a guard scan and not a model load.

    `arm` may be passed in ALREADY RESOLVED, and the server does exactly that. The
    alternative — resolving again here — meant two more round trips to the
    decision_lab service per request, both inside the server-wide lock, and an
    `ArmError` raised from here landed inside that lock where nothing catches it, so a
    400 became a 500.
    """
    if arm is None:
        arm = guard_pii.resolve_arm(model or "english", path)
    guarded = agent if agent is not None else build_agent(arm=arm)
    guarded.with_arm(arm)
    result = guarded.agent.invoke({"messages": [{"role": "user", "content": message}]})
    messages = result.get("messages") or []
    answer = _message_text(messages[-1]) if messages else ""
    return {
        "answer": answer,
        "blocked": bool(result.get("blocked", False)),
        "arm": arm.key,
        "path": arm.path,
        "guard_events": list(result.get("guard_events") or []),
    }
