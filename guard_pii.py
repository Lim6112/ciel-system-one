"""The detectors behind the security guard: what counts as PII, and who decides.

A guard is only as good as the thing it delegates to, so this module's first job is to
be honest about which of our models can actually answer the question. Measured on
2026-10-22, asking the `sensitive_data` question of each arm on the same text — one
containing a name, an email, an SSN, a card and a phone number — against a clean one:

    laya english           0.843   vs  0.099 clean   <- separates
    laya multilingual      0.973   vs  0.001 clean   <- separates
    laya typed-decisions   0.588   vs  0.197 clean   <- weak; not trained on PII
    decision_lab doc       0.496   vs  0.180 clean   <- weak; blind on prose
    decision_lab typed     0.171   vs  0.430 clean   <- ANTI-correlated

The decision_lab `doc` head is sharp on its own synthetic template (0.955 vs 0.001) and
blind on prose, because its PII latent was trained on one contact line the generator
emits ("You can reach {name} at {email} or {phone}."). A template matcher, not a PII
detector. The `typed` head scores the clean text HIGHER than the PII text, so blocking on
its opinion would reject the harmless request and pass the harmful one.

So the arms stay selectable — the difference is the interesting part — but only the laya
english and multilingual checkpoints may block on their PII opinion, and only on the
input stage. The laya typed-decisions checkpoint is advisory too: on 20 ordinary
prompts it puts 2 over the 0.3 threshold where english puts 0 over. The regex scanner
blocks on every arm, in every stage.

THREE THINGS THIS MODULE DELIBERATELY DOES NOT CLAIM.

First, a model can say *whether* PII is present, not *which span* leaked. The regex
scanner is what locates spans; the model is what catches what a regex cannot see. Neither
is sufficient alone.

Second, the jailbreak latent is not safe to block on. It scores 0.999 on "Create a prompt
template for a customer support bot" and 0.881 on "write a Fibonacci function", while real
attacks score 0.943-1.000. The bands overlap, so no threshold separates them. It is
computed and reported; it does not block unless GUARD_BLOCK_JAILBREAK says so.

Third, the detector is a *model arm*, chosen per request, the way the other pages in this
repo choose one. The arms disagree — that is the point of showing them side by side.

The deterministic half — the scanner that finds the spans a model cannot point at —
lives in guard_scan.py, along with the vocabulary of what the guard finds. This module
is the model half: the questions, the arms, the detectors and the policy.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# `path` is a field name on Arm and a parameter of resolve_arm, so the stdlib module is
# imported under an alias to keep the two from colliding.
import urllib.error
import urllib.request as urllib_request

import ciel
import jev

from laya.presets import guard_questions

from guard_scan import (
    CREDENTIAL_CATEGORY,
    INJECTION_CATEGORY,
    JAILBREAK_CATEGORY,
    PII_CATEGORY,
    SECRET_CATEGORY,
    UNAVAILABLE_CATEGORY,
    Finding,
    SecretScanner,
)

# ---- the questions the model arms are asked ---------------------------------------------

# laya's own guard preset. Using the shipped phrasing rather than inventing one is not
# a detail: these latents were trained against this wording, and a paraphrase degrades the
# score silently rather than raising.
#
# Only the noul questions are asked. The preset also carries harm_severity (a score) and
# topic (a choice), and neither has a "probability of true" to threshold — asking them
# anyway would put meaningless zeros in the guard trace and invite a reader to trust a
# number that was never a probability.
GUARD_QUESTIONS: Dict[str, Dict[str, Any]] = {
    qid: question
    for qid, question in guard_questions().items()
    if question.get("type") == "noul"
}

# The PII question for the service-backed arms, in the exact wording decision_lab's
# question bank uses for `contains_pii`. The canonical phrasing is not a formality
# there: the head was trained against this wording, and an invented paraphrase
# degrades the score silently instead of raising.
PII_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "sensitive_data": {
        "type": "noul",
        "instructions": "Does this state contain personally identifiable information?",
    },
}

# The two typed-decision questions the security_incidents workflow asks. Used by the
# service-backed arm, which serves the typed-decisions head.
SECURITY_INCIDENT_QUESTIONS: Dict[str, Dict[str, Any]] = {
    "credential_compromise": {
        "type": "noul",
        "instructions": "The evidence indicates a credential or account has been compromised.",
    },
    "true_positive": {
        "type": "noul",
        "instructions": "This alert reflects genuinely malicious or unauthorised activity.",
    },
}

# Thresholds, from the sweep in the README (15 PII / 15 clean prompts on the english arm).
# sensitive_data scored 0.146-1.000 on PII and 0.0001-0.156 on clean, so 0.3 has no false
# positives on that set and still catches email, name, phone, ssn, card and key.
PII_THRESHOLD = 0.3
JAILBREAK_THRESHOLD = 0.95
INJECTION_THRESHOLD = 0.95

# The checkpoint the guard loads up front. The English one, because it is the arm that
# separates on PII (0.843 against 0.099 for clean text) and because it is the smallest
# download of the three. The others load on demand, when a request asks for them.
DEFAULT_LAYA_MODEL = "english"

class GuardError(RuntimeError):
    """A guard failure the caller should surface rather than swallow."""


def _ensure_loopback_bypass() -> None:
    """Keep this process's loopback calls off any ambient HTTP proxy.

    This machine sets http_proxy, which urllib (ciel.py) and httpx (LangChain's Ollama
    client) both honour — so a request to 127.0.0.1 is sent to the proxy and comes
    back as "connection reset", which reads like the service is down rather than like a
    routing mistake.

    This is a property of the machine the guard runs on, not of ciel.py, so it is fixed
    here — in the module that actually makes the loopback calls — rather than in the
    shared client. On a machine with no proxy set, no_proxy simply gains two hosts and
    nothing else changes. It is process-local, so no other module is affected.
    """
    hosts = [h.strip() for h in os.environ.get("no_proxy", "").split(",") if h.strip()]
    for host in ("127.0.0.1", "localhost"):
        if host not in hosts:
            hosts.append(host)
    os.environ["no_proxy"] = ",".join(hosts)


_ensure_loopback_bypass()


# The three places the guard runs, and what each one is looking at.
INPUT_STAGE = "input"    # the user's message, before the model sees it
TOOL_STAGE = "tool"      # a tool's arguments, before the tool runs
OUTPUT_STAGE = "output"  # the model's answer, before the user sees it


@dataclass(frozen=True)
class GuardPolicy:
    """Which findings block, which are reported, and where the lines are.

    `block_on_jailbreak` is off by default and that is a measurement, not a preference:
    the jailbreak latent's false-positive band (0.999 on a legitimate prompt-engineering
    request) overlaps its true-positive band (0.943 on a real attack), so blocking on it
    rejects ordinary requests. GUARD_BLOCK_JAILBREAK=1 turns it on for a deployment that
    would rather over-block.

    `model_pii_blocks_on` says where the MODEL's PII opinion is allowed to block. It is
    the input stage only, and that is a measurement too. The same latent that separates
    cleanly on user prose (PII 0.146-1.000, clean 0.0001-0.156) does not separate at all
    on the other two stages:

        model replies    clean 0.001-0.653   leaky 0.265-0.861   (overlapping)
        tool arguments   clean 0.071-0.257   leaky 0.163-0.906   (overlapping)

    Blocking on it there would reject "I cannot reveal my system prompt" (0.653) while
    missing a leaked card number (0.163). The regex scanner, by contrast, separates in
    all three stages — 0 false positives and every leak caught — so it blocks everywhere
    and the model's opinion is reported everywhere.
    """

    pii_threshold: float = PII_THRESHOLD
    jailbreak_threshold: float = JAILBREAK_THRESHOLD
    injection_threshold: float = INJECTION_THRESHOLD
    block_on_pii: bool = True
    block_on_secret: bool = True
    block_on_jailbreak: bool = False
    # Its own switch, not folded into block_on_jailbreak: the two latents have
    # different false-positive behaviour and a deployment that trusts one has no
    # reason to trust the other.
    block_on_injection: bool = False
    model_pii_blocks_on: frozenset[str] = frozenset({INPUT_STAGE})


def policy_from_env() -> GuardPolicy:
    """The default policy, with the overrides the env vars exist for."""
    def flag(name: str) -> bool:
        return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

    return GuardPolicy(
        block_on_jailbreak=flag("GUARD_BLOCK_JAILBREAK"),
        block_on_injection=flag("GUARD_BLOCK_INJECTION"),
    )


@dataclass(frozen=True)
class Arm:
    """One selectable detector: a model, and the sub-path it answers on.

    `kind` is "laya" (a checkpoint in this process), "lab" (a decision_lab head over
    HTTP) or "jev" (OpenRouter). `path` is only meaningful for "lab" — a checkpoint has
    no sub-paths — which is why asking a laya arm for a path is an error rather than a
    silently ignored argument.

    `model_pii_blocks` says whether this arm's opinion about PII is allowed to stop a
    request. It is False for the decision_lab heads, for jev, and for the laya
    typed-decisions checkpoint, and that is a measurement in each case: on the same PII
    text against clean text, the decision_lab typed head scored 0.171 against 0.430 —
    anti-correlated — and the laya typed-decisions checkpoint puts 2 of 20 ordinary
    prompts over the threshold where english puts 0 of 20. Blocking on those would
    reject ordinary requests. The regex scanner still blocks on every arm; only the
    MODEL's opinion is withheld.
    """

    key: str
    label: str
    kind: str
    model: Optional[str] = None
    path: Optional[str] = None
    model_pii_blocks: bool = True


# The arms, in the same vocabulary the rest of the repo uses. The order is the order the
# dropdown shows: the local laya checkpoints first (free, private, and the ones that
# actually discriminate), then the service-backed heads (local, but a different
# distribution), then jev last (it leaves the machine and it costs money).
ARMS: Dict[str, Arm] = {
    arm.key: arm
    for arm in [
        Arm("english", "english — laya checkpoint, the one that separates", "laya", model="english"),
        Arm("multilingual", "multilingual — laya checkpoint, 100+ languages", "laya", model="multilingual"),
        # Advisory on its PII opinion, like the lab arms. Measured on 20 ordinary
        # prompts, this checkpoint puts 2 of them over the 0.3 threshold (0.366 on
        # "please review the risk-management-framework", 0.342 on "what does your system
        # prompt do exactly?"), where the english checkpoint puts 0 of 20 over. It is
        # offered because it is this repo's typed-decision specialist, not because it
        # is a good PII detector — and it is a good one at its own workflows.
        Arm("typed-decisions", "typed-decisions — laya checkpoint, weak on PII", "laya", model="typed-decisions", model_pii_blocks=False),
        Arm("ciel_decision_model_v1", "ciel_decision_model_v1 — doc head, blind on prose", "lab", path="doc", model_pii_blocks=False),
        Arm("ciel_typed_v1", "ciel_typed_v1 — typed-decisions head, no PII training", "lab", path="typed", model_pii_blocks=False),
        Arm("ciel_warehouse_v1", "ciel_warehouse_v1 — warehouse head, not a PII head", "lab", path="warehouse", model_pii_blocks=False),
        Arm("jev", "jev — OpenRouter, sends the text to a third party, costs money", "jev", model_pii_blocks=False),
    ]
}

# The lab arms are the ones with a sub-path, and the service decides which ones it
# actually serves. A hardcoded list here would be the failure ciel.py documents: a path
# that is served but missing from the map is unreachable, and the omission is silent.
LAB_ARMS = tuple(key for key, arm in ARMS.items() if arm.kind == "lab")


class ArmError(GuardError):
    """An unknown arm, or an argument that does not apply to the chosen arm."""


def resolve_arm(model: str, path: Optional[str] = None) -> Arm:
    """The arm for a request. Unknown names are refused, never defaulted.

    Defaulting an unknown model to the first arm would mean a typo silently gets
    checked by a different detector than the one the caller asked for — the block would
    still look like it worked, which is the worst kind of wrong.
    """
    key = (model or "").strip()
    if key not in ARMS:
        raise ArmError(f"unknown model {model!r}; choose one of {sorted(ARMS)}")
    arm = ARMS[key]
    if path and arm.kind != "lab":
        raise ArmError(f"model {key!r} has no sub-paths; only {list(LAB_ARMS)} do")
    if arm.kind == "lab" and path:
        known = lab_paths()
        if not known:
            # The service is down, which is not the caller's fault and not a bad path.
            # Saying "unknown path" here would blame the caller for an upstream outage.
            raise ArmError(
                "decision_lab is not reachable, so its sub-paths cannot be listed; "
                f"start it, or use a laya arm"
            )
        if path not in known:
            raise ArmError(f"unknown path {path!r}; the service serves {known}")
        return Arm(
            arm.key, arm.label, arm.kind, path=path, model_pii_blocks=arm.model_pii_blocks
        )
    return arm


def lab_paths() -> List[str]:
    """The sub-paths the decision_lab service actually serves, right now.

    Read live rather than hardcoded, so the dropdown cannot drift from the service.

    NEVER raises. Every caller wants the same thing from a service that is down or
    misbehaving — an empty list, meaning "no sub-paths are available" — and this
    function is called from the health endpoint, from arm resolution and from the
    dropdown. Raising here took the health endpoint down with a 500 and turned a
    misbehaving service into a traceback in the CLI, for no gain: the sub-path list is
    a convenience, and the guard's real work is done by the scanner and the laya arm.
    A service that answers with something other than the expected shape is reported
    the same way — as "no paths" — and the arm resolution that follows says the service
    is not reachable, which is the truth the user can act on.

    The URL comes from ciel.base_url(), which is this repo's single reader of
    DECISION_LAB_URL (see ciel.py's docstring on why there is exactly one). Nothing
    else here knows the env var.
    """
    try:
        with urllib_request.urlopen(f"{ciel.base_url()}/api/paths", timeout=5.0) as response:
            body = json.load(response)
        paths = body.get("paths")
        if not isinstance(paths, list):
            return []
        return [str(p) for p in paths]
    except Exception:
        # OSError covers ConnectionRefusedError, which is what a service that is not
        # running raises — it is a subclass of OSError, not of URLError. ValueError
        # covers a non-JSON body. AttributeError covers a body that is not an object.
        return []


# ---- the model-backed arms -------------------------------------------------------------


class _HubOnline:
    """Lift HF_HUB_OFFLINE for the duration of a model load, then restore it.

    The same pattern asr.py uses, for the same reason: laya_startup.py sets
    HF_HUB_OFFLINE=1 so that laya never reaches for the network, and that flag would
    also stop the first download of a checkpoint dead. Restoring it matters — the flag
    is a global the rest of the repo relies on.

    The lock is PROCESS-WIDE, not per-instance, and that is the point. The flag is a
    global, so two overlapping windows would restore it wrongly: A enters (saves "1",
    unsets it), B enters (saves nothing, because the flag is already unset), A exits
    (restores "1"), B exits (unsets it, because B never saw it set) — and the flag is
    left unset, so laya starts reaching for the network. asr.py has a module-global
    lock for exactly this reason.
    """

    _ENV = "HF_HUB_OFFLINE"
    _LOCK = threading.Lock()

    def __init__(self) -> None:
        self._saved: Optional[str] = None
        self._was_set = False

    def __enter__(self) -> "_HubOnline":
        self._LOCK.acquire()
        self._saved = os.environ.get(self._ENV)
        self._was_set = self._ENV in os.environ
        os.environ.pop(self._ENV, None)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._was_set:
            os.environ[self._ENV] = self._saved if self._saved is not None else "1"
        else:
            os.environ.pop(self._ENV, None)
        self._LOCK.release()


class LayaDetector:
    """PII and attack detection by a laya checkpoint, in this process.

    Wraps laya's Router rather than loading a checkpoint directly, because the Router
    already resolves the model id, already holds the loaded agents, and already has an
    LRU. `max_loaded=3` keeps all three checkpoints resident so switching arms in the UI
    does not pay a 30-second reload each time.

    The router is built lazily. A guard server that preloads a 421M-parameter model at
    import time cannot be imported by a test, and there is no reason to pay for the
    checkpoint until a request actually arrives.
    """

    name = "laya"

    def __init__(self, router: Any = None) -> None:
        """`router` is an already-built laya Router to scan with, if the caller has one.

        The web server hands over the Router it preloaded for /api/predict. That is the
        whole point of the parameter: the process then holds ONE Router and one copy of
        each checkpoint, rather than two Routers each holding a 421M-parameter english
        (which is also why the first guard turn used to pay a ~30s load while holding the
        server's device lock).

        Reuse is exact, not approximate: `Router(preload=True)` has already made every
        model in laya's `DEFAULT_MODELS` resident, and the guard's laya arms ask for
        exactly those three — english, multilingual, typed-decisions.

        Left unset (the CLI), the detector builds its own lazily, exactly as before.
        """
        self._router: Any = None
        self._lock = threading.Lock()
        if router is not None:
            self._router = router

    def _get_router(self, model: str = DEFAULT_LAYA_MODEL) -> Any:
        """The router: the injected one if there is one, else built on first use.

        The checkpoint is loaded HERE, inside the online window, rather than lazily at
        the first prediction. That ordering matters: `Router(preload=False)` builds
        nothing at construction and defers the download to the first `predict()`, by
        which time the offline flag has been restored — so a machine that has never
        downloaded the checkpoint would fail with "outgoing traffic has been
        disabled". Loading it eagerly inside the window makes the first run work and
        still leaves the flag set for every run after it.

        The checkpoint that gets loaded is the one that was ASKED for, not just the
        default. `Router.load` is lazy, so a request for the multilingual arm on a
        machine with HF_HUB_OFFLINE=1 and no cached multilingual checkpoint would
        otherwise fail at the first prediction — outside this window, and with the
        flag restored.

        An injected router skips all of that: its own startup already went through the
        online window, and it is preloaded, so there is nothing to load or to guard.
        """
        if self._router is None:
            with self._lock:
                if self._router is None:
                    from laya.router import Router

                    with _HubOnline():
                        router = Router(preload=False, max_loaded=3)
                        router.load(DEFAULT_LAYA_MODEL)
                    self._router = router
        return self._router

    def _ensure_loaded(self, router: Any, model: str) -> None:
        """Load a checkpoint inside the online window if it is not resident yet."""
        if model in router.loaded:
            return
        with _HubOnline():
            router.load(model)

    def loaded_models(self) -> List[str]:
        """Which checkpoints are resident — for the health endpoint."""
        return list(self._router.loaded) if self._router is not None else []

    def scan(self, text: str, model: str = DEFAULT_LAYA_MODEL) -> List[Finding]:
        if not text.strip():
            return []
        try:
            router = self._get_router(model)
            self._ensure_loaded(router, model)
            result = router.predict({"prompt": text}, GUARD_QUESTIONS, model=model)
        except Exception as exc:
            # A detector that cannot run is a GAP IN COVERAGE, not a clean bill of
            # health. Reporting it as an unavailable finding — rather than letting the
            # exception escape and turn every request into a 500 — is what lets the
            # trace say "this was not checked" instead of "nothing was found".
            return [
                Finding(
                    self.name,
                    UNAVAILABLE_CATEGORY,
                    0.0,
                    f"the laya detector could not run: {type(exc).__name__}",
                )
            ]
        answers = result.get("answers") or {}
        findings: List[Finding] = []
        for qid, answer in answers.items():
            score = float(answer.get("noul", 0.0))
            findings.append(Finding(self.name, _category_for(qid), score, qid))
        return findings


class DecisionLabDetector:
    """PII and credential detection by a decision_lab head, over HTTP.

    Reuses ciel.py rather than writing a second client for the same service: it already
    speaks the laya typed-question contract, already maps the service's answer shape
    back, and already raises CielError with the status and body a server turns into a
    502.

    Both question sets go upstream in one call, so the arms stay comparable: the PII
    question (so a lab head can be scored on the same axis as a laya checkpoint) and
    the typed-decisions security questions (which only the typed head was trained on).

    Advisory only. The typed head was never trained on PII — measured 0.171 on PII text
    against 0.430 on clean, which is ANTI-correlated — so a finding here is reported
    and does not block. The doc head is sharper on its own template format and still
    blind on prose (0.496 on PII text against 0.180 on clean).
    """

    name = "decision_lab"

    def scan(self, text: str, path: str = "typed") -> List[Finding]:
        if not text.strip():
            return []
        questions = {**PII_QUESTIONS, **SECURITY_INCIDENT_QUESTIONS}
        try:
            result = ciel.predict(
                {"prompt": text},
                questions,
                model=_lab_model_for(path),
            )
        except (ciel.CielError, ValueError, AttributeError, TypeError, KeyError) as exc:
            # The service being down is not a guard finding, but it is not "clean"
            # either. It is a gap in coverage, and the trace has to say so — a caller
            # that cannot tell "checked, nothing found" from "could not check" will
            # read a blind spot as a clean bill of health.
            #
            # The wider exception set matters: ciel.py raises a bare ValueError for a
            # non-JSON body, and an unexpected response shape raises AttributeError.
            # Those are the same situation as an unreachable service — the detector
            # could not produce an answer.
            return [
                Finding(
                    self.name,
                    UNAVAILABLE_CATEGORY,
                    0.0,
                    f"decision_lab could not be reached: {type(exc).__name__}",
                )
            ]
        answers = result.get("answers") or {}
        findings: List[Finding] = []
        for qid, answer in answers.items():
            score = float(answer.get("noul", 0.0))
            if qid in PII_QUESTIONS:
                findings.append(Finding(self.name, PII_CATEGORY, score, qid))
            else:
                findings.append(Finding(self.name, CREDENTIAL_CATEGORY, score, qid))
        return findings


class JevDetector:
    """Detection by OpenRouter, through the existing jev.py client.

    Offered for parity with the other pages, and labelled honestly: jev sends the text
    to a third party, which for a PII guard means sending the PII itself, and it costs
    money per call. Every other arm is local.
    """

    name = "jev"

    def scan(self, text: str) -> List[Finding]:
        if not text.strip():
            return []
        try:
            result = jev.predict({"prompt": text}, GUARD_QUESTIONS)
        except jev.JevError:
            return [
                Finding(self.name, UNAVAILABLE_CATEGORY, 0.0, "jev is unreachable")
            ]
        answers = result.get("answers") or {}
        return [
            Finding(self.name, _category_for(qid), float(a.get("noul", 0.0)), qid)
            for qid, a in answers.items()
        ]


# decision_lab serves one head per sub-path, and ciel.py's MODEL_PATHS maps a model name
# to one. The guard's sub-path dropdown offers each head directly, so the mapping here is
# the inverse of ciel.py's.
_LAB_MODEL_BY_PATH = {arm.path: key for key, arm in ARMS.items() if arm.kind == "lab"}


def _lab_model_for(path: str) -> str:
    """The ciel.py model name for a sub-path.

    A path with no mapping is an ERROR, not a fallback. The service can be configured
    with heads this map does not know about, and silently scoring such a head with the
    typed one would report a different detector's answer as if it were the requested
    one — the same silent-wrong-head failure ciel.py's own docstring warns about.
    """
    model = _LAB_MODEL_BY_PATH.get(path)
    if model is None:
        raise ArmError(
            f"no arm is mapped to the {path!r} sub-path; known: {sorted(_LAB_MODEL_BY_PATH)}"
        )
    return model


def _category_for(qid: str) -> str:
    """The guard category a question id belongs to.

    An unknown id is NOT PII. The fallback used to be, which meant that if a question
    id ever reached this function that the guard did not recognise, it would be
    treated as personal data and could block the request. A question the guard does
    not understand is one it has no opinion about, and that is not the same as PII.
    """
    if qid == "sensitive_data":
        return PII_CATEGORY
    if qid == "jailbreak":
        return JAILBREAK_CATEGORY
    if qid == "prompt_injection":
        return INJECTION_CATEGORY
    return "unknown"


# ---- the guard itself ------------------------------------------------------------------

# One answer for every block. It does not name the finding, because telling the model
# which rule it tripped teaches it which rule to avoid — and telling the browser leaks
# the shape of the filter for no benefit to the user.
BLOCKED_ANSWER = (
    "This request was blocked by the security guard and was not processed."
)


@dataclass
class GuardReport:
    """What the guard saw, and what it decided."""

    findings: List[Finding] = field(default_factory=list)
    blocked: bool = False
    reason: str = ""
    # Whether this arm's opinion about PII is allowed to block at all. The lab and jev
    # arms are not: their PII scores are anti-correlated, and blocking on them would
    # reject the clean request and pass the harmful one.
    arm_model_pii_blocks: bool = True
    # The finding that actually stopped the run, so the UI can mark it rather than
    # guessing which one in the list was responsible.
    blocker: Optional[Finding] = None


class Guard:
    """Runs the detectors for one arm and decides whether the text may proceed.

    The regex scanner always runs, whatever the arm: a matched secret format is a fact
    about the text, not a calibrated probability, so it is not the arm's to overrule.
    """

    def __init__(
        self,
        policy: Optional[GuardPolicy] = None,
        router: Any = None,
    ) -> None:
        """`router` is an already-built laya Router to scan with, if the process has one.

        The web server passes its own, so the process holds one Router and one copy of
        each checkpoint instead of the detector building a duplicate 421M-parameter
        english. Left unset (the CLI), the detector builds its own lazily, as before.
        """
        self.policy = policy or policy_from_env()
        self.scanner = SecretScanner()
        self.laya = LayaDetector(router=router)
        self.lab = DecisionLabDetector()
        self.jev = JevDetector()

    def scan(self, text: str, arm: Arm, stage: str = INPUT_STAGE) -> GuardReport:
        """Run every applicable detector over `text` and decide.

        `stage` is one of input, tool or output. It decides whether the MODEL's PII
        opinion is allowed to block — see GuardPolicy for the measurement behind that.
        """
        report = GuardReport(arm_model_pii_blocks=arm.model_pii_blocks)
        report.findings.extend(self.scanner.scan(text))

        if arm.kind == "laya":
            report.findings.extend(self.laya.scan(text, model=arm.model or "english"))
        elif arm.kind == "lab":
            report.findings.extend(self.lab.scan(text, path=arm.path or "typed"))
        elif arm.kind == "jev":
            report.findings.extend(self.jev.scan(text))

        self._decide(report, stage)
        return report

    def _decide(self, report: GuardReport, stage: str) -> None:
        """Apply the policy to the findings. First match wins; the reason is recorded."""
        policy = self.policy
        for finding in report.findings:
            if finding.category == SECRET_CATEGORY and policy.block_on_secret:
                report.blocked = True
                report.blocker = finding
                report.reason = "a secret pattern was found in the text"
                return
            if finding.category == PII_CATEGORY and policy.block_on_pii:
                if finding.detector == self.scanner.name:
                    # The regex scanner separates in every stage, so it blocks in every
                    # stage. A matched format is a fact, not a probability to tune.
                    report.blocked = True
                    report.blocker = finding
                    report.reason = "a personal identifier was found in the text"
                    return
                # The MODEL's opinion. It may block only where it separates (the input
                # stage, on user prose) and only for an arm whose opinion is trustworthy
                # on PII at all.
                if (
                    stage in policy.model_pii_blocks_on
                    and report.arm_model_pii_blocks
                    and finding.score >= policy.pii_threshold
                ):
                    report.blocked = True
                    report.blocker = finding
                    report.reason = (
                        f"the {finding.detector} detector scored this text "
                        f"{finding.score:.2f} for personal data"
                    )
                    return
            if finding.category == JAILBREAK_CATEGORY and policy.block_on_jailbreak:
                if finding.score >= policy.jailbreak_threshold:
                    report.blocked = True
                    report.blocker = finding
                    report.reason = (
                        f"the {finding.detector} detector scored this text "
                        f"{finding.score:.2f} for a jailbreak attempt"
                    )
                    return
            if finding.category == INJECTION_CATEGORY and policy.block_on_injection:
                if finding.score >= policy.injection_threshold:
                    report.blocked = True
                    report.blocker = finding
                    report.reason = (
                        f"the {finding.detector} detector scored this text "
                        f"{finding.score:.2f} for a prompt injection"
                    )
                    return
