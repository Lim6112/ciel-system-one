// The defence ontology: what may be done to a flow, declared as axioms rather than
// written as an `if`.
//
// WHY DECLARE IT. `warehouse_ontology.js` states the reason better than this file can,
// and it is the same reason here: the hand-written predicate was "not wrong, but it
// cannot be checked". Its twin drifted away from it and robots ended up pinned in a
// permanent standoff. Traffic defence has the same shape of risk — "a source over the
// rate limit may not be allowed", "a confirmed secret is not re-inspected" — and left as
// `if`s in `traffic_sim.js` they would drift from what the page enforces, with the same
// result: the UI enforcing a disposition the rule never permitted.
//
// So the rule is declared here, and `divergence()` checks the declaration against a
// hand-written twin over every passage the simulator produces. Without that check this
// file would be a second home for the rule, which is strictly worse than the one home it
// had.
//
// THE AXIOM FORM IS CONSTRAINED, AND THAT SHAPED THE DECLARATIONS. This subset
// implements maxCardinality / minCardinality / range / disjoint — no negation. So "a
// flood source implies Allow is not permitted" cannot be written directly. It is
// expressible as a disjointness, which is the same claim: if the permitted response set
// of a flooding source is {Inspect, Drop, Quarantine}, then Allow is not in it. Same
// move `warehouse_ontology.js` makes for "a static blocker disallows stay".
//
// SCOPE: an OWL-RL subset, closed-world. Absence of a fact is absence, not openness.
//
// THE RESPONSES THEMSELVES LIVE IN traffic_sim.js (it needs them to route a flow), and
// the import runs one way — ontology imports the sim, never the reverse.

import { checkTBox as coreCheckTBox } from "./ontology_core.js";
import { BASE_RESPONSE_SET, RESPONSE, RESPONSE_ORDER } from "./traffic_sim.js";

// ---- vocabulary ------------------------------------------------------------------------

/** The full vocabulary, used to read which side of a disjointness is a response. */
export const RESPONSE_SET = RESPONSE_ORDER;

/** What a flow may be served with before any rule fires. See BASE_RESPONSE_SET. */
export const BASE_SET = BASE_RESPONSE_SET;

/** The two structural classes, and the content classes a scan can produce. */
export const STRUCTURAL_CLASSES = ["FloodRate", "PortSweep"];
export const CONTENT_CLASSES = ["ConfirmedPII", "ConfirmedSecret", "UnverifiedContent"];

export const VOCABULARY = {
  classes: [
    "Flow", "Source", "RepeatSource", "QuarantinedSource",
    "FloodRate", "PortSweep",
    "ConfirmedPII", "ConfirmedSecret", "UnverifiedContent", "CleanContent",
    "Allow", "Inspect", "RateLimit", "Drop", "Quarantine",
    "AllowedFlow", "BlockedFlow", "toServer", "toSink",
  ],
  properties: ["response", "route"],
};

/** The forms this domain's reason() actually evaluates. See ontology_core.js on why this
 *  list is per-domain and deliberately not shared. */
export const IMPLEMENTED_FORMS = ["maxCardinality", "minCardinality", "range", "disjoint"];

// ---- the TBox ---------------------------------------------------------------------------
//
// Declarations, not control flow. Read by checkTBox and by reason(), and editable live —
// removing one disables its check and reason() reports it as unasserted rather than
// crashing, which is the demonstration the editor exists for.

export const TBOX = [
  {
    id: "flood-source-cannot-allow",
    form: "disjoint",
    value: ["FloodRate", "Allow"],
    comment:
      "Tier 0. 속도 윈도우를 넘긴 소스는 전달하지 않는다. 함의(implication)가 아니라 " +
      "서로소로 선언한 이유는 이 부분집합에 부정(negation)이 없기 때문이다 — Allow가 " +
      "대응 분류의 클래스인 이상, \"flood 흐름의 허용 대응 집합은 Allow를 배제한다\"는 " +
      "\"FloodRate disjoint Allow\"와 같은 주장이다. 이 규칙이 flood를 내용 계층에 " +
      "들여보내지 않는다: 어떤 페이로드도 읽기 전에 결정되며, 아무것도 담지 않은 SYN에 " +
      "대해 할 수 있는 일은 그것뿐이다.",
  },
  {
    id: "sweep-source-cannot-allow",
    form: "disjoint",
    value: ["PortSweep", "Allow"],
    comment:
      "Tier 0. 서로 다른 목적지 포트를 훑는 소스는 전달하지 않는다. flood와 같은 형식, " +
      "같은 이유다 — 스윕은 시간에 걸친 형태이므로 텍스트 탐지기가 읽을 페이로드 " +
      "문자열이 없다.",
  },
  {
    id: "confirmed-secret-cannot-allow",
    form: "disjoint",
    value: ["ConfirmedSecret", "Allow"],
    comment:
      "Tier 0. 페이로드에서 provider 키 또는 할당된 자격증명이 발견됐다. regex 스캐너는 " +
      "모든 arm의 모든 stage에서 차단한다 — 매치된 키 형식은 조정할 확률이 아니라 " +
      "텍스트에 대한 사실이기 때문이다.",
  },
  {
    id: "confirmed-pii-cannot-allow",
    form: "disjoint",
    value: ["ConfirmedPII", "Allow"],
    comment:
      "Tier 0. 개인정보가 발견됐다 — 스캐너가 찾았거나, 정책이 PII에 대해 신뢰하는 arm이 " +
      "찾았다. 어느 arm이 그런지, 그리고 나머지가 왜 권고 전용인지는 guard_pii.py의 " +
      "Arm.model_pii_blocks 참고.",
  },
  {
    id: "unverified-content-cannot-allow",
    form: "disjoint",
    value: ["UnverifiedContent", "Allow"],
    comment:
      "Tier 0. Inspect 대응을 필요하게 만드는 항목이다. 내용 계층이 답하지 못했을 때 — " +
      "탐지기에 도달하지 못했거나, 선택된 arm이 PII를 보고하면서도 그것으로 차단할 " +
      "자격이 없을 때 — 그 흐름은 깨끗한 것이 아니라 검사되지 않은 것이다. " +
      "guard_pii.py가 자기 자신의 unavailable finding에 대해 같은 말을 한다: " +
      "\"checked, nothing found\"와 \"could not check\"를 구분하지 못하는 호출자는 " +
      "사각지대를 청정 판정으로 읽는다. Allow는 철회되고, Inspect는 아니다.",
  },
  {
    id: "confirmed-secret-cannot-inspect",
    form: "disjoint",
    value: ["ConfirmedSecret", "Inspect"],
    comment:
      "Tier 0이며, Inspect가 Allow의 별칭이 아니라 실제 계층인 이유다. 확인된 비밀은 " +
      "이미 찾아냈으므로 \"더 자세히 보자\"는 처분이 아니다 — 그것은 무언가를 한 것처럼 " +
      "보이면서 키를 통과시키는 방법이다.",
  },
  {
    id: "quarantined-source-holds",
    form: "disjoint",
    value: ["QuarantinedSource", "Allow"],
    comment:
      "Tier 0. 격리된 소스는 앞에 놓인 패킷의 형태가 어떻든 전달하지 않는다. 이 파일에서 " +
      "가장 강한 규칙이며, 의도적으로 흐름이 아니라 소스에 관한 것이다 — 소스가 한 번 " +
      "격리되면 그 이후의 모든 패킷은 형태를 고려하기도 전에 붙잡힌다. 이 공리가 없으면 " +
      "격리된 소스의 무해해 보이는 후속 패킷이 전체 대응 집합을 달고 도착해 허용으로 " +
      "읽히며, 그것이 바로 격리가 풀리는 방식이다.",
  },
  {
    id: "first-offence-not-dropped",
    form: "disjoint",
    value: ["FirstOffenceSource", "Drop"],
    comment:
      "Tier 0이며, 아래의 escalation을 장식이 아니라 측정 가능하게 만드는 공리다. 첫 위반 " +
      "소스는 drop이 아니라 rate-limit이다 — 값싼 대응을 비싼 대응보다 먼저 시도한다. " +
      "이것이 없으면 첫 위반에도 이미 Drop이 허용되므로, 반복 위반 소스에 Drop을 " +
      "부여해도 아무것도 달라지지 않고 escalation 공리는 무효가 된다 — 결정을 한 번도 " +
      "바꾸지 않는 규칙을 선언하는 셈이다. 두 공리는 의도적으로 서로 다른 대응을 " +
      "겨냥한다(여기서는 Drop, 아래에서는 Drop+Quarantine). 그래야 부여에 더할 것이 " +
      "남는다. 타입에 주목: 첫 위반은 FirstOffenceSource, 반복은 RepeatSource로 " +
      "타입되므로, 이 공리는 escalation이 작동하는 바로 그 순간 적용을 멈춘다.",
  },
  {
    id: "repeat-offender-admits-drop",
    form: "range",
    subject: "RepeatSource",
    property: "response",
    value: [RESPONSE.DROP, RESPONSE.QUARANTINE],
    comment:
      "부여(grant)로서의 Tier 0이며, 그래서 양방향으로 검사한다. 이미 흐름이 차단된 적 " +
      "있는 소스는 승격된다: Drop과 Quarantine이 그 소스에 허용되는 대응에 추가된다. " +
      "warehouse의 deadlocked-robot-may-reverse와 같은 형태다 — 그 파일에서 철회가 아니라 " +
      "부여를 하는 유일한 공리다. 부여가 틀릴 수 있는 방법은 두 가지다: 적용돼야 할 때 " +
      "보류되는 것(escalation이 발동하지 않음), 그리고 적용되면 안 될 때 적용되는 것" +
      "(첫 위반이 반복으로 승격됨). reason() 참고.",
  },
  {
    id: "flow-has-response",
    form: "minCardinality",
    subject: "Flow",
    property: "response",
    value: 1,
    comment:
      "Tier 0. 규칙은 어떤 대응도 전부 제거해서는 안 된다. 합법적 처분이 하나도 없는 " +
      "흐름은 차단된 흐름이 아니라 처리되지 않은 흐름이며, 이 공리가 그것을 아무도 읽지 " +
      "않는 폴백에 맡기는 대신 명시한다.",
  },
  {
    id: "allow-means-forward",
    form: "range",
    subject: "AllowedFlow",
    property: "route",
    value: ["toServer"],
    comment:
      "Tier 0이며, 이 온톨로지와 페이지가 그리는 흐름 그래프를 잇는 공리다. Allow만 " +
      "흐름을 서버로 전달하고, 나머지 대응은 모두 격리소로 보낸다. 규칙은 Allow라고 " +
      "결론했는데 페이지가 패킷을 격리소로 그리는 것은 이 파일이 보고 가능하게 만들려는 " +
      "바로 그 drift이므로, 렌더러가 알아서 맞추도록 두지 않고 선언한다.",
  },
];

/**
 * Which member of a disjointness is the condition and which is the response it excludes.
 *
 * Read from the axiom's own value rather than from a table in the reasoner, so editing an
 * axiom in the live panel changes what is enforced — an earlier version kept a parallel
 * table of "which axiom withdraws which response", and editing the axiom then did nothing
 * while the table silently went on applying the old rule.
 *
 * Returns `{condition, excluded}`, either of which is null if the pair names no response.
 */
export function disjointSides(rule) {
  const [a, b] = rule.value;
  const aIsResponse = RESPONSE_SET.includes(a);
  const bIsResponse = RESPONSE_SET.includes(b);
  if (bIsResponse && !aIsResponse) return { condition: a, excluded: b };
  if (aIsResponse && !bIsResponse) return { condition: b, excluded: a };
  return { condition: null, excluded: null };
}

export const axiom = id => TBOX.find(a => a.id === id);

export { describeAxiom } from "./ontology_core.js";

export const checkTBox = (tbox = TBOX) => coreCheckTBox(tbox, IMPLEMENTED_FORMS);

/**
 * The declared axiom set, captured at module load.
 *
 * The reasoner iterates THIS rather than the live `TBOX`, so that deleting an axiom in the
 * editor is observable: asking the live array "which axioms apply?" cannot notice a removed
 * one, because the removed axiom is absent from the question. Against this copy, an axiom
 * that is declared but not asserted is reported in `unasserted` and its rule stops applying.
 */
const PRISTINE = TBOX.map(a => ({ ...a, value: Array.isArray(a.value) ? [...a.value] : a.value }));

// ---- the ABox ---------------------------------------------------------------------------

/** The class a flow's content facts put it in. */
function contentType(content) {
  if (!content) return null;
  if (content.unavailable) return "UnverifiedContent";
  if (content.secret) return "ConfirmedSecret";
  if (content.pii) return "ConfirmedPII";
  return "CleanContent";
}

/**
 * The structural class, if the structural layer fired.
 *
 * Quarantine is deliberately NOT one: it is a property of the SOURCE (already judged, held
 * for every subsequent packet) rather than a fact about this flow's shape. Folding it in
 * here is what made a quarantined source come back with the full response set and read as
 * allowed — see `quarantined-source-holds`.
 */
function structuralType(structural) {
  if (!structural || !structural.blocked) return null;
  if (structural.rule === "PortSweep") return "PortSweep";
  if (structural.rule === "FloodRate") return "FloodRate";
  return null;
}

/**
 * Build the ABox for one passage.
 *
 * `description` is what traffic_sim.js knows plus what the content scan returned:
 *
 *   { sourceId, kind, quarantined, priorViolations,
 *     structural: {blocked, rule, detail, score},
 *     content: null | {unavailable, secret, pii, blockingFindings, arm, armModelPiiBlocks} }
 */
export function buildResponseABox(description) {
  const structural = structuralType(description.structural);
  const content = contentType(description.content);
  const repeat = (description.priorViolations || 0) > 0;

  const individuals = new Map();
  individuals.set("source", {
    id: description.sourceId,
    type: description.quarantined ? "QuarantinedSource"
      : repeat ? "RepeatSource"
      : "FirstOffenceSource",
    quarantined: !!description.quarantined,
    repeat,
    violations: description.priorViolations || 0,
  });
  if (structural) {
    individuals.set("structural", { id: description.structural.rule, type: structural });
  }
  if (content) {
    individuals.set("content", { id: "content", type: content });
  }
  individuals.set("flow", {
    id: `flow-${description.sourceId}`,
    type: "Flow",
    // THE ASSERTED SET IS THE HAND-WRITTEN PREDICATE'S ANSWER. This is the arrangement
    // warehouse_ontology uses and it is what makes the violation check mean anything:
    // the ABox is built from the hand-written rule, and the axioms are then evaluated
    // against it. If the two disagree, reason() reports it — so the declaration is
    // checked against the code it replaces on every single passage, not only in the
    // divergence harness.
    asserted: new Set(handResponse(description)),
    response: new Set(),
    route: new Set(["toSink"]),
    allowed: false,
  });
  return {
    individuals,
    derived: { structuralType: structural, contentType: content, repeat },
  };
}

// ---- the reasoner -------------------------------------------------------------------------

/**
 * Evaluate the TBox against an ABox.
 *
 * Returns {violations, unasserted, derived} where `derived.permitted` is the response set
 * and `derived.route` is where the flow goes. `unasserted` names any axiom that is absent
 * from the TBox — deleting one in the live editor disables its check and is reported here
 * rather than silently changing behaviour.
 */
export function reason(facts) {
  const violations = [];
  const unasserted = [];
  const flow = facts.individuals.get("flow");
  const source = facts.individuals.get("source");
  const structural = facts.individuals.get("structural");
  const content = facts.individuals.get("content");

  const ruleFor = (id) => {
    const rule = axiom(id);
    if (!rule) {
      if (!unasserted.includes(id)) unasserted.push(id);
      return null;
    }
    return rule;
  };

  // ---- apply the rules -------------------------------------------------------------
  //
  // DRIVEN BY THE TBOX, NOT BY A TABLE HERE. Each disjointness names a condition class
  // and the response it excludes; the class that condition holds is looked up among the
  // ABox individuals. An earlier version kept a parallel table of "which axiom withdraws
  // which response", which is exactly the second home for the rule this file exists to
  // remove — editing the axiom then changed nothing, and the divergence check caught it.

  // Every individual by type, so a disjointness can find whoever holds its condition.
  const holders = new Map();
  for (const individual of facts.individuals.values()) {
    if (individual.type) holders.set(individual.type, individual);
  }

  const applied = [];
  // The base set, not the full vocabulary: Drop and Quarantine are escalation-only.
  const permitted = new Set(BASE_SET);
  // Iterate the LIVE TBox, so editing an axiom's value in the panel takes effect
  // immediately. (An earlier version iterated a pristine copy, which meant the value
  // shown in the editor and the value enforced could differ — the editor would save a
  // rule that changed nothing.) PRISTINE is used only below, to notice REMOVALS: asking
  // the live array "which axioms apply?" cannot notice a deleted one, because the deleted
  // axiom is absent from the question.
  for (const rule of [...TBOX]) {
    if (rule.form !== "disjoint") continue;
    const { condition, excluded } = disjointSides(rule);
    if (!excluded) continue;
    // The condition holds if some individual in the ABox carries its type. A quarantined
    // source is typed QuarantinedSource in buildResponseABox, so it needs no special case.
    if (!holders.has(condition)) continue;
    if (!permitted.has(excluded)) continue;   // already withdrawn by an earlier rule
    permitted.delete(excluded);
    applied.push({ axiom: rule.id, removed: excluded });
  }

  // Which declared axioms are no longer asserted, asked against the pristine set.
  for (const declared of PRISTINE) {
    if (!axiom(declared.id) && !unasserted.includes(declared.id)) unasserted.push(declared.id);
  }

  // The grant: a repeat source MAY be served Drop or Quarantine. A grant only ever ADDS,
  // and it is permissive — a first offence is not thereby forbidden those responses, it
  // simply has not earned the escalation. The reverse direction is checked below.
  const grant = ruleFor("repeat-offender-admits-drop");
  // Measured against the WITHDRAWALS, not against the fact that the source repeated. The
  // two differ whenever an axiom is edited in the live panel, and the question the page
  // asks is "did the escalation change what this flow may be served with?".
  const beforeGrant = new Set(permitted);
  if (grant && source.repeat) {
    for (const r of grant.value) permitted.add(r);
  }
  const escalated = [...permitted].some(r => !beforeGrant.has(r));

  // The floor: the rule may never empty the set. Restored to the BASE set rather than the
  // full vocabulary, so a rule that removed everything cannot be rescued by handing the
  // flow a Drop it never earned.
  const floor = ruleFor("flow-has-response");
  if (floor && permitted.size < floor.value) {
    for (const r of BASE_SET) permitted.add(r);
  }

  // ---- check the declaration against the hand-written predicate ----------------------
  //
  // THE POINT OF THE WHOLE LAYER. The ABox asserted what the hand-written rule concluded;
  // the axioms above computed what the DECLARED rule concludes. If the two differ, the
  // declaration does not faithfully reimplement the code, and that is reported here — on
  // every passage, not only in the test harness. A non-empty list is the signal that an
  // axiom and the predicate it replaced have drifted.
  const asserted = [...flow.asserted].sort().join("|");
  const computed = [...permitted].sort().join("|");
  if (flow.asserted.size && asserted !== computed) {
    violations.push({
      axiom: "disjoint",
      subject: flow.id,
      detail: `선언된 공리는 [${[...permitted].join(", ")}]를 허용하지만 손으로 쓴 술어는 ` +
              `[${[...flow.asserted].join(", ")}]를 주장한다`,
      kind: "declaration-drift",
    });
  }

  const allowed = permitted.has(RESPONSE.ALLOW);
  flow.response = permitted;
  flow.allowed = allowed;

  // ---- check the axioms against that set --------------------------------------------
  //
  // A violation is the state CONTRADICTING a declared axiom, not the axiom firing. This
  // is the distinction warehouse_ontology draws: it reports "blocked by a static obstacle
  // but stay is permitted", never "a static obstacle withdrew stay".

  // disjoint A ⊥ B: A holds, and the permitted set intersects B.
  //
  // Checked as the disjointness it is declared as — the permitted set must not intersect
  // what the axiom separates off — rather than by re-deriving the condition. That is what
  // makes this able to catch BOTH failure directions with one rule: an applied axiom whose
  // side effect went missing, and a deleted axiom whose withdrawal is still happening. It
  // reads what the rule produced, not what the rule should have read.
  for (const rule of TBOX) {
    if (rule.form !== "disjoint") continue;
    if (!axiom(rule.id)) continue;
    const { condition, excluded } = disjointSides(rule);
    if (!excluded) continue;
    if (!holders.has(condition)) continue;
    if (permitted.has(excluded)) {
      violations.push({
        axiom: rule.id,
        subject: flow.id,
        detail: `${condition} 이(가) 성립하지만 ${excluded} 이(가) 여전히 허용됨 — ` +
                `"${rule.value.join(" ⊥ ")}" 이(가) 모순됨`,
        kind: "disjoint-violated",
      });
    }
  }

  // range: RepeatSource ⊑ response ∈ [Drop, Quarantine] — the REVERSE direction of the
  // grant, and the reason a grant needs checking both ways. Because the grant is
  // permissive it never narrows, so over the shipped TBox every response is already
  // available and this cannot fire. It becomes reachable the moment a withdrawal narrows
  // the set for a repeat source — i.e. when an axiom is edited in the live panel — which
  // is exactly when "is this source actually being escalated?" is worth hearing about.
  if (grant && source.repeat && permitted.size < RESPONSE_SET.length) {
    const earned = new Set(grant.value);
    const escalatedHere = [...permitted].some(r => earned.has(r));
    if (!escalatedHere) {
      violations.push({
        axiom: grant.id,
        subject: flow.id,
        detail: `반복 위반 소스(차단된 흐름 ${source.violations}건)가 ` +
                `${[...earned].join(", ")} 중 어느 것도 받지 못했으므로 escalation이 ` +
                `적용되지 않았다`,
        kind: "grant-not-applied",
      });
    }
  }

  // range: AllowedFlow ⊑ route ∈ [toServer]. Only Allow forwards.
  const routeAxiom = ruleFor("allow-means-forward");
  const route = new Set();
  if (allowed) route.add("toServer");
  route.add("toSink");
  flow.route = route;

  if (routeAxiom && allowed) {
    const admitted = new Set(routeAxiom.value);
    if (!route.has("toServer") || !admitted.has("toServer")) {
      violations.push({
        axiom: routeAxiom.id,
        subject: flow.id,
        detail: `흐름이 허용되었지만 서버로 라우팅되지 않았다; 이 공리가 허용하는 것은 ` +
                `${[...admitted].join(", ")}`,
        kind: "route",
      });
    }
  }

  return {
    violations,
    unasserted,
    applied,
    derived: {
      permitted: RESPONSE_SET.filter(r => permitted.has(r)),
      response: permitted,
      route,
      allowed,
      escalated,
      structuralType: facts.derived.structuralType,
      contentType: facts.derived.contentType,
      repeat: facts.derived.repeat,
    },
    applied,
  };
}

/**
 * The permitted response set, as the decision loop sees it.
 *
 * Falls back to the full set if a rule ever emptied it — a flow is never left with no
 * legal disposition, which is the same contract `warehouse_ontology.permittedActions`
 * keeps.
 */
export function permittedResponses(description) {
  const facts = buildResponseABox(description);
  const verdict = reason(facts);
  return {
    options: verdict.derived.permitted.length ? verdict.derived.permitted : [...RESPONSE_SET],
    // A flow with the whole BASE set has nothing to decide, and the page uses this to skip
    // the model call entirely.
    //
    // SET EQUALITY, NOT A LENGTH COMPARISON. Comparing counts was a real bug, and a quiet
    // one: a confirmed secret yields {RateLimit, Drop, Quarantine} — three members, the
    // same count as the base set — so a key_leak was reported as "no rule fired" and
    // allowed. The sizes matching was a coincidence of how the withdrawals compose, and a
    // coincidental match is exactly what a length check cannot tell from the real thing.
    undecided: verdict.derived.permitted.length === BASE_SET.length &&
      verdict.derived.permitted.every(r => BASE_SET.includes(r)),
    route: [...verdict.derived.route],
    allowed: verdict.derived.allowed,
    // Which withdrawals actually applied, and whether the escalation changed the set —
    // so the panel can show WHY the menu narrowed rather than only that it did.
    applied: verdict.applied,
    escalated: verdict.derived.escalated,
    repeat: verdict.derived.repeat,
    structuralType: verdict.derived.structuralType,
    contentType: verdict.derived.contentType,
    violations: verdict.violations,
    unasserted: verdict.unasserted,
  };
}

// ---- the hand-written twin, and the check that they agree ---------------------------------

/**
 * The response set, written by hand.
 *
 * This is what the ontology replaces, kept so the two can be compared. It is deliberately
 * a plain if-chain — the point is that it CANNOT be wrong in a way you can detect, which
 * is why the declared version is the one that ships.
 */
export function handResponse(description) {
  const structural = structuralType(description.structural);
  const content = contentType(description.content);
  const repeat = (description.priorViolations || 0) > 0;

  const out = new Set(BASE_SET);
  if (description.quarantined) out.delete(RESPONSE.ALLOW);
  if (structural === "FloodRate" || structural === "PortSweep") out.delete(RESPONSE.ALLOW);
  if (content === "ConfirmedSecret" || content === "ConfirmedPII" ||
      content === "UnverifiedContent") {
    out.delete(RESPONSE.ALLOW);
  }
  if (content === "ConfirmedSecret") out.delete(RESPONSE.INSPECT);
  // The escalation: destroying traffic is only reachable once a source has already had a
  // flow blocked. This is what makes the grant axiom do work rather than sit inert.
  if (repeat) { out.add(RESPONSE.DROP); out.add(RESPONSE.QUARANTINE); }
  if (out.size === 0) for (const r of BASE_SET) out.add(r);
  return RESPONSE_ORDER.filter(r => out.has(r));
}

/**
 * Does the declared rule agree with the hand-written predicate it replaces?
 *
 * The warehouse demo's divergence harness, in the shape this domain needs: run both over
 * every passage the simulator can produce and compare. Zero disagreements is the claim;
 * a non-zero count means the declaration does NOT faithfully reimplement the code, and the
 * number to report is the disagreement count, not "it looks right".
 *
 * This is the check that makes the ontology worth having — see the header.
 */
export function divergence(cases) {
  const broken = [];
  let compared = 0;
  for (const description of cases) {
    compared++;
    const declared = permittedResponses(description).options.slice().sort();
    const hand = handResponse(description).slice().sort();
    if (declared.join("|") !== hand.join("|")) {
      broken.push({
        source: description.sourceId,
        kind: description.kind,
        structural: description.structural?.rule ?? null,
        contentType: contentType(description.content),
        declared,
        hand,
      });
    }
  }
  return { compared, disagreements: broken.length, broken };
}

/** One row describing what the ontology currently believes, for the panel. */
export function aboxFor(description) {
  if (!description) return [];
  const verdict = permittedResponses(description);
  const conditions = [verdict.structuralType, verdict.contentType].filter(Boolean);
  if (description.quarantined) conditions.unshift("QuarantinedSource");
  else if (verdict.repeat) conditions.unshift("RepeatSource");
  return [{
    id: description.sourceId,
    detail: conditions.length
      ? `${conditions.join(" + ")} · permitted ${verdict.options.join("/")}` +
        `${verdict.allowed ? " → 서버로 전달" : " → 가드에서 보류"}`
      : "발동한 규칙 없음 · 기본 집합 전체가 허용되므로 흐름은 기본적으로 허용",
    violations: verdict.violations,
    unasserted: verdict.unasserted,
  }];
}
