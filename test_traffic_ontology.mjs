// Tests for the traffic defence ontology.
//
// THE LOAD-BEARING TEST HERE IS THE DIVERGENCE ONE. Declaring a rule that also exists as
// code is only worth anything if the declaration and the code are checked against each
// other — an unchecked second home for the rule is strictly worse than the one home it
// had. That is not hypothetical in the sibling domain: the two halves of the warehouse
// blocker rule lived in separate functions, drifted, and pinned robots in a permanent
// head-on standoff.
//
// So `test_the_declaration_agrees_with_the_hand_written_predicate` runs the declared
// reasoner and the hand-written predicate over every passage shape the simulator can
// produce and requires zero disagreements.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  BASE_SET,
  IMPLEMENTED_FORMS,
  RESPONSE_SET,
  TBOX,
  aboxFor,
  checkTBox,
  divergence,
  handResponse,
  permittedResponses,
  reason,
  buildResponseABox,
} from "./static/traffic_ontology.js";
import { RESPONSE, createSim, spawn, step, counters, TRAFFIC_KINDS } from "./static/traffic_sim.js";

// A passage shape, with every field defaulted so a test names only what it is about.
const passage = (over = {}) => ({
  sourceId: "c1",
  kind: "order_lookup",
  quarantined: false,
  priorViolations: 0,
  structural: { blocked: false, rule: null, detail: "", score: 0 },
  content: null,
  ...over,
});

const flood = over => passage({ structural: { blocked: true, rule: "FloodRate", detail: "x", score: 1 }, ...over });
const sweep = over => passage({ structural: { blocked: true, rule: "PortSweep", detail: "x", score: 1 }, ...over });
const holds = over => passage({ structural: { blocked: true, rule: "Quarantine", detail: "x", score: 1 }, ...over });

/** Every passage shape the simulator can produce, as a cross product. */
function everyPassageShape() {
  const shapes = [];
  const structurals = [
    { blocked: false, rule: null },
    { blocked: true, rule: "FloodRate" },
    { blocked: true, rule: "PortSweep" },
    { blocked: true, rule: "Quarantine" },
  ];
  const contents = [null, { secret: true }, { pii: true }, { unavailable: true }];
  for (const structural of structurals) {
    for (const content of contents) {
      for (const priorViolations of [0, 1, 3]) {
        for (const quarantined of [false, true]) {
          shapes.push(passage({ structural, content, priorViolations, quarantined }));
        }
      }
    }
  }
  return shapes;
}

// ---- the TBox ---------------------------------------------------------------------------

test("the TBox is coherent", () => {
  const status = checkTBox();
  assert.equal(status.satisfiable, true,
    `unsatisfiable: ${JSON.stringify(status.unsatisfiable)}`);
});

test("every declared form is one this domain implements", () => {
  // An axiom in a form nothing implements reads as enforced and is inert — the failure
  // ontology_core.checkTBox exists to reject.
  for (const axiom of TBOX) {
    assert.ok(IMPLEMENTED_FORMS.includes(axiom.form),
      `${axiom.id} declares form ${axiom.form}, which this domain does not implement`);
  }
});

test("every axiom carries a comment explaining what it replaces", () => {
  for (const axiom of TBOX) {
    assert.ok(axiom.comment && axiom.comment.length > 40,
      `${axiom.id} has no substantive comment`);
  }
});

// ---- the divergence check ------------------------------------------------------------------

test("the declaration agrees with the hand-written predicate", () => {
  const result = divergence(everyPassageShape());
  assert.equal(result.disagreements, 0,
    `declared and hand-written rules disagree:\n${JSON.stringify(result.broken, null, 1)}`);
  assert.ok(result.compared >= 96, `only ${result.compared} shapes compared`);
});

test("the divergence check can actually fail", () => {
  // A check that cannot fail proves nothing. Perturb the hand-written twin and confirm
  // the comparison notices — otherwise a future edit that broke agreement would pass.
  const original = TBOX.find(a => a.id === "flood-source-cannot-allow");
  const saved = [...original.value];
  try {
    // Point the axiom at a DIFFERENT response, so it stops withdrawing Allow. Inspect is
    // already withdrawn for some shapes, which would mask the change.
    original.value = ["FloodRate", "RateLimit"];
    const result = divergence(everyPassageShape());
    assert.ok(result.disagreements > 0,
      "breaking an axiom did not produce a divergence, so the check is not running");
  } finally {
    original.value = saved;
  }
});

// ---- the individual axioms -------------------------------------------------------------------

test("a flood source is never permitted to allow", () => {
  const verdict = permittedResponses(flood());
  assert.ok(!verdict.options.includes(RESPONSE.ALLOW), `flood permitted ${verdict.options}`);
  assert.equal(verdict.allowed, false);
  assert.equal(verdict.structuralType, "FloodRate");
});

test("a swept source is never permitted to allow", () => {
  const verdict = permittedResponses(sweep());
  assert.ok(!verdict.options.includes(RESPONSE.ALLOW));
  assert.equal(verdict.structuralType, "PortSweep");
});

test("a quarantined source is held whatever the packet looks like", () => {
  // The rule is about the SOURCE, not the flow: a benign-looking follow-up from a
  // quarantined source must not read as allowed.
  const verdict = permittedResponses(holds({ quarantined: true, priorViolations: 4 }));
  assert.ok(!verdict.options.includes(RESPONSE.ALLOW),
    `a quarantined source was permitted ${verdict.options}`);
  assert.equal(verdict.allowed, false);
});

test("a confirmed secret is never permitted to inspect", () => {
  // Inspect would be a way to let a confirmed key through while appearing to act.
  const verdict = permittedResponses(passage({ content: { secret: true } }));
  assert.ok(!verdict.options.includes(RESPONSE.ALLOW));
  assert.ok(!verdict.options.includes(RESPONSE.INSPECT),
    `a confirmed secret was permitted Inspect: ${verdict.options}`);
});

test("a confirmed pii payload cannot be allowed", () => {
  const verdict = permittedResponses(passage({ content: { pii: true } }));
  assert.ok(!verdict.options.includes(RESPONSE.ALLOW));
});

test("unverified content cannot allow but CAN inspect", () => {
  // The link between the guard's own "could not check ≠ clean" contract and the response
  // vocabulary. Withdrawing Allow is the point; withdrawing Inspect would leave an
  // unchecked flow with no disposition that reflects being unchecked.
  const verdict = permittedResponses(passage({ content: { unavailable: true } }));
  assert.ok(!verdict.options.includes(RESPONSE.ALLOW),
    "an unchecked payload was permitted Allow, so a blind spot reads as clean");
  assert.ok(verdict.options.includes(RESPONSE.INSPECT),
    "an unchecked payload cannot be Inspected, so the tier has no purpose");
  assert.equal(verdict.contentType, "UnverifiedContent");
});

test("a clean packet has the whole base set and nothing to decide", () => {
  // Undecided means "no rule fired", which is what lets the page skip the model call.
  // The base set is Allow/Inspect/RateLimit — Drop and Quarantine are escalation-only and
  // are therefore NOT available to a source that has done nothing.
  const verdict = permittedResponses(passage());
  assert.deepEqual([...verdict.options].sort(), [...BASE_SET].sort());
  assert.equal(verdict.options.length, BASE_SET.length);
  assert.equal(verdict.undecided, true,
    "a packet with no fired rule reports a decision, so the model would be called for nothing");
});

test("a response set of the same SIZE is still not undecided", () => {
  // The regression, in the exact shape it took. A confirmed secret from a REPEAT source
  // yields {RateLimit, Drop, Quarantine} — three members, the same count as the base set
  // {Allow, Inspect, RateLimit} — because the grant adds back what the withdrawals took.
  // A length-based "did anything fire?" check read that as undecided and let a leaked key
  // through, which is what happened on the page before this was fixed.
  const secret = permittedResponses(passage({ content: { secret: true }, priorViolations: 2 }));
  assert.equal(secret.options.length, BASE_SET.length,
    "this test is only meaningful while the sizes happen to coincide");
  assert.notDeepEqual([...secret.options].sort(), [...BASE_SET].sort());
  assert.equal(secret.undecided, false,
    "a confirmed secret was reported as undecided, so the flow would be allowed");
  assert.ok(!secret.options.includes(RESPONSE.ALLOW));
});

test("destroying traffic requires having already blocked a flow", () => {
  // Drop and Quarantine are escalation-only. A first offence may be rate-limited; only a
  // source that has already had a flow blocked may be dropped or quarantined.
  const first = permittedResponses(flood());
  assert.ok(!first.options.includes(RESPONSE.DROP), "a first offence could be dropped");
  assert.ok(!first.options.includes(RESPONSE.QUARANTINE), "a first offence could be quarantined");
  assert.ok(first.options.includes(RESPONSE.RATE_LIMIT), "a first offence cannot be rate-limited");

  const repeat = permittedResponses(flood({ priorViolations: 2 }));
  assert.ok(repeat.options.includes(RESPONSE.DROP), "a repeat offender could not be dropped");
  assert.ok(repeat.options.includes(RESPONSE.QUARANTINE));
});

test("every blocked flow keeps at least one response", () => {
  // minCardinality: the rule may never empty the set. Checked over every shape, because
  // the way this breaks is a new withdrawal nobody noticed was total.
  for (const shape of everyPassageShape()) {
    const verdict = permittedResponses(shape);
    assert.ok(verdict.options.length >= 1,
      `no legal disposition for ${JSON.stringify(shape)}`);
  }
});

test("an allowed flow is routed to the server", () => {
  const verdict = permittedResponses(passage());
  assert.equal(verdict.allowed, true);
  assert.ok(verdict.route.includes("toServer"), `allowed but routed ${verdict.route}`);
});

test("a blocked flow is not routed to the server", () => {
  for (const shape of [flood(), sweep(), passage({ content: { secret: true } })]) {
    const verdict = permittedResponses(shape);
    assert.equal(verdict.allowed, false);
    assert.ok(!verdict.route.includes("toServer"),
      `blocked flow was routed to the server: ${verdict.route}`);
  }
});

// ---- escalation ------------------------------------------------------------------------------

test("a repeat offender may be dropped or quarantined", () => {
  const verdict = permittedResponses(flood({ priorViolations: 2 }));
  assert.ok(verdict.options.includes(RESPONSE.DROP));
  assert.ok(verdict.options.includes(RESPONSE.QUARANTINE));
  assert.equal(verdict.escalated, true);
});

test("a first offence is not marked as escalated", () => {
  const verdict = permittedResponses(flood({ priorViolations: 0 }));
  assert.equal(verdict.escalated, false,
    "a first offence was reported as an escalation");
});

test("escalation only ever ADDS, so it cannot unblock a flooded source", () => {
  // The grant must not interact with the withdrawals to reopen Allow.
  const verdict = permittedResponses(flood({ priorViolations: 9 }));
  assert.ok(!verdict.options.includes(RESPONSE.ALLOW),
    "escalating a repeat offender let Allow back in");
});

// ---- live editing ------------------------------------------------------------------------------

test("removing an axiom reports it unasserted rather than crashing", () => {
  // The contract the live editor depends on: deleting a declaration disables its check
  // and says so, instead of throwing on the next packet.
  const rule = TBOX.find(a => a.id === "flood-source-cannot-allow");
  const index = TBOX.indexOf(rule);
  TBOX.splice(index, 1);
  try {
    const verdict = permittedResponses(flood());
    assert.deepEqual(verdict.unasserted, ["flood-source-cannot-allow"],
      `expected the removal to be reported, got ${JSON.stringify(verdict.unasserted)}`);
    assert.ok(verdict.options.includes(RESPONSE.ALLOW),
      "the withdrawal still applied after its axiom was removed");
  } finally {
    TBOX.splice(index, 0, rule);
  }
});

test("restoring the axiom restores the rule", () => {
  const verdict = permittedResponses(flood());
  assert.ok(!verdict.options.includes(RESPONSE.ALLOW));
  assert.deepEqual(verdict.unasserted, []);
});

// ---- the reasoner's violation reporting ----------------------------------------------------------

test("a healthy passage reports no violations", () => {
  for (const shape of everyPassageShape()) {
    const verdict = permittedResponses(shape);
    assert.equal(verdict.violations.length, 0,
      `a normal passage reported ${JSON.stringify(verdict.violations)}`);
  }
});

test("a declaration that disagrees with the predicate is reported as a violation", () => {
  // The drift the file exists to catch. The ABox asserts what the hand-written rule
  // concluded; perturb that assertion so the declared axioms no longer agree with it.
  // This is the same comparison the divergence harness makes, available on every passage.
  const facts = buildResponseABox(flood());
  facts.individuals.get("flow").asserted.add(RESPONSE.ALLOW);
  const result = reason(facts);
  const hit = result.violations.find(v => v.kind === "declaration-drift");
  assert.ok(hit, `expected drift to be reported, got ${JSON.stringify(result.violations)}`);
});

test("the abox row names the fired class, the permitted set, and the escalation", () => {
  // The panel is presented in Korean; the CLASS NAMES stay as the vocabulary spells them,
  // because they are the ontology's own terms rather than prose. So these assertions are
  // on the identifiers, which is also what keeps them meaningful across a translation.
  const rows = aboxFor(flood({ sourceId: "a1" }));
  assert.equal(rows.length, 1);
  assert.match(rows[0].detail, /FloodRate/, "the fired class is not named");
  assert.match(rows[0].detail, /Inspect/, "the permitted set is not shown");
  assert.match(rows[0].detail, /보류/, "the route is not shown");

  const escalated = aboxFor(flood({ sourceId: "a1", priorViolations: 3 }));
  assert.match(escalated[0].detail, /RepeatSource/, "a repeat offender is not identified");
  assert.match(escalated[0].detail, /Drop/, "the escalation is not shown");
});

test("the abox row says so plainly when nothing has fired", () => {
  const rows = aboxFor(passage());
  assert.match(rows[0].detail, /발동한 규칙 없음/);
});

test("aboxFor tolerates no passage at all", () => {
  assert.deepEqual(aboxFor(null), []);
});

// ---- against the real simulator --------------------------------------------------------------------

test("every passage the simulator produces agrees with the hand-written predicate", () => {
  // The divergence check over REAL traffic rather than a synthetic cross product: drive
  // the sim across seeds that produce floods, sweeps, PII, keys and clears, and compare
  // on each packet that reaches the guard.
  let compared = 0;
  for (const seed of [1, 3, 5, 7, 11, 42, 99]) {
    const sim = createSim({ seed, config: { packetsPerSecond: 45 } });
    for (let i = 0; i < 700; i++) {
      step(sim, 16);
      for (const arrival of sim.arrivals) {
        const description = {
          sourceId: arrival.sourceId,
          kind: arrival.kind,
          quarantined: sim.quarantine.has(arrival.sourceId),
          priorViolations: sim.violations.get(arrival.sourceId) || 0,
          structural: arrival.structural,
          content: null,
        };
        const declared = permittedResponses(description).options.slice().sort().join("|");
        const hand = handResponse(description).slice().sort().join("|");
        assert.equal(declared, hand,
          `seed ${seed} packet ${arrival.id} (${arrival.kind}): ${declared} vs ${hand}`);
        compared++;
      }
    }
  }
  assert.ok(compared > 50, `only ${compared} hits the guard sat through`);
});

test("a flood never reaches the content tier, so the queue stays empty", () => {
  const sim = createSim({ seed: 17, config: { packetsPerSecond: 80 } });
  for (let i = 0; i < 200; i++) spawn(sim, TRAFFIC_KINDS.FLOOD, "a1");
  for (let i = 0; i < 400; i++) step(sim, 16);
  assert.equal(sim.queue.length, 0, "a structurally blocked packet entered the scan queue");
  assert.ok(counters(sim).dropped > 0, "the flood was not actually blocked");
});
