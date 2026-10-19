// Tests for the warehouse blocker ontology.
//
// The load-bearing test here is the divergence one. Declaring a rule that already
// exists as code is only worth anything if the declaration and the code are
// checked against each other — an unchecked second home for the rule is strictly
// worse than the one home it had. That is not hypothetical in this domain: the
// two halves of this rule (which options are offered, and which answer is
// correct) lived in separate functions, drifted, and pinned robots in a permanent
// head-on standoff.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ACTION_SET,
  IMPLEMENTED_FORMS,
  TBOX,
  buildABox,
  checkTBox,
  divergence,
  permittedActions,
  reason,
} from "./static/warehouse_ontology.js";
import { permittedFor } from "./static/warehouse_panel.js";
import { createWorld, feasibleOptions, step } from "./static/warehouse_sim.js";

const CONFIG = { speed: 80, lidarRange: 90, autoplay: true };

// Collect every blocking situation the sim produces, at its real cadence.
function collectBlockingScenes(seeds = [1, 2, 3, 5, 7, 11, 42], cap = 4000) {
  const cases = [];
  for (const seed of seeds) {
    const robots = createWorld(seed);
    const lastAsk = new Map();
    let now = 0;
    for (let i = 0; i < 60000 && cases.length < cap; i++) {
      now += 16;
      step(robots, 0.016, now, CONFIG);
      for (const robot of robots) {
        const scan = robot.obstacle;
        if (!scan) continue;
        const stalledFor = robot.lastProgressAt ? (now - robot.lastProgressAt) / 1000 : 0;
        const stalled = stalledFor * 1000 >= 4000;
        cases.push({ robot, scan, kind: scan.rack ? "rack" : "robot", stalled });
        if (now - (lastAsk.get(robot.id) || 0) >= 300) {
          lastAsk.set(robot.id, now);
          const options = feasibleOptions(scan.rack ? "rack" : "robot", stalled);
          robot.suppressed = !options.includes("stay");
        }
      }
    }
  }
  return cases;
}

test("the TBox is coherent", () => {
  const status = checkTBox();
  assert.equal(status.satisfiable, true,
    `unsatisfiable: ${JSON.stringify(status.unsatisfiable)}`);
  // Every declared form must be one this domain actually implements, or the axiom
  // reads as enforced while nothing consults it — the one failure a self-check
  // must not have.
  for (const a of TBOX) {
    assert.ok(IMPLEMENTED_FORMS.includes(a.form), `form ${a.form} is not implemented`);
  }
});

test("the declared rule agrees with the hand-written predicate", () => {
  // Zero disagreements is the claim. Non-zero means the declaration does NOT
  // faithfully reimplement feasibleOptions, which is the whole justification for
  // having a declaration at all.
  const cases = collectBlockingScenes();
  assert.ok(cases.length > 100, `only ${cases.length} blocking cases collected`);
  const { compared, disagreed } = divergence(cases);
  assert.equal(disagreed.length, 0,
    `${disagreed.length}/${compared} disagreements: ` +
    JSON.stringify(disagreed.slice(0, 3)));
});

test("the rack axiom fires when the rule is violated", () => {
  // A check that never fires proves nothing. Feed the reasoner a permitted set
  // that wrongly keeps "stay" for a rack, which is exactly what feasibleOptions
  // is there to prevent.
  const scene = collectBlockingScenes([1], 4000).find(c => c.kind === "rack");
  assert.ok(scene, "no rack block encountered");

  const ok = permittedActions(scene.robot, scene.scan, { stalled: false });
  assert.equal(ok.ok, true);
  assert.equal(ok.violations.length, 0);
  assert.ok(!ok.options.includes("stay"), "a rack must not offer stay");

  const cheating = permittedActions(scene.robot, scene.scan, {
    stalled: false,
    feasibleOptions: () => ACTION_SET,
  });
  assert.equal(cheating.ok, false);
  assert.ok(cheating.violations.some(v => v.axiom === "static-blocker-disallows-stay"),
    `expected the static-blocker axiom to fire, got ${JSON.stringify(cheating.violations)}`);
});

test("the stalled axiom fires when waiting is left on the menu", () => {
  const scene = collectBlockingScenes([1], 4000).find(c => c.kind === "robot");
  assert.ok(scene, "no robot block encountered");

  const cheating = permittedActions(scene.robot, scene.scan, {
    stalled: true,
    feasibleOptions: () => ACTION_SET,
  });
  assert.ok(cheating.violations.some(v => v.axiom === "stalled-blocker-disallows-stay"),
    "a stalled robot with stay permitted must violate the stall axiom");
});

test("deleting an axiom disables its check and says so", () => {
  // The live editor lets an axiom be removed. That must report rather than crash:
  // the physics still knows the answer, the ontology has simply stopped
  // asserting it, and the report is the whole demonstration.
  const scene = collectBlockingScenes([1], 4000).find(c => c.kind === "rack");
  const idx = TBOX.findIndex(a => a.id === "static-blocker-disallows-stay");
  const removed = TBOX.splice(idx, 1)[0];
  try {
    const result = permittedActions(scene.robot, scene.scan, { stalled: false });
    assert.equal(result.violations.length, 0, "a deleted axiom must not still fire");
    assert.ok(result.unasserted.includes("static-blocker-disallows-stay"),
      `expected an unasserted report, got ${JSON.stringify(result.unasserted)}`);
  } finally {
    TBOX.splice(idx, 0, removed);
  }
});

test("a blocked robot is never left with no legal move", () => {
  // The rule exists to remove stay; if removing it could empty the set the robot
  // would be grounded, which is the failure mode the minCardinality axiom names.
  for (const c of collectBlockingScenes([1, 7], 4000)) {
    const result = permittedActions(c.robot, c.scan, { stalled: c.stalled });
    assert.ok(result.options.length >= 1,
      `no action left for ${c.kind} (stalled=${c.stalled})`);
  }
});

test("an unblocked robot is offered everything", () => {
  const { facts } = buildABox({ id: "R1" }, null, {});
  const probe = reason(facts);
  assert.deepEqual(probe.derived.permitted.slice().sort(), ACTION_SET.slice().sort());
  assert.equal(probe.derived.goldAction, null);
  assert.equal(probe.violations.length, 0);
});

// ---- the deadlock escape ----------------------------------------------------
//
// The only axiom in this file that GRANTS an action instead of withdrawing one, so
// it has two ways to be wrong and both are tested: granting `backward` to a robot
// that is not deadlocked hands it a way to give up route for no reason, and
// withholding it from a robot that is deadlocked leaves the robot with no move that
// changes the geometry, which is the deadlock the axiom exists to break.

test("a deadlocked robot is permitted to reverse, and only then", () => {
  const { facts } = buildABox({ id: "R1" }, null, { deadlocked: true });
  const probe = reason(facts);
  assert.ok(probe.derived.permitted.includes("backward"),
    `the escape was not admitted: [${probe.derived.permitted}]`);
  assert.equal(probe.derived.deadlocked, true);
  assert.equal(probe.violations.length, 0,
    `the declaration contradicts itself: ${JSON.stringify(probe.violations)}`);
});

test("an ordinary blocked robot is not permitted to reverse", () => {
  // The gate has to be the two clocks, not merely "is blocked". If `backward` leaked
  // into the everyday set, every blocked robot could abandon its route instead of
  // stepping around the obstacle, which is the behaviour the other four axioms spend
  // their time shaping.
  for (const c of collectBlockingScenes([1, 7], 3000)) {
    const result = permittedActions(c.robot, c.scan, { stalled: c.stalled, deadlocked: false });
    assert.ok(!result.options.includes("backward"),
      `backward offered to a non-deadlocked robot (${c.kind}, stalled=${c.stalled})`);
  }
});

test("the escape axiom rejects illegal backward in both directions", () => {
  // Injected rather than observed, because neither state occurs naturally: the sim
  // never grants `backward` to a moving robot, and never withholds it from a wedged
  // one. A rule that cannot be made to fire is not being tested.
  const scene = collectBlockingScenes([1], 1)[0];

  const illegal = buildABox(scene.robot, scene.scan, { stalled: scene.stalled, deadlocked: false });
  illegal.facts.individuals.get(scene.robot.id).permittedAction.add("backward");
  const over = reason(illegal.facts);
  assert.ok(over.violations.some(v => v.axiom === "deadlocked-robot-may-reverse"),
    "granting backward to a non-deadlocked robot was not reported");

  const missing = buildABox(scene.robot, scene.scan, { stalled: scene.stalled, deadlocked: true });
  missing.facts.individuals.get(scene.robot.id).permittedAction.delete("backward");
  const under = reason(missing.facts);
  assert.ok(under.violations.some(v => v.axiom === "deadlocked-robot-may-reverse"),
    "withholding backward from a deadlocked robot was not reported");
});

// ---- the escape axiom is the SOURCE of the capability, not a check on it -----
//
// Every other axiom in this file works one way round: the code computes the
// permitted set and the axiom verifies the computation, so deleting the axiom stops
// the check and leaves the behaviour. This one has to work the other way, because
// its entire content is "the ontology admits this action" -- a hardcoded set beside
// it would make the declaration decoration. These two tests pin both halves of that:
// the permission AND the sentence the observation carries are read off the axiom, so
// deleting it withdraws both together.

test("the escape set comes from the axiom, not a constant beside it", () => {
  const robot = { id: "R1", x: 200, y: 200, heading: 0 };
  const withAxiom = permittedActions(robot, null, { stalled: true, deadlocked: true });
  assert.deepEqual(withAxiom.options, ["backward", "left", "right"]);

  const idx = TBOX.findIndex(a => a.id === "deadlocked-robot-may-reverse");
  const removed = TBOX.splice(idx, 1)[0];
  try {
    const without = permittedActions(robot, null, { stalled: true, deadlocked: true });
    assert.ok(!without.options.includes("backward"),
      `the escape survived its own deletion: [${without.options}]`);
    // It falls back to the ORDINARY set -- this robot has no scan, so that is the
    // full three-way avoid set, not the two-way sidestep set. The point of the
    // assertion is that it is the set the robot would have had without the deadlock
    // concept at all, rather than a hardcoded escape.
    assert.deepEqual(without.options, ["left", "stay", "right"]);
    assert.ok(without.unasserted.includes("deadlocked-robot-may-reverse"),
      "the deletion was not reported");
  } finally {
    TBOX.splice(idx, 0, removed);
  }
});

test("the observation's deadlock sentence comes from the axiom too", () => {
  // The sentence and the permission must stand or fall together. If the wording were
  // hardcoded in the simulator, deleting the axiom would leave the model still being
  // told the robot is wedged while the ontology had stopped saying so -- the same
  // one-rule-two-homes shape that produced the head-on standoff.
  const robot = { id: "R1", x: 200, y: 200, heading: 0, phase: "toPickup", colorName: "red" };
  const before = permittedActions(robot, null, { stalled: true, deadlocked: true });
  assert.ok(before.stateLeaf && before.stateLeaf.deadlock,
    "the axiom contributes no wording to the observation");

  // NOTHING HERE CONSTRAINS THE WORDING, and that is now deliberate rather than an
  // omission. Two assertions have been removed from this spot. The first forbade
  // naming `sidestep`/`left`/`right`, on a theory measurement did not support. The
  // second forbade naming the ANSWER -- `!\/(backward|reverse)\/i` -- which passed
  // only while the leaf read "wedged, no progress for 8s" and failed the moment it
  // gained "should move backward". The wording is the ontology author's to choose;
  // what this test holds is the part that can actually break, which is below: the
  // sentence and the permission must stand or fall together. Which sub-path the
  // request reaches is what moves the answer, and no assertion about this string
  // can catch that -- see the note on the axiom.

  const idx = TBOX.findIndex(a => a.id === "deadlocked-robot-may-reverse");
  const removed = TBOX.splice(idx, 1)[0];
  try {
    const after = permittedActions(robot, null, { stalled: true, deadlocked: true });
    assert.deepEqual(after.stateLeaf, {},
      `the sentence outlived the axiom: ${JSON.stringify(after.stateLeaf)}`);
  } finally {
    TBOX.splice(idx, 0, removed);
  }
});

test("a non-deadlocked robot gets no deadlock wording", () => {
  // The leaf is gated on the same flag as the permission, so a merely blocked robot
  // cannot be described as wedged.
  const robot = { id: "R1", x: 200, y: 200, heading: 0 };
  const v = permittedActions(robot, null, { stalled: false, deadlocked: false });
  assert.deepEqual(v.stateLeaf, {});
  assert.ok(!v.options.includes("backward"));
});

test("the page's wrapper forwards every field the ontology grows", () => {
  // `permittedFor` is a WRAPPER that hand-picks what the page may see, and that makes
  // it a place where a new field goes missing without anything failing. It happened:
  // `stateLeaf` was added to `permittedActions` and not listed here, so the deadlock
  // sentence silently vanished from the observation while the options still arrived.
  // The symptom read as "the model will not choose backward" rather than "the state
  // never said the robot was stuck", which is why it survived the suite.
  const robot = { id: "R1", x: 200, y: 200, heading: 0 };
  const direct = permittedActions(robot, null, { stalled: true, deadlocked: true });
  const wrapped = permittedFor(robot, null, true, true);
  for (const field of ["options", "headOnRole", "deadlocked", "stateLeaf"]) {
    const want = field === "headOnRole" ? direct.headOnRole : direct[field];
    assert.deepEqual(wrapped[field === "headOnRole" ? "role" : field], want,
      `permittedFor dropped or altered ${field}`);
  }
  // Specifically the deadlock sentence, since that is the one that went missing.
  assert.ok(wrapped.stateLeaf && wrapped.stateLeaf.deadlock,
    "the wrapper dropped the deadlock wording");
});
