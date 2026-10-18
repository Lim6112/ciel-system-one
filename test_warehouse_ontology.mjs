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
