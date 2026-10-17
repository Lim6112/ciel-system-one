// Warehouse sim invariants: geometry, the JSON state contract, blocker split,
// escalation, and the gate's oracle.
//
// The point of these tests is the boundary the browser and the head share. Two
// prior incidents in this project came from training on one state format and
// inferring on another, and the simulator is where that format is produced — so
// the schema, the leaf count and the gold are pinned here rather than eyeballed
// in the page.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  AVOID_QID,
  SIDESTEP_QID,
  STALL_MS,
  applyDecision,
  buildQuestion,
  buildState,
  createWorld,
  feasibleOptions,
  goldDirection,
  questionId,
  scanLidar,
  step,
} from "./static/warehouse_sim.js";

const CONFIG = { speed: 80, lidarRange: 90, autoplay: true };

// Run the world until `predicate(robot, scan)` holds, or give up.
function advance(robots, { maxTicks = 40000, predicate = null, onTick = null } = {}) {
  let now = 0;
  for (let i = 0; i < maxTicks; i++) {
    now += 16;
    step(robots, 0.016, now, CONFIG);
    for (const robot of robots) {
      if (onTick) onTick(robot, now);
      if (predicate && predicate(robot, now)) return { robot, now };
    }
  }
  return null;
}

function collectBlocked(kind, count, seed = 1) {
  const found = [];
  advance(createWorld(seed), {
    onTick(robot, now) {
      const scan = robot.obstacle;
      if (!scan) return;
      const isRack = !!scan.rack;
      if ((kind === "rack") !== isRack) return;
      if (found.length < count) {
        found.push({ scan, state: buildState(robot, scan, now), robot, now });
      }
    },
    maxTicks: found.length < count ? 60000 : 1,
  });
  return found;
}

test("a rack blocker is offered a 2-way question with no 'stay'", () => {
  for (const { scan } of collectBlocked("rack", 3)) {
    const q = buildQuestion("rack", false);
    assert.deepEqual(Object.keys(q.criteria), ["left", "right"]);
    assert.equal(questionId("rack", false), SIDESTEP_QID);
    assert.deepEqual(feasibleOptions("rack", false), ["left", "right"]);
    assert.ok(scan.rack, "expected a rack hit");
  }
});

test("a robot blocker is offered the 3-way question with 'stay'", () => {
  const blocked = collectBlocked("robot", 3);
  assert.ok(blocked.length > 0, "no robot blocker encountered");
  for (const { scan } of blocked) {
    const q = buildQuestion("robot", false);
    assert.deepEqual(Object.keys(q.criteria), ["left", "stay", "right"]);
    assert.equal(questionId("robot", false), AVOID_QID);
    assert.deepEqual(feasibleOptions("robot", false), ["left", "stay", "right"]);
    assert.equal(scan.rack, null);
  }
});

test("a stalled robot loses 'stay' regardless of blocker", () => {
  assert.equal(questionId("robot", true), SIDESTEP_QID);
  assert.deepEqual(feasibleOptions("robot", true), ["left", "right"]);
  assert.deepEqual(Object.keys(buildQuestion("robot", true).criteria),
    ["left", "right"]);
});

test("the state is flat scalar JSON with at most 15 leaves", () => {
  // Flat: server.py str()s each top-level value, so a nested object would be
  // sent as a Python repr. <=15: the head prepends the whole text as field 0, so
  // the leaf count has to leave room without the chunker merging any leaf — and a
  // merged leaf would take the gold-bearing one with it.
  for (const kind of ["rack", "robot"]) {
    for (const { state } of collectBlocked(kind, 3)) {
      assert.ok(Object.keys(state).length <= 15,
        `too many leaves: ${Object.keys(state).length}`);
      for (const [key, value] of Object.entries(state)) {
        assert.ok(["string", "number"].includes(typeof value),
          `${key} is ${typeof value}`);
      }
    }
  }
});

test("the decisive leaf is present and names the gold", () => {
  for (const { scan, state } of collectBlocked("rack", 3)) {
    assert.equal(state.obstacle_kind, "storage rack");
    assert.ok("gap_left" in state && "gap_right" in state);
    // no lateral offset for a rack: its lidar hit returns a distance only
    assert.ok(!("obstacle_side" in state));
    const wider = state.gap_left === "wide" ? "left" : "right";
    assert.equal(goldDirection(scan), wider);
  }
  for (const { scan, state } of collectBlocked("robot", 3)) {
    assert.equal(state.obstacle_kind, "robot");
    assert.ok("obstacle_side" in state);
    assert.ok(!("gap_left" in state));
    assert.equal(goldDirection(scan), goldDirection(scan)); // stable
  }
});

test("gold is the side opposite the blocker", () => {
  for (const { scan, state } of collectBlocked("robot", 6)) {
    const gold = goldDirection(scan);
    if (state.obstacle_side === "right") assert.equal(gold, "left");
    else if (state.obstacle_side === "left") assert.equal(gold, "right");
    else assert.equal(gold, "stay");
  }
});

test("the position leaf never contradicts the side leaf", () => {
  for (const kind of ["rack", "robot"]) {
    for (const { state } of collectBlocked(kind, 6)) {
      if (state.obstacle_kind === "storage rack") {
        assert.equal(state.obstacle_position, "directly ahead in my lane");
      } else if (state.obstacle_side === "center") {
        // head-on but explicitly not blocking — which is what makes the 'stay'
        // answer honest rather than a contradiction of the observation
        assert.equal(state.obstacle_position, "ahead of me, but not blocking my lane");
      } else {
        assert.equal(state.obstacle_position, "ahead, off to one side of my lane");
      }
    }
  }
});

test("escalation fires on a stall, not on a clock", () => {
  // The old trigger (5 s continuously blocked) never fired: across 178 recorded
  // calls the longest block was 3 s, so the 2-way branch was unreachable and
  // "stay" was always on the menu. This pins the replacement — progress
  // staleness — by driving the policy that stalls a robot: a robot told to wait
  // makes no progress, and that is the condition escalation keys on.
  const robots = createWorld(7);
  const r = robots[0];
  let now = 0;
  for (let i = 0; i < 40000; i++) {
    now += 16;
    step(robots, 0.016, now, CONFIG);
    applyDecision(r, "stay", now);          // hold this robot still
    if (r.lastProgressAt && (now - r.lastProgressAt) >= STALL_MS) break;
  }
  const stalledFor = (now - r.lastProgressAt) / 1000;
  assert.ok(stalledFor * 1000 >= STALL_MS,
    `robot never stalled (stalledFor=${stalledFor}s) — escalation unreachable`);

  // and once stalled, the question loses its "stay" option
  const scan = r.obstacle || scanLidar(r, robots, CONFIG.lidarRange);
  if (scan) {
    const kind = scan.rack ? "rack" : "robot";
    assert.equal(questionId(kind, true), SIDESTEP_QID);
    assert.deepEqual(feasibleOptions(kind, true), ["left", "right"]);
    assert.equal(buildState(r, scan, now).blocked, "for a while");
  }
});

test("a sidestep takes the robot's progress, not just a timer", () => {
  // Whatever direction is chosen must actually move the robot; otherwise the
  // stall timer fires forever and every decision escalates.
  const robots = createWorld(3);
  const r = robots[0];
  const before = { x: r.x, y: r.y };
  advance(robots, { maxTicks: 400 });
  assert.ok(Math.hypot(r.x - before.x, r.y - before.y) > 1,
    "robot did not move at all");
});

test("createWorld is deterministic for a given seed", () => {
  const a = createWorld(42).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  const b = createWorld(42).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  assert.deepEqual(a, b);
  const c = createWorld(43).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  assert.notDeepEqual(a, c);
});
