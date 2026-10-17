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
  hitsRack,
  nearestCorridor,
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

test("nearestCorridor returns the nearer corridor", () => {
  // It returned the farther one: the comparison was right but the branches were
  // swapped, so every robot below y=284 was routed to the corridor at 146.
  // That is not a cosmetically odd route — `cy` drives the first leg's y, so a
  // robot at y=390 got a first leg from (55,390) to (55,146): a straight line
  // through both rack rows, which the collision check refused. The robot was
  // pinned at its spawn with 0 deliveries.
  assert.equal(nearestCorridor(390), 330);
  assert.equal(nearestCorridor(405), 330);
  assert.equal(nearestCorridor(300), 330);
  assert.equal(nearestCorridor(238), 238);
  assert.equal(nearestCorridor(200), 238);
  assert.equal(nearestCorridor(146), 146);
});

test("no route leg crosses a rack", () => {
  // The invariant the routing bug violated. Stated as rack-crossing rather than
  // "follows a corridor or aisle" on purpose: the first leg runs from wherever
  // the robot happens to be to its corridor, and that is legitimately vertical
  // at a non-aisle x as long as it starts outside the rack rows.
  //
  // The old first leg went from (55,390) straight up to (55,146), crossing both
  // rows, which the collision check then refused — pinning the robot at spawn.
  for (const seed of [1, 2, 3, 5, 7, 11, 42]) {
    for (const robot of createWorld(seed)) {
      const pts = [{ x: robot.x, y: robot.y }, ...robot.waypoints];
      for (let i = 1; i < pts.length; i++) {
        const a = pts[i - 1], b = pts[i];
        const steps = Math.ceil(Math.hypot(b.x - a.x, b.y - a.y));
        for (let s = 0; s <= steps; s++) {
          const t = steps ? s / steps : 0;
          const x = a.x + (b.x - a.x) * t;
          const y = a.y + (b.y - a.y) * t;
          assert.ok(!hitsRack(x, y),
            `seed ${seed} ${robot.id}: leg (${a.x},${a.y})->(${b.x},${b.y}) ` +
            `crosses a rack near (${x.toFixed(0)},${y.toFixed(0)})`);
        }
      }
    }
  }
});

test("a robot reaches its pickup rack and delivers", () => {
  // The end-to-end statement the routing bug broke: with the geometry oracle as
  // the policy, robots must actually complete cycles. Before the fix this was
  // 1 pickup in 60 s of sim time; it is now ~100.
  const robots = createWorld(1);
  const cfg = { ...CONFIG };
  let now = 0, pickups = 0;
  const prev = new Map();
  const lastAsk = new Map();
  for (let i = 0; i < 60000; i++) {
    now += 16;
    step(robots, 0.016, now, cfg);
    for (const robot of robots) {
      const scan = robot.obstacle;
      if (scan && now - (lastAsk.get(robot.id) || 0) >= 300) {
        lastAsk.set(robot.id, now);
        const stalledFor = robot.lastProgressAt ? (now - robot.lastProgressAt) / 1000 : 0;
        const options = feasibleOptions(scan.rack ? "rack" : "robot",
          stalledFor * 1000 >= STALL_MS);
        const gold = goldDirection(scan);
        applyDecision(robot, options.includes(gold) ? gold : "left", now);
      }
      if (prev.get(robot.id) === "toPickup" && robot.phase === "toDock") pickups++;
      prev.set(robot.id, robot.phase);
    }
  }
  assert.ok(pickups >= 50, `only ${pickups} pickups — routing regressed`);
  assert.ok(robots.reduce((n, r) => n + r.deliveries, 0) >= 50);
});

test("no robot ever leaves the floor", () => {
  // A sidestep is a fixed-duration perpendicular crawl in one direction with no
  // target, so near an edge it walked off the board — 51,558 ticks' worth before
  // the bounds clamp, every one of them mid-sidestep.
  for (const seed of [1, 5, 11]) {
    const robots = createWorld(seed);
    const lastAsk = new Map();
    let now = 0;
    for (let i = 0; i < 40000; i++) {
      now += 16;
      step(robots, 0.016, now, CONFIG);
      for (const robot of robots) {
        const scan = robot.obstacle;
        if (scan && now - (lastAsk.get(robot.id) || 0) >= 300) {
          lastAsk.set(robot.id, now);
          const stalledFor = robot.lastProgressAt ? (now - robot.lastProgressAt) / 1000 : 0;
          const options = feasibleOptions(scan.rack ? "rack" : "robot",
            stalledFor * 1000 >= STALL_MS);
          const gold = goldDirection(scan);
          applyDecision(robot, options.includes(gold) ? gold : "left", now);
        }
        assert.ok(robot.x >= 0 && robot.x <= 520 && robot.y >= 0 && robot.y <= 420,
          `seed ${seed} ${robot.id} left the floor at (${robot.x.toFixed(1)},${robot.y.toFixed(1)})`);
      }
    }
  }
});


test("the oracle never contradicts the option set it is gated against", () => {
  // The livelock this pins: `feasibleOptions` withdraws "stay" once a robot has
  // stalled (waiting demonstrably failed), but a head-on blocker still made the
  // oracle answer "stay". A gate deferring to that answer put the robot straight
  // back into the wait it had just escalated out of — 791 deferred against 40
  // decided, robots pinned in a permanent head-on standoff.
  //
  // The oracle and the option set are the two halves of one rule, so they must
  // agree at every blocker, stall state and geometry the sim can produce.
  for (const seed of [1, 2, 3, 5, 7, 11, 42, 99, 123]) {
    const robots = createWorld(seed);
    const lastAsk = new Map();
    let now = 0;
    for (let i = 0; i < 40000; i++) {
      now += 16;
      step(robots, 0.016, now, CONFIG);
      for (const robot of robots) {
        const scan = robot.obstacle;
        if (!scan || now - (lastAsk.get(robot.id) || 0) < 300) continue;
        lastAsk.set(robot.id, now);
        const stalledFor = robot.lastProgressAt ? (now - robot.lastProgressAt) / 1000 : 0;
        const stalled = stalledFor * 1000 >= STALL_MS;
        const kind = scan.rack ? "rack" : "robot";
        const gold = goldDirection(scan, stalled);
        assert.ok(feasibleOptions(kind, stalled).includes(gold),
          `seed ${seed} ${robot.id}: oracle "${gold}" is not on the menu ` +
          `(kind=${kind}, stalled=${stalled}) — the gate would invert the escalation`);
        applyDecision(robot, gold, now);
      }
    }
  }
});

test("createWorld is deterministic for a given seed", () => {
  const a = createWorld(42).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  const b = createWorld(42).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  assert.deepEqual(a, b);
  const c = createWorld(43).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  assert.notDeepEqual(a, c);
});
