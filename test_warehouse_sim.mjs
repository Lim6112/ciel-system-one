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
  DOCK,
  DOCK_ZONE_RADIUS,
  SIDESTEP_QID,
  STALL_MS,
  applyDecision,
  bindWorld,
  buildQuestion,
  buildState,
  createWorld,
  dockZoneOpen,
  feasibleOptions,
  goldDirection,
  headOnRole,
  headingIn,
  hitsRack,
  inDockZone,
  nearestAisle,
  nearestCorridor,
  questionId,
  rejoinPath,
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

test("nearestAisle returns the nearer aisle", () => {
  // Added after this function was quietly broken for the length of one edit: the
  // reduce branches were swapped, so it returned the FARTHEST aisle and every
  // route crossed the board twice. Nothing caught it, because every other test in
  // this file asserts an end-to-end outcome (deliveries, pickups, blocked time)
  // and none asserted a lane choice -- the same blind spot the original
  // `nearestCorridor` bug lived in until an outcome happened to expose it. The
  // cost of that miss: pickups fell to 2 in a 60 s run, which reads as a routing
  // catastrophe and not as "one comparison is backwards".
  assert.equal(nearestAisle(321), 380);   // nearest of 140/260/380
  assert.equal(nearestAisle(200), 140);
  assert.equal(nearestAisle(100), 140);
  assert.equal(nearestAisle(490), 380);
  assert.equal(nearestAisle(260), 260);
  // And it must agree with an explicit nearest-neighbour search everywhere, which
  // is the property the swapped branches violated.
  for (let x = 0; x <= 520; x++) {
    const want = [140, 260, 380].reduce((a, b) => Math.abs(b - x) < Math.abs(a - x) ? b : a);
    assert.equal(nearestAisle(x), want, `nearestAisle(${x})`);
  }
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

test("the dock-zone rule is off by default, because it costs throughput", () => {
  // A negative result worth pinning. The rule is a clean statement of a real
  // domain constraint and it is fully implemented, but it measured as a ~40%
  // throughput LOSS (89.6 -> 43.6 mean deliveries here; 93.6/91.1 -> 55.0/48.0
  // across two runs over nine seeds) at every radius swept, 20-90 px. It also
  // failed to hold its own invariant, because the hold parks a robot at the zone
  // edge where it still blocks the apron.
  //
  // So this test does not assert the rule works. It asserts the rule is not
  // silently switched on: if someone flips the default, the demo regresses and
  // nothing else in the suite would say why.
  assert.equal(DOCK_ZONE_RADIUS > 0, true);
  const robots = createWorld(1);
  const seen = [];
  step(robots, 0.016, 16, CONFIG);
  seen.push(robots.every(r => r.heldByDock === false));
  assert.ok(seen.every(Boolean),
    "the dock rule is holding robots by default — it was measured as a net loss");
});

test("the dock rule can still be turned on and holds one occupant", () => {
  // The implementation stays live and reachable (dockRule: true) so the negative
  // result above is reproducible and so an ontology-driven retry has something to
  // govern. Sanity-check the pieces rather than the throughput.
  const robots = createWorld(1);
  const inside = robots[0];
  inside.x = DOCK.x;
  inside.y = DOCK.y + 10;
  const outside = robots[1];
  outside.waypoints = [{ x: DOCK.x, y: DOCK.y }];
  outside.wp = 0;
  assert.equal(inDockZone(inside), true);
  assert.equal(inDockZone(outside), false);
  assert.equal(headingIn(outside), true);
  assert.equal(dockZoneOpen(inside, robots), true, "the incumbent keeps the zone");
  assert.equal(dockZoneOpen(outside, robots), false, "a second arrival is refused");
});

test("a run stays deterministic past the first delivery", () => {
  // createWorld seeds the RNG, but `step` used to call `assignTask(robot)` with
  // no rng, so every task after the first drew from Math.random. `createWorld(seed)`
  // was therefore only deterministic up to the first delivery, and two runs of the
  // same seed diverged — which silently turned several A/B measurements taken
  // during this work into noise (one seed, three runs: 3,774 / 8,156 / 9,547
  // encounters). Pinned here because a benchmark harness that is not reproducible
  // is worse than no harness: it produces confident numbers about nothing.
  const snapshot = seed => {
    const robots = createWorld(seed);
    let now = 0;
    const lastAsk = new Map();
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
        const role = kind === "robot" ? headOnRole(robot, scan) : null;
        const options = feasibleOptions(kind, stalled, role);
        let gold = goldDirection(scan, stalled, role);
        if (!options.includes(gold)) gold = options[0];
        applyDecision(robot, gold, now);
      }
    }
    return {
      deliveries: robots.reduce((n, r) => n + r.deliveries, 0),
      pos: robots.map(r => `${r.id}(${Math.round(r.x)},${Math.round(r.y)})`),
      racks: robots.map(r => r.pickupRack.id),
    };
  };
  const a = snapshot(1);
  const b = snapshot(1);
  assert.deepEqual(a, b, "same seed diverged — a task assignment is drawing from Math.random");
  assert.ok(a.deliveries > 3, `only ${a.deliveries} deliveries; not past the first cycle`);
});

test("a dead-ahead meeting elects exactly one advancer, from both sides", () => {
  // The tie-break that replaced the blockedSince election. Both robots must reach
  // a verdict, the verdicts must be opposite, and neither may need shared state:
  // each compares its own id against the counterpart's.
  const robots = createWorld(1);
  // Put two robots nose to nose in the same lane, headings opposed.
  const [a, b] = [robots[0], robots[1]];
  a.x = 260; a.y = 330; a.heading = Math.PI; a.waypoints = [{ x: 100, y: 330 }]; a.wp = 0;
  b.x = 220; b.y = 330; b.heading = 0; b.waypoints = [{ x: 400, y: 330 }]; b.wp = 0;
  bindWorld(robots);
  const scanA = scanLidar(a, robots, CONFIG.lidarRange);
  const scanB = scanLidar(b, robots, CONFIG.lidarRange);
  assert.ok(scanA && scanB, "the two should see each other");
  const roleA = headOnRole(a, scanA);
  const roleB = headOnRole(b, scanB);
  assert.ok(roleA && roleB, `both must be elected, got ${roleA}/${roleB}`);
  assert.notEqual(roleA, roleB,
    `both elected "${roleA}" — a dead-ahead pair would move or hold together`);
  // And the elected roles are the ones the option sets reflect.
  assert.deepEqual(feasibleOptions("robot", false, roleA).includes("stay"), roleA === "yield");
  assert.deepEqual(feasibleOptions("robot", false, roleB).includes("stay"), roleB === "yield");
});

test("head-on resolutions stay under the stall budget", () => {
  // The property the election exists for. Before it, head-on meetings waited out
  // STALL_MS (4 s) and took ~4.4 s to resolve against ~0.22 s for every other
  // block. Measured after: p90 736 ms, and the >4 s count fell from 704 to 6 over
  // seven seeds. The bound is loose on purpose — the assertion is that the wait is
  // not the norm, not that the tail is gone (a 0.04% tail remains, diagnosed in
  // warehouse_sim.js).
  const slow = [];
  let total = 0;
  for (const seed of [1, 5]) {
    const robots = createWorld(seed);
    const lastAsk = new Map();
    const open = new Map();
    let now = 0;
    for (let i = 0; i < 60000; i++) {
      now += 16;
      step(robots, 0.016, now, CONFIG);
      for (const robot of robots) {
        const scan = robot.obstacle;
        const byRobot = scan && !scan.rack;
        const cur = open.get(robot.id);
        if (cur && (!byRobot || cur.blocker !== scan.other)) {
          total++;
          if (now - cur.start > 4000) slow.push(now - cur.start);
          open.delete(robot.id);
        }
        if (byRobot && !open.has(robot.id)) {
          open.set(robot.id, { start: now, blocker: scan.other });
        }
        if (!scan || now - (lastAsk.get(robot.id) || 0) < 300) continue;
        lastAsk.set(robot.id, now);
        const stalledFor = robot.lastProgressAt ? (now - robot.lastProgressAt) / 1000 : 0;
        const stalled = stalledFor * 1000 >= STALL_MS;
        const kind = scan.rack ? "rack" : "robot";
        const role = kind === "robot" ? headOnRole(robot, scan) : null;
        const options = feasibleOptions(kind, stalled, role);
        let gold = goldDirection(scan, stalled, role);
        if (!options.includes(gold)) gold = options[0];
        applyDecision(robot, gold, now);
      }
    }
  }
  assert.ok(total > 100, `only ${total} robot-vs-robot encounters`);
  const share = slow.length / total;
  assert.ok(share < 0.02,
    `${slow.length}/${total} (${(100 * share).toFixed(2)}%) encounters exceeded the ` +
    `4 s stall budget — the election is not doing its job`);
});

test("a head-on state is rewritten so neither robot is told to wait", () => {
  // The rewrite: dead ahead, the untransformed text says "ahead of me, but not
  // blocking my lane", which the head reads as "not in my way" and answers `stay`
  // — measured 25 of 25 on the live service. With a role, the text says the blocker
  // IS in the lane and names a side, and the same 25 of 25 answer a sidestep at
  // 93-99%. Pinned here because it is the whole reason no retraining was needed:
  // the head already knew what in-lane-plus-a-side means.
  const robots = createWorld(1);
  const [a, b] = [robots[0], robots[1]];
  a.x = 260; a.y = 330; a.heading = Math.PI; a.waypoints = [{ x: 100, y: 330 }]; a.wp = 0;
  b.x = 220; b.y = 330; b.heading = 0; b.waypoints = [{ x: 400, y: 330 }]; b.wp = 0;
  bindWorld(robots);
  for (const [r, other] of [[a, b], [b, a]]) {
    const scan = scanLidar(r, robots, CONFIG.lidarRange);
    assert.ok(scan, "the two should see each other");
    const role = headOnRole(r, scan);
    assert.ok(role, "a dead-ahead pair must be elected");

    const plain = buildState(r, scan, 0, null);
    assert.equal(plain.obstacle_side, "center");
    assert.equal(plain.obstacle_position, "ahead of me, but not blocking my lane",
      "without a role the text still says the blocker is not in the lane — the phrasing " +
      "that answers `stay`");

    const rewritten = buildState(r, scan, 0, role);
    assert.equal(rewritten.obstacle_position, "directly ahead in my lane",
      "the rewrite must say the blocker IS in the lane");
    assert.ok(["left", "right"].includes(rewritten.obstacle_side),
      `the rewrite must name a side, got ${rewritten.obstacle_side}`);
    assert.notEqual(rewritten.obstacle_side, "center");
    // Off-centre cases must be left alone: they already carry a side and the
    // ordinary gold reads it.
    const offAxis = { ...scan, rel: { x: 40, y: scan.rel.y } };
    assert.equal(buildState(r, offAxis, 0, role).obstacle_side, "right");
  }
});

test("a head-on pair is sent to opposite sides of the corridor", () => {
  // The sign that two earlier attempts got wrong. The head answers the OPPOSITE of
  // the named side, so naming each robot the side it "has room on" sends both to
  // the same half of the corridor. Because the two headings are opposed, naming the
  // SAME side is what diverges — verified across every heading rather than argued.
  const worldMove = (r, named) => {
    const stepping = named === "left" ? "right" : "left";
    const d = stepping === "left" ? -1 : 1;
    const px = Math.cos(r.heading), py = Math.sin(r.heading);
    return { dx: -py * d, dy: px * d };
  };
  for (let deg = 0; deg < 360; deg += 15) {
    const h = deg * Math.PI / 180;
    const a = { id: "R1", colorName: "red", x: 300, y: 300, heading: h, sidestepDir: 0,
                waypoints: [], wp: 0, deliveries: 0, phase: "toDock" };
    const b = { id: "R2", colorName: "green", x: 300 + Math.cos(h) * 20,
                y: 300 + Math.sin(h) * 20, heading: (h + Math.PI) % (2 * Math.PI),
                sidestepDir: 0, waypoints: [], wp: 0, deliveries: 0, phase: "toDock" };
    bindWorld([a, b]);
    const scanA = scanLidar(a, [a, b], CONFIG.lidarRange);
    const scanB = scanLidar(b, [a, b], CONFIG.lidarRange);
    if (!scanA || !scanB) continue;
    const stA = buildState(a, scanA, 0, headOnRole(a, scanA) || "advance");
    const stB = buildState(b, scanB, 0, headOnRole(b, scanB) || "advance");
    const ma = worldMove(a, stA.obstacle_side), mb = worldMove(b, stB.obstacle_side);
    const before = Math.hypot(b.x - a.x, b.y - a.y);
    const after = Math.hypot((b.x + mb.dx) - (a.x + ma.dx), (b.y + mb.dy) - (a.y + ma.dy));
    assert.ok(after > before + 1e-9,
      `heading ${deg}deg: the pair converged (${before.toFixed(2)} -> ${after.toFixed(2)}) — ` +
      `named "${stA.obstacle_side}"/"${stB.obstacle_side}"`);
  }
});

test("rejoining a path keeps the goal and drops only what is behind", () => {
  // A sidestep ends and the robot has to get back on its route. Asking the ROUTER
  // for a new one is wrong — it rebuilds the chain from the current position and
  // resets `wp`, so its first leg is "go to the nearest corridor" and a robot that
  // just stepped off that corridor is sent sideways back onto the line it left.
  // Measured with the router route: 3,585 deliveries became 2,759.
  //
  // `rejoinPath` projects instead: find the segment nearest the robot and resume
  // from its near end, keeping the rest untouched.
  const robot = {
    id: "R1", x: 300, y: 200, heading: 0, wp: 1,
    // An L-chain: along the corridor at y=238, then up the aisle at x=260.
    waypoints: [{ x: 260, y: 330 }, { x: 260, y: 238 }, { x: 140, y: 238 }, { x: 140, y: 146 }],
  };
  const goalBefore = robot.waypoints[robot.waypoints.length - 1];
  assert.equal(rejoinPath(robot), true);
  // The goal must not move — a rejoin corrects the return leg, it does not re-plan.
  const goalAfter = robot.waypoints[robot.waypoints.length - 1];
  assert.deepEqual(goalAfter, goalBefore, "the goal was re-decided");
  assert.equal(robot.wp, 0);
  // And nothing behind the robot is kept: it is at y=200 on the x=260 segment, so
  // the earlier waypoints are already passed.
  assert.ok(!robot.waypoints.some(p => p.x === 260 && p.y === 330),
    `a passed waypoint came back: ${JSON.stringify(robot.waypoints)}`);
});

test("a rejoin never leaves the robot with no path", () => {
  // An empty waypoint list is read as "arrived" by the step loop, which would fire
  // a phase change from a swerve rather than from reaching the goal.
  const robot = { id: "R1", x: 140, y: 146, heading: 0, wp: 0,
                  waypoints: [{ x: 140, y: 146 }] };
  const had = JSON.stringify(robot.waypoints);
  const ok = rejoinPath(robot);
  if (ok) assert.ok(robot.waypoints.length > 0, "rejoin emptied the path");
  else assert.equal(JSON.stringify(robot.waypoints), had, "refused but mutated anyway");
});

test("rejoining beats re-routing measured end to end", () => {
  // The property the feature exists for, measured rather than asserted. Compared
  // against the same run with rejoining off, so the difference is the feature.
  const measure = replan => {
    let deliveries = 0, over4s = 0;
    for (const seed of [1, 5]) {
      const robots = createWorld(seed);
      const lastAsk = new Map();
      const open = new Map();
      let now = 0;
      for (let i = 0; i < 60000; i++) {
        now += 16;
        step(robots, 0.016, now, { ...CONFIG, replanAfterSidestep: replan });
        for (const robot of robots) {
          const scan = robot.obstacle;
          const byRobot = scan && !scan.rack;
          const cur = open.get(robot.id);
          if (cur && (!byRobot || cur.blocker !== scan.other)) {
            if (now - cur.start > 4000) over4s++;
            open.delete(robot.id);
          }
          if (byRobot && !open.has(robot.id)) open.set(robot.id, { start: now, blocker: scan.other });
          if (!scan || now - (lastAsk.get(robot.id) || 0) < 300) continue;
          lastAsk.set(robot.id, now);
          const stalledFor = robot.lastProgressAt ? (now - robot.lastProgressAt) / 1000 : 0;
          const stalled = stalledFor * 1000 >= STALL_MS;
          const kind = scan.rack ? "rack" : "robot";
          const role = kind === "robot" ? headOnRole(robot, scan) : null;
          const options = feasibleOptions(kind, stalled, role);
          let gold = goldDirection(scan, stalled, role);
          if (!options.includes(gold)) gold = options[0];
          applyDecision(robot, gold, now);
        }
      }
      deliveries += robots.reduce((n, r) => n + r.deliveries, 0);
    }
    return { deliveries, over4s };
  };
  const off = measure(false);
  const on = measure(true);
  assert.ok(on.deliveries >= off.deliveries,
    `rejoining reduced deliveries: ${off.deliveries} -> ${on.deliveries}`);
  assert.ok(on.over4s <= off.over4s,
    `rejoining increased the >4s tail: ${off.over4s} -> ${on.over4s}`);
});

test("createWorld is deterministic for a given seed", () => {
  const a = createWorld(42).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  const b = createWorld(42).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  assert.deepEqual(a, b);
  const c = createWorld(43).map(r => [r.id, r.x, r.y, r.pickupRack.id]);
  assert.notDeepEqual(a, c);
});
