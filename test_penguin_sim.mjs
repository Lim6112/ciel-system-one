// node --test test_penguin_sim.mjs
//
// The penguin's world. penguin_sim.js is pure logic with no DOM, so everything asserted
// here is the shipped behaviour and not a re-description of it.
//
// The claims worth making are about the four commands actually being different from one
// another, and about the world telling the model the truth about what is in front of it.
// A demo where `back` and `go` agree, or where turning left four times does not return
// you to your heading, looks fine on screen and is worthless as an example.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CELL, COLS, ROWS, W, H, HEADINGS, STEPS, COMMANDS,
  MAX_STEPS, FISH_COUNT, START_COL, START_ROW, START_HEADING, WALK_DUR, TURN_DUR,
  DEFAULT_CONFIG, QUESTION,
  createSim, resetSim, applyCommand, step, isAnimating, headingAngle,
  cellAt, cellAhead, aheadOf, leftOf, rightOf, fishLeft,
  measure, buildState, describeCell,
} from "./static/penguin_sim.js";

// One fish parked in a far corner. NOT decoration: a floe with zero fish is instantly
// "all fish eaten", so an emptied world would end every movement test on its first
// command. (That behaviour is correct — an empty floe IS won — which is exactly why the
// helper must not create one.) Tests that care about fish reassign `sim.fish` themselves.
const PARKED_FISH = { col: 0, row: 0 };

/** A sim with nothing to bump into and only a distant fish, so movement is predictable. */
function bare(seed = 7, config) {
  const sim = createSim({ seed, config });
  sim.fish = [{ ...PARKED_FISH }];
  sim.water = [];
  return sim;
}

/** Run the animation clock forward until nothing is animating. */
function settle(sim, maxSeconds = 2) {
  for (let t = 0; t < maxSeconds && isAnimating(sim); t += 1 / 60) step(sim, 1 / 60);
  return sim;
}

// ---- geometry ---------------------------------------------------------------------

test("the canvas is exactly the grid it claims to be", () => {
  assert.equal(W, COLS * CELL);
  assert.equal(H, ROWS * CELL);
});

test("STEPS agrees with HEADINGS — index i steps the way heading i points", () => {
  // This is the coupling that makes `turn_left` = -1 work. If the two arrays drift, turns
  // rotate the sprite but move the penguin in the old direction.
  assert.equal(HEADINGS.length, 4);
  assert.deepEqual(STEPS[0], { dx: 0, dy: -1 });   // north is up: -y
  assert.deepEqual(STEPS[1], { dx: 1, dy: 0 });    // east
  assert.deepEqual(STEPS[2], { dx: 0, dy: 1 });    // south
  assert.deepEqual(STEPS[3], { dx: -1, dy: 0 });   // west
});

test("headingAngle puts north up, east right, south down, west left", () => {
  assert.equal(headingAngle(HEADINGS.indexOf("north")), -90);
  assert.equal(headingAngle(HEADINGS.indexOf("east")), 0);
  assert.equal(headingAngle(HEADINGS.indexOf("south")), 90);
  assert.equal(headingAngle(HEADINGS.indexOf("west")), 180);
});

test("the penguin starts on the floe, not on the edge", () => {
  const sim = createSim({ seed: 3 });
  assert.equal(sim.col, START_COL);
  assert.equal(sim.row, START_ROW);
  assert.equal(sim.heading, START_HEADING);
  assert.ok(sim.col > 0 && sim.col < COLS - 1);
  assert.ok(sim.row > 0 && sim.row < ROWS - 1);
});

// ---- turning ----------------------------------------------------------------------

test("four left turns return the penguin to the heading it started on", () => {
  const sim = bare();
  const start = sim.heading;
  for (let i = 0; i < 4; i++) { applyCommand(sim, "turn_left"); settle(sim); }
  assert.equal(sim.heading, start);
});

test("four right turns return the penguin to the heading it started on", () => {
  const sim = bare();
  const start = sim.heading;
  for (let i = 0; i < 4; i++) { applyCommand(sim, "turn_right"); settle(sim); }
  assert.equal(sim.heading, start);
});

test("left and right are opposites, not the same direction", () => {
  const a = bare(); applyCommand(a, "turn_left"); settle(a);
  const b = bare(); applyCommand(b, "turn_right"); settle(b);
  assert.notEqual(a.heading, b.heading, "turn left and turn right agreed");
  // Two of one is the other side of nothing: left then right is a no-op.
  const c = bare();
  applyCommand(c, "turn_left"); settle(c);
  applyCommand(c, "turn_right"); settle(c);
  assert.equal(c.heading, START_HEADING);
});

test("turning does not move the penguin", () => {
  const sim = bare();
  applyCommand(sim, "turn_left"); settle(sim);
  assert.equal(sim.col, START_COL);
  assert.equal(sim.row, START_ROW);
});

test("turning is free — it does not spend a step", () => {
  // Deliberate: the penguin can always stand still and reorient. Charging for it would
  // price a correct turn differently from a correct step.
  const sim = bare();
  applyCommand(sim, "turn_left"); settle(sim);
  applyCommand(sim, "turn_right"); settle(sim);
  assert.equal(sim.stepsUsed, 0);
});

// ---- walking ----------------------------------------------------------------------

test("go walks one cell in the facing direction", () => {
  const sim = bare();
  sim.heading = HEADINGS.indexOf("north");
  applyCommand(sim, "go"); settle(sim);
  assert.equal(sim.col, START_COL);
  assert.equal(sim.row, START_ROW - 1);
});

test("go and back move in opposite directions", () => {
  const fwd = bare(); applyCommand(fwd, "go"); settle(fwd);
  const bwd = bare(); applyCommand(bwd, "back"); settle(bwd);
  // back from north is south: same column, row + 1
  assert.equal(fwd.row, START_ROW - 1);
  assert.equal(bwd.row, START_ROW + 1);
  assert.equal(fwd.col, bwd.col);
});

test("back moves against the heading, so it follows the penguin around a turn", () => {
  const sim = bare();
  applyCommand(sim, "turn_right"); settle(sim);      // north -> east
  applyCommand(sim, "back"); settle(sim);            // east backwards is west
  assert.equal(sim.col, START_COL - 1);
  assert.equal(sim.row, START_ROW);
});

test("a step spends exactly one step", () => {
  const sim = bare();
  applyCommand(sim, "go"); settle(sim);
  assert.equal(sim.stepsUsed, 1);
});

// ---- the edge and the water -------------------------------------------------------

test("walking off the floe is blocked and does not move the penguin", () => {
  const sim = bare();
  sim.col = 0; sim.row = 3; sim.heading = HEADINGS.indexOf("west");
  const ev = applyCommand(sim, "go");
  assert.equal(ev.blocked, true);
  assert.equal(ev.moved, false);
  assert.equal(sim.col, 0, "penguin left the floe");
  assert.equal(sim.row, 3);
});

test("walking into open water is blocked and does not move the penguin", () => {
  const sim = bare();
  sim.col = 3; sim.row = 3; sim.heading = HEADINGS.indexOf("east");
  sim.water = [{ col: 4, row: 3 }];
  const ev = applyCommand(sim, "go");
  assert.equal(ev.blocked, true);
  assert.equal(ev.moved, false);
  assert.equal(sim.col, 3);
});

test("a blocked step still costs a step, or the water carries no risk", () => {
  // If a blocked move were free, there would be no reason for the model to care what is
  // ahead, and asking it would be theatre.
  const sim = bare();
  sim.col = 3; sim.row = 3; sim.heading = HEADINGS.indexOf("east");
  sim.water = [{ col: 4, row: 3 }];
  applyCommand(sim, "go"); settle(sim);
  assert.equal(sim.stepsUsed, 1);
});

test("the edge and the water are different things to say, not one 'blocked'", () => {
  assert.equal(describeCell("edge"), "the edge of the floe");
  assert.equal(describeCell("water"), "open water");
  assert.notEqual(describeCell("edge"), describeCell("water"));
  assert.equal(describeCell("ice"), "open ice");
  assert.equal(describeCell("fish"), "a fish");
});

// ---- fish -------------------------------------------------------------------------

test("walking onto a fish eats it and scores it", () => {
  const sim = bare();
  sim.col = 3; sim.row = 3; sim.heading = HEADINGS.indexOf("east");
  sim.fish = [{ col: 4, row: 3 }];
  const ev = applyCommand(sim, "go");
  assert.equal(ev.ate?.total, 1);
  assert.equal(sim.collected, 1);
  assert.equal(fishLeft(sim), 0);
  assert.equal(cellAt(sim, 4, 3), "ice", "the fish is still there after being eaten");
});

test("eating every fish ends the run", () => {
  const sim = bare();
  sim.col = 3; sim.row = 3; sim.heading = HEADINGS.indexOf("east");
  sim.fish = [{ col: 4, row: 3 }];
  const ev = applyCommand(sim, "go");
  assert.equal(ev.done, true);
  assert.equal(sim.doneReason, "all fish eaten");
});

test("a fish does not block movement — it is eaten, not bounced off", () => {
  const sim = bare();
  sim.col = 3; sim.row = 3; sim.heading = HEADINGS.indexOf("east");
  sim.fish = [{ col: 4, row: 3 }];
  const ev = applyCommand(sim, "go");
  assert.equal(ev.moved, true);
  assert.equal(ev.blocked, false);
});

// ---- the step budget ---------------------------------------------------------------

test("the run ends when the steps run out, and says why", () => {
  const sim = bare(7, { maxSteps: 3 });
  for (let i = 0; i < 3; i++) { applyCommand(sim, "go"); settle(sim); }
  assert.equal(sim.stepsUsed, 3);
  assert.equal(sim.done, true);
  assert.equal(sim.doneReason, "out of steps");
});

test("a finished run refuses further commands", () => {
  const sim = bare(7, { maxSteps: 1 });
  applyCommand(sim, "go"); settle(sim);
  assert.equal(sim.done, true);
  const ev = applyCommand(sim, "go");
  assert.equal(ev.accepted, false);
  assert.equal(sim.stepsUsed, 1, "a refused command still spent a step");
});

test("stepsLeft counts down and never goes negative", () => {
  const sim = bare(7, { maxSteps: 2 });
  assert.equal(measure(sim).stepsLeft, 2);
  applyCommand(sim, "go"); settle(sim);
  assert.equal(measure(sim).stepsLeft, 1);
  applyCommand(sim, "go"); settle(sim);
  assert.equal(measure(sim).stepsLeft, 0);
});

// ---- the animation clock -----------------------------------------------------------

test("a command resolves instantly in the logic but animates over time", () => {
  // The split is what lets tests assert outcomes without stepping a clock, while the
  // browser still gets something to draw.
  const sim = bare();
  applyCommand(sim, "go");
  assert.equal(sim.row, START_ROW - 1, "the logic did not resolve immediately");
  assert.equal(isAnimating(sim), true, "nothing to animate");
  settle(sim);
  assert.equal(isAnimating(sim), false);
});

test("the sprite arrives exactly on the cell centre when the animation ends", () => {
  const sim = bare();
  applyCommand(sim, "go"); settle(sim);
  assert.equal(sim.view.x, sim.col * CELL + CELL / 2);
  assert.equal(sim.view.y, sim.row * CELL + CELL / 2);
});

test("a second command during an animation is refused, not queued", () => {
  // Queuing would make the penguin obey an instruction the speaker issued about a
  // position it has since left.
  const sim = bare();
  applyCommand(sim, "go");
  const ev = applyCommand(sim, "go");
  assert.equal(ev.accepted, false);
  assert.equal(sim.stepsUsed, 1);
});

test("step is inert when nothing is animating", () => {
  const sim = bare();
  const before = { ...sim.view };
  step(sim, 1 / 60); step(sim, 1 / 60);
  assert.equal(sim.view.x, before.x);
  assert.equal(sim.view.y, before.y);
});

test("the walk animation is not stepped over in a single frame", () => {
  // At a plausible frame rate the bird has to be visibly between cells, or "waddling"
  // is a claim the renderer cannot honour.
  assert.ok(WALK_DUR > 1 / 30, "one cell completes in under a frame");
  assert.ok(TURN_DUR > 1 / 30, "one turn completes in under a frame");
});

test("an unknown command is refused and changes nothing", () => {
  const sim = bare();
  const ev = applyCommand(sim, "teleport");
  assert.equal(ev.accepted, false);
  assert.equal(sim.col, START_COL);
  assert.equal(sim.heading, START_HEADING);
});

test("COMMANDS is exactly the four the question offers", () => {
  assert.deepEqual([...COMMANDS].sort(), Object.keys(QUESTION.criteria).sort());
});

// ---- determinism -------------------------------------------------------------------

test("the same seed lays out the same floe", () => {
  const a = createSim({ seed: 42 });
  const b = createSim({ seed: 42 });
  assert.deepEqual(a.fish, b.fish);
  assert.deepEqual(a.water, b.water);
});

test("different seeds lay out different floes", () => {
  const a = createSim({ seed: 1 });
  const b = createSim({ seed: 2 });
  assert.notDeepEqual([a.fish, a.water], [b.fish, b.water]);
});

test("the layout is deterministic across a fresh process-independent replay", () => {
  const sim = createSim({ seed: 9 });
  const before = JSON.parse(JSON.stringify(sim.fish));
  resetSim(sim);
  assert.deepEqual(sim.fish, before);
});

test("resetSim puts back the fish that were eaten", () => {
  // REGRESSION. `sim.fish` is CONSUMED — an eaten fish is filtered out of it for good.
  // Restoring only the counters therefore left the floe permanently emptier: the HUD's
  // fish total silently shrank, and a floe cleared then replayed was instantly "done".
  const sim = createSim({ seed: 4 });
  const full = sim.fish.length;
  sim.fish = [{ col: sim.col, row: sim.row - 1 }];
  sim.heading = HEADINGS.indexOf("north");
  applyCommand(sim, "go");
  assert.equal(sim.collected, 1);
  assert.equal(sim.fish.length, 0);

  resetSim(sim);
  assert.equal(sim.fish.length, full, "the eaten fish did not come back");
  assert.equal(sim.collected, 0);
  assert.equal(sim.done, false, "a replayed floe was already finished");
  assert.equal(sim.stepsUsed, 0);
});

test("resetSim revives a finished run rather than starting it already won", () => {
  // The end state a player actually reaches: every fish gone, so `done` is latched. If
  // resetSim restored only the counters, the replay would begin on an empty floe and the
  // first command would immediately re-end it — the run would be unplayable, twice.
  const sim = createSim({ seed: 31 });
  sim.fish = [];
  sim.done = true;
  sim.doneReason = "all fish eaten";
  sim.collected = FISH_COUNT;

  resetSim(sim);
  assert.equal(sim.fish.length, FISH_COUNT, "the cleared floe came back empty");
  assert.equal(sim.done, false);
  assert.equal(sim.doneReason, null);
  const ev = applyCommand(sim, "go");
  assert.equal(ev.accepted, true, "the replayed floe refused a command immediately");
});

test("resetSim with keepLayout false rebuilds the seeded floe", () => {
  const sim = createSim({ seed: 4 });
  sim.fish = [];
  resetSim(sim, { keepLayout: false });
  assert.equal(sim.fish.length, FISH_COUNT);
  assert.deepEqual(sim.fish, createSim({ seed: 4 }).fish);
});

test("resetSim does not alias the pristine layout — replaying twice is safe", () => {
  // If resetSim handed back the SAME objects, eating one would corrupt the snapshot and
  // the second replay would be short a fish.
  const sim = createSim({ seed: 22 });
  resetSim(sim);
  const target = sim.fish[0];
  sim.col = target.col; sim.row = target.row + 1;
  sim.fish = [{ col: target.col, row: target.row }];
  sim.heading = HEADINGS.indexOf("north");
  applyCommand(sim, "go");
  resetSim(sim);
  assert.equal(sim.fish.length, FISH_COUNT, "the snapshot lost a fish to the first replay");
});

test("nothing is placed on the starting cell", () => {
  for (let seed = 1; seed <= 40; seed++) {
    const sim = createSim({ seed });
    assert.equal(cellAt(sim, START_COL, START_ROW), "ice",
      `seed ${seed} started the penguin on something`);
  }
});

test("fish and water never overlap each other", () => {
  for (let seed = 1; seed <= 40; seed++) {
    const sim = createSim({ seed });
    const keys = [...sim.fish, ...sim.water].map(c => `${c.col},${c.row}`);
    assert.equal(new Set(keys).size, keys.length, `seed ${seed} placed two things on one cell`);
  }
});

// ---- what the model is told --------------------------------------------------------

test("measure reports the neighbours the accessors report", () => {
  const sim = createSim({ seed: 11 });
  const m = measure(sim);
  assert.equal(m.heading, HEADINGS[sim.heading]);
  assert.equal(m.ahead, aheadOf(sim));
  assert.equal(m.left, leftOf(sim));
  assert.equal(m.right, rightOf(sim));
  // Every kind is a string that maps to a sentence. A cell kind with no prose would reach
  // the model as a bare word.
  for (const kind of [m.ahead, m.left, m.right, m.behind]) {
    assert.equal(typeof kind, "string");
    assert.match(describeCell(kind), /^[a-z]/);
  }
});

test("what is behind is the opposite of what is ahead when both are in bounds", () => {
  const sim = bare();
  sim.col = 5; sim.row = 5; sim.heading = HEADINGS.indexOf("north");
  assert.equal(aheadOf(sim), "ice");
  assert.equal(aheadOf(sim, -1), "ice");
  // Now put something one cell behind (south of a north-facing penguin).
  sim.fish = [{ col: 5, row: 6 }];
  assert.equal(aheadOf(sim, -1), "fish");
  assert.equal(aheadOf(sim), "ice");
});

test("buildState carries the transcript verbatim and names the world in prose", () => {
  const sim = bare();
  sim.col = 5; sim.row = 5; sim.heading = HEADINGS.indexOf("north");
  sim.fish = [{ col: 5, row: 4 }];        // directly ahead
  sim.water = [{ col: 4, row: 5 }];       // to the left of north

  const state = buildState(sim, "turn left please");
  assert.equal(state["spoken instruction"], "turn left please", "transcript was altered");
  assert.equal(state["penguin is facing"], "north");
  assert.equal(state["ahead of the penguin"], "a fish");
  assert.equal(state["to the left of the penguin"], "open water");
  assert.equal(state["to the right of the penguin"], "open ice");
});

test("buildState is a formatter over measure — the two cannot disagree", () => {
  const sim = createSim({ seed: 5 });
  const m = measure(sim);
  const s = buildState(sim, "go");
  assert.equal(s["ahead of the penguin"], describeCell(m.ahead));
  assert.equal(s["to the left of the penguin"], describeCell(m.left));
  assert.equal(s["to the right of the penguin"], describeCell(m.right));
  assert.equal(s["penguin is facing"], m.heading);
});

test("every state value is a string the API contract accepts", () => {
  // server.py coerces state values with str(); the sim should not be handing it objects
  // that stringify to "[object Object]".
  const state = buildState(createSim({ seed: 5 }), "go");
  for (const [k, v] of Object.entries(state)) {
    assert.equal(typeof v, "string", `${k} is not a string`);
    assert.ok(!v.includes("[object"), `${k} stringified badly`);
  }
});

test("cellAhead respects the grid edges without wrapping", () => {
  const sim = bare();
  assert.equal(cellAt(sim, -1, 0), "edge");
  assert.equal(cellAt(sim, COLS, 0), "edge");
  assert.equal(cellAt(sim, 0, -1), "edge");
  assert.equal(cellAt(sim, 0, ROWS), "edge");
  assert.equal(cellAhead(sim, 0, 0, HEADINGS.indexOf("north")).row, -1);
});

test("a full seeded run replays identically", () => {
  const run = () => {
    const sim = createSim({ seed: 17 });
    const script = ["go", "turn_left", "go", "back", "turn_right", "go", "go", "turn_left", "go"];
    for (const c of script) { applyCommand(sim, c); settle(sim); }
    return { col: sim.col, row: sim.row, heading: sim.heading, collected: sim.collected, steps: sim.stepsUsed };
  };
  assert.deepEqual(run(), run());
});

test("a long script always terminates inside the step budget", () => {
  for (let seed = 1; seed <= 25; seed++) {
    const sim = createSim({ seed });
    const script = COMMANDS;
    for (let i = 0; i < 200 && !sim.done; i++) {
      applyCommand(sim, script[i % script.length]);
      settle(sim);
    }
    assert.equal(sim.done, true, `seed ${seed} never finished`);
    assert.ok(sim.stepsUsed <= MAX_STEPS, `seed ${seed} overspent its steps`);
  }
});

test("DEFAULT_CONFIG's budget is the exported MAX_STEPS", () => {
  assert.equal(DEFAULT_CONFIG.maxSteps, MAX_STEPS);
});
