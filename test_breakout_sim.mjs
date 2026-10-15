// node --test test_breakout_sim.mjs
//
// The control arm. breakout_sim.js is a MOVE of logic that used to live inline in
// breakout.html, so the job here is to prove the move was inert: same physics, same
// numbers, no DOM. test_breakout_ontology.mjs is free to assume all of this.
//
// What this file is not is a test of the ontology — see test_ontology.mjs, whose header
// makes the same point about the fleet reasoner.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  W, H, ROWS, COLS, PADDLE_W, PADDLE_H, PADDLE_Y, PADDLE_SPEED, BALL_R,
  BRICK_H, BRICK_GAP, BRICK_TOP, CATCH_HALF, PADDLE_MIN, PADDLE_MAX,
  BALL_MIN_X, BALL_MAX_X, BAND_ENTER_Y, BAND_EXIT_Y, LAST_BRICK_BOTTOM,
  CATCH_TOL_PX, DEFAULT_CONFIG, QUESTION, STAY_BAND_PX,
  createSim, startGame, resetBall, aliveBricks, setBallSpeed, step,
  measure, buildState, decidable, reflectX, rolloutReachable, buildLandingX, aimTarget,
} from "./static/breakout_sim.js";

const fresh = (over = {}) => {
  const sim = createSim({ seed: 7 });
  startGame(sim);
  return Object.assign(sim, over);
};

// ---- geometry ----------------------------------------------------------------

test("derived geometry matches the constants it was derived from", () => {
  assert.equal(CATCH_HALF, PADDLE_W / 2 + BALL_R);
  assert.equal(PADDLE_MIN, PADDLE_W / 2);
  assert.equal(PADDLE_MAX, W - PADDLE_W / 2);
  assert.equal(BALL_MIN_X, BALL_R);
  assert.equal(BALL_MAX_X, W - BALL_R);
  // The band is the vertical window in which a catch is registered.
  assert.equal(BAND_ENTER_Y, PADDLE_Y - BALL_R);
  assert.equal(BAND_EXIT_Y, PADDLE_Y + PADDLE_H + 6 - BALL_R);
  assert.equal(LAST_BRICK_BOTTOM, BRICK_TOP + (ROWS - 1) * (BRICK_H + BRICK_GAP) + BRICK_H);
});

// ---- the contact band cannot be stepped over ---------------------------------
//
// This is load-bearing for the rollout. If the ball could jump clean over the band in one
// frame, a catch would depend on where the frame boundaries happened to fall and the
// predicate would be measuring quantisation rather than reachability.

test("the contact band cannot be stepped over at any legal ball speed", () => {
  const bandHeight = BAND_EXIT_Y - BAND_ENTER_Y;
  // The slider tops out at 300 px/s and the frame dt is clamped to 33ms.
  const perFrame = 300 * 0.033;
  assert.ok(perFrame < bandHeight,
    `ball advances ${perFrame.toFixed(1)}px per frame but the band is only ${bandHeight}px`);
});

test("a ball descending fast still gets caught when it crosses the band", () => {
  const sim = fresh();
  setBallSpeed(sim, 300);
  sim.paddleX = W / 2;
  sim.ball = { x: W / 2, y: PADDLE_Y - BALL_R - 4, vx: 0, vy: 300 };
  const dt = 1 / 60;
  let caught = false;
  for (let i = 0; i < 10 && !caught; i++) caught = step(sim, dt).caught;
  assert.equal(caught, true, "a ball sitting on the paddle must not tunnel through it");
});

// ---- wall reflections ---------------------------------------------------------

test("the ball reflects off both side walls", () => {
  const sim = fresh();
  sim.ball = { x: BALL_MIN_X - 1, y: 200, vx: -50, vy: 0 };
  step(sim, 1 / 60, { keys: {} });
  assert.equal(sim.ball.x, BALL_MIN_X);
  assert.ok(sim.ball.vx > 0, "leftward velocity flips at the left wall");

  sim.ball = { x: BALL_MAX_X + 1, y: 200, vx: 50, vy: 0 };
  step(sim, 1 / 60, { keys: {} });
  assert.equal(sim.ball.x, BALL_MAX_X);
  assert.ok(sim.ball.vx < 0, "rightward velocity flips at the right wall");
});

test("reflectX is a triangle wave that never leaves the arena", () => {
  assert.equal(reflectX(100), 100, "inside the arena, position is unchanged");
  for (const start of [10, 240, 470]) {
    for (const vx of [-400, -137, 0, 91, 400]) {
      for (let t = 0; t < 4; t += 0.017) {
        const x = reflectX(start + vx * t);
        assert.ok(x >= BALL_MIN_X - 1e-9 && x <= BALL_MAX_X + 1e-9,
          `reflectX(${start}, ${vx}, t=${t}) escaped the arena: ${x}`);
      }
    }
  }
  // Crosses a wall and comes back rather than continuing through it.
  assert.ok(reflectX(BALL_MAX_X + 100) < BALL_MAX_X);
});

// ---- paddle --------------------------------------------------------------------

test("the paddle clamps inside the arena at both ends", () => {
  const sim = fresh();
  sim.paddleX = PADDLE_MIN;
  step(sim, 1 / 60, { keys: { ArrowLeft: true } });
  assert.equal(sim.paddleX, PADDLE_MIN, "cannot leave through the left wall");

  sim.paddleX = PADDLE_MAX;
  step(sim, 1 / 60, { keys: { ArrowRight: true } });
  assert.equal(sim.paddleX, PADDLE_MAX, "cannot leave through the right wall");
});

test("arrow keys override CJet's command when keys are supplied", () => {
  const sim = fresh();
  sim.paddleDir = 1;                        // CJet says right
  const before = sim.paddleX;
  step(sim, 1 / 60, { keys: { ArrowLeft: true } });
  assert.ok(sim.paddleX < before, "the human's keys win in manual mode");
});

// ---- bricks --------------------------------------------------------------------

test("a brick hit removes exactly one brick and scores one point", () => {
  const sim = fresh();
  assert.equal(aliveBricks(sim), ROWS * COLS);
  sim.ball = { x: W / 2, y: BRICK_TOP - BALL_R - 1, vx: 0, vy: 300 };
  const ev = step(sim, 1 / 240);
  assert.equal(ev.brickHit !== null, true);
  assert.equal(sim.score, 1);
  assert.equal(aliveBricks(sim), ROWS * COLS - 1);
});

test("clearing the board ends the run", () => {
  const sim = fresh();
  for (const row of sim.bricks) for (let c = 0; c < COLS; c++) row[c] = false;
  sim.bricks[0][0] = true;
  sim.ball = { x: BRICK_GAP + 1, y: BRICK_TOP - BALL_R - 1, vx: 0, vy: 300 };
  const ev = step(sim, 1 / 240);
  assert.equal(ev.cleared, true);
  assert.equal(sim.running, false);
});

// ---- lives ---------------------------------------------------------------------

test("falling costs a life and respawns until the lives run out", () => {
  const sim = fresh();
  for (let i = 0; i < 3; i++) {
    sim.ball = { x: W / 2, y: H + BALL_R + 1, vx: 0, vy: 300 };
    sim.paddleX = 0;                        // nowhere near the ball
    const ev = step(sim, 1 / 240);
    assert.equal(ev.lost, true);
  }
  assert.equal(sim.lives, 0);
  assert.equal(sim.running, false, "the run ends when the last life is gone");
});

// ---- the speed slider ----------------------------------------------------------

test("setBallSpeed rescales the live ball without changing its direction", () => {
  const sim = fresh();
  sim.ball = { x: 200, y: 100, vx: 30, vy: -160 };
  const before = Math.sign(sim.ball.vx), beforeY = Math.sign(sim.ball.vy);
  setBallSpeed(sim, 300);
  const speed = Math.hypot(sim.ball.vx, sim.ball.vy);
  assert.ok(Math.abs(speed - 300) < 1e-9, `expected 300, got ${speed}`);
  assert.equal(Math.sign(sim.ball.vx), before);
  assert.equal(Math.sign(sim.ball.vy), beforeY);
});

test("the default ball speed matches the slider's default", () => {
  assert.equal(DEFAULT_CONFIG.ballSpeed, 160, "breakout.html ships the slider at 160");
});

// ---- determinism ---------------------------------------------------------------

test("the same seed replays identically", () => {
  const run = () => {
    const sim = createSim({ seed: 42 });
    startGame(sim);
    const trace = [];
    for (let i = 0; i < 400; i++) {
      const ev = step(sim, 1 / 120, { keys: {} });
      trace.push([sim.ball.x, sim.ball.y, sim.score, ev.caught, ev.lost]);
    }
    return trace;
  };
  assert.deepEqual(run(), run(), "same seed, same game");
});

test("different seeds produce different games", () => {
  // Sampled across a range rather than two adjacent seeds: mulberry32's first draw for
  // seeds 1-5 all land above 0.5, so ball.vx is +80 for every one of them. That is the
  // generator being fine and the sample being unrepresentative, not a seeding bug.
  const vx = new Set();
  for (let seed = 1; seed <= 40; seed++) vx.add(createSim({ seed }).ball.vx);
  assert.ok(vx.size > 1, "the seed has to actually reach the ball");
});

test("the simulation module contains no Math.random and no DOM access", async () => {
  const { readFile } = await import("node:fs/promises");
  const src = await readFile(new URL("./static/breakout_sim.js", import.meta.url), "utf8");
  // Strip comments before scanning: the file *discusses* Math.random in a comment saying
  // why it is not used, and a substring check would flag that as the very thing banned.
  const code = src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
  assert.equal(/\bMath\s*\.\s*random\b/.test(code), false,
    "an unseeded RNG would make every harness comparison meaningless");
  for (const dom of ["document.", "window.", "getElementById", "canvas"]) {
    assert.equal(code.includes(dom), false, `breakout_sim.js must not touch ${dom}`);
  }
});

// ---- reachability: the decidable band -----------------------------------------

test("a rising ball is never a loss event, so no verdict is given", () => {
  const ball = { x: 240, y: 200, vx: 0, vy: -160 };
  assert.equal(decidable(ball), false);
  assert.equal(rolloutReachable(ball, 240), null,
    "null means 'no verdict exists' — NOT the same as unreachable");
});

test("a ball above the bricks is not decidable, because a brick may deflect it", () => {
  const above = { x: 240, y: LAST_BRICK_BOTTOM + BALL_R - 1, vx: 0, vy: 160 };
  assert.equal(decidable(above), false, "just above the lowest brick edge");
  const below = { x: 240, y: LAST_BRICK_BOTTOM + BALL_R + 1, vx: 0, vy: 160 };
  assert.equal(decidable(below), true, "just below it");
});

test("a ball already at paddle level is past deciding", () => {
  assert.equal(decidable({ x: 240, y: BAND_ENTER_Y + 1, vx: 0, vy: 160 }), false);
});

test("a ball directly overhead is catchable", () => {
  const ball = { x: 240, y: 300, vx: 0, vy: 160 };
  assert.equal(rolloutReachable(ball, 240), true);
});

test("a ball too far away, with too little time, is unreachable", () => {
  // 300px away with 0.19s left: the paddle covers ~39px, less than its own catch width.
  const ball = { x: 460, y: 300, vx: 0, vy: 160 };
  assert.equal(rolloutReachable(ball, 120), false);
});

test("the rollout does not mutate the state it was handed", () => {
  const ball = { x: 412, y: 200, vx: 41, vy: 160 };
  const frozen = JSON.stringify(ball);
  rolloutReachable(ball, 80);
  assert.equal(JSON.stringify(ball), frozen, "the reasoner must not corrupt the live game");
});

test("the rollout agrees with the game it is supposed to describe", () => {
  // The strongest available check short of the full sweep in test_breakout_ontology.mjs:
  // set up a real ball, let the real step() play it out, and ask the rollout about the
  // same state. If the rollout disagreed with step() here, every verdict in the panel
  // would be about a different game.
  let agree = 0, total = 0;
  for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
    const sim = createSim({ seed });
    startGame(sim);
    for (let i = 0; i < 600 && sim.running; i++) {
      const verdict = rolloutReachable(sim.ball, sim.paddleX);
      if (verdict !== null) {
        total++;
        // Play the real physics from here with a paddle that does the predicted thing.
        // `aimTarget`, not `buildLandingX` — see its doc comment for why aiming at the
        // landing point alone disagrees with the rollout on a third of all states.
        //
        // Stop at `lost`, not just at `caught`. Losing costs a life and `step()` respawns a
        // fresh ball at the paddle, which this paddle may well catch — so a probe that only
        // stopped on a catch would score that second ball as agreement with a verdict about
        // the first one.
        const probe = createSim({ seed: 1 });
        startGame(probe);
        probe.ball = { ...sim.ball };
        probe.paddleX = sim.paddleX;
        let caught = false;
        for (let k = 0; k < 900 && !caught; k++) {
          const dir = Math.sign(aimTarget(probe.ball) - probe.paddleX);
          probe.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN,
            probe.paddleX + dir * PADDLE_SPEED * (1 / 240)));
          const ev = step(probe, 1 / 240);
          caught = ev.caught;
          if (ev.lost) break;
        }
        if (caught === verdict) agree++;
      }
      step(sim, 1 / 120, { keys: {} });
    }
  }
  assert.ok(total > 200, `expected a decent sample, got ${total} decidable states`);
  assert.equal(agree, total,
    `rollout disagreed with the physics on ${total - agree} of ${total} states`);
});

// ---- what the model is told -----------------------------------------------------

test("the two regimes disagree on this ball — and that disagreement is the demo", () => {
  // A ball drifting hard left while far from the paddle: its CURRENT x says LEFT, its
  // landing point says RIGHT. The fixture is pinned so a physics change cannot quietly
  // make the disagreement vanish — it is the whole thing being demonstrated.
  const sim = fresh();
  sim.paddleX = 72;
  sim.ball = { x: 34, y: 131, vx: -241, vy: 62 };
  const m = measure(sim);
  // Pinned so a later physics change cannot quietly make this fixture vacuous.
  assert.equal(Math.round(m.landingX), 197);
  assert.equal(Math.round(m.gap), 125);
  assert.notEqual(Math.sign(sim.ball.x - sim.paddleX), Math.sign(m.gap),
    "this fixture is only meaningful while current and landing disagree");
  assert.equal(m.gap, m.landingX - sim.paddleX);
  assert.match(buildState(sim).situation, /RIGHT/,
    "ontology ON: the observation follows the LANDING point — right of the paddle");
  // The landing sentence restates the landing SIDE, not the raw landingX: absolute
  // coordinates appear in no training state (0 occurrences of "x≈" in the breakout
  // JSONL) and collapsed the local ciel head to one class.
  assert.match(buildState(sim).situation, /It will land clearly to the RIGHT of the paddle \(gap \d+ px\)/);
  assert.doesNotMatch(buildState(sim).situation, /x≈/,
    "absolute coordinates are gone — they are out of the head's training distribution");
  sim.config.enforceReachability = false;
  const off = buildState(sim).situation;
  assert.match(off, /LEFT/,
    "ontology OFF: the observation follows the CURRENT position — and says the wrong side");
  assert.doesNotMatch(off, /It will land/);
});

test("with the ontology on, the input states the landing side in the gap prose", () => {
  // ON: the side is named from the LANDING point (gap = landingX − paddleX, what the
  // 12.4% fix made it) and the landing sentence restates that gap in the phrasing the
  // renderer emits. It used to print the two absolute x positions; that form appears in
  // no training state and collapsed the local ciel head, so it was replaced by prose.
  // OFF: the side is named from the ball's CURRENT position — the deliberate naive
  // baseline, which on this fixture names the opposite side.
  const sim = fresh({ ball: { x: 200, y: 200, vx: 200, vy: 160 } });
  assert.equal(decidable(sim.ball), true, "fixture must be a decidable ball");
  const m = measure(sim);
  assert.ok(Math.abs(m.gap) > 15 && Math.abs(sim.ball.x - sim.paddleX) > 15,
    "fixture must name a side in BOTH regimes, or the contrast is invisible");
  assert.notEqual(Math.sign(sim.ball.x - sim.paddleX), Math.sign(m.gap),
    "fixture is only meaningful while current and landing disagree");
  const on = buildState(sim).situation;
  assert.match(on, /It will land /, "ON states where the ball will land");
  assert.match(on, new RegExp(`gap ${Math.abs(Math.round(m.gap))} px`),
    "the landing sentence restates the rollout's gap");
  assert.doesNotMatch(on, /x≈/,
    "no absolute coordinates: absent from training, and they collapsed the head");
  assert.match(on, /gap \d+ px/, "the gap prose survives — the model keys on it (measured)");
  assert.match(on, m.gap > 0 ? /RIGHT/ : /LEFT/, "ON names the landing side");
  sim.config.enforceReachability = false;
  const off = buildState(sim).situation;
  assert.match(off, /gap \d+ px/, "switch off: the prose shape survives");
  assert.doesNotMatch(off, /It will land/, "switch off: no prediction, no landing sentence");
  assert.match(off, sim.ball.x - sim.paddleX > 0 ? /RIGHT/ : /LEFT/,
    "OFF names the CURRENT side — the deliberate naive baseline");
});

test("the descent clause is a sentence, not a clause merged into the geometry", () => {
  // The renderer emits "Falling toward the paddle." as its OWN sentence
  // (states/breakout.py `_motion_echo`). Merging it into the geometry sentence
  // ("...and down, falling toward the paddle") puts a phrase the head never saw in
  // that position into the attended field, and it dominated the geometry: measured
  // against the local decision_lab head, the merged form answered "left" for a ball
  // plainly right of the paddle (right-prob 0.41, vs OK with the clause separated).
  const sim = fresh({ ball: { x: 600, y: 200, vx: -200, vy: 160 } });
  sim.paddleX = 100;
  const m = measure(sim);
  assert.equal(m.descending, true, "fixture must be a descending ball");
  const on = buildState(sim).situation;
  assert.match(on, /down, toward the paddle\. Falling toward the paddle\./,
    "descent is stated as two sentences, matching the training prose");
  assert.doesNotMatch(on, /down, falling/,
    "the merged clause is what breaks the head — it must not come back");
  // The rising case keeps its own single phrase (also verbatim from _motion_echo).
  const rising = fresh({ ball: { x: 600, y: 200, vx: -200, vy: -160 } });
  rising.paddleX = 100;
  assert.equal(measure(rising).descending, false, "fixture must be a rising ball");
  assert.match(buildState(rising).situation, /up, away from the paddle/);
});

test("a centered ball takes the renderer's CENTER path, not the side path", () => {
  // states/breakout.py renders side == "center" differently in BOTH parts of the
  // motion clause: _short_motion ("drifting left, falling") instead of
  // _motion_phrase ("moving left and down, toward the paddle"), and
  // _motion_echo(centre) == "The paddle is under the ball" instead of the
  // toward/away clause. "moving left and down" occurs 0 times among the training
  // center states, and the side path was measured to answer a centered ball
  // "right" with stay-prob as low as 0.00; the center path answers "stay"
  // throughout (0.88-0.94 over gaps 0..24).
  // OFF regime, so the side reads straight from ball.x - paddleX and the fixture
  // is exact. (Under ON the gap is the landing point, which this fixture does not
  // try to control — the center path is reached in both regimes the same way.)
  const sim = fresh({ ball: { x: 348, y: 200, vx: -40, vy: 160 } });
  sim.paddleX = 350;                       // 2 px off: inside STAY_BAND_PX
  sim.config.enforceReachability = false;
  assert.ok(Math.abs(sim.ball.x - sim.paddleX) <= STAY_BAND_PX,
    "fixture must be centered under the sim's own band");
  const off = buildState(sim).situation;
  assert.match(off, /almost directly above the paddle \(offset \d+ px\)/);
  assert.match(off, /drifting left, falling/,
    "centred uses the short motion words, not the side phrase");
  assert.match(off, /The paddle is under the ball/,
    "and the centre echo, which is what marks the ball as centered");
  assert.doesNotMatch(off, /moving left and down/,
    "the side motion phrase is what the head has no training for at center");
});

test("the centered band matches the training threshold, not a tighter one", () => {
  // The head learned "stay" for gaps 0..24 (STAY_THRESHOLD_PX) and flips only
  // past it — measured: stay-prob 0.90 at both 16 and 24, then 0.01 at 26. A
  // narrower band would describe a 20 px gap as a side, which is both wrong to
  // the head and prose the renderer never emits for that number.
  assert.equal(STAY_BAND_PX, 24);
  const sim = fresh({ ball: { x: 370, y: 200, vx: -40, vy: 160 } });
  sim.paddleX = 350;                       // 20 px — inside the band
  sim.config.enforceReachability = false;
  assert.match(buildState(sim).situation, /almost directly above the paddle/,
    "20 px is centered: inside the training band");
  sim.ball = { x: 382, y: 200, vx: -40, vy: 160 };   // 32 px — past it
  assert.match(buildState(sim).situation, /clearly to the RIGHT of the paddle/,
    "32 px is a side: the training band ends before it");
});

test("the ON observation names the wrong side 0% of the time; the OFF baseline 12.4%", () => {
  // The number this whole feature is priced against. Comparing against the CURRENT x —
  // what the OFF regime deliberately does again, as the naive baseline — named the wrong
  // side 12.4% of the time. The ON observation names the landing side and must never err.
  // Both rates are measured on the ACTUAL prose each regime emits, not on the gap it was
  // computed from, because the prose is what the model reads.
  const sideOf = s => /RIGHT of the paddle/.test(s) ? 1
    : /LEFT of the paddle/.test(s) ? -1 : 0;
  let wrongOn = 0, wrongOff = 0, checked = 0;
  for (let seed = 1; seed <= 4000; seed++) {
    const sim = createSim({ seed });
    startGame(sim);
    sim.ball = { x: sim.rng() * W, y: 121 + sim.rng() * (BAND_ENTER_Y - 121),
                 vx: (sim.rng() - 0.5) * 520, vy: 60 + sim.rng() * 240 };
    sim.paddleX = PADDLE_MIN + sim.rng() * (PADDLE_MAX - PADDLE_MIN);
    const landing = buildLandingX(sim.ball);
    if (landing === null) continue;
    const truth = Math.sign(landing - sim.paddleX);
    if (truth === 0) continue;              // genuinely centred: no side to name
    checked++;
    sim.config.enforceReachability = true;
    const onSide = sideOf(buildState(sim).situation);
    if (onSide !== 0 && onSide !== truth) wrongOn++;
    sim.config.enforceReachability = false;
    const offSide = sideOf(buildState(sim).situation);
    if (offSide !== 0 && offSide !== truth) wrongOff++;
  }
  assert.ok(checked > 3000, `expected a large sample, got ${checked}`);
  assert.equal(wrongOn, 0,
    `the ON observation named the wrong side ${wrongOn} times — must be 0`);
  assert.ok(wrongOff > 0,
    "the OFF baseline must name a wrong side sometimes, or the contrast is not real");
  const rate = wrongOff / checked;
  assert.ok(rate > 0.05 && rate < 0.25,
    `OFF wrong-side rate ${(100 * rate).toFixed(1)}% — expected ~12.4%; ` +
    `a big move means the fixture or the regime drifted`);
});

test("the tolerance is non-negative, because a false 'unreachable' is the costly error", () => {
  assert.ok(CATCH_TOL_PX >= 0,
    "a negative tolerance would let the reasoner veto a catchable ball — the one error "
    + "clause 2 of the invariant forbids");
});

// ---- the question ---------------------------------------------------------------

test("the question still offers exactly the three paddle commands", () => {
  assert.equal(QUESTION.type, "choice");
  assert.deepEqual(Object.keys(QUESTION.criteria), ["left", "stay", "right"]);
});
