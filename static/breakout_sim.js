// The Breakout simulation, as a pure module.
//
// WHY THIS IS A SEPARATE FILE. breakout.html used to hold its physics inline, which
// meant the only way to check any claim about it was to load a browser and count bricks
// on screen. fleet_sim.js was split out for exactly this reason, and the comment there
// says it: the browser and the A/B harness must not be able to drift apart. This file is
// the half that can be tested; breakout.html is only the DOM layer over it.
//
// NOTHING HERE TOUCHES THE DOM. No document, no window, no canvas. `step()` returns
// events and lets the caller update whatever it likes — the HUD in the browser, nothing
// at all in the node harness. This is what makes the reachability rollout below
// trustworthy: it runs the same code the game runs.

import { mulberry32 } from "./rng.js";

// ---- fixed game geometry -------------------------------------------------------

export const W = 480, H = 360;
export const ROWS = 5, COLS = 8;
export const BRICK_TOP = 30, BRICK_H = 14, BRICK_GAP = 4;
export const BRICK_W = (W - 2 * BRICK_GAP) / COLS - BRICK_GAP;
export const PADDLE_W = 70, PADDLE_H = 10, PADDLE_Y = H - 24;
export const PADDLE_SPEED = 200;           // px/s, same in both modes
export const BALL_R = 5;
// |gap| at or under this reads as "centered". Mirrors the ciel head's training
// renderer (states/breakout.py STAY_THRESHOLD_PX = 24 — that dataset's gold is
// "stay" for gaps 0..24 and a side from 32 up). The observation is prose for a
// head trained on that data, so the band has to match it: the head answers
// "stay" across 0..24 and flips only past it, and describing a 20 px gap as
// "clearly to the RIGHT" is phrasing the renderer never emits.
export const STAY_BAND_PX = 24;
export const ROW_COLORS = ["#f87171", "#fb923c", "#facc15", "#4ade80", "#60a5fa"];

// Derived geometry, named once because three separate things need them and two of those
// things are the reason the ontology cannot be trusted if they drift apart.
export const CATCH_HALF = PADDLE_W / 2 + BALL_R;              // 40 — paddle centre ± this
export const PADDLE_MIN = PADDLE_W / 2, PADDLE_MAX = W - PADDLE_W / 2;
export const BALL_MIN_X = BALL_R, BALL_MAX_X = W - BALL_R;
// The ball is caught anywhere in this band, so the FIRST legal contact is the bottom edge
// of the ball meeting the top of the paddle.
export const BAND_ENTER_Y = PADDLE_Y - BALL_R;
export const BAND_EXIT_Y = PADDLE_Y + PADDLE_H + 6 - BALL_R;
// Lowest brick edge. A ball descending from above this can be deflected before it ever
// reaches the paddle, so nothing can be said about whether it is catchable.
export const LAST_BRICK_BOTTOM = BRICK_TOP + (ROWS - 1) * (BRICK_H + BRICK_GAP) + BRICK_H;

/** Can the reachability verdict be given at all? Only below the bricks, while falling. */
export function decidable(ball) {
  return ball.vy > 0 && ball.y - BALL_R > LAST_BRICK_BOTTOM && ball.y < BAND_ENTER_Y;
}

// ---- reachability: the rollout -------------------------------------------------
//
// THIS IS THE PREDICATE, and it is a forward simulation on purpose.
//
// The obvious implementation is closed-form: work out the time to the paddle, multiply by
// PADDLE_SPEED, compare against the gap. Four versions of that were written and measured
// against this rollout over 40,000 random states, and every one was wrong — 62% to 83%
// agreement. The reason is that three separate things break the algebra: the ball keeps
// drifting sideways while it falls, it reflects off the side walls, and a catch happens if
// the ball is in reach at ANY moment in the contact band rather than only at one instant.
//
// Worse, the residual error runs in one direction. The closed forms declared thousands of
// balls uncatchable that the physics actually catches — which is precisely the failure
// this whole feature exists to fix, since it would exculpate the model on balls it lost.
//
// Cost, measured: ~9us for a typical ball, ~95us in the worst case (a ball descending at
// ~50px/s, which is the longest horizon — see `rolloutHorizon`). Decision ticks are 150-500ms
// apart, so even the worst case is ~0.06% of the gap between asks. There is no honest
// argument for the shortcut.

/** Triangle wave: where the ball is after `u` px of sideways travel, reflections included. */
export function reflectX(u) {
  const span = BALL_MAX_X - BALL_MIN_X;
  const m = (((u - BALL_MIN_X) % (2 * span)) + 2 * span) % (2 * span);
  return BALL_MIN_X + (m <= span ? m : 2 * span - m);
}

/** One physics step, shared by the game and the rollout. Mutates `s`. */
function physicsStep(s, dt, dir) {
  s.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN, s.paddleX + dir * PADDLE_SPEED * dt));
  s.ball.x += s.ball.vx * dt;
  s.ball.y += s.ball.vy * dt;
  if (s.ball.x < BALL_MIN_X) { s.ball.x = BALL_MIN_X; s.ball.vx = Math.abs(s.ball.vx); }
  if (s.ball.x > BALL_MAX_X) { s.ball.x = BALL_MAX_X; s.ball.vx = -Math.abs(s.ball.vx); }
  if (s.ball.y < BALL_R) { s.ball.y = BALL_R; s.ball.vy = Math.abs(s.ball.vy); }
}

// dt for the rollout. 1/240 measured 0 verdict flips against a 2000Hz reference over
// 8000 states; 1/120 flipped 3. Chosen for stability, not speed — it is still ~1us.
const ROLLOUT_DT = 1 / 240;

// How long the rollout simulates.
//
// DERIVED, NOT PICKED. A fixed 2.5s was wrong, and `test_breakout_ontology.mjs` found it:
// 1136 of 21000 sampled balls were declared unreachable when the physics caught them, every
// one of them a slow ball. A paddle bounce can leave the ball descending as slowly as
// ~50px/s (max exit angle is 60deg off vertical, so vy bottoms out near speed/2), and
// falling the ~208px from the bottom brick to the catch band then takes over 4 seconds. The
// rollout gave up at 2.5s and reported "unreachable" — precisely the unsound error this
// whole feature must never make, and it would have shown up only as slightly too many
// "provably lost" labels in the demo.
//
// The horizon is the time the ball actually needs, plus a fixed allowance for the paddle to
// keep up once it arrives. Capped so a pathological vy cannot spin the loop.
const ROLLOUT_GRACE_S = 0.5;
const ROLLOUT_MAX_S = 12;

function rolloutHorizon(ball) {
  const fall = Math.max(0, (BAND_ENTER_Y - ball.y) / ball.vy);
  return Math.min(fall + ROLLOUT_GRACE_S, ROLLOUT_MAX_S);
}

/**
 * Could a paddle catch this ball at all, if it aimed perfectly?
 *
 * The paddle re-aims at the predicted intercept every step, which is what the real game
 * does every 150-500ms. That matters: measured against an exhaustive search over the three
 * held directions, a re-aiming paddle catches 22.9% more balls than any single committed
 * direction can, because it is free to reverse mid-fall. Modelling a held command here
 * would have understated catchability by a fifth.
 *
 * `s.bricks` is deliberately ignored: `decidable()` has already established the ball is
 * below the brick field, so no brick can intercept it.
 *
 * Does not mutate anything the caller owns.
 */
export function rolloutReachable(ball, paddleX) {
  if (!decidable(ball)) return null;            // no verdict exists, not a "no"
  const s = { ball: { x: ball.x, y: ball.y, vx: ball.vx, vy: ball.vy }, paddleX };
  for (let t = 0; t < rolloutHorizon(ball); t += ROLLOUT_DT) {
    if (s.ball.vy <= 0) return false;
    physicsStep(s, ROLLOUT_DT, Math.sign(aimTarget(s.ball) - s.paddleX));
    if (s.ball.y > H + BALL_R) return false;
    if (s.ball.vy > 0 && s.ball.y + BALL_R >= PADDLE_Y &&
        s.ball.y + BALL_R <= PADDLE_Y + PADDLE_H + 6 &&
        Math.abs(s.ball.x - s.paddleX) <= CATCH_HALF) return true;
  }
  return false;
}

// Tolerance on the catch half-width. Non-negative ON PURPOSE: it can only turn
// "unreachable" into "reachable", never the reverse, and a false "unreachable" is the
// costly error because it silences a question the model could have answered. Sourced
// from the integrator — one frame of paddle travel — rather than picked to make a test
// pass. A test asserts it stays >= 0 so it cannot be "tidied" negative later.
export const CATCH_TOL_PX = Math.ceil(PADDLE_SPEED * 0.033);

/** Where the ball will first be catchable, or null when that is not yet knowable. */
export function buildLandingX(ball) {
  if (!decidable(ball)) return null;
  const t = (BAND_ENTER_Y - ball.y) / ball.vy;
  return reflectX(ball.x + ball.vx * t);
}

/**
 * Where a paddle should aim right now — the definition of perfect play, in one place.
 *
 * Above the catch band, aim at the landing point. Once the ball is IN the band the
 * landing point is behind it, so aim at where it actually is.
 *
 * The second clause is not a detail. An earlier version of the test harness used
 * `buildLandingX` alone and stopped steering the moment the ball entered the band — that
 * is, it gave up during the only window where a catch is still possible, and disagreed
 * with the rollout on a third of all states. The bug was in the harness, but it is easy
 * to repeat, so the rule lives here instead of being open-coded at each call site.
 */
export function aimTarget(ball) {
  const landing = buildLandingX(ball);
  return landing === null ? ball.x : landing;
}

/** The one `measure()` both the prose and the reasoner read. See `buildState`. */
export function measure(sim) {
  const { ball, paddleX } = sim;
  const descending = ball.vy > 0;
  const landingX = buildLandingX(ball);
  return {
    descending,
    // The gap to where the ball will ARRIVE, not where it is. Using the current x is the
    // bug this replaced: the ball drifts and reflects while it falls, so in 12.4% of
    // decidable states the two point opposite ways and the model was told to go the wrong
    // side. See `buildState`.
    gap: landingX === null ? ball.x - paddleX : landingX - paddleX,
    landingX,
    timeToContact: descending && ball.y < BAND_ENTER_Y
      ? (BAND_ENTER_Y - ball.y) / ball.vy : null,
    reachable: rolloutReachable(ball, paddleX),
  };
}

// ---- simulation state ------------------------------------------------------------

// `enforceReachability` is the ontology's master switch, driven by the "Ontology: on/off"
// button in breakout.html's Controls row. It gates EVERY behavioural involvement the
// ontology has — question suppression (`shouldAsk`) AND the stale-command fix in the frame
// loop — because the principle is one-directional: when the ontology judges, its verdicts
// control; when it is off, it controls nothing. There is no third state where it judges
// but does not act.
export const DEFAULT_CONFIG = { ballSpeed: 160, decideEvery: 0.25, enforceReachability: true };

export function createSim({ seed = 1, config = {} } = {}) {
  const sim = {
    ball: null, paddleX: W / 2, bricks: [], score: 0, lives: 3,
    running: false, paused: false,
    paddleDir: 0,            // CJet's current command: -1 | 0 | 1
    // What that command was issued against, and whether the world has moved since.
    //
    // `issuedVXGapSign` is the SIGN of the gap — which side of the paddle the ball's
    // landing point was on — at the moment the answer was applied. That sign, not the
    // landing coordinate, is what a later reflection can invalidate; see
    // `commandIsStale` for why the coordinate cannot be. `null` means no command has been
    // issued yet, which is not the same as "issued and still valid".
    issuedVXGapSign: 0,
    // Set when the ball's horizontal velocity reverses while a command is held. It is the
    // EVENT the axiom is about; `commandIsStale` is only consulted once this is true.
    bouncedSinceIssue: false,
    config: { ...DEFAULT_CONFIG, ...config },
    rng: mulberry32(seed),
  };
  resetBall(sim);
  return sim;
}

/**
 * Record a command and the direction it was issued against.
 *
 * Called whenever the model's answer is applied, so the pair stays in step with `paddleDir`.
 * The sign is taken with the same deadband the decision loop uses (`|gap| <= 1` reads as
 * Stay), so a command issued while the paddle is already aligned is not recorded as
 * pointing left or right and then reported stale for merely being aligned.
 */
export function issueCommand(sim, dir) {
  sim.paddleDir = dir;
  const gap = measure(sim).gap;
  sim.issuedVXGapSign = Math.abs(gap) <= 1 ? 0 : Math.sign(gap);
  sim.bouncedSinceIssue = false;
}

/**
 * Has the command's target been cancelled?
 *
 * ONLY consult this after `bouncedSinceIssue` is true — it is false otherwise, because a
 * command steering toward a landing point that has not moved is not stale, however long ago
 * it was issued.
 *
 * WHY IT COMPARES SIGNS AND NOT COORDINATES, which is the whole design and the mistake an
 * earlier version made. The obvious reading of "the command was issued for a landing point,
 * and the ball has since reflected" is to store the landing coordinate and compare it with
 * the current one. That detects nothing, and not because of a subtlety — `buildLandingX`
 * folds the trajectory through `reflectX`, so the predicted landing point is CONTINUOUS
 * across a reflection by construction. Measured over 12 reflections observed mid-descent,
 * the issued and current landing points agreed to within 0.7 px every time; the premise the
 * coordinate comparison rests on is simply false.
 *
 * What a reflection actually invalidates is the SIDE of the paddle the ball will arrive on.
 * That is what the paddle is driving toward and what the model's answer named, and a
 * reflection can invert it while leaving the predicted coordinate nearly where it was. So
 * the axiom is stated over the sign, which is the part that can change.
 *
 * A stale command is only reported when the held direction is non-zero and disagrees with
 * the live truth: a held Stay is never stale, because there is no direction to be wrong
 * about, and reporting one would fire the correction on every settled ball.
 */
export function commandIsStale(sim) {
  if (!sim.bouncedSinceIssue || !sim.ball) return false;
  if (sim.issuedVXGapSign === 0 || sim.paddleDir === 0) return false;
  const gap = measure(sim).gap;
  const truth = Math.abs(gap) <= 1 ? 0 : Math.sign(gap);
  return truth !== 0 && truth !== sim.paddleDir;
}

export function resetBall(sim) {
  // Seeded, not Math.random(): a scenario has to replay identically for the harness to be
  // comparing anything. Mirrors fleet_sim.js, which made the same change for the same reason.
  sim.ball = {
    x: W / 2, y: PADDLE_Y - BALL_R - 1,
    vx: (sim.rng() < 0.5 ? -1 : 1) * sim.config.ballSpeed * 0.5,
    vy: -sim.config.ballSpeed,
  };
  sim.paddleDir = 0;
  // A respawned ball invalidates any command aimed at the old one, and it has by definition
  // not bounced yet, so both fields go back to their no-command-issued values.
  sim.issuedVXGapSign = 0;
  sim.bouncedSinceIssue = false;
}

export function startGame(sim) {
  sim.paddleX = W / 2;
  sim.score = 0; sim.lives = 3;
  sim.bricks = [];
  for (let r = 0; r < ROWS; r++) sim.bricks.push(Array.from({ length: COLS }, () => true));
  resetBall(sim);
  sim.running = true; sim.paused = false;
}

export const aliveBricks = sim =>
  sim.bricks.reduce((n, row) => n + row.filter(Boolean).length, 0);

/** Rescale the live ball to a new speed, as the slider did mid-flight. */
export function setBallSpeed(sim, speed) {
  sim.config.ballSpeed = speed;
  if (!sim.ball) return;
  const s = Math.hypot(sim.ball.vx, sim.ball.vy) || 1;
  const scale = speed / s;
  sim.ball.vx *= scale; sim.ball.vy *= scale;
}

/**
 * Advance one frame. Returns what happened, so the caller can update its HUD — physics
 * must not reach into the DOM, or the harness could not drive this.
 */
export function step(sim, dt, { keys = null } = {}) {
  const ev = { caught: false, lost: false, brickHit: null, cleared: false };
  if (!sim.running || sim.paused) return ev;

  // paddle: CJet's command, or the player's arrow keys when autoplay is off
  let dir = sim.paddleDir;
  if (keys) dir = (keys.ArrowLeft ? -1 : 0) + (keys.ArrowRight ? 1 : 0);
  sim.paddleX = Math.min(PADDLE_MAX, Math.max(PADDLE_MIN, sim.paddleX + dir * PADDLE_SPEED * dt));

  const ball = sim.ball;
  ball.x += ball.vx * dt;
  ball.y += ball.vy * dt;

  // Record the horizontal reflection, so `commandIsStale` has the EVENT it is about. Set
  // from the velocity sign rather than from the wall-clamp branches below because a paddle
  // bounce also reverses vx and would otherwise be indistinguishable from a wall bounce —
  // and only the wall bounce cancels the landing point the command was issued against.
  const vxBefore = ball.vx;
  if (ball.x < BALL_MIN_X) { ball.x = BALL_MIN_X; ball.vx = Math.abs(ball.vx); }
  if (ball.x > BALL_MAX_X) { ball.x = BALL_MAX_X; ball.vx = -Math.abs(ball.vx); }
  if (ball.y < BALL_R) { ball.y = BALL_R; ball.vy = Math.abs(ball.vy); }
  if (Math.sign(ball.vx) !== Math.sign(vxBefore)) sim.bouncedSinceIssue = true;

  // paddle bounce: exit angle depends on where the ball lands on the paddle
  if (ball.vy > 0 && ball.y + BALL_R >= PADDLE_Y &&
      ball.y + BALL_R <= PADDLE_Y + PADDLE_H + 6 &&
      Math.abs(ball.x - sim.paddleX) <= CATCH_HALF) {
    const hit = (ball.x - sim.paddleX) / (PADDLE_W / 2);   // -1 .. 1
    const speed = Math.hypot(ball.vx, ball.vy);
    const angle = hit * (Math.PI / 3);                      // max 60° from vertical
    ball.vx = speed * Math.sin(angle);
    ball.vy = -Math.abs(speed * Math.cos(angle));
    ball.y = PADDLE_Y - BALL_R;
    ev.caught = true;
  }

  // bricks: kill the first one the ball overlaps, reflect on the shallower axis.
  // Transcribed exactly as the inline version had it — the rollout and the live game must
  // agree here or the verdict is about a different game than the one being played.
  const rowH = BRICK_H + BRICK_GAP;
  const rHit = Math.floor((ball.y - BRICK_TOP) / rowH);
  for (const r of [rHit, rHit - 1, rHit + 1]) {
    if (r < 0 || r >= ROWS) continue;
    for (let c = 0; c < COLS; c++) {
      if (!sim.bricks[r][c]) continue;
      const bx = BRICK_GAP + c * (BRICK_W + BRICK_GAP);
      const by = BRICK_TOP + r * rowH;
      if (ball.x + BALL_R < bx || ball.x - BALL_R > bx + BRICK_W ||
          ball.y + BALL_R < by || ball.y - BALL_R > by + BRICK_H) continue;
      sim.bricks[r][c] = false;
      sim.score++;
      ev.brickHit = { r, c };
      const overlapX = Math.min(ball.x + BALL_R - bx, bx + BRICK_W - (ball.x - BALL_R));
      const overlapY = Math.min(ball.y + BALL_R - by, by + BRICK_H - (ball.y - BALL_R));
      if (overlapX < overlapY) ball.vx = -ball.vx; else ball.vy = -ball.vy;
      if (aliveBricks(sim) === 0) { sim.running = false; ev.cleared = true; }
      return ev;
    }
  }

  // fell past the paddle
  if (ball.y > H + BALL_R) {
    sim.lives--;
    ev.lost = true;
    if (sim.lives <= 0) sim.running = false; else resetBall(sim);
  }
  return ev;
}

// ---- what the model is told -------------------------------------------------------

export const QUESTION = {
  type: "choice",
  instructions: "In which direction should the paddle move to catch or intercept the ball?",
  criteria: {
    left: "the ball is to the left of the paddle or moving away from it on the right",
    stay: "the paddle is already aligned under the ball's landing point",
    right: "the ball is to the right of the paddle or moving away from it on the left",
  },
};

/**
 * The observation, as prose.
 *
 * A FORMATTER over `measure()` — it computes nothing itself. That is the whole point: the
 * number the model reads and the number the reasoner reasons over are the same number, so
 * they cannot drift, and the ontology has a real claim on the model's input rather than
 * sitting beside it in a panel.
 *
 * The landing point is what made this correct. The previous version compared the paddle to
 * the ball's CURRENT x, which is wrong whenever the ball drifts or reflects on the way
 * down: measured over 80,000 decidable states, that named the wrong side 12.4% of the
 * time. Asking about the landing point instead takes that to zero by construction.
 *
 * Worth being straight about what that did and did not buy: it did NOT change the score.
 * Playing 25 seeded games with a perfect controller, aiming at the current x lost 1025
 * balls and aiming at the landing point lost 1025. A paddle driving toward the landing
 * point sweeps through the ball's current x on the way, so the re-aiming controller
 * absorbs the difference on its own. The gain is that the model is no longer *told* the
 * wrong thing — which stops mattering the moment the ball is close, or the moment a
 * controller commits to one side for the whole descent.
 */
export function buildState(sim) {
  const m = measure(sim);
  const { ball } = sim;
  // Two observation regimes, split on the ontology's master switch, and the difference IS
  // the demo:
  //
  // OFF — the NAIVE observation. The side is named from the ball's CURRENT position, the
  // behaviour before the 12.4% wrong-side fix: a ball drifting left lands right of the
  // paddle about one ask in eight, and the observation says LEFT all the same. Kept
  // deliberately as the contrast arm — it is what "no prediction" looks like, and the
  // closed-loop A/B exists to price the difference.
  //
  // ON — the prediction. The side is named from the wall-folded LANDING point (the
  // `ball-predicted-at-paddle-level` axiom's output — gap = landingX − paddleX, which is
  // what the fix made it), and the landing point is stated in the same sentence. It used
  // to print the two absolute x positions instead; that was measured to collapse the local
  // ciel head, so it now restates the gap as prose — see the landingLine comment below.
  const predicted = !!sim.config?.enforceReachability && m.landingX !== null;
  // Threshold is STAY_BAND_PX (the training renderer's value), not an arbitrary 15.
  const sideOf = gap => gap > STAY_BAND_PX
    ? `clearly to the RIGHT of the paddle (gap ${gap.toFixed(0)} px)`
    : gap < -STAY_BAND_PX
      ? `clearly to the LEFT of the paddle (gap ${Math.abs(gap).toFixed(0)} px)`
      : `almost directly above the paddle (offset ${Math.abs(gap).toFixed(0)} px)`;
  // When no prediction exists the only side there is, is the current one — that part is
  // not a regime difference, it is the undecidable band saying so.
  const gapUsed = predicted ? m.gap : ball.x - sim.paddleX;
  const side = sideOf(gapUsed);
  const centered = Math.abs(gapUsed) <= STAY_BAND_PX;
  const timeToPaddle = !m.descending
    ? "rising away from the paddle"
    : m.timeToContact === null
      ? "already at paddle level"
      : `about ${m.timeToContact.toFixed(1)} seconds until the paddle can reach it`;
  // The landing point, stated the way the head trained on it. The absolute
  // coordinates this used to print ("It will land at x≈480, and the paddle is at
  // x≈360") appear in NO training state — 0 occurrences of "will land at" and of
  // "x≈" across the breakout JSONL, and the weakest match in an embedding screen
  // (max-cos 0.772 against 4,280 trained field texts, nearest neighbour an
  // unrelated "paddle.x: 480" leaf). The local ciel head answered it by collapsing
  // to one class. `sideOf` restates the same number as the prose the renderer
  // actually emits, which is what the head has a learned response for.
  //
  // The word "clearly" is load-bearing and is why this calls sideOf(m.gap) rather
  // than reusing `side`: sideOf's centre branch says "offset", and this sentence
  // needs its own lead-in.
  const landingLine = predicted
    ? ` It will land ${sideOf(m.gap)}.`
    : "";
  // Centered balls take the renderer's CENTER path, which differs from the side
  // path in both parts of the motion clause:
  //   motion — _short_motion ("drifting left, falling"), NOT _motion_phrase
  //            ("moving left and down, toward the paddle"). "moving left and
  //            down" occurs 0 times among the training center states.
  //   echo   — _motion_echo(centre) is "The paddle is under the ball"; the side
  //            path would append "up, away from the paddle" instead.
  // Verified against the local ciel head over gaps 0..24: the side path was
  // answered "right" for half the centered balls (stay-prob as low as 0.00),
  // the center path "stay" throughout (0.88-0.94).
  const shortMotion = () => {
    const parts = [];
    if (ball.vx > 0) parts.push("drifting right");
    else if (ball.vx < 0) parts.push("drifting left");
    if (m.descending) parts.push("falling");
    else parts.push("rising");
    return parts.join(", ");
  };
  const motionClause = centered
    ? `${shortMotion()}. The paddle is under the ball`
    : "moving " + (ball.vx > 0 ? "right" : "left") + " and " +
      // "Falling toward the paddle" is its own sentence in the renderer
      // (states/breakout.py `_motion_echo`), never a clause of the geometry
      // sentence. Merging it here put a phrase the head never saw in that
      // position into the attended field, and it dominated the geometry: with
      // the merged form the ciel head answered "left" for a ball plainly right
      // of the paddle (right-prob 0.41 vs OK when the clause stands alone).
      `${m.descending ? "down, toward the paddle. Falling toward the paddle" : "up, away from the paddle"}`;
  return {
    situation: `The ball is ${side} and ${motionClause}. ` +
      `${timeToPaddle.charAt(0).toUpperCase() + timeToPaddle.slice(1)}.${landingLine}`,
    // "Bricks remaining: X. Score: Y. Lives: Z." matches the ciel head's
    // training prose phrasing. (An earlier claim that the "N of M remain;
    // score S, L lives left" variant flipped correct paddle directions no
    // longer reproduces on the v3 head — both phrasings answer left @ 0.9998
    // with identical latents. The field-name-prefix hazard in ciel.py's
    // docstring is real and does reproduce; this one was a stale-cache-era
    // artifact.)
    bricks: `Bricks remaining: ${aliveBricks(sim)}. Score: ${sim.score}. Lives: ${sim.lives}.`,
  };
}
