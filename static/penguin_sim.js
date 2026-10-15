// The penguin's ice floe, as a pure module.
//
// WHY THIS IS A SEPARATE FILE, same reason as breakout_sim.js and fleet_sim.js: the
// browser and the headless harness must not be able to drift apart. If the walking and
// turning rules lived inside penguin.html, the only way to check any claim about them
// would be to load a browser and watch a bird waddle. Here they are a module node can
// import and assert against.
//
// NOTHING HERE TOUCHES THE DOM. No document, no window, no canvas, no AudioContext.
// `applyCommand()` returns what happened and lets the caller decide what to draw. The
// microphone lives entirely in penguin.html; this file has never heard of it.

import { mulberry32 } from "./rng.js";

// ---- the grid -------------------------------------------------------------------
//
// A cell grid rather than free coordinates, because the commands are quantised: "go" is
// one step, "turn left" is 90 degrees. Free movement would make "one step" mean nothing
// and every collision question a geometry problem. On a grid the whole world model is
// four directions and integer arithmetic, which is a thing a test can pin down.

export const CELL = 40;
export const COLS = 15, ROWS = 11;
export const W = COLS * CELL, H = ROWS * CELL;

// Index is the heading. Order matters: `turn_left` is -1 mod 4 and `turn_right` is +1,
// and "north, east, south, west" is the order that makes that true. Do not reorder.
export const HEADINGS = ["north", "east", "south", "west"];
// Unit step per heading index, in grid cells. Matches HEADINGS above.
export const STEPS = [{ dx: 0, dy: -1 }, { dx: 1, dy: 0 }, { dx: 0, dy: 1 }, { dx: -1, dy: 0 }];

export const MAX_STEPS = 30;
export const FISH_COUNT = 6;
export const WATER_COUNT = 7;

// Where the penguin starts, and the heading it starts on.
export const START_COL = Math.floor(COLS / 2), START_ROW = Math.floor(ROWS / 2);
export const START_HEADING = 0;                                  // north

export const COMMANDS = ["go", "back", "turn_left", "turn_right"];

// ---- animation timing -------------------------------------------------------------
//
// The sim owns these because `step()` advances them, and `step()` has to be testable —
// "the penguin arrives after WALK_DUR seconds" is a claim with a right answer, and a
// test asserts it. The per-frame easing and the waddle are the HTML's business.

export const WALK_DUR = 0.28;   // seconds to travel one cell
export const TURN_DUR = 0.22;   // seconds to rotate 90 degrees

export const DEFAULT_CONFIG = { maxSteps: MAX_STEPS };

// ---- the world --------------------------------------------------------------------

/** What is in a cell. Out of bounds is not "ice" — it is the edge of the floe. */
export function cellAt(sim, col, row) {
  if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return "edge";
  if (sim.water.some(w => w.col === col && w.row === row)) return "water";
  if (sim.fish.some(f => f.col === col && f.row === row)) return "fish";
  return "ice";
}

/** The cell `distance` ahead of a heading from a given cell. */
export function cellAhead(sim, col, row, heading, distance = 1) {
  const { dx, dy } = STEPS[((heading % 4) + 4) % 4];
  return { col: col + dx * distance, row: row + dy * distance };
}

/** The cell contents `distance` ahead of where the penguin currently stands. */
export function aheadOf(sim, distance = 1) {
  const { col, row } = cellAhead(sim, sim.col, sim.row, sim.heading, distance);
  return cellAt(sim, col, row);
}

/** The cell contents directly left / right of the penguin (a 90 degree turn away). */
export function leftOf(sim) {
  const { col, row } = cellAhead(sim, sim.col, sim.row, sim.heading - 1);
  return cellAt(sim, col, row);
}
export function rightOf(sim) {
  const { col, row } = cellAhead(sim, sim.col, sim.row, sim.heading + 1);
  return cellAt(sim, col, row);
}

export const fishLeft = sim => sim.fish.length;

/**
 * Lay out the floe for a seed.
 *
 * Seeded, not Math.random(): a scenario has to replay identically or the harness is not
 * comparing anything. Same PRNG the other two sims use.
 *
 * Placement is rejection sampling against a growing set of taken cells. The grid is
 * 165 cells and we place 13 things, so this terminates immediately in practice; the
 * attempt cap is there so a future change to the counts cannot spin forever.
 */
function populate(sim, rng) {
  const taken = new Set([`${START_COL},${START_ROW}`]);
  const place = count => {
    const out = [];
    for (let guard = 0; out.length < count && guard < 1000; guard++) {
      const col = Math.floor(rng() * COLS);
      const row = Math.floor(rng() * ROWS);
      const key = `${col},${row}`;
      if (taken.has(key)) continue;
      taken.add(key);
      out.push({ col, row });
    }
    return out;
  };
  sim.fish = place(FISH_COUNT);
  sim.water = place(WATER_COUNT);
  // The pristine layout, kept because `sim.fish` is CONSUMED — `applyCommand` filters an
  // eaten fish out of it permanently. Replaying "the same floe" therefore cannot be done
  // by resetting the counters; the fish have to come from somewhere that was never
  // mutated. See `resetSim`.
  sim.layout = {
    fish: sim.fish.map(f => ({ ...f })),
    water: sim.water.map(w => ({ ...w })),
  };
}

export function createSim({ seed = 1, config = {} } = {}) {
  const sim = {
    seed,                              // kept so resetSim can rebuild the same floe
    col: START_COL, row: START_ROW, heading: START_HEADING,
    fish: [], water: [],
    layout: null,                      // pristine fish/water, see `populate`
    collected: 0, stepsUsed: 0,
    done: false, doneReason: null,
    history: [],                       // [{command, result}] for the HUD
    config: { ...DEFAULT_CONFIG, ...config },
  };
  populate(sim, mulberry32(seed));
  resetView(sim);
  return sim;
}

/**
 * Start over on the same floe. Clears the run, restores the fish.
 *
 * `keepLayout` is the default because that is what the page's "Same floe" button means —
 * same fish in the same places, so a scenario replays exactly. It restores from
 * `sim.layout` rather than from the live `sim.fish`, which is the bug this replaced:
 * eaten fish are filtered OUT of `sim.fish` as they are eaten, so restoring the counters
 * alone left the floe permanently emptier, the HUD's fish total silently shrinking, and a
 * fully-cleared floe re-ending the run on the first command.
 */
export function resetSim(sim, { keepLayout = true } = {}) {
  if (keepLayout && sim.layout) {
    sim.fish = sim.layout.fish.map(f => ({ ...f }));
    sim.water = sim.layout.water.map(w => ({ ...w }));
  } else {
    populate(sim, mulberry32(sim.seed ?? 1));
  }
  sim.col = START_COL; sim.row = START_ROW; sim.heading = START_HEADING;
  sim.collected = 0; sim.stepsUsed = 0;
  sim.done = false; sim.doneReason = null;
  sim.history = [];
  resetView(sim);
  return sim;
}

// ---- the view ----------------------------------------------------------------------
//
// Presentational state, but it lives here rather than in the HTML because `step()`
// advances it and the tests drive `step()`. Pixel position is DERIVED from the logical
// cell, never the other way around — the logic is the truth and the picture follows it.

function cellCenter(col, row) {
  return { x: col * CELL + CELL / 2, y: row * CELL + CELL / 2 };
}

/** Canvas angle for a heading: north is up, which is -90 degrees in canvas coords. */
export const headingAngle = heading => heading * 90 - 90;

function resetView(sim) {
  const { x, y } = cellCenter(sim.col, sim.row);
  sim.view = {
    x, y,
    headingDeg: headingAngle(sim.heading),
    phase: 0,          // walk cycle, advanced only while walking
    pulse: 0,          // 1 -> 0 after eating; the HTML scales the body by it
    shake: 0,          // 1 -> 0 after a blocked step; the HTML nudges x by it
    anim: null,        // {kind, t, fromX, fromY, toX, toY, fromDeg, toDeg}
  };
}

/**
 * Advance the animation by `dt`. Returns what the caller might care about.
 *
 * Deliberately separate from `applyCommand()`: a command resolves INSTANTLY in the logic
 * (so tests assert outcomes without stepping a clock), and the motion to show that
 * outcome plays out over the next few frames. Splitting them is what lets a test say
 * "blocked, no movement" on the same tick the boolean is decided.
 */
export function step(sim, dt) {
  const ev = { arrived: false };
  const v = sim.view;
  const a = v.anim;

  // Decays run whether or not anything is animating, so a pulse still fades on a
  // command that happened to block.
  v.pulse = Math.max(0, v.pulse - dt / 0.35);
  v.shake = Math.max(0, v.shake - dt / 0.30);

  if (!a) return ev;

  a.t = Math.min(1, a.t + dt / a.dur);
  // Ease-out: the bird accelerates off the mark and settles into the cell. Linear reads
  // as a sliding sprite rather than a walk.
  const e = 1 - (1 - a.t) ** 2;

  if (a.kind === "walk") {
    v.x = a.fromX + (a.toX - a.fromX) * e;
    v.y = a.fromY + (a.toY - a.fromY) * e;
    v.phase = (v.phase + dt / WALK_DUR) % 1;    // one waddle cycle per cell
  } else {
    v.headingDeg = a.fromDeg + (a.toDeg - a.fromDeg) * e;
  }

  if (a.t >= 1) {
    v.anim = null;
    ev.arrived = true;
  }
  return ev;
}

/** True while the bird is mid-move or mid-turn. The HTML blocks input on this. */
export const isAnimating = sim => sim.view.anim !== null;

// ---- commands -----------------------------------------------------------------------

/**
 * Carry out one command. Resolves the outcome immediately and starts the animation.
 *
 * Returns an event object rather than mutating anything the caller watches: this is the
 * seam that lets the browser update its HUD and the node harness assert, off one call.
 *
 * Steps and turns are costed differently ON PURPOSE. A translation always costs a step —
 * including a blocked one — because being wrong about where the water is has to hurt, or
 * "ahead is open water" is not worth asking the model about. A turn is free, because the
 * penguin can always stand still and reorient, and charging for it would make the bar
 * for a correct turn different from the bar for a correct step for no good reason.
 */
export function applyCommand(sim, command) {
  const ev = {
    accepted: false, command,
    moved: false, blocked: false, turned: false,
    ate: null, done: false, doneReason: null,
    stepsLeft: Math.max(0, sim.config.maxSteps - sim.stepsUsed),
  };

  // Rejecting rather than throwing: a wedged state must surface in the HUD, not as an
  // unhandled rejection in the page. The caller sees accepted:false and can say so.
  if (sim.done || isAnimating(sim) || !COMMANDS.includes(command)) return ev;
  ev.accepted = true;

  const v = sim.view;
  const from = cellCenter(sim.col, sim.row);

  if (command === "turn_left" || command === "turn_right") {
    const delta = command === "turn_left" ? -1 : 1;
    sim.heading = ((sim.heading + delta) % 4 + 4) % 4;
    ev.turned = true;
    const toDeg = v.headingDeg + delta * 90;
    v.anim = { kind: "turn", t: 0, dur: TURN_DUR, fromDeg: v.headingDeg, toDeg };
    // Snap the *logical* heading now; the animation is only how we draw getting there.
    // `headingDeg` is left continuous (it can run past 360) so consecutive turns never
    // have to pick a shortest arc — see the note in `resetView`.
    v.headingDeg = toDeg;
    v.anim.fromDeg = toDeg - delta * 90;
    sim.history.push({ command, result: "turned" });
    return ev;
  }

  // go is one step forward, back is one step backward — the same rule with the heading
  // rotated 180 degrees, so both share the whole body below.
  const travel = command === "go" ? sim.heading : sim.heading + 2;
  const { col, row } = cellAhead(sim, sim.col, sim.row, travel);
  const target = cellAt(sim, col, row);

  sim.stepsUsed++;
  ev.stepsLeft = Math.max(0, sim.config.maxSteps - sim.stepsUsed);

  // The edge of the floe and open water both stop the bird. They are different FIELDS
  // because they are different sentences to the model ("the edge of the floe" vs "open
  // water") — a test asserts the prose keeps them apart.
  if (target === "edge" || target === "water") {
    ev.blocked = true;
    v.shake = 1;
    sim.history.push({ command, result: target === "water" ? "blocked by water" : "blocked by the edge" });
  } else {
    sim.col = col; sim.row = row;
    ev.moved = true;
    const to = cellCenter(col, row);
    v.anim = { kind: "walk", t: 0, dur: WALK_DUR, fromX: from.x, fromY: from.y, toX: to.x, toY: to.y };
    if (target === "fish") {
      sim.fish = sim.fish.filter(f => !(f.col === col && f.row === row));
      sim.collected++;
      ev.ate = { col, row, total: sim.collected };
      v.pulse = 1;
      sim.history.push({ command, result: "ate a fish" });
    } else {
      sim.history.push({ command, result: "moved" });
    }
  }

  // Two ways to finish, and the reason is kept so the HUD can say which.
  if (sim.fish.length === 0) {
    sim.done = true; sim.doneReason = "all fish eaten";
  } else if (sim.stepsUsed >= sim.config.maxSteps) {
    sim.done = true; sim.doneReason = "out of steps";
  }
  ev.done = sim.done;
  ev.doneReason = sim.doneReason;
  return ev;
}

// ---- what the model is told ------------------------------------------------------------

export const QUESTION = {
  type: "choice",
  instructions: "Which command should the penguin carry out?",
  criteria: {
    go: "walk one cell forward, in the direction the penguin is facing",
    back: "walk one cell backward, away from the direction the penguin is facing",
    turn_left: "stay in place and rotate 90 degrees to the penguin's left",
    turn_right: "stay in place and rotate 90 degrees to the penguin's right",
  },
};

/**
 * A short human phrase for a cell — the vocabulary BOTH the prose and the reasoner use.
 *
 * One function, two callers, on purpose: the sentence the model reads and the value the
 * tests assert over cannot disagree if there is only one place that names a cell.
 */
export function describeCell(kind) {
  switch (kind) {
    case "fish": return "a fish";
    case "water": return "open water";
    case "edge": return "the edge of the floe";
    default: return "open ice";
  }
}

/** The observation, as plain numbers. `buildState` is a formatter over this. */
export function measure(sim) {
  return {
    heading: HEADINGS[sim.heading],
    ahead: aheadOf(sim),
    left: leftOf(sim),
    right: rightOf(sim),
    behind: aheadOf(sim, -1),
    stepsLeft: Math.max(0, sim.config.maxSteps - sim.stepsUsed),
    fishLeft: sim.fish.length,
    collected: sim.collected,
  };
}

/**
 * The state the model is asked to route over, as prose.
 *
 * The transcript goes in verbatim as `spoken instruction`, because the question being
 * asked IS "which of the four commands does this utterance mean" — and the surrounding
 * cells are there so the model can weigh what the penguin can actually do from here.
 *
 * This is the same contract the other demos use (index.html's guardrail and router
 * examples also route free text), so the shape is not novel. What IS worth flagging: the
 * other demos were built against checkpoints trained on prose of a known shape, and a
 * raw ASR transcript is not that. See the limitations note in the README — the `rules`
 * arm exists so a bad read is not the only path.
 */
export function buildState(sim, transcript) {
  const m = measure(sim);
  return {
    "spoken instruction": transcript,
    "penguin is facing": m.heading,
    "ahead of the penguin": describeCell(m.ahead),
    "to the left of the penguin": describeCell(m.left),
    "to the right of the penguin": describeCell(m.right),
    "behind the penguin": describeCell(m.behind),
    "steps remaining": String(m.stepsLeft),
    "fish remaining": String(m.fishLeft),
  };
}
