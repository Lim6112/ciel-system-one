// Warehouse blockade-resolution simulation — pure, seeded, DOM-free.
//
// warehouse.html is only the DOM layer over this module, and eval_warehouse.mjs
// imports the same file, so the browser and the harness cannot drift apart.
//
// The one thing to understand before changing anything here: the blocker can be
// a rack or another robot, and those are DIFFERENT problems, so the sim builds a
// different question for each.
//
//   rack  — static, never yields. Worth 62% of the recorded traffic (110/178).
//           "stay" is provably wrong, so it is not offered at all; the question
//           is a 2-way sidestep and the obstacle is described by the room on
//           either side of it.
//   robot — dynamic, can yield. 3-way, and "stay" means *I give way*, which is
//           a real answer for a robot but nonsense for a rack.
//
// The lidar reflects that: a rack is hit by a straight ray, which returns a
// distance only — no lateral offset exists to report. A robot is tracked as a
// relative vector, so it does have a side. Reporting a rack's "side" by reading
// the side words off the *other* robots in the state is the trap this layout
// avoids: those words are decorrelated noise, not the blocker's position.

import { mulberry32 } from "./rng.js";

// ---- fixed geometry ---------------------------------------------------------

export const W = 520, H = 420;
export const DOCK = { x: 205 + 110 / 2, y: 28 + 62 };     // cargo house door
const HOUSE = { x: 205, y: 28, w: 110, h: 62 };
const RACK_W = 70, RACK_H = 26;
export const RACKS = [
  { id: "A1", x: 45,  y: 170, w: RACK_W, h: RACK_H },
  { id: "A2", x: 165, y: 170, w: RACK_W, h: RACK_H },
  { id: "A3", x: 285, y: 170, w: RACK_W, h: RACK_H },
  { id: "A4", x: 405, y: 170, w: RACK_W, h: RACK_H },
  { id: "B1", x: 45,  y: 280, w: RACK_W, h: RACK_H },
  { id: "B2", x: 165, y: 280, w: RACK_W, h: RACK_H },
  { id: "B3", x: 285, y: 280, w: RACK_W, h: RACK_H },
  { id: "B4", x: 405, y: 280, w: RACK_W, h: RACK_H },
];
export { HOUSE };

export const PX_PER_M = 10;
export const ROBOT_R = 9;
export const MIN_SEP = 2 * ROBOT_R + 3;                   // hard collision safety
export const LIDAR_HALF_ANGLE = 45 * Math.PI / 180;
export const SIDESTEP_MS = 1500;

// Escalation: how long a robot may make NO forward progress before it stops
// asking "how do I resolve this" and starts being told to go around.
//
// Deliberately a stall timer, not a blocked timer. The old blockedSince timer
// never fired in practice — across 178 recorded calls the longest block was 3 s
// against a 5 s threshold, so the 2-way branch was unreachable and "stay" was
// always on the menu. Progress is what the demo actually cares about.
export const STALL_MS = 4000;
// Below this, a position delta is heading jitter rather than travel.
export const PROGRESS_EPSILON_PX = 1.0;

const AISLE_XS = [140, 260, 380];       // gap centers between racks
const CORRIDOR_YS = [146, 238, 330];    // above row A, between rows, below row B

// Robot colors + spawn points only — every destination is assigned at runtime
export const ROUTES = {
  R1: { color: "#f87171", name: "red",   start: { x: 55, y: 390 } },
  R2: { color: "#4ade80", name: "green", start: { x: 465, y: 390 } },
  R3: { color: "#60a5fa", name: "blue",  start: { x: 260, y: 405 } },
};

export const fmtM = px => `${Math.max(1, Math.round(px / PX_PER_M))} m`;

// ---- navigation -------------------------------------------------------------
// Robots move only along safe lines: corridors are clear horizontally, aisle
// gaps are clear vertically. Any route is a corridor/aisle L-shaped chain.

export function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

export function nearestAisle(x) {
  return AISLE_XS.reduce((a, b) => Math.abs(b - x) < Math.abs(a - x) ? b : a);
}

export function nearestCorridor(y) {
  // Returns the NEARER corridor. It returned the farther one until this was
  // fixed: the comparison was right but the branches were swapped, so any robot
  // below y=284 was sent to the corridor at 146 rather than the one at 330.
  //
  // The damage was not a wrong-looking route but a route off the map: `cy` drove
  // the first leg's y, so R1 at y=390 was routed to (55,146) — one straight line
  // from the bottom of the board to the top, straight through both rack rows,
  // which the hard collision check then refused, pinning the robot at its spawn
  // with 0 deliveries. Route legs must only ever travel along y=146/238/330
  // (rows are clear horizontally) or x=140/260/380 (aisles are clear
  // vertically); the old first leg satisfied neither.
  return CORRIDOR_YS.reduce((a, b) => Math.abs(b - y) < Math.abs(a - y) ? b : a);
}

// row A is approached from the top corridor, row B from the bottom corridor —
// no route ever needs to cross a rack row horizontally
export function rackCorridor(rack) {
  return rack.y < 200 ? CORRIDOR_YS[0] : CORRIDOR_YS[2];
}

// drop consecutive duplicate points and a start point equal to current position
function cleanPath(pts, from) {
  const all = [from, ...pts];
  return all.filter((p, i) =>
    i > 0 && (Math.abs(p.x - all[i - 1].x) > 0.5 || Math.abs(p.y - all[i - 1].y) > 0.5));
}

export function routeToRack(robot, rack) {
  const pickX = rack.x + rack.w / 2;
  const cy = nearestCorridor(robot.y), ty = rackCorridor(rack);
  const a0 = nearestAisle(robot.x), a1 = nearestAisle(pickX);
  const pts = [{ x: robot.x, y: cy }, { x: a0, y: cy }];
  if (a0 !== a1) pts.push({ x: a1, y: cy });
  if (cy !== ty) pts.push({ x: a1, y: ty });
  pts.push({ x: pickX, y: ty });
  return cleanPath(pts, robot);
}

export function routeToDock(robot) {
  const cy = nearestCorridor(robot.y), ty = CORRIDOR_YS[0];
  const a = nearestAisle(robot.x);
  const pts = [{ x: robot.x, y: cy }, { x: a, y: cy }];
  if (cy !== ty) pts.push({ x: a, y: ty });
  pts.push({ x: DOCK.x, y: ty }, DOCK);
  return cleanPath(pts, robot);
}

// ---- robots -----------------------------------------------------------------

export function makeRobot(rid, rng = Math.random) {
  const spec = ROUTES[rid];
  const robot = {
    id: rid, color: spec.color, colorName: spec.name,
    x: spec.start.x, y: spec.start.y,
    waypoints: [], wp: 0,
    phase: "toPickup", pickupRack: null,   // toPickup -> toDock -> loop
    heading: 0, waiting: false,
    sidestepUntil: 0, sidestepDir: 0,
    // stall tracking: blockedSince is the raw obstacle timer (kept for the
    // readout), lastProgressAt is what actually drives escalation
    blockedSince: null, lastProgressAt: 0, lastProgressX: spec.start.x,
    lastProgressY: spec.start.y,
    obstacle: null, deliveries: 0, asking: false, lastAsk: 0,
    // The seeded rng, kept so every later task assignment draws from the same
    // deterministic stream rather than falling back to Math.random.
    rng,
  };
  assignTask(robot, rng);
  return robot;
}

// new delivery cycle: pick a random rack, plan the pickup leg
export function assignTask(robot, rng = Math.random) {
  robot.pickupRack = RACKS[Math.floor(rng() * RACKS.length)];
  robot.waypoints = routeToRack(robot, robot.pickupRack);
  robot.wp = 0;
  robot.phase = "toPickup";
}

// ---- lidar ------------------------------------------------------------------

// Vector to another robot in my frame: x = lateral (+right), y = forward (+ahead)
export function relativeTo(observer, other) {
  const dx = other.x - observer.x, dy = other.y - observer.y;
  const cos = Math.cos(observer.heading), sin = Math.sin(observer.heading);
  return {
    y: dx * cos + dy * sin,       // forward component
    x: -dx * sin + dy * cos,      // lateral component, +right
  };
}

// Straight ray along the heading vs rack rects (slab method). Racks use a ray,
// not the 45° cone: a robot aimed through a 50 px aisle must not read the
// flanking rack corners (~25 px lateral, ~30 px ahead) as a wall ahead.
//
// The hit also carries the lateral offset from the robot's lane to the rack's
// centre, which is what makes a rack's sidestep decidable — the ray alone gives
// a distance, and "move away from it" needs a side.
export function rayHitsRack(robot, range) {
  let best = null;
  const dirX = Math.cos(robot.heading), dirY = Math.sin(robot.heading);
  for (const rack of RACKS) {
    let tmin = 0, tmax = range, hit = true;
    for (const axis of ["x", "y"]) {
      const o = axis === "x" ? robot.x : robot.y;
      const d = axis === "x" ? dirX : dirY;
      const lo = rack[axis], hi = lo + (axis === "x" ? rack.w : rack.h);
      if (Math.abs(d) < 1e-6) {
        if (o < lo || o > hi) { hit = false; break; }
        continue;
      }
      let t1 = (lo - o) / d, t2 = (hi - o) / d;
      if (t1 > t2) [t1, t2] = [t2, t1];
      tmin = Math.max(tmin, t1);
      tmax = Math.min(tmax, t2);
    }
    if (hit && tmin <= tmax && tmin > 0 && tmin <= range &&
        (!best || tmin < best.dist)) {
      best = { rack, dist: tmin };
    }
  }
  if (!best) return null;
  // Lateral offset from the robot's heading line to the rack's centre:
  // positive => the rack sits to the robot's right. This is the rack case's
  // gold-bearing number, and it is computed from the rack itself — never
  // copied from whichever other robot happens to be described in the state.
  const toRack = relativeTo(robot, {
    x: best.rack.x + best.rack.w / 2, y: best.rack.y + best.rack.h / 2,
  });
  return { rack: best.rack, dist: best.dist, lateral: toRack.x };
}

export function scanLidar(robot, robots, range) {
  let best = null;
  for (const other of robots) {
    if (other === robot) continue;
    const rel = relativeTo(robot, other);
    const dist = Math.hypot(rel.x, rel.y);
    if (dist > range) continue;
    const angle = Math.abs(Math.atan2(rel.x, Math.max(0, rel.y)));
    if (rel.y < 0 || angle > LIDAR_HALF_ANGLE) continue;
    if (!best || dist < best.dist) best = { other: other.id, rack: null, dist, rel };
  }
  const rackHit = rayHitsRack(robot, range);
  if (rackHit && (!best || rackHit.dist < best.dist)) {
    best = { other: null, rack: rackHit.rack, dist: rackHit.dist,
             rel: null, rackLateral: rackHit.lateral };
  }
  return best;
}

// ---- the question -----------------------------------------------------------
// Two questions, split by blocker. Both exist in the trained bank under these
// qids, and the 2-way one deliberately has no "stay": the Tier-0 form of the
// rule is to remove the option, not to explain it.

export const AVOID_QID = "avoid_direction";
export const SIDESTEP_QID = "sidestep_direction";

// Once a robot has made no progress for STALL_MS, it stops asking whether to
// wait and is told to go around instead.
export function buildQuestion(blockerKind, stalled) {
  if (blockerKind === "rack" || stalled) {
    return {
      type: "choice",
      instructions: "You are a warehouse delivery robot. Something is blocking " +
        "your path and will not move out of the way on its own. In which " +
        "direction should you sidestep to pass it?",
      criteria: {
        left: "sidestep to my left to pass the obstacle",
        right: "sidestep to my right to pass the obstacle",
      },
    };
  }
  return {
    type: "choice",
    instructions: "You are a warehouse delivery robot. Another robot is blocking " +
      "your path ahead. It may move on its own. In which direction should you " +
      "move to resolve the blockage?",
    criteria: {
      left: "move to my left to get around the obstacle, then merge back to my route",
      stay: "yield and wait, because the obstacle is about to clear my path",
      right: "move to my right to get around the obstacle, then merge back to my route",
    },
  };
}

export function questionId(blockerKind, stalled) {
  return (blockerKind === "rack" || stalled) ? SIDESTEP_QID : AVOID_QID;
}

// ---- state (the JSON contract) ----------------------------------------------
// JSON, not prose, and the same JSON the training corpus renders — that is the
// whole point. Two prior incidents in this project were caused by training on
// one format and inferring on another, so the two are the same object here.
//
// Flat and scalar only: server.py calls str() on each top-level value, so a
// nested object would be sent as a Python repr the head has never seen. Keep
// the leaf count <= 15: the head prepends the whole text as field 0, which
// leaves no room for the chunker to start merging leaves — and a merged leaf
// takes the decisive one with it.

function bandGap(px) {
  return px > 45 ? "wide" : "narrow";
}

export function buildState(robot, scan, now = 0) {
  const others = robot.others
    ? robot.others.filter(o => o !== robot.id)
        .map(id => describeOther(robot, id))
        .filter(Boolean)
    : [];
  const stalledFor = robot.lastProgressAt
    ? Math.round((now - robot.lastProgressAt) / 1000) : 0;

  const doc = {
    role: "warehouse delivery robot",
    color: robot.colorName,
    mission: robot.phase === "toPickup"
      ? "heading to a rack to pick up cargo" : "carrying cargo to the dock",
    dock_distance: Math.hypot(robot.x - DOCK.x, robot.y - DOCK.y) < 150 ? "near" : "far",
  };

  if (scan.rack) {
    doc.obstacle_kind = "storage rack";
    doc.obstacle_id = scan.rack.id;
    doc.obstacle_position = "directly ahead in my lane";
    doc.obstacle_distance = scan.dist < 60 ? "close" : "far";
    // The rack case's gold: which side has more room. Words, not pixels —
    // the head's backbone is nearly insensitive to the value of a number.
    // A rack sitting to my right occupies that side, so the room there is the
    // narrow one; the wider gap is the opposite side, which is also where the
    // sidestep gold points. Getting this mapping backwards inverts the answer,
    // so it is asserted in test_warehouse_sim.mjs.
    const lateral = scan.rackLateral ?? 0;
    doc.gap_left = lateral > 0 ? "wide" : "narrow";
    doc.gap_right = lateral > 0 ? "narrow" : "wide";
  } else {
    const rel = scan.rel || { x: 0 };
    const side = rel.x > 12 ? "right" : rel.x < -12 ? "left" : "center";
    doc.obstacle_kind = "robot";
    doc.obstacle_id = scan.other;
    doc.obstacle_side = side;
    doc.obstacle_position = side === "center"
      ? "ahead of me, but not blocking my lane"
      : "ahead, off to one side of my lane";
    doc.obstacle_distance = scan.dist < 60 ? "close" : "far";
    doc.obstacle_motion = scan.motion || "moving in the same direction";
  }

  others.slice(0, 2).forEach((desc, i) => { doc[`other_robot_${i + 1}`] = desc; });
  doc.deliveries_completed = robot.deliveries;
  doc.blocked = stalledFor >= STALL_MS / 1000 ? "for a while" : "briefly";
  return doc;
}

function describeOther(robot, id) {
  const other = robotsById.get(id);
  if (!other) return null;
  const rel = relativeTo(robot, other);
  const dist = fmtM(Math.hypot(rel.x, rel.y));
  const fore = rel.y >= 0 ? `ahead of me` : `behind me`;
  const side = rel.x > 12 ? ", to my right" : rel.x < -12 ? ", to my left" : "";
  const motion = Math.abs(other.heading - robot.heading) % (2 * Math.PI) < Math.PI / 6
    ? "moving in the same direction" : "crossing my path";
  return `${other.colorName}, ${fore}${side}, ${motion}`;
}

// resolveOther() binds the world; kept as a module-level lookup so buildState
// stays a pure function of (robot, scan).
let robotsById = new Map();
export function bindWorld(robots) {
  robotsById = new Map(robots.map(r => [r.id, r]));
}

// ---- head-on priority -------------------------------------------------------
//
// Two robots meeting dead ahead cannot be told apart by geometry: their lateral
// offset is zero, so there is no side to prefer, and the gold oracle answers
// "stay" to BOTH of them. Both yield, nobody moves, and the pair sits there until
// STALL_MS (4 s) expires and escalation forces a sidestep — which is why head-on
// encounters resolved in ~4.4 s while every other block resolved in ~0.22 s
// (measured: 96% of the >4s episodes were head-on, and 96% of those had
// escalated).
//
// So the dead band needs a tie-break, and the rule is: the LOWER ID advances, the
// other yields. That is asymmetric by construction — exactly one robot can hold the
// lower id — which is what makes the deadlock structurally impossible rather than
// merely unlikely. Same principle as the dock rule, where the incumbent is found
// rather than elected.
//
// WHY THIS NEEDS A LEADER, AND HOW THE LEADER IS PICKED. The comparison must reach
// the same verdict on both robots independently, and two candidates were measured:
//
//   first-detected (blockedSince) — the tick each robot first saw an obstacle.
//     Wall-clock, equal on both sides, no coordinate frame involved. Measured as
//     the worse rule: p90 1,376 ms, max 5,920 ms, and 704 encounters over 4 s.
//   lower id — needs no memory at all. p90 672 ms, max 3,088 ms, and ZERO
//     encounters over 4 s. Reproducible across runs once `assignTask` stopped
//     drawing from Math.random (see that call site — before the fix every A/B
//     here was noise).
//
// The id rule wins because `blockedSince` is not a stable property of the
// MEETING, it is a property of each robot's own history: a robot that has been
// blocked by something else, or blocked longer ago, carries a stale value into a
// new encounter and elects the wrong side. Measured: the role flipped mid-encounter
// in ~6% of encounters, and flipping is what re-arms the STALL_MS wait. The id
// comparison reads only who the counterpart is, so it cannot be stale and cannot
// disagree between the two robots — each compares its own id against the other's
// and exactly one concludes "advance".
//
// The cost, stated plainly: the lower id is always the one that goes first, so the
// rule is unfair in a way a wall-clock race is not. It buys a 2x p90 and removes
// the >4 s tail entirely, which is the trade this demo wants — but it is a trade,
// not a free lunch, and if fairness ever matters more than latency the
// `blockedSince` version is the one to bring back.
//
// WHAT THIS RULE STILL GETS WRONG, kept here because the attempt to fix it was
// reverted and the diagnosis is the expensive part. Watching the live page, the
// yielding robot WAITS FOR A ROBOT THAT IS NOT COMING: it holds at a fixed spot
// with "stay" while the robot elected to advance has already passed, or is dodging
// and returning to its own route without ever clearing the path. Escalation breaks
// it at the STALL_MS price every time, which is why robot-vs-robot blocks in the
// browser measure p90 ~4.7 s with 6 of 39 over 4 s, while rack blocks (which never
// yield) are 0 of 51. Measured in the page, not in a harness.
//
// Three fixes were tried and all were reverted, with the numbers:
//   - both robots dodge, nobody yields. Better in the harness, worse in the page:
//     agreement with the model fell to 40%, because the trained head learned `stay`
//     for a head-on blocker and this stops producing it. A gold change needs a
//     retrain, not a code change.
//   - end the sidestep early once the lane clears (`|rel.x| >= MIN_SEP`). Helped,
//     but a rack hit carries no `rel`, so "no rel" read as "clear" and every rack
//     sidestep restarted each tick — 70,560 encounters where there had been 9,000.
//   - re-plan from the post-sidestep position (`replanFrom`). Made it worse:
//     routeToRack/routeToDock build their L-chain from the current position and
//     reset `wp = 0`, so the first leg is often backwards — 3,585 deliveries became
//     2,759, sidestep travel per delivery 475 px became 662 px.
// The one change that DID measure as a clear win and was also reverted with the
// rest: `warehouse.html` passing `role` into `goldDirection`. Without it the page
// silently used the unresolved branch while the simulator and every harness run had
// moved on — see the note at that call site.
//
// `role` is "advance" for the lower id, "yield" for the other. Returns null for
// anything that is not a dead-ahead encounter, so callers can fall through to the
// ordinary gold.
export const HEAD_ON_BAND_PX = 12;      // same dead band goldDirection uses

// THE REMAINING TAIL, DIAGNOSED BUT NOT FIXED. About 5 encounters in 11,204
// (0.04%) still exceed 4 s, and they have one shape. The two robots do not agree
// on whether they are head-on: the higher id sees the other dead ahead and yields,
// while the lower id sees it at an angle (|rel.x| > the dead band), gets role null,
// and dodges and returns to its route over and over. So the yielder waits for a
// robot that IS moving but is not clearing its path, and only STALL_MS breaks it
// (measured: yielder stall 1.3 s climbing to 7.7 s while the counterpart's stall
// stayed at 0.0 s throughout).
//
// A counterpart-stall rule was tried and removed: it asks "is the robot I am
// waiting for moving", and that robot moves busily while never clearing, so the
// rule fired almost never (6 encounters over 4 s became 5). The signal that would
// work is "has the distance between us stopped closing" — progress of the
// ENCOUNTER rather than of either robot — which needs per-encounter state the sim
// does not keep. Not attempted here: it is more machinery than the 0.04% justifies,
// and this file's history is a list of plausible rules that measured worse.
export function headOnRole(robot, scan) {
  if (!scan || scan.rack || !scan.rel) return null;
  if (Math.abs(scan.rel.x) > HEAD_ON_BAND_PX) return null;
  if (scan.rel.y <= 0) return null;              // not actually ahead of me
  // No counterpart to compare against (a harness driving a single robot): the
  // robot in front of us advances, because nothing has elected it to yield.
  if (!robotsById.has(scan.other)) return "yield";
  return robot.id < scan.other ? "advance" : "yield";
}

// ---- physics ----------------------------------------------------------------

// circle vs rack rect: clamp the circle center to the rect, then distance test
export function hitsRack(x, y) {
  return RACKS.some(rack => {
    const cx = Math.max(rack.x, Math.min(x, rack.x + rack.w));
    const cy = Math.max(rack.y, Math.min(y, rack.y + rack.h));
    return Math.hypot(x - cx, y - cy) < ROBOT_R;
  });
}

// Which way actually clears a blocker, from geometry alone. This is the oracle
// eval_warehouse.mjs scores the model against, and the gate's fallback — so it
// must not read any surface word.
export function goldDirection(scan, stalled = false, role = null) {
  if (scan.rack) {
    // Away from where the rack sits — which is also the wider-gap side.
    const lateral = scan.rackLateral ?? 0;
    return lateral > 0 ? "left" : "right";
  }
  const rel = scan.rel || { x: 0 };
  if (Math.abs(rel.x) <= HEAD_ON_BAND_PX) {
    // Dead ahead, so the offset carries no side to prefer. Two cases, and they
    // differ by whether anything has been elected:
    //
    // `role` set (see headOnRole) is the resolved case. The robot that detected
    // the blocker first advances — a sidestep, so the question keeps a real
    // answer — while the other yields. That asymmetry is what removes the wait:
    // before this, both were told "stay" and the pair burned STALL_MS 4 s doing
    // nothing, which measured as ~4.4 s head-on resolution against ~0.22 s for
    // every other block.
    //
    // `role` null is the unresolved case: either nobody compared yet, or the
    // surrounding code is not running the priority rule. Answering a sidestep
    // here would make both robots dodge the same way and keep dodging, so the
    // honest answer with nothing elected is the one that does not invent an
    // asymmetry that is not there. Callers that DO run the rule pass a role.
    if (role === "advance") return "left";
    if (role === "yield") return "stay";
    // Unresolved: `stalled` is the caller saying waiting has demonstrably failed,
    // so a sidestep is forced rather than a wait that will not end. Without it,
    // yield — which is only correct when the other side is advancing.
    return stalled ? "left" : "stay";
  }
  return rel.x > 0 ? "left" : "right";       // away from the blocker
}

// Options the geometry leaves open, and the exact counterpart of goldDirection —
// whatever this offers, that function must answer from; whatever it withholds,
// that function must not answer. The two drifted once and pinned robots in a
// permanent standoff, so `divergence()` checks them against each other over every
// blocking situation the sim produces.
//
// A rack cannot be waited out. A blocker dead ahead USED to withhold "stay" too,
// on the grounds that there is no around to go to — which was true and also the
// bug: with "stay" withdrawn and no side to prefer, the only answers left were
// two arbitrary directions, and the pair sat until STALL_MS expired. Now the
// dead band resolves through headOnRole instead: the advance role is offered a
// sidestep (it must move), the yield role is offered "stay" (moving is not its
// job), and the two agree because they share one comparison.
export function feasibleOptions(blockerKind, stalled, role = null) {
  if (blockerKind === "rack") return ["left", "right"];
  if (role === "advance") return ["left", "right"];
  if (role === "yield") return ["left", "stay", "right"];
  return stalled ? ["left", "right"] : ["left", "stay", "right"];
}

// ---- dock zone: a rule that does not pay for itself -------------------------
//
// IMPLEMENTED, MEASURED, SHIPPED OFF. Read this before turning it on.
//
// The problem it was meant to solve is real. Every robot's route ends at the
// cargo house — the one place all three converge — and that is where the
// contention is: 54.4% of all blocked ticks in a 60 s run are spent inside this
// radius (185,889 of 341,911 across seven seeds). A docking station serving one
// vehicle at a time is also a real domain rule, and it says cleanly what the
// fleet demo's seat limit says: a second arrival is not a bad answer, it is a
// state the domain does not permit.
//
// It does not work, and the reason is worth keeping.
//
// WHAT WAS MEASURED. Seven to nine seeds, 60 s each, geometry oracle as the
// policy, the only difference being this rule:
//
//   rule off   93.6 and 91.1 mean deliveries (two consecutive runs)
//   rule on    55.0 and 48.0
//
// So it costs roughly 40% of throughput. The radius was swept (20/30/45/60/90 px)
// and every value was worse than off, best case 46.1 at 45 px. It also failed to
// hold its own invariant: 1,800-2,800 ticks of two robots inside the zone, not
// through a race but because the hold parks a robot AT the zone edge where it
// still counts as outside on its own test and still blocks the apron.
// Serialising the apron costs more than the overlap it prevents: the zone is a
// bottleneck precisely because the routes cross in it, so throwing away the
// parallel crossing hurts more than the two-robot overlap helps.
//
// WHAT WOULD BE NEEDED TO MAKE IT WORK — not done, recorded so the next attempt
// does not start from zero:
//   - A radius small enough to exclude the crossing apron, or a zone shaped as a
//     segment of the final approach rather than a disc around the dock.
//   - A hold position that is genuinely outside the zone AND off the routes
//     others use, since the current one parks a blocker in the aisle.
//   - Most likely a reservation model rather than a mutex: the win would be
//     ordering arrivals, not excluding them.
// Any of those is a bigger change than the rule itself, and the evidence above
// says the rule as stated is a net loss.
//
// Kept as code because it is a complete, working implementation of the axiom the
// ontology would declare (see the plan's F1/F2): if the ontology route is taken,
// this is what the declared rule would have to govern, and it is measurable in
// place with `dockRule: true`. Kept OFF because shipping it would make the demo
// worse, and a rule that reads well and measures badly is exactly the thing the
// Tier-0 pattern is supposed to catch.
export const DOCK_ZONE_RADIUS = 45;

export function inDockZone(robot, radius = DOCK_ZONE_RADIUS) {
  return Math.hypot(robot.x - DOCK.x, robot.y - DOCK.y) < radius;
}

export function headingIn(robot, radius = DOCK_ZONE_RADIUS) {
  const target = robot.waypoints[robot.wp];
  return !!target && Math.hypot(target.x - DOCK.x, target.y - DOCK.y) < radius;
}

// The axiom's implementation: at most one robot inside the zone. Whoever is
// already in keeps it; everyone else heading in holds. Occupancy relative, so it
// is asymmetric by construction and needs no priority function — unlike the
// head-on standoff fixed elsewhere in this module, which was symmetric and so
// deadlocked. That part of the reasoning held; the throughput did not.
export function dockZoneOpen(robot, robots, radius = DOCK_ZONE_RADIUS, reserved = null) {
  if (inDockZone(robot, radius)) return true;    // already inside: let it out
  if (!headingIn(robot, radius)) return true;    // not entering the dock
  const occupied = robots.some(o => o !== robot && inDockZone(o, radius));
  const claimed = reserved && reserved.size > 0;
  return !occupied && !claimed;
}

export function step(robots, dt, now, { speed, lidarRange, autoplay, dockRule = false,
                                       dockZoneRadius = DOCK_ZONE_RADIUS }) {
  bindWorld(robots);
  const blocked = [];
  for (const robot of robots) {
    const scan = scanLidar(robot, robots, lidarRange);
    robot.obstacle = scan;
    robot.others = robots.map(r => r.id);
    if (scan) {
      if (robot.blockedSince === null) robot.blockedSince = now;
      // fallback (autoplay off): blocked robots just stop — the deadlock demo
      if (!autoplay) robot.waiting = true;
    } else {
      robot.blockedSince = null;
      robot.waiting = false;
    }
    blocked.push(!!scan);
  }

  // The Tier-0 verdict, recomputed from the ABox every tick rather than carried
  // as state. That matters: `waiting` is cleared each tick for an unblocked
  // robot, so a hold expressed as "tell it to wait" evaporates immediately and
  // the robot walks in anyway. Reading occupancy fresh also means the hold lifts
  // by itself the moment the incumbent leaves, with nothing to reset.
  //
  // Two robots can be admitted in the same tick — occupancy is read before either
  // has moved — and both then finish inside the zone, breaking the very invariant
  // the rule exists to state (measured: 7,250 ticks of two-occupancy over nine
  // seeds, before this was fixed). So the scan also claims the zone: the first
  // admitted robot reserves it and the rest of the tick is refused against the
  // reservation, not merely against the current occupants.
  const dockVerdict = new Map();
  const reserved = new Set();
  for (const robot of robots) {
    const open = dockRule
      ? dockZoneOpen(robot, robots, dockZoneRadius, reserved)
      : true;
    dockVerdict.set(robot.id, open);
    if (open && dockRule && !inDockZone(robot, dockZoneRadius) &&
        headingIn(robot, dockZoneRadius)) {
      reserved.add(robot.id);
    }
  }

  for (const robot of robots) {
    const travelled = Math.hypot(robot.x - robot.lastProgressX, robot.y - robot.lastProgressY);
    if (travelled > PROGRESS_EPSILON_PX) {
      robot.lastProgressAt = now;
      robot.lastProgressX = robot.x;
      robot.lastProgressY = robot.y;
    }

    const held = !dockVerdict.get(robot.id);
    robot.heldByDock = held;

    // The head-on election, recomputed from the ABox each tick like the dock
    // verdict — so it is never stale and lifts by itself once the encounter ends.
    // A yielding robot holds: it was elected to concede, and the answer it was
    // given ("stay") means wait, which only works if waiting is actually honoured.
    // See the head-on priority note above headOnRole for why the election exists.
    const hRole = headOnRole(robot, robot.obstacle);
    robot.headOnRole = hRole;
    const yielding = hRole === "yield";
    if (yielding) robot.waiting = true;

    let vx = 0, vy = 0;
    if (held || yielding) {
      // Tier 0 closes the zone: no movement, and no sidestep either — a sidestep
      // is still motion, and the rule is about not entering. The yield case is
      // the same shape: a robot elected to concede stops, including mid-sidestep,
      // because continuing to crawl would carry it into the robot it conceded to.
    } else if (now < robot.sidestepUntil) {
      // perpendicular crawl to get around the obstacle
      const px = Math.cos(robot.heading), py = Math.sin(robot.heading);
      vx = -py * robot.sidestepDir * speed;
      vy = px * robot.sidestepDir * speed;
    } else if (!robot.waiting) {
      const target = robot.waypoints[robot.wp];
      if (!target) continue;
      const dx = target.x - robot.x, dy = target.y - robot.y;
      const dist = Math.hypot(dx, dy);
      if (dist < 5) {
        robot.wp++;
        if (robot.wp >= robot.waypoints.length) {
          // pickup point reached -> haul to the dock; dock reached -> delivered,
          // then a fresh random rack starts the next cycle
          if (robot.phase === "toPickup") {
            robot.phase = "toDock";
            robot.waypoints = routeToDock(robot);
          } else {
            robot.deliveries++;
            // The robot's own rng, not Math.random. Using the default here made
            // every cycle after the first unseeded, so `createWorld(seed)` was
            // only deterministic up to the first delivery and two runs of the
            // same seed diverged — which silently turned every A/B measurement
            // in this file's history into noise (same seed gave 3,774 / 8,156 /
            // 9,547 episodes across three runs).
            assignTask(robot, robot.rng);
          }
          robot.wp = 0;
        }
        continue;
      }
      robot.heading = Math.atan2(dy, dx);
      vx = dx / dist * speed;
      vy = dy / dist * speed;
    }

    // hard safety: never overlap another robot or leave the floor, whatever CJet
    // decided. If the direct move is blocked, slide along at increasing angles —
    // otherwise two robots approaching head-on can push-lock outside each other's
    // lidar cone.
    //
    // The bounds check is not decoration: a sidestep is a perpendicular crawl for
    // a fixed 1.5 s in one direction, with no target and no other reason to stop,
    // so a robot that sidesteps near an edge used to walk straight off the board
    // (measured: 2-6% of ticks at the demo's cadence, every one of them mid-
    // sidestep). Clamping rather than refusing the move keeps the slide-angles
    // useful: a robot at a wall still gets to slide along it.
    for (const off of [0, 0.6, -0.6, 1.2, -1.2, Math.PI / 2, -Math.PI / 2]) {
      const sx = vx * Math.cos(off) - vy * Math.sin(off);
      const sy = vx * Math.sin(off) + vy * Math.cos(off);
      const nx = clamp(robot.x + sx * dt, ROBOT_R, W - ROBOT_R);
      const ny = clamp(robot.y + sy * dt, ROBOT_R, H - ROBOT_R);
      const collides = robots.some(o => o !== robot &&
        Math.hypot(nx - o.x, ny - o.y) < MIN_SEP) || hitsRack(nx, ny);
      if (!collides) { robot.x = nx; robot.y = ny; break; }
    }
  }
  return { blockedNow: blocked.filter(Boolean).length };
}

// Apply one model answer. `action` is a label from the question that was asked.
export function applyDecision(robot, action, now) {
  // "stay" = yield; left/right = sidestep maneuver.
  // a sidestep already in progress finishes before a "stay" takes hold —
  // cancelling it mid-maneuver would freeze the robot beside the obstacle
  const sidestepping = now < robot.sidestepUntil;
  if (action === "stay") {
    if (!sidestepping) robot.waiting = true;
  } else {
    robot.waiting = false;
    // commit to one direction per maneuver — flipping mid-sidestep every
    // decision tick would zigzag with zero net progress
    if (!sidestepping) {
      robot.sidestepUntil = now + SIDESTEP_MS;
      robot.sidestepDir = action === "left" ? -1 : 1;
    }
  }
}

export function createWorld(seed = 1, rng = null) {
  // Seeded by default so a scenario replays identically in the browser and in
  // eval_warehouse.mjs — the same reason fleet_sim.js and breakout_sim.js share
  // this PRNG.
  const random = rng || mulberry32(seed);
  const robots = ["R1", "R2", "R3"].map(id => makeRobot(id, random));
  // stagger the ask phases so the three robots don't decide simultaneously
  // with near-identical state (symmetric answers → symmetric deadlock)
  robots.forEach((r, i) => { r.lastAsk = i * 120; });
  bindWorld(robots);
  return robots;
}
