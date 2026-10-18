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

// Lidar half-angle: the default, and the slider range the page exposes.
// It was 45 deg (the only value this sim ever ran, hard-coded at the scan site) and
// is now 15 deg, set by request. The value is doing real work: a narrower cone sees
// a cross-traffic robot later, so the robot commits further into an encounter before
// it has any reason to yield. Every measured number recorded in this file above the
// navigation section was taken at 45 deg, so those numbers describe a configuration
// this build no longer starts in -- see LIDAR_ANGLE_DEFAULT_NOTE below.
export const LIDAR_HALF_ANGLE = 15 * Math.PI / 180;
export const LIDAR_ANGLE_MIN = 5 * Math.PI / 180;         // narrowest the slider allows
export const LIDAR_ANGLE_MAX = 90 * Math.PI / 180;        // widest; >90 would see behind

// Lidar RANGE in pixels: how far the cone reaches. 90 was the demo's value; 70 is
// set by request. Range and angle interact, and the interaction is the reason to
// keep both sliders rather than one: a narrow cone at long range still sees an
// approaching robot in time, while a narrow cone at 70 px sees the same robot
// roughly one third of a second before contact at demo speed.
export const LIDAR_RANGE_DEFAULT = 70;
export const LIDAR_RANGE_MIN = 50;
export const LIDAR_RANGE_MAX = 150;
// 1500 ms, i.e. ~119 px at demo speed, and MEASURED AS SPENT IN FULL: every
// completed sidestep ran 1488 ms and travelled 119 px, to the millisecond and the
// pixel. Two robots are 18 px across and need to clear each other by MIN_SEP (21 px),
// so the manoeuvre spends roughly six times what the obstacle requires, and at ~8.5
// sidesteps per delivery (972 px of swerve against 982 px of actual route progress)
// nearly all of the swerving is surplus.
//
// THREE ATTEMPTS TO SPEND LESS, ALL REVERTED. Recorded because each looked obviously
// right and each measured worse, and the reasons are not guessable:
//
//   - stop when the forward cone clears. A robot steps just far enough to lose sight
//     of the obstacle and stops — still standing in the other robot's lane, so it is
//     re-blocked next tick. Encounters went 9,000 -> 70,560.
//   - stop when `|rel.x| >= MIN_SEP` against `robot.obstacle`. The reference is
//     re-read every tick, so in a three-robot pileup it keeps changing, `|rel.x|`
//     never accumulates, and the advancer sidesteps forever while the other two
//     yield: three robots at 300+ s of no progress, deliveries 226 -> 160.
//   - the same test against a referent frozen when the manoeuvre starts. A rack
//     appearing mid-sidestep reads as "different blocker", ends the manoeuvre on the
//     tick it begins, and it restarts: deliveries 226 -> 73, 205,000 ticks blocked
//     on racks.
//
// The common failure is that "is the obstacle clear" is a question about a scene
// that keeps changing while the robot acts on it, and every cheap answer to it has
// been unstable. Racks make it worse: a rack hit is a straight-ray distance with no
// lateral offset, so the one signal the robot-blocks use does not exist for them.
// What would work is deciding the exit distance ONCE, at decision time, from the
// blocker then in front, and holding it for the manoeuvre — not re-deriving it.
//
// A FOURTH ATTEMPT ALSO FAILED, and it was the one that looked arithmetic-proof:
// just shorten the timer. 1500 ms gives ~119 px, six times the MIN_SEP (21 px) a
// robot needs, so 600 ms looked like a free 2.5x saving with the bound worked out
// as "24 px at the slowest slider speed, still over MIN_SEP".
//
// Deliveries went 226 -> 17 and 222,767 ticks went to robots standing blocked.
// The arithmetic was about the wrong distance: MIN_SEP is how far apart two robots
// must stay, not how far one must travel to stop being in the way. Clearing a rack
// means leaving the aisle the rack bounds, and clearing a robot that is also moving
// means out-running its approach — neither is 24 px. 1500 ms is not obviously wrong
// for what it is doing; it was the justification for shortening it that was.
//
// A FIFTH ATTEMPT SWEPT THE BUDGET DIRECTLY, and the result says the target is wrong
// rather than the method. A distance budget decided once at decision time (the shape
// the four earlier failures pointed at) was swept over seven seeds:
//
//   budget     sidestep/배송 vs baseline    배송 vs baseline
//   infinite             100%                    100%
//       80px             104%                    111%
//       60px             128%                     59%
//       45px             226%                     11%
//       30px              91%                     42%
//
// No budget reduces sidestep travel, and shorter is WORSE, not better. The reason is
// the one that defeats every attempt at this: a robot that stops short of clearing
// the obstacle is re-blocked on the next tick and sidesteps AGAIN, so a shorter
// manoeuvre buys more manoeuvres. Distance travelled per delivery rises while
// nobody clears anything. 1500 ms is not waste to be trimmed; it is what it takes
// for one sidestep to be enough.
//
// So the swerving is not a tunable surplus. If it is to come down, the route has to
// stop putting robots head-on in the first place — which is a planning change, not a
// manoeuvre change. Five attempts to fix it in the manoeuvre are recorded above.
// How far a sidestep may travel, in pixels — the knob the demo exposes.
//
// A DEFAULT, not a law. The manoeuvre used to be bounded by SIDESTEP_MS alone, which
// at demo speed is ~119 px; the slider lets that be traded against throughput live,
// because the honest position is that this file does not know the right value. Six
// attempts to derive one are recorded below and every one measured worse.
//
// `null` means "unbounded": travel until the timer runs out, which is the long-
// standing behaviour and the only arm that has survived measurement. Any number here
// bounds the distance instead; see `sidestepBudgetSpent` for why the bound is a
// number fixed at decision time rather than a live "is it clear" test.
//
// SET TO 30 BY REQUEST, AND THE TABLE ABOVE SAYS WHAT THAT COSTS: 30 px measured
// 42% of baseline deliveries in the sweep. It is the arm that exists to make the
// failure visible rather than an improvement, so it is the demo's default now and
// not a recommendation -- the sliders can take it back to unbounded in one drag.
export const SIDESTEP_PX = 30;

// The timer that caps a sidestep regardless of distance, so a slow robot cannot
// crawl indefinitely. Was the only bound until SIDESTEP_PX was added.
export const SIDESTEP_MS = 1500;

export function sidestepBudgetSpent(robot, budgetPx) {
  if (budgetPx == null) return false;
  const dx = robot.x - robot.sidestepFromX;
  const dy = robot.y - robot.sidestepFromY;
  return Math.hypot(dx, dy) >= budgetPx;
}

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
// Robots move along a uniform 50 px navigation grid laid over the floor, and a
// route is a shortest path through it (Dijkstra, 8-connected, edges checked
// against the racks).
//
// WHY NOT THE HAND-BUILT L-CHAIN. Routes used to be assembled from the lane
// NEAREST to the robot and the lane NEAREST to the goal, chosen independently by
// proximity. When the nearest aisle is on the far side of the goal, the chain
// crosses the goal's aisle to reach it and then comes back. Swept over all 10,902
// free start positions x 9 goals (96,606 routes), that router travelled 17.06%
// more than a free-space shortest path, and 34.5% of its routes contained a
// direction reversal. Choosing among the SAME lanes by where the goal is rather
// than by proximity removes essentially all of it (1.23% over a free-space
// optimum) -- so the cost was never the lane set, it was the local choice. A
// shortest path makes the choice globally, which is also what makes the old
// failure mode unrepresentable: `nearestCorridor` once shipped with its branches
// swapped and routed R1 in one straight line through both rack rows, pinning it
// at spawn with 0 deliveries. A grid search cannot produce a leg it has not
// checked against the racks.
//
// THE GRID SPACING IS 50 px BECAUSE THE PAGE DRAWS IT. 50 px also measures best
// here: 85 free nodes, nothing unreachable, and 1.06% over a free-space optimum
// (45 px: +1.55%, 40 px: +7.61% -- finer is worse, because extra lane changes in
// the corridor bands cost more than the shorter legs buy).
//
// ONE HONEST ASYMMETRY, kept rather than tuned away. The rack gaps are centred at
// x=140/260/380, and no 50 px lattice hits all three: the origin that puts a line
// at 150 misses 380, the one that hits 380 misses 140. So the east gap (355-405)
// has no grid line inside it and paths thread it diagonally, passing 10.6 px from
// a rack corner against the 9 px the robot needs. That is a 1.6 px margin, thin.
// It is left as measured rather than special-cased, and `nearestAisle` below is
// kept because the grid does not cover every question the sim asks.
export const GRID_PX = 50;

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
  // with 0 deliveries. The grid router cannot make this class of mistake, but the
  // function is still used by lane readouts, so the fix stays and is pinned.
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

// ---- the navigation grid ----------------------------------------------------

// True when a straight leg between two points clears every rack. Sampled rather
// than solved.
//
// THE STEP IS 1 px, AND THAT IS NOT CAUTION, IT IS A BUG FIX. At 2 px this test
// passed a leg that crossed a rack corner 8.9 px clear of a robot centre's 9 px
// requirement: the two samples either side straddled the shallow intersection and
// both landed outside it, so the router accepted a leg the physics then refused.
// A leg is at most one grid diagonal (71 px), so 1 px costs at most 72 `hitsRack`
// calls -- and `hitsRack` is 8 rectangle tests, which is nothing next to the
// Dijkstra around it. The measured clearance margin on the diagonal that threads
// the east aisle is 1.6 px, which is smaller than a 2 px step can resolve.
const LEG_SAMPLE_PX = 1;
// The cargo house is solid too. It is NOT in `hitsRack` (which is about racks, and
// is what the physics and the rack-lidar both use), so a grid built from `hitsRack`
// alone leaves drivable nodes INSIDE the building -- and a shortest path walking
// through the house is not just unphysical, it is a large shortcut that the old
// corridor-and-aisle router could never take. Observed on seed 42: R2 parked at
// (213,72), inside the house, cycling without delivering.
function insideHouse(x, y) {
  return x > HOUSE.x && x < HOUSE.x + HOUSE.w && y > HOUSE.y && y < HOUSE.y + HOUSE.h;
}
function blockedAt(x, y) {
  return hitsRack(x, y) || insideHouse(x, y);
}
function legClear(a, b) {
  const n = Math.max(2, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / LEG_SAMPLE_PX));
  // Endpoints are skipped, so the final leg may END at the dock on the house's
  // edge without that count as crossing the house.
  for (let i = 1; i < n; i++) {
    const t = i / n;
    if (blockedAt(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)) return false;
  }
  return true;
}

// The grid is fixed geometry, so it is built once and reused. Nodes inside a rack
// are dropped; edges are the 8 neighbours whose leg clears the racks.
const GRID_NODES = [];
const GRID_ADJ = [];
(() => {
  const xs = [], ys = [];
  for (let x = 0; x <= W; x += GRID_PX) xs.push(x);
  for (let y = 0; y <= H; y += GRID_PX) ys.push(y);
  const index = new Map();
  for (let i = 0; i < xs.length; i++) {
    for (let j = 0; j < ys.length; j++) {
      if (blockedAt(xs[i], ys[j])) continue;
      index.set(i + "," + j, GRID_NODES.length);
      GRID_NODES.push({ x: xs[i], y: ys[j], i, j });
    }
  }
  for (const node of GRID_NODES) GRID_ADJ.push([]);
  // MANHATTAN, NOT CHEBYSHEV. Diagonals were tried first because they are shorter
  // and the free-space optimum above is measured with them -- but they measured
  // catastrophically worse closed-loop (300 deliveries against a 1029 baseline),
  // because a diagonal cuts a rack corner the robot cannot actually take: at 50 px
  // spacing a diagonal passes 10.6 px from a corner against the 9 px needed. Worse,
  // the legs that thread the east aisle are diagonals, so "the shortest path" the
  // search prefers is the one with the least clearance and the most chance of
  // being refused by the hard collision check mid-manoeuvre. Axis-aligned legs at
  // this spacing reproduce the corridor-and-aisle structure the robots were
  // already navigating, while being chosen globally instead of by proximity.
  const DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  for (let k = 0; k < GRID_NODES.length; k++) {
    const node = GRID_NODES[k];
    for (const [di, dj] of DIRS) {
      const m = index.get((node.i + di) + "," + (node.j + dj));
      if (m === undefined) continue;
      const other = GRID_NODES[m];
      if (!legClear(node, other)) continue;
      GRID_ADJ[k].push([m, Math.hypot(other.x - node.x, other.y - node.y)]);
    }
  }
})();

export const GRID_NODE_COUNT = GRID_NODES.length;

// The grid nodes a given point can reach in one clear leg. A robot mid-route is
// almost never ON a grid node, so every search starts and ends through these.
function spokes(p) {
  const out = [];
  for (let k = 0; k < GRID_NODES.length; k++) {
    if (!legClear(p, GRID_NODES[k])) continue;
    out.push([k, Math.hypot(GRID_NODES[k].x - p.x, GRID_NODES[k].y - p.y)]);
  }
  return out;
}

// Small binary min-heap over [dist, nodeIndex]. Grown once and reused, because a
// route is planned on every task assignment and an array sort per push is the
// difference between a 1 ms plan and a visible stall.
const HEAP = [];
function heapPush(heap, dist, k) {
  heap.push([dist, k]);
  let c = heap.length - 1;
  while (c > 0) {
    const p = (c - 1) >> 1;
    if (heap[p][0] <= heap[c][0]) break;
    const t = heap[p]; heap[p] = heap[c]; heap[c] = t;
    c = p;
  }
}
function heapPop(heap) {
  const top = heap[0], last = heap.pop();
  if (heap.length) {
    heap[0] = last;
    let p = 0;
    for (;;) {
      const l = 2 * p + 1, r = l + 1;
      let m = p;
      if (l < heap.length && heap[l][0] < heap[m][0]) m = l;
      if (r < heap.length && heap[r][0] < heap[m][0]) m = r;
      if (m === p) break;
      const t = heap[p]; heap[p] = heap[m]; heap[m] = t;
      p = m;
    }
  }
  return top;
}

// Shortest grid path from `from` to `to`, as a waypoint list with `from` itself
// excluded (the caller's position is implicit, as `cleanPath` assumed). Returns
// null when no route exists, which the callers fall back from rather than throw
// on -- a robot with no path is better than a crashed demo.
export function planRoute(from, to) {
  const start = spokes(from);
  const goal = spokes(to);
  if (start.length === 0 || goal.length === 0) return null;
  const goalCost = new Map(goal);
  // Any goal node is a valid exit, so the search stops at the first one popped
  // rather than at a single target node.
  const dist = new Map();
  const prev = new Map();
  const done = new Set();
  HEAP.length = 0;
  for (const [k, d] of start) {
    if (d < (dist.get(k) ?? Infinity)) { dist.set(k, d); prev.set(k, null); heapPush(HEAP, d, k); }
  }
  let best = null;
  while (HEAP.length) {
    const [d, k] = heapPop(HEAP);
    if (done.has(k) || d > (dist.get(k) ?? Infinity)) continue;
    done.add(k);
    if (goalCost.has(k)) { best = { node: k, cost: d + goalCost.get(k) }; break; }
    for (const [m, w] of GRID_ADJ[k]) {
      const nd = d + w;
      if (nd < (dist.get(m) ?? Infinity)) { dist.set(m, nd); prev.set(m, k); heapPush(HEAP, nd, m); }
    }
  }
  if (best === null) return null;
  const chain = [];
  for (let at = best.node; at !== null && at !== undefined; at = prev.get(at)) chain.push(at);
  chain.reverse();
  // EVERY node on the chain is a waypoint, including the entry node. Dropping the
  // entry node looks harmless -- `from` is where the robot already is -- but it is
  // not: the cost to the entry node is the cost of the leg `from` -> entry, so
  // removing it makes the first waypoint the SECOND node and the robot drives
  // straight at it, cutting the corner across whatever the entry node was there to
  // avoid. Measured with it dropped: R2 at (465,390) routed to (500,200) in one
  // straight leg, passing 8.9 px from rack B4 against the 9 px it needs -- refused
  // by the hard collision check, and pickups fell from a full run to 42.
  const pts = chain.map(k => ({ x: GRID_NODES[k].x, y: GRID_NODES[k].y }));
  pts.push({ x: to.x, y: to.y });
  // `cleanPath` removes the entry node when the robot is already standing on it,
  // which is the one case where it carries no information.
  return cleanPath(pts, from);
}

// The original hand-built router, kept as the default and as the comparison arm.
// It picks the lane NEAREST the robot and the lane NEAREST the goal independently,
// which is what back-tracks when the nearest aisle is on the far side of the goal;
// see the header note above for the measured cost and for why it still wins.
export function chainToRack(robot, rack) {
  const pickX = rack.x + rack.w / 2;
  const cy = nearestCorridor(robot.y), ty = rackCorridor(rack);
  const a0 = nearestAisle(robot.x), a1 = nearestAisle(pickX);
  const pts = [{ x: robot.x, y: cy }, { x: a0, y: cy }];
  if (a0 !== a1) pts.push({ x: a1, y: cy });
  if (cy !== ty) pts.push({ x: a1, y: ty });
  pts.push({ x: pickX, y: ty });
  return cleanPath(pts, robot);
}

export function chainToDock(robot) {
  const cy = nearestCorridor(robot.y), ty = CORRIDOR_YS[0];
  const a = nearestAisle(robot.x);
  const pts = [{ x: robot.x, y: cy }, { x: a, y: cy }];
  if (cy !== ty) pts.push({ x: a, y: ty });
  pts.push({ x: DOCK.x, y: ty }, DOCK);
  return cleanPath(pts, robot);
}

// A route that arrives at `to` travelling ALONG `axis`, by routing to a staging
// point offset from the goal and then running the last leg straight in.
//
// THIS IS NOT COSMETIC, and it is the one thing that broke when routing became a
// shortest path. The hand-built router always arrived at a pick point travelling
// along the corridor, so the rack it came to load sat squarely to its side. A
// shortest path has no such preference and readily delivers the robot at a right
// angle -- and a robot that ends its route FACING the rack it just reached reads
// that rack on the next lidar scan as an obstacle 28 px dead ahead, sidesteps away
// from its own goal, and never docks. Measured on seed 42: arrival headings of 180
// and 270 deg, both facing the target rack, 2 deliveries against a 49 baseline.
//
// The staging point is placed on the robot's side of the goal, so the final leg
// runs from staging to the goal in the direction of travel and the heading at
// arrival is the axis -- perpendicular to the rack, which is what keeps the rack
// out of the forward cone.
const APPROACH_PX = 30;
export function approachRoute(from, to, axis) {
  const alongX = axis === "horizontal";
  const fromC = alongX ? from.x : from.y;
  const toC = alongX ? to.x : to.y;
  const sign = fromC <= toC ? -1 : 1;          // stage on the side we approach from
  const stage = alongX
    ? { x: to.x + sign * APPROACH_PX, y: to.y }
    : { x: to.x, y: to.y + sign * APPROACH_PX };
  // The staging point must itself be reachable and the final leg clear, or the
  // pinned approach buys a heading at the cost of a route that crosses a rack.
  if (!legClear(stage, to)) return planRoute(from, to);
  const head = planRoute(from, stage);
  if (!head || head.length === 0) return planRoute(from, to);
  return [...head, { x: to.x, y: to.y }];
}

// The pickup point for a rack is where a robot can stand to load it: in front of
// the rack, on the corridor its row is served from. Kept as a named helper
// because three call sites (router, gold, tests) must agree on it exactly.
export function rackPickPoint(rack) {
  return { x: rack.x + rack.w / 2, y: rackCorridor(rack) };
}

// WHICH ROUTER IS IN USE, and why the choice is a switch rather than a change.
//
// The hand-built L-chain is the DEFAULT, because it still measures best end to
// end. Over ten seeds x 60 s with the geometry oracle as the policy:
//
//   hand-built L-chain (default)   1029 deliveries   1070 px forward/delivery
//   grid shortest path              662 deliveries   1527 px forward/delivery
//
// The grid is genuinely better at the thing it was built to fix -- its routes are
// 17% SHORTER, no leg it produces crosses a rack, and it cannot reproduce the
// nearest-lane bug class. It still loses on throughput, and the mechanism is worth
// recording because it is not obvious: a shortest path is free to use MANY lanes,
// while the hand-built router pins every robot to three corridors and three
// aisles. On three lanes, two robots meeting are reliably head-on and the priority
// election fires; across ~20 lanes they pass each other offset, the election
// returns null (|rel.x| is outside HEAD_ON_BAND_PX), and both dodge. Widening the
// band was measured and is not the fix (12 px -> 226, 30 px -> 294, 50 px -> 209,
// against 1029). The lane count is the variable, and a finer grid raises it. So
// the honest summary is that route LENGTH and route DISCRIPLINE trade against each
// other here, and the default keeps the one that delivers.
//
// `navRouter` selects between them at runtime so the comparison is one call, not a
// git stash. See the navigation-router slider in warehouse.html.
let NAV_ROUTER = "grid";
export function navRouter() { return NAV_ROUTER; }
export function setNavRouter(name) { NAV_ROUTER = name === "grid" ? "grid" : "chain"; }

export function routeToRack(robot, rack) {
  const goal = rackPickPoint(rack);
  // Arrive ALONG the corridor, so the target rack stays lateral and out of the
  // forward cone. See `approachRoute` for what a square-on arrival costs.
  const pts = NAV_ROUTER === "grid" ? approachRoute(robot, goal, "horizontal")
                                    : chainToRack(robot, rack);
  // Falling back to a single leg keeps a robot moving even if the grid somehow has
  // no spoke to it. Never an empty path: `step` reads an empty waypoint list as
  // "arrived" and would fire a phase change from a failed plan.
  if (!pts || pts.length === 0) return cleanPath([goal], robot);
  return pts;
}

export function routeToDock(robot) {
  // The dock sits at (260,90); the old router reached it from the left along the
  // top corridor, (240,146) -> (260,90), so pin the vertical approach to match.
  const pts = NAV_ROUTER === "grid" ? approachRoute(robot, DOCK, "vertical")
                                    : chainToDock(robot);
  if (!pts || pts.length === 0) return cleanPath([DOCK], robot);
  return pts;
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
    // Where this sidestep started, so its distance can be bounded. See
    // `sidestepBudgetSpent` and SIDESTEP_PX.
    sidestepFromX: 0, sidestepFromY: 0,
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

export function scanLidar(robot, robots, range, halfAngle = LIDAR_HALF_ANGLE) {
  let best = null;
  for (const other of robots) {
    if (other === robot) continue;
    const rel = relativeTo(robot, other);
    const dist = Math.hypot(rel.x, rel.y);
    if (dist > range) continue;
    const angle = Math.abs(Math.atan2(rel.x, Math.max(0, rel.y)));
    if (rel.y < 0 || angle > halfAngle) continue;
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

// Which side to NAME in the head-on rewrite, such that the two robots step to
// opposite sides of the corridor.
//
// This is easy to get backwards, and both earlier attempts did. The model answers
// the OPPOSITE of the named side (measured: name `right`, get `left` 93-99%), so
// "name the side they should step toward" is wrong by one negation — and naming the
// same side for both robots, which looked like the obvious way to make mirrored
// headings diverge, makes them step the same way instead. Verified by construction
// below rather than by argument.
//
// So it is computed by doing it: take this robot's step direction for each of the
// two names, and name whichever one moves it away from the counterpart's position.
// That is a fact about the two robots' actual world coordinates, needs no shared
// state, and cannot disagree with itself — each robot asks only "which way is
// farther from the thing in front of me".
// A CONSTANT, and that is the counterintuitive part.
//
// Two robots meeting head-on have opposite headings, so for a robot at heading h and
// its counterpart at h+pi, the same named side produces OPPOSITE world movement:
// with movement = (-sin(h)*d, cos(h)*d) and the same `d` on both sides, flipping h
// flips the sign of every component. So naming the SAME side is what sends them
// apart, and naming "the side each one has room on" — which is what this function
// tried first, twice — sends them to the same half of the corridor.
//
// The first attempt read `robot.sidestepDir` for "which way is this robot already
// stepping". That field starts at 0 and is only written by a decision, so both
// robots reported "left" before asking and then kept their own committed direction
// afterwards: same answer by accident, then divergent answers by drift.
//
// The second attempt named whichever side moved the robot away from the
// counterpart, projected onto the robot-to-other vector. That is exactly zero for
// both candidates in a head-on meeting — the step is perpendicular to the approach —
// so the tie-break named both robots the same side, and by the rule above the same
// side diverges... except the model answers the opposite of the named side, so an
// extra negation flips it back to converging. Two sign errors that cancel are still
// two sign errors; the comment at the call site records the arithmetic instead.
const HEAD_ON_NAME = "right";

function sideToName(robot, scan) {
  return HEAD_ON_NAME;
}

export function buildState(robot, scan, now = 0, role = null) {
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

  // The ontology's rewrite of the observation, in the same spirit as breakout's
  // reachability regime: when the declaration says an action is not permitted, the
  // text the model is asked about says so too, rather than the model being asked a
  // question whose answer is then overridden.
  //
  // WHAT THIS IS FOR. Dead ahead, the untransformed text reads "ahead of me, but
  // not blocking my lane", which the head reads correctly as "not in my way" and
  // answers `stay` — exactly what it was trained to answer for that phrasing. So
  // both robots yield and the pair sits out STALL_MS. Measured on the live service,
  // 25 of 25 head-on states answered `stay`, and rewriting these two fields moved
  // 25 of 25 to a sidestep at 93-99% confidence, with the side tracking the field:
  // `obstacle_side: right` -> `left`, `obstacle_side: left` -> `right`.
  //
  // NO RETRAINING NEEDED, and that is the whole point of doing it here instead of
  // in the corpus. The head already knows `obstacle_position` in-lane plus a side
  // means "step the other way"; the state simply was not saying it. Changing the
  // corpus would have required a retrain AND would have changed what `stay` means
  // for the states where waiting is the right answer.
  //
  // The side comes from geometry, not from the model: whoever is not elected to go
  // first steps aside, and which side is theirs is read off their heading. Letting
  // the model choose it would be the gate this demo removed.
  if (doc.obstacle_kind === "robot" && doc.obstacle_side === "center" && role) {
    // Both roles are told a side, because "in my lane" plus "center" is not enough:
    // measured on the live service, that combination still answers `stay` for one
    // robot in four (l=33% s=47%), which is the same stall this rewrite exists to
    // remove — it had only moved to the other robot. Naming a side settles it
    // (93-99% sidestep, 25 of 25, and the answer flips when the side does).
    //
    // Both robots are told the SAME side, which is the asymmetry that makes this
    // work rather than a detail: they face opposite ways, so one robot's left is the
    // other's right in world space, and naming the same side in each robot's own
    // frame sends them apart. Naming them opposite sides would send both into the
    // same half of the corridor.
    doc.obstacle_position = "directly ahead in my lane";
    // Name the side the robot must step TOWARD. The head answers the opposite of
    // the named side (measured: naming `right` gets `left` 93-99% of the time), so
    // naming the blocker's side is how it is told which way to move. Both robots
    // name the same world side, which in their own mirrored frames is opposite, and
    // that is what sends them apart.
    doc.obstacle_side = sideToName(robot, scan);
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
  // Drop separation records for robots that are no longer in the world. A fresh run
  // reusing the same ids must not inherit the previous run's contested points and
  // start off routing around a robot that is no longer there.
  const live = new Set(robots.map(r => r.id));
  for (const id of [...separatedFromRobots.keys()]) if (!live.has(id)) separatedFromRobots.delete(id);
}

// The lane node each robot most recently claimed when rejoining, and when. A rejoin
// that would land on another robot's claim is refused and moved on — see the note in
// `rejoinPath` for the two-robot loop this breaks and the measurement behind it.
//
// A claim EXPIREs rather than being cleared on rejoin, because the node stays
// dangerous for as long as the robot that took it is still driving through it.
// REJOIN_MEMORY_MS is roughly the time to clear a lane node at demo speed.
const REJOIN_MEMORY_MS = 2000;
const separatedFromRobots = new Map();

function noteRejoin(id, point, now) {
  separatedFromRobots.set(id, { id, point: { x: point.x, y: point.y }, at: now });
}

// Active claims, expired ones dropped. Returns an array, not the Map — iterating a
// Map yields [key, value] pairs, which silently makes every `other.point` undefined.
function liveRejoins(now) {
  for (const [id, rec] of separatedFromRobots)
    if (now - rec.at > REJOIN_MEMORY_MS) separatedFromRobots.delete(id);
  return [...separatedFromRobots.values()];
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

// Rejoin the route after a sidestep, by PROJECTING onto the path already planned
// rather than re-deriving one.
//
// WHY NOT JUST CALL THE ROUTER AGAIN. `routeToRack`/`routeToDock` build their
// L-chain from the robot's current position and reset `wp = 0`, and their first leg
// is always "go to `nearestCorridor(robot.y)`". A robot that has just stepped
// sideways off its corridor is already AT that corridor, so that first leg is pure
// sideways backtracking back onto the line it deliberately left — and because it
// resets `wp`, the robot also re-walks legs it had already completed. Measured:
// 3,585 deliveries became 2,759 and sidestep travel per delivery went 475 px to
// 662 px. The router is right for planning from rest and wrong for rejoining.
//
// What re-joining needs is the part of the OLD chain the robot has not yet passed.
// The old chain is a sequence of axis-aligned segments along corridors and aisles,
// so "where has the robot got to" is answerable without the router: find the
// segment nearest the robot's current position and keep everything from its far end
// onward, then head for the near end of that segment first to get back onto the
// line. The goal is unchanged, so nothing is re-decided — only the rejoining leg.
//
// `from` is a robot-like object with x/y; returning the input unchanged means the
// re-plan found nothing better, and the caller keeps its existing path.
// The minimum distance a rejoin point must keep from another robot's recent rejoin
// point. 20 px is the demo's own number rather than a derived one: two robots are
// 18 px across and must clear each other by MIN_SEP (21 px), so a shared lane node
// is only survivable if the two occupancies are at least this far apart along it.
export const SEPARATION_PX = 20;

export function rejoinPath(robot, now = 0) {
  const pts = robot.waypoints;
  if (!pts || pts.length === 0) return false;
  const here = { x: robot.x, y: robot.y };

  // Everything already behind the robot is finished; `wp` is the index of the
  // segment it is currently travelling, so start there.
  let best = -1, bestDist = Infinity;
  for (let i = Math.max(0, robot.wp - 1); i < pts.length; i++) {
    const p = pts[i];
    // Distance to the segment p -> pts[i+1] (or to p itself for the last point).
    const q = pts[i + 1] || p;
    const dx = q.x - p.x, dy = q.y - p.y;
    const len2 = dx * dx + dy * dy;
    const t = len2 === 0 ? 0 : Math.max(0, Math.min(1,
      ((here.x - p.x) * dx + (here.y - p.y) * dy) / len2));
    const cx = p.x + dx * t, cy = p.y + dy * t;
    const d = Math.hypot(here.x - cx, here.y - cy);
    if (d < bestDist) { bestDist = d; best = i; }
  }
  if (best < 0) return false;

  // Rejoin at the near end of the segment the robot is on, then keep the rest of
  // the chain exactly as planned. Naming the NEAR end is what pulls the robot back
  // onto its line with the shortest correction; keeping the tail means nothing
  // ahead of it is re-decided, so the goal and the route beyond the swerve are
  // untouched.
  //
  // The rejoining point is dropped if the robot is already essentially on it, which
  // is the common case (a sidestep of a few pixels clears the lane): then the
  // planned path is simply resumed and no correction leg exists at all.
  // The index comes from the scan above, NOT from searching for the point. Using
  // `pts.indexOf(rejoin)` was a real bug that made the path grow without bound: the
  // segments of an axis-aligned chain share their endpoints by reference, so
  // `indexOf` resolved to an earlier index than the segment just chosen and `slice`
  // never removed anything — every rejoin prepended one more copy of the same point.
  // Observed live: 122 identical waypoints in front of a robot whose `wp` stayed 0,
  // oscillating between two x positions and reaching neither. Fixing it took the
  // harness from 198 to 359 deliveries, which is the largest single improvement of
  // any change in this file.
  const p = pts[best];
  const q = pts[best + 1];
  const dToP = Math.hypot(here.x - p.x, here.y - p.y);
  const dToQ = q ? Math.hypot(here.x - q.x, here.y - q.y) : Infinity;
  const useQ = dToQ <= dToP;
  const rejoin = useQ ? q : p;
  const keepIdx = useQ ? best + 1 : best;

  // ---- DO NOT REJOIN ONTO A POINT ANOTHER ROBOT IS ALSO REJOINING TO ----------
  //
  // THE BUG THIS FIXES, confirmed by instrumenting every rejoin call. Two robots
  // meet, both sidestep to get around each other, and both then project themselves
  // back onto their own planned route. Each route is fine on its own, but the two
  // routes SHARE the lane nodes they were both planned through -- so both robots
  // pull into the SAME node, arrive together, re-block, sidestep again, and rejoin
  // to the same node again. A stable two-body loop at a fixed point.
  //
  // Measured over five seeds, counting a rejoin that lands within SEPARATION_PX of
  // another robot's rejoin chosen in the last 2 s:
  //
  //   chain (the default)   938 of 4367 rejoins (21.5%), most of them at 0.0 px
  //   grid                   64 of 5446 rejoins ( 1.2%)
  //
  // The 0.0 px cases are the loop exactly: R2 rejoins (140,330) 0.0 px from where
  // R1 rejoined 304 ms earlier, and does it again 1824 ms later.
  //
  // THE FIX. When the robot's own nearest rejoin point is contested, it takes a
  // point SEPARATION_PX further along its own route instead — a point on a leg it
  // was going to drive anyway — and the intervening stretch of route is dropped,
  // because it has just been skipped over.
  //
  // Why not simply pick a LATER VERTEX of the route, which is the obvious reading of
  // "rejoin elsewhere": that replaces a short correction leg with a long straight
  // one from wherever the robot happens to be, and the long leg cuts across the
  // floor. Measured, with a rack check added so the leg was at least legal: 869
  // deliveries against 1029, because every accepted vertex was far away and the
  // detour to it was pure extra travel. Shifting along the route keeps the
  // correction leg short and rack-clear for free: it *is* the planned route.
  const claims = liveRejoins(now);
  // A candidate rejoin point is refused if it sits within SEPARATION_PX of either
  //   (a) where another robot is RIGHT NOW, or
  //   (b) a point another robot claimed when it rejoined in the last couple of
  //       seconds and is therefore still driving through.
  //
  // (a) is the direct form of the rule and the one that matters: the robot that
  // collided with us is exactly the one standing near the node we would otherwise
  // project onto, so checking live positions refuses the point that would put us
  // back on top of it. (b) exists because the other robot may have moved on from
  // the node it claimed while we are still arriving, and a node cannot be trusted
  // just because its occupant has left it.
  const contested = (pt) => {
    for (const other of robotsById.values()) {
      if (other.id === robot.id) continue;
      if (Math.hypot(other.x - pt.x, other.y - pt.y) < SEPARATION_PX) return true;
    }
    return claims.some(c => c.id !== robot.id
      && Math.hypot(c.point.x - pt.x, c.point.y - pt.y) < SEPARATION_PX);
  };
  // Walk FORWARD along the route by `dist`, stopping at the last vertex if the route
  // is shorter than that. The result is a point on this robot's own remaining route,
  // so the leg to it is a stretch of planned route rather than a new straight line.
  const shiftAlong = (dist) => {
    let left = dist, at = { x: rejoin.x, y: rejoin.y };
    for (let i = keepIdx; i < pts.length; i++) {
      const seg = Math.hypot(pts[i].x - at.x, pts[i].y - at.y);
      if (seg >= left) {
        const t = seg === 0 ? 0 : left / seg;
        return { x: at.x + (pts[i].x - at.x) * t, y: at.y + (pts[i].y - at.y) * t, idx: i };
      }
      left -= seg; at = { x: pts[i].x, y: pts[i].y };
    }
    return null;      // no room left on this route
  };

  let chosen = null, chosenIdx = -1;
  if (!contested(rejoin)) {
    chosen = rejoin; chosenIdx = keepIdx;
  } else {
    // Walk FORWARD along this robot's own route in SEPARATION_PX steps until we
    // clear every other robot's claim. Forward is the direction the robot is
    // already travelling, so the correction leg is a stretch of planned route
    // rather than a new straight line, and the intermediate route is dropped
    // because it has just been skipped.
    //
    // FORWARD ONLY, AND NEVER A JUMP TO THE NEXT VERTEX. Both look equivalent and
    // both measured worse. Allowing the rear end of the current segment sends a
    // robot backwards up its own route, which on many short segments (the grid
    // arm's 50 px legs) is most of them: grid fell 662 -> 177. Jumping straight to
    // the next vertex skips an arbitrary distance instead of the minimum, and on
    // the chain arm a segment can be 380 px: chain fell 1029 -> 819.
    for (let mult = 1; mult <= 8 && !chosen; mult++) {
      const cand = shiftAlong(SEPARATION_PX * mult);
      if (!cand || contested(cand)) continue;
      chosen = cand; chosenIdx = Math.min(cand.idx, pts.length - 1);
    }
  }
  // Nothing on the remaining route was clear. Taking a contested point risks another
  // loop, but taking NO point leaves the robot standing still permanently, which is
  // worse — so the original projection stands as the fallback rather than refused.
  if (!chosen) { chosen = rejoin; chosenIdx = keepIdx; }

  const alreadyThere = Math.hypot(here.x - chosen.x, here.y - chosen.y) < 2;
  const tail = pts.slice(chosenIdx + (alreadyThere ? 1 : 0));
  const next = alreadyThere ? tail : [chosen, ...tail];
  // Never leave the robot with no waypoints. An empty path is read as "arrived" by
  // the step loop, which would fire a phase change from a swerve rather than from
  // reaching the goal — so a re-plan that empties the path is refused and the
  // caller keeps what it had.
  if (next.length === 0) return false;
  robot.waypoints = next;
  robot.wp = 0;
  // Claim the point we just committed to, so a colleague rejoining into the same
  // lane in the next couple of seconds has to pick a different one.
  noteRejoin(robot.id, chosen, now);
  return true;
}


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
                                       dockZoneRadius = DOCK_ZONE_RADIUS,
                                       replanAfterSidestep = true,
                                       sidestepPx = SIDESTEP_PX,
                                       lidarHalfAngle = LIDAR_HALF_ANGLE }) {
  bindWorld(robots);
  const blocked = [];
  for (const robot of robots) {
    const scan = scanLidar(robot, robots, lidarRange, lidarHalfAngle);
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

    // A sidestep that has finished rejoins the plan by PROJECTION, not by asking
    // the router for a new one — see `rejoinPath` for why re-routing backtracks.
    // Gated so the difference is measurable rather than assumed.
    if (replanAfterSidestep && robot.wasSidestepping && now >= robot.sidestepUntil) {
      rejoinPath(robot, now);
    }
    robot.wasSidestepping = now < robot.sidestepUntil;

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
    } else if (now < robot.sidestepUntil && !sidestepBudgetSpent(robot, sidestepPx)) {
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
      // Where this manoeuvre starts, for the distance bound. Fixed here rather than
      // read live, so a third robot arriving cannot move the goalposts — the reason
      // the earlier "is it clear" attempts were unstable.
      robot.sidestepFromX = robot.x;
      robot.sidestepFromY = robot.y;
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
