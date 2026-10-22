// A/B harness for the fleet dispatch rules.
//
//   uv run python -m uvicorn server:app        # terminal 1
//   node eval_rules.mjs --n 200 --seed 7
//
// Imports the same static/fleet_sim.js the browser does, so there is no second
// implementation to drift out of sync.
//
// Scenarios are generated counterfactually: a sim is driven forward by a fixed policy
// and every decision tick is snapshotted. Both arms then answer the *same* snapshots,
// differing only in the question text. That isolates what we actually want to measure —
// did injecting the rule change the model's choice — instead of letting one arm's
// assignments steer the other arm's future inputs.
//
// The closed loop is reported separately, and per mode, because the counterfactual and
// the end-to-end result answer different questions: the counterfactual asks whether the
// rule reaches the model, the loop asks whether the world is one the rule helps.

import {
  createSim, step, assignTo, buildState, buildBaselineState, buildQuestion,
  buildBaselineQuestion, detourCost, nearestCost, feasible, rankedByNearest, headEstimate,
  insertStops, RULES,
} from "./static/fleet_sim.js";
import { canServe, reason, checkTBox, TBOX } from "./static/ontology.js";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const m = argv[i].match(/^--([^=]+)(?:=(.*))?$/);
    if (!m) continue;
    if (m[2] !== undefined) out[m[1]] = m[2];
    else if (argv[i + 1] && !argv[i + 1].startsWith("--")) out[m[1]] = argv[++i];
    else out[m[1]] = true;
  }
  return out;
}
const args = parseArgs(process.argv.slice(2));
const N = Number(args.n ?? 200);
const SEED = Number(args.seed ?? 7);
const ENDPOINT = args.url ?? "http://localhost:8000/api/predict";
// Reassigned per model by runCounterfactual() when --compare is used.
let MODEL = args.model ?? "auto";
const WARMUP = Number(args.warmup ?? 5);
const DEMAND_EVERY = Number(args.demand ?? 1.0);

// ---- scenario generation -----------------------------------------------------

/** Deep-copy the parts of a bus the cost model reads, so arms see identical inputs. */
const snapBus = b => ({
  id: b.id, label: b.label, x: b.x, y: b.y,
  stops: b.stops.map(s => ({ kind: s.kind, point: { ...s.point }, p: s.p })),
  onboard: new Set(b.onboard),
  queue: [...b.queue],
});

function generateScenarios(count, seed, multiPickup) {
  const out = [];
  let s = seed;
  while (out.length < count) {
    const sim = createSim({ seed: s++, config: { demandEvery: DEMAND_EVERY, multiPickup } });
    for (let i = 0; i < 4000 && out.length < count; i++) {
      for (const p of step(sim, 0.05)) {
        if (sim.demandCount < 3) { assignTo(sim, cheapest(sim, p), p); continue; }
        out.push({
          buses: sim.buses.map(snapBus),
          passenger: p,
          config: { multiPickup },
          delivered: sim.deliveredCount,
        });
        assignTo(sim, cheapest(sim, p), p);
      }
    }
  }
  return out;
}

/** The policy that drives scenario generation: cheapest bus that can actually take it. */
function cheapest(sim, p) {
  const ok = sim.buses.filter(b => feasible(b, p, sim));
  const pool = ok.length ? ok : sim.buses;
  return [...pool].sort((a, b) => detourCost(a, p, sim) - detourCost(b, p, sim))[0];
}

// ---- arms --------------------------------------------------------------------

const simOf = sc => ({ buses: sc.buses, passengers: [], deliveredCount: sc.delivered || 0,
                       config: { multiPickup: sc.config.multiPickup } });

/**
 * `rules`          — per-option derived numbers, infeasible buses omitted (Tier 0 + 1)
 * `rules-nofilter` — same numbers, but every bus is offered (measures Tier 0 alone)
 * `no-rules`       — the rule-free state and the original static question
 */
const ARMS = {
  "no-rules": sc => ({
    state: buildBaselineState(simOf(sc), sc.passenger),
    question: buildBaselineQuestion(),
  }),
  "rules": sc => ({
    state: buildState(simOf(sc), sc.passenger),
    question: buildQuestion(simOf(sc), sc.passenger),
  }),
  "rules-nofilter": sc => {
    const s = simOf(sc);
    const q = buildQuestion(s, sc.passenger);
    for (const bus of s.buses) {
      const k = `bus_${bus.id.toLowerCase()}`;
      if (!q.criteria[k]) q.criteria[k] = `${bus.label}: full, no room`;
    }
    return { state: buildState(s, sc.passenger), question: q };
  },
  // Identical to `rules` except that Tier 0 is answered by the TBox/ABox reasoner rather
  // than the hand-written seat loop. Everything the model sees is the same, so any
  // difference in outcome is attributable to the constraint layer alone.
  "ontology": sc => {
    const s = simOf(sc);
    s.config.feasibility = ontologyFeasibility;
    return { state: buildState(s, sc.passenger), question: buildQuestion(s, sc.passenger) };
  },
};

/**
 * Tier 0 through the ontology. Wraps the reasoner in the shape `insertSeq` expects, and
 * keeps the bus in scope for the ABox, which is built per candidate route.
 */
function ontologyFeasibility(cand, bus) {
  return canServe(bus, cand, RULES.seats).ok;
}

/**
 * Would the two constraint layers reach the same conclusion? Run the insertion search both
 * ways over every scenario and compare. This is the check that makes the ontology claim
 * falsifiable: a hand-rolled `if` cannot be wrong in a way you can detect, but a declared
 * axiom set can be compared against the thing it replaces.
 */
function divergenceCheck(scenarios) {
  const out = { compared: 0, disagreed: [], ontologyViolations: 0, extraAxioms: new Set() };
  for (const sc of scenarios) {
    for (const p of [sc.passenger]) {
      for (const bus of sc.buses) {
        const hand = insertStops(bus, p, RULES.seats);
        const ont = insertStops(bus, p, RULES.seats, ontologyFeasibility);
        out.compared++;
        const sameNull = (hand === null) === (ont === null);
        const sameDelta = sameNull && (hand === null || Math.abs(hand.delta - ont.delta) < 1e-9);
        if (!sameDelta) {
          out.disagreed.push({
            bus: bus.id, demand: p.id,
            hand: hand ? `+${hand.delta.toFixed(2)}` : "no route",
            ontology: ont ? `+${ont.delta.toFixed(2)}` : "no route",
          });
        }
        // Surface anything the reasoner flagged that the seat limit did not account for.
        if (ont) {
          for (const v of canServe(bus, ont.stops, RULES.seats, p).violations) {
            if (v.axiom !== "bus-capacity") out.extraAxioms.add(v.axiom);
          }
        }
      }
    }
  }
  return out;
}

// ---- scoring -----------------------------------------------------------------

/** The best achievable detour and straight-line distance, over buses that can take it. */
function bests(sc) {
  let detour = Infinity, near = Infinity;
  for (const b of sc.buses) {
    if (!feasible(b, sc.passenger, sc)) continue;
    detour = Math.min(detour, detourCost(b, sc.passenger, sc));
    near = Math.min(near, nearestCost(b, sc.passenger));
  }
  if (detour === Infinity) {                       // nothing can take it
    for (const b of sc.buses) {
      detour = Math.min(detour, detourCost(b, sc.passenger, sc));
      near = Math.min(near, nearestCost(b, sc.passenger));
    }
  }
  return { detour, near };
}

async function ask(arm, sc) {
  const { state, question } = ARMS[arm](sc);
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      state,
      questions: { assign: question },
      ...(MODEL !== "auto" ? { model: MODEL } : {}),
    }),
  });
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  const body = await res.json();
  const a = body.result.answers.assign;
  return { choice: a.choice, confidence: a.confidence ?? 0,
           act: (a.action || {}).act_probability ?? 1,
           latency: body.latency_ms, head: headEstimate(question),
           // Jev reports what the call cost; laya returns nothing because it runs
           // locally, so absent means $0 marginal — not "unknown".
           cost: body.result.cost_usd ?? 0 };
}

// ---- counterfactual ----------------------------------------------------------

const MODE = args.mode ?? "multi";
const multi = MODE !== "serial";
const scenarios = generateScenarios(N, SEED, multi);

const mean = xs => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
const pct = x => `${(100 * x).toFixed(1)}%`;
const pick = (rows, k) => mean(rows.map(r => r[k]));

const arms = ["no-rules", "rules", "rules-nofilter", "ontology", "greedy"];

// ---- ontology: self-check and divergence --------------------------------------
// This costs no API calls and is the load-bearing result for the ontology arm: the two
// constraint layers are run over every scenario's bus/demand pair and compared. Zero
// disagreements means the TBox faithfully reimplements what it replaces. Non-zero means
// the ontology is wrong somewhere, and the cause is a finding, not a number to tune.

const tbox = checkTBox(TBOX);
const divergence = divergenceCheck(scenarios);
const ontologyCost = (() => {
  const bus = scenarios[0]?.buses[0];
  if (!bus) return 0;
  const p = scenarios[0].passenger;
  const stops = insertStops(bus, p, RULES.seats)?.stops ?? [];
  const N = 20000, t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) canServe(bus, stops, RULES.seats, p);
  return Number(process.hrtime.bigint() - t0) / 1000 / N;
})();

console.log(`\n=== ontology: TBox self-check ===`);
console.log(tbox.satisfiable
  ? "  satisfiable — the axiom set is coherent"
  : "  UNSATISFIABLE:\n" + tbox.unsatisfiable.map(u => `    ${u.id}: ${u.reason}`).join("\n"));

console.log(`\n=== ontology: divergence from the hand-written constraint layer ===`);
console.log(`  ${divergence.compared} bus/demand pairs compared, ` +
            `${divergence.disagreed.length} disagreements`);
if (divergence.disagreed.length) {
  for (const d of divergence.disagreed.slice(0, 10)) {
    console.log(`    bus ${d.bus} demand #${d.demand}: hand-written ${d.hand}, ontology ${d.ontology}`);
  }
  if (divergence.disagreed.length > 10) {
    console.log(`    … and ${divergence.disagreed.length - 10} more`);
  }
}
if (divergence.extraAxioms.size) {
  console.log(`  axioms the seat limit alone does not account for: ` +
              `${[...divergence.extraAxioms].join(", ")}`);
} else {
  console.log(`  no other axiom was triggered — the seat limit was the only binding constraint here`);
}
console.log(`  reasoning cost ${ontologyCost.toFixed(2)} us per route check, against a ` +
            `~136000 us decision`);

/** Run every arm against one model, print its table, and return the rows. */
async function runCounterfactual(model) {
  MODEL = model;
  console.error(`\n[${model}] ${scenarios.length} ${MODE}-mode scenarios, seed ${SEED}`);
  console.error(`[${model}] warming up (${WARMUP} calls)…`);
  for (let i = 0; i < WARMUP; i++) await ask("rules", scenarios[0]);

  const results = {};
  for (const arm of Object.keys(ARMS).filter(a => a !== "greedy")) {
    const rows = [];
    for (const sc of scenarios) {
      let r;
      try {
        r = await ask(arm, sc);
      } catch (err) {
        console.error(`\nFAILED on arm ${arm} (${model}): ${err.message}`);
        process.exit(1);
      }
      const bus = sc.buses.find(b => `bus_${b.id.toLowerCase()}` === r.choice);
      if (!bus) { console.error(`unknown choice ${r.choice}`); process.exit(1); }
      const { detour, near } = bests(sc);
      const got = detourCost(bus, sc.passenger, sc);
      const gotNear = nearestCost(bus, sc.passenger);
      rows.push({
        detourRegret: got - detour,
        nearRegret: gotNear - near,
        pickedMinDetour: Math.abs(got - detour) < 1e-6,
        pickedNearest: Math.abs(gotNear - near) < 1e-6,
        violatedCapacity: !feasible(bus, sc.passenger, sc),
        confidence: r.confidence,
        act: r.act,
        latency: r.latency,
        head: r.head,
        cost: r.cost,
        // is there a real conflict? if nearest == least-detour the rule changes nothing
        tensionful: Math.abs(detour - near) > 0.5,
      });
    }
    results[arm] = rows;
    process.stderr.write(`[${model}] ${arm}: done\n`);
  }

  // The nearest-bus dispatcher, scored with no model at all. Note it does not check
  // feasibility — that is the point of the naive baseline, and it is why its
  // "nearest regret" can go negative: it is measured against an optimum restricted to
  // buses that can actually take the work.
  results.greedy = scenarios.map(sc => {
    const s = simOf(sc);
    const { detour, near } = bests(sc);
    const b = rankedByNearest(s, sc.passenger)[0];
    const got = detourCost(b, sc.passenger, s);
    return {
      detourRegret: got - detour,
      nearRegret: nearestCost(b, sc.passenger) - near,
      pickedMinDetour: Math.abs(got - detour) < 1e-6,
      pickedNearest: Math.abs(nearestCost(b, sc.passenger) - near) < 1e-6,
      violatedCapacity: !feasible(b, sc.passenger, s),
      tensionful: Math.abs(detour - near) > 0.5,
    };
  });

  const tension = results["rules"].filter(r => r.tensionful);
  console.log(`\n=== model: ${model} ===`);
  console.log(`${scenarios.length} scenarios (seed ${SEED}, ${MODE} mode), ` +
              `${tension.length} with a real nearest-vs-least-detour conflict\n`);
  const w = 22;
  console.log("".padEnd(w) + arms.map(a => a.padStart(16)).join(""));
  console.log("-".repeat(w + 16 * arms.length));
  for (const [label, fn] of [
    ["mean detour regret", a => pick(results[a], "detourRegret").toFixed(2) + " blk"],
    ["mean nearest regret", a => pick(results[a], "nearRegret").toFixed(2) + " blk"],
    ["picks least-detour", a => pct(mean(results[a].map(r => r.pickedMinDetour)))],
    ["picks nearest", a => pct(mean(results[a].map(r => r.pickedNearest)))],
    ["capacity violations", a => String(results[a].filter(r => r.violatedCapacity).length)],
    ["mean confidence", a => mean(results[a].map(r => r.confidence ?? 0)).toFixed(3)],
    ["ms", a => results[a].some(r => r.latency) ? mean(results[a].map(r => r.latency)).toFixed(0) : "—"],
    ["$ / decision", a => {
      const any = results[a].some(r => r.cost);
      return any ? `$${mean(results[a].map(r => r.cost)).toFixed(6)}` : "$0 local";
    }],
    ["head tokens (est)", a => results[a].some(r => r.head) ? String(Math.max(...results[a].map(r => r.head))) : "—"],
  ]) {
    console.log(label.padEnd(w) + arms.map(a => String(fn(a)).padStart(16)).join(""));
  }

  if (tension.length) {
    console.log(`\nRestricted to the ${tension.length} conflicting scenarios:\n`);
    console.log("".padEnd(w) + arms.map(a => a.padStart(16)).join(""));
    console.log("-".repeat(w + 16 * arms.length));
    for (const [label, fn] of [
      ["mean detour regret", a => pick(results[a].filter(r => r.tensionful), "detourRegret").toFixed(2) + " blk"],
      ["picks least-detour", a => pct(mean(results[a].filter(r => r.tensionful).map(r => r.pickedMinDetour)))],
    ]) {
      console.log(label.padEnd(w) + arms.map(a => String(fn(a)).padStart(16)).join(""));
    }
  }

  console.log(`\nconfidence (rules arm): ${percentiles(results["rules"], "confidence")}`);
  console.log(`act_probability      : ${percentiles(results["rules"], "act")}`);
  return results;
}

const MODELS = args.compare
  ? String(args.models ?? "english,jev").split(",").map(s => s.trim())
  : [MODEL];
const byModel = {};
for (const m of MODELS) byModel[m] = await runCounterfactual(m);

if (MODELS.length > 1) {
  console.log(`\n\n=== head to head: does criteria injection transfer across models? ===\n`);
  const w = 26;
  const head = ["model", "arm", "detour regret", "least-detour", "capacity viol.", "$ / decision"];
  console.log("".padEnd(w) + head.slice(1).map(h => h.padStart(20)).join(""));
  console.log("-".repeat(w + 20 * (head.length - 1)));
  for (const m of MODELS) {
    for (const arm of ["no-rules", "rules"]) {
      const r = byModel[m][arm];
      const cost = r.some(x => x.cost) ? `$${mean(r.map(x => x.cost)).toFixed(6)}` : "$0 local";
      console.log((`${m} / ${arm}`).padEnd(w) + [
        pick(r, "detourRegret").toFixed(2) + " blk",
        pct(mean(r.map(x => x.pickedMinDetour))),
        String(r.filter(x => x.violatedCapacity).length),
        cost,
      ].map(s => String(s).padStart(20)).join(""));
    }
  }
  console.log(`\nThe two arms differ only in what is put in the prompt: the same scenarios,`);
  console.log(`the same scoring, the same two models. A drop from no-rules to rules in both`);
  console.log(`rows means the technique is a property of System-1 models, not of laya alone.`);
}

function percentiles(rows, key) {
  const xs = rows.map(r => r[key]).filter(x => x !== undefined).sort((a, b) => a - b);
  if (!xs.length) return "—";
  const q = p => xs[Math.floor(p * (xs.length - 1))];
  return [0.10, 0.50, 0.95].map(p => `p${(p * 100).toFixed(0)} ${q(p).toFixed(3)}`).join("  ");
}

// ---- closed loop -------------------------------------------------------------

// The counterfactual isolates the decision. This closes the loop: each policy drives
// its own sim, in each world, so it pays for the assignments it makes downstream.
// This is the number that decides whether the rule is worth anything.
async function closedLoop(policy, seed, mPickup, seconds = 200) {
  const sim = createSim({ seed, config: { demandEvery: DEMAND_EVERY, multiPickup: mPickup } });
  const ticks = Math.round(seconds / 0.05);
  let calls = 0;
  const pickBus = async (p) => {
    if (policy === "greedy") return rankedByNearest(sim, p)[0];
    const { state, question } = ARMS[policy]({ buses: sim.buses, passenger: p, config: sim.config });
    const res = await fetch(ENDPOINT, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state, questions: { assign: question },
        ...(MODEL !== "auto" ? { model: MODEL } : {}) }),
    });
    calls++;
    if (!res.ok) return cheapest(sim, p);
    const ch = (await res.json()).result.answers.assign.choice;
    return sim.buses.find(b => `bus_${b.id.toLowerCase()}` === ch) || cheapest(sim, p);
  };
  for (let i = 0; i < ticks; i++) {
    for (const p of step(sim, 0.05)) {
      if (sim.demandCount < 3) { assignTo(sim, cheapest(sim, p), p); continue; }
      assignTo(sim, await pickBus(p), p);
    }
    for (const p of sim.passengers) {
      if (p.state === "waiting") assignTo(sim, await pickBus(p), p);
    }
  }
  return { delivered: sim.deliveredCount, demands: sim.demandCount, calls,
           blocks: sim.blocksDriven };
}

// The closed loop answers a different question from the counterfactual (does the rule
// pay off end to end), and it costs one API call per decision — at ~1s for Jev a 200s
// loop is many minutes and doubles the spend. So it runs on a single model only.
if (args.closed !== false && MODELS.length === 1) {
  const policies = ["rules", "greedy"];
  console.log(`\nClosed loop on model "${MODEL}", ${args.loopsecs ?? 200}s per policy, ` +
              `one demand every ${DEMAND_EVERY}s, 3 seeds each.\n` +
              `Cells are "delivered / blocks per demand":\n`);
  const w2 = 20;
  console.log("".padEnd(w2) + policies.map(p => p.padStart(22)).join(""));
  console.log("-".repeat(w2 + 22 * policies.length));
  for (const [label, m] of [["multi-pickup on", true], ["multi-pickup off", false]]) {
    const out = [];
    for (const p of policies) {
      const rows = [];
      for (const sd of [11, 12, 13]) rows.push(await closedLoop(p, sd, m));
      out.push(`${mean(rows.map(r => r.delivered)).toFixed(1)} / ` +
               `${mean(rows.map(r => r.blocks / r.demands)).toFixed(1)}`);
    }
    console.log(label.padEnd(w2) + out.map(s => s.padStart(22)).join(""));
  }
  const cell = async (p, m) => {
    const runs = await Promise.all([11, 12, 13].map(sd => closedLoop(p, sd, m)));
    return mean(runs.map(r => r.delivered));
  };
  const [rMulti, rSerial, gMulti, gSerial] = await Promise.all([
    cell("rules", true), cell("rules", false), cell("greedy", true), cell("greedy", false)]);
  const delta = (a, b) => {
    const pctChange = (a / b - 1) * 100;
    return `${pctChange >= 0 ? "+" : ""}${pctChange.toFixed(0)}%`;
  };
  console.log(`\nmulti-pickup is worth ${delta(rMulti, rSerial)} throughput to the rules ` +
              `dispatcher and ${delta(gMulti, gSerial)} to greedy.`);
  console.log(`The detour rule is worth ${delta(rMulti, gMulti)} over nearest-bus with ` +
              `multi-pickup, and ${delta(rSerial, gSerial)} without it.`);
}
