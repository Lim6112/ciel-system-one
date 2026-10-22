// Warehouse closed-loop oracle: does the head choose the side the geometry says?
//
// This is the number that proves the fix, and the one the README quotes. The
// failure it exists to detect is a single-class collapse — the doc/breakout head
// answered "right" to 171 of 178 recorded warehouse decisions regardless of
// which side the obstacle was on — so per-side accuracy and a confusion table
// matter more than the mean.
//
// It imports static/warehouse_sim.js, the same module the browser runs, so the
// scenarios and the gold cannot drift from the demo.
//
//   uv run python -m uvicorn server:app          # terminal 1
//   node eval_warehouse.mjs --model ciel_warehouse_v1
//   node eval_warehouse.mjs --model ciel_decision_model_v1   # the before picture
//
// The confidence percentiles it prints are what CONF_FLOOR in warehouse.html
// should be set from — measured, not guessed. A floor reused across models does
// not mis-set that gate, it silently disables it.

import * as S from "./static/warehouse_sim.js";

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};

const URL = argOf("--url", "http://localhost:8000/api/predict");
const MODEL = argOf("--model", "auto");
const N = Number(argOf("--n", 120));
const SEED = Number(argOf("--seed", 1));
const CONFIG = { speed: 80, lidarRange: 90, autoplay: true };

// Collect decision points by driving the sim. Driving (rather than sampling
// random states) is what makes this an oracle over the states the demo actually
// produces — a fixed distribution is the trap the fleet A/B harness documents.
function collectScenarios(n, seed) {
  const robots = S.createWorld(seed);
  const seen = [];
  let now = 0;
  for (let i = 0; i < 400000 && seen.length < n; i++) {
    now += 16;
    S.step(robots, 0.016, now, CONFIG);
    for (const robot of robots) {
      const scan = robot.obstacle;
      if (!scan || robot.asking || robot._taken === now) continue;
      const stalledFor = robot.lastProgressAt ? (now - robot.lastProgressAt) / 1000 : 0;
      const stalled = stalledFor * 1000 >= S.STALL_MS;
      const kind = scan.rack ? "rack" : "robot";
      const qid = S.questionId(kind, stalled);
      seen.push({
        qid,
        kind,
        question: S.buildQuestion(kind, stalled),
        state: S.buildState(robot, scan, now),
        gold: S.goldDirection(scan, stalled),
        feasible: S.feasibleOptions(kind, stalled),
      });
      robot._taken = now;
    }
  }
  return seen;
}

async function ask(scenario) {
  const response = await fetch(URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      state: scenario.state,
      questions: { [scenario.qid]: scenario.question },
      model: MODEL,
    }),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status}: ${detail.slice(0, 300)}`);
  }
  const body = await response.json();
  return body.result.answers[scenario.qid];
}

function percentiles(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const at = q => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
  return { p10: at(0.10), p50: at(0.50), p90: at(0.90), min: sorted[0], max: sorted[sorted.length - 1] };
}

async function main() {
  const scenarios = collectScenarios(N, SEED);
  if (!scenarios.length) {
    console.error("no decision points reached — did the sim evolve?");
    process.exit(1);
  }
  console.error(`asking ${scenarios.length} scenarios (model=${MODEL}) → ${URL}`);

  const rows = [];
  for (const scenario of scenarios) {
    try {
      const answer = await ask(scenario);
      rows.push({ scenario, choice: answer.choice, confidence: answer.confidence,
                  probs: answer.probabilities || {} });
    } catch (err) {
      console.error(`  failed: ${err.message}`);
      process.exit(1);
    }
  }

  const byQid = {};
  for (const row of rows) {
    const qid = row.scenario.qid;
    const bucket = (byQid[qid] ||= { n: 0, correct: 0, conf: [], confCorrect: [],
                                     gold: {}, pred: {}, confusion: {} });
    bucket.n++;
    const right = row.choice === row.scenario.gold;
    if (right) bucket.correct++;
    bucket.conf.push(row.confidence || 0);
    if (right) bucket.confCorrect.push(row.confidence || 0);
    bucket.gold[row.scenario.gold] = (bucket.gold[row.scenario.gold] || 0) + 1;
    bucket.pred[row.choice] = (bucket.pred[row.choice] || 0) + 1;
    const c = (bucket.confusion[row.scenario.gold] ||= {});
    c[row.choice] = (c[row.choice] || 0) + 1;
  }

  const overall = rows.filter(r => r.choice === r.scenario.gold).length;
  console.log(`\n=== warehouse oracle — model=${MODEL} ===`);
  console.log(`accuracy vs geometry: ${overall}/${rows.length} ` +
    `(${(100 * overall / rows.length).toFixed(1)}%)\n`);

  for (const [qid, b] of Object.entries(byQid)) {
    console.log(`${qid}: ${b.correct}/${b.n} (${(100 * b.correct / b.n).toFixed(1)}%)`);
    console.log(`  gold distribution     : ${JSON.stringify(b.gold)}`);
    console.log(`  predicted distribution: ${JSON.stringify(b.pred)}`);
    const preds = [...new Set(Object.values(b.confusion).flatMap(r => Object.keys(r)))].sort();
    console.log(`  confusion (rows=gold):`);
    for (const g of Object.keys(b.confusion).sort()) {
      const cells = preds.map(p => `${p}=${b.confusion[g][p] || 0}`).join("  ");
      console.log(`    ${g.padEnd(6)} ${cells}`);
    }
    const p = percentiles(b.conf);
    const pc = percentiles(b.confCorrect);
    console.log(`  confidence p10/p50/p90: ${p.p10.toFixed(4)} / ${p.p50.toFixed(4)} / ${p.p90.toFixed(4)}`);
    console.log(`  confident+correct p10  : ${pc ? pc.p10.toFixed(4) : "n/a"}`);
    console.log();
  }

  console.log("A single populated predicted-distribution key means the head has");
  console.log("collapsed again (the bug was right=171/178). Set CONF_FLOOR in");
  console.log("warehouse.html from the confidence p10 above, for THIS model.");
}

main();
