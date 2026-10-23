// node --test test_traffic_sim.mjs
//
// The traffic simulator's shipped behaviour. Everything here runs against the module the
// browser loads — no DOM, no browser, no network, no model — which is what makes these
// real assertions about the page rather than about a copy of it.
//
// The load-bearing tests are the grammar ones. A demo whose "PII" packets do not actually
// match the scanner's patterns would report false negatives that are really badly written
// samples, and it would look like the detector failing.

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ATTACK_KINDS,
  BENIGN_KINDS,
  CLIENT_IDS,
  EDGES,
  MAX_INFLIGHT_PACKETS,
  NODES,
  RATE_FLOOD_THRESHOLD,
  SCAN_QUEUE_CAPACITY,
  SWEEP_PORT_THRESHOLD,
  TRAFFIC_KINDS,
  W,
  H,
  createSim,
  counters,
  mulberry32,
  packetText,
  pacing,
  pathPoints,
  payloadFor,
  pointAt,
  sampleScenario,
  spawn,
  step,
  structuralVerdict,
  buildScanRequest,
} from "./static/traffic_sim.js";

// ---- the grammar shapes, mirrored from guard_scan.py ------------------------------------
//
// These MUST mirror the Python patterns. They are duplicated rather than imported because
// the scanner is Python and the sim is JS; the duplication is pinned by the tests below,
// and a drift shows up as a failing test rather than as a silent demo failure.

const SCANNER_SHAPES = {
  email: /\b[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9.-]{1,255}\.[A-Za-z]{2,63}\b/,
  ssn: /\b\d{3}-\d{2}-\d{4}\b/,
  card: /\b(?:\d[ -]*?){13,19}\b/,
  phone: /(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3}-\d{3}-\d{4})(?![\d.])/,
  openai_proj: /\bsk-proj-[A-Za-z0-9_-]{20,}/,
  github: /\bgh[pousr]_[A-Za-z0-9]{16,}/,
  aws: /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/,
  // JS has no inline `(?i)` — the Python pattern carries a case-insensitive flag, which
  // becomes the `i` modifier here.
  assigned: /(?<![A-Za-z0-9])(?:[A-Za-z0-9]+[ _-]){0,2}(?:api[ _-]?key|access[ _-]?key|secret|password|passwd|token|credential)\s*[:=]\s*["']?[A-Za-z0-9._+/=-]{16,}/i,
};

const PII_SHAPES = [SCANNER_SHAPES.email, SCANNER_SHAPES.ssn, SCANNER_SHAPES.card, SCANNER_SHAPES.phone];
const SECRET_SHAPES = [
  SCANNER_SHAPES.openai_proj, SCANNER_SHAPES.github, SCANNER_SHAPES.aws, SCANNER_SHAPES.assigned,
];

// ---- topology ---------------------------------------------------------------------------

test("every edge endpoint is a node", () => {
  for (const edge of EDGES) {
    assert.ok(NODES[edge.from], `edge from ${edge.from} names no node`);
    assert.ok(NODES[edge.to], `edge to ${edge.to} names no node`);
  }
});

test("no source reaches the server except through the guard", () => {
  // The page's whole claim is "traffic is inspected at the guard". That is only true if
  // every path actually goes through it, so it is asserted rather than assumed.
  //
  // Proximity, not equality: the lane offset shifts a waypoint perpendicular to the leg,
  // so a waypoint is NEAR the edge node rather than exactly on it.
  const near = (point, node) => Math.hypot(point.x - node.x, point.y - node.y) <= 12;
  const inbound = EDGES.filter(e => e.to === "edge").map(e => e.from);
  for (const sourceId of inbound) {
    const path = pathPoints(sourceId, "guard");
    assert.ok(path.length >= 3, `${sourceId} -> guard is not routed via the edge router`);
    assert.ok(path.some(p => near(p, NODES.edge)), `${sourceId} -> guard skips the edge router`);
    assert.ok(near(path[path.length - 1], NODES.guard), `${sourceId} -> guard does not end there`);
  }
  assert.ok(inbound.includes("a1") && inbound.includes("c1"));
  for (const destination of ["srv", "drop"]) {
    const path = pathPoints("guard", destination);
    assert.ok(near(path[0], NODES.guard), `guard -> ${destination} does not start at the guard`);
    assert.ok(near(path[path.length - 1], NODES[destination]),
      `guard -> ${destination} does not end there`);
  }
});

test("every sampled position stays inside the canvas", () => {
  for (const edge of EDGES) {
    const path = pathPoints(edge.from, edge.to, 1);
    for (let i = 0; i <= 20; i++) {
      const p = pointAt(path, i / 20);
      assert.ok(p.x >= 0 && p.x <= W, `${edge.from}->${edge.to} x=${p.x} outside canvas`);
      assert.ok(p.y >= 0 && p.y <= H, `${edge.from}->${edge.to} y=${p.y} outside canvas`);
    }
  }
});

test("a path is arc-length parameterised, so the packet does not jump on a bend", () => {
  const path = pathPoints("c1", "guard");
  let previous = pointAt(path, 0);
  const total = Math.hypot(
    ...["x", "y"].map((k, i) => (path[path.length - 1][k] - path[0][k]) * (i === 0 ? 1 : 1)),
  );
  void total;
  // Walk in even steps; the distance moved between steps should be roughly constant
  // regardless of which segment the step landed in.
  const moves = [];
  for (let i = 1; i <= 30; i++) {
    const p = pointAt(path, i / 30);
    moves.push(Math.hypot(p.x - previous.x, p.y - previous.y));
    previous = p;
  }
  const max = Math.max(...moves);
  const min = Math.min(...moves);
  assert.ok(max - min < max * 0.05,
    `step distances vary too much (${min.toFixed(2)}..${max.toFixed(2)}), so t is per-segment`);
});

// ---- determinism ------------------------------------------------------------------------

test("the same seed replays the same stream", () => {
  const run = () => {
    const sim = createSim({ seed: 7 });
    const kinds = [];
    for (let i = 0; i < 400; i++) { step(sim, 16); }
    for (const p of sim.packets) kinds.push(`${p.kind}:${p.text}`);
    return kinds.join("|");
  };
  assert.equal(run(), run());
});

test("different seeds diverge", () => {
  const run = (seed) => {
    const sim = createSim({ seed });
    for (let i = 0; i < 400; i++) step(sim, 16);
    return sim.packets.map(p => p.text).join("|");
  };
  assert.notEqual(run(1), run(2));
});

// ---- the grammars ------------------------------------------------------------------------

test("every traffic kind produces a non-empty payload", () => {
  const rng = mulberry32(5);
  const kinds = [...BENIGN_KINDS, ...ATTACK_KINDS];
  for (const kind of kinds) {
    for (let i = 0; i < 40; i++) {
      const text = payloadFor(rng, kind);
      assert.ok(typeof text === "string" && text.length > 0, `${kind} produced ${text}`);
    }
  }
});

test("a pii_exfil packet always contains something the scanner matches", () => {
  // The test that stops the demo shipping a "PII" packet no detector fires on. 300
  // samples, every one of which must match at least one of the scanner's PII shapes.
  const rng = mulberry32(11);
  for (let i = 0; i < 300; i++) {
    const text = payloadFor(rng, TRAFFIC_KINDS.PII_EXFIL);
    const hit = PII_SHAPES.some(shape => shape.test(text));
    assert.ok(hit, `pii_exfil sample matched no scanner shape: ${JSON.stringify(text)}`);
  }
});

test("a key_leak packet always matches a provider prefix", () => {
  const rng = mulberry32(13);
  for (let i = 0; i < 300; i++) {
    const text = payloadFor(rng, TRAFFIC_KINDS.KEY_LEAK);
    const hit = SECRET_SHAPES.some(shape => shape.test(text));
    assert.ok(hit, `key_leak sample matched no provider shape: ${JSON.stringify(text)}`);
  }
});

test("a card in a pii_exfil packet is Luhn-valid, so the scanner does not skip it", () => {
  const luhn = digits => {
    let total = 0;
    [...digits].reverse().forEach((c, index) => {
      let v = Number(c);
      if (index % 2 === 1) { v *= 2; if (v > 9) v -= 9; }
      total += v;
    });
    return total % 10 === 0;
  };
  const rng = mulberry32(17);
  let sawCard = false;
  for (let i = 0; i < 400; i++) {
    const text = payloadFor(rng, TRAFFIC_KINDS.PII_EXFIL);
    const match = text.match(/\b(?:\d[ -]*?){13,19}\b/);
    if (!match) continue;
    sawCard = true;
    const digits = match[0].replace(/\D/g, "");
    assert.ok(digits.length >= 13 && digits.length <= 19, `card length ${digits.length}`);
    assert.ok(luhn(digits), `card ${digits} fails Luhn, so the scanner would skip it`);
  }
  assert.ok(sawCard, "no card-shaped sample was produced in 400 draws");
});

test("a flood packet carries nothing the content layer could block on", () => {
  // A SYN flood is not stopped by a text detector, and the demo must not be able to
  // claim otherwise by accident. So the payload must contain no PII and no secret.
  const rng = mulberry32(19);
  for (let i = 0; i < 200; i++) {
    const text = payloadFor(rng, TRAFFIC_KINDS.FLOOD);
    for (const shape of [...PII_SHAPES, ...SECRET_SHAPES]) {
      assert.ok(!shape.test(text),
        `flood payload matched a detector shape, so the page would overclaim: ${text}`);
    }
  }
});

test("a benign packet carries nothing the content layer would block on", () => {
  // The false-positive canary. If a benign kind starts matching a shape, the demo shows
  // the guard blocking ordinary traffic and the reason is the sample, not the model.
  const rng = mulberry32(23);
  for (const kind of BENIGN_KINDS) {
    for (let i = 0; i < 200; i++) {
      const text = payloadFor(rng, kind);
      for (const shape of [...PII_SHAPES, ...SECRET_SHAPES]) {
        assert.ok(!shape.test(text), `benign ${kind} matched a detector shape: ${text}`);
      }
    }
  }
});

test("sampleScenario only ever returns a declared kind", () => {
  const rng = mulberry32(29);
  const kinds = new Set([...BENIGN_KINDS, ...ATTACK_KINDS]);
  for (let i = 0; i < 500; i++) {
    assert.ok(kinds.has(sampleScenario(rng).kind));
  }
});

// ---- the structural detector (pure, numbers in) -------------------------------------------

test("a burst at the threshold is a flood, and one below it is not", () => {
  assert.equal(structuralVerdict({ packetsFromSource: RATE_FLOOD_THRESHOLD }).rule, "FloodRate");
  assert.equal(structuralVerdict({ packetsFromSource: RATE_FLOOD_THRESHOLD - 1 }).blocked, false);
});

test("the score is the ratio to the threshold, clamped to 1", () => {
  const at = structuralVerdict({ packetsFromSource: RATE_FLOOD_THRESHOLD });
  assert.equal(at.score, 1);
  const way = structuralVerdict({ packetsFromSource: RATE_FLOOD_THRESHOLD * 50 });
  assert.equal(way.score, 1, "score must clamp, or the bar renderer overflows");
});

test("a port sweep trips on DISTINCT ports", () => {
  assert.equal(structuralVerdict({ distinctPorts: SWEEP_PORT_THRESHOLD }).rule, "PortSweep");
  assert.equal(structuralVerdict({ distinctPorts: SWEEP_PORT_THRESHOLD - 1 }).blocked, false);
});

test("a sweep is reported as a sweep even when it is also a flood", () => {
  // Reaching 8 distinct ports means sending at least 8 packets, so a sweep is nearly
  // always a burst too. Checking rate first would label every sweep a flood, the PortSweep
  // axiom would never fire, and the page would claim only one of the two attacks it offers.
  const swept = structuralVerdict({ packetsFromSource: 40, distinctPorts: 12 });
  assert.equal(swept.rule, "PortSweep", "a sweep was reported as a flood");
  assert.match(swept.detail, /40 packets/, "the flood part of it went unmentioned");
});

test("a pure burst with one destination port is still a flood", () => {
  const flooded = structuralVerdict({ packetsFromSource: RATE_FLOOD_THRESHOLD, distinctPorts: 1 });
  assert.equal(flooded.rule, "FloodRate");
});

test("a burst of port_scan packets actually reaches the sweep rule", () => {
  // The rule being reachable THROUGH spawn, not just correct in isolation. An earlier
  // version recorded a fixed port for every packet, so `structuralVerdict` was right and
  // the sweep never fired in practice — the demo could not show its own scenario, and no
  // test noticed because they all drove the predicate directly.
  const sim = createSim({ seed: 71 });
  for (let i = 0; i < SWEEP_PORT_THRESHOLD + 4; i++) spawn(sim, TRAFFIC_KINDS.PORT_SCAN, "a1");
  const swept = sim.packets.filter(p => p.structural.rule === "PortSweep");
  assert.ok(swept.length > 0, "a burst of port_scan packets tripped no sweep");
  const ports = new Set(sim.packets.map(p => p.dstPort));
  assert.ok(ports.size > 1, `every port_scan packet used the same port (${[...ports]})`);
});

test("a burst of flood packets actually reaches the rate rule", () => {
  const sim = createSim({ seed: 73 });
  for (let i = 0; i < RATE_FLOOD_THRESHOLD + 4; i++) spawn(sim, TRAFFIC_KINDS.FLOOD, "a1");
  const flooded = sim.packets.filter(p => p.structural.rule === "FloodRate");
  assert.ok(flooded.length > 0, "a burst of flood packets tripped no rate rule");
});

test("a burst of benign packets below the rate limit trips neither rule", () => {
  // The reachability check from the other side: if the thresholds were reachable by
  // ordinary traffic, the demo would block clean requests.
  //
  // Note the bound. At the threshold or above, a burst from one source IS a flood by
  // definition — the rule is about rate, not intent — so asserting "18 benign packets are
  // never flagged" would be asserting that the rate rule does not work.
  const sim = createSim({ seed: 79 });
  for (let i = 0; i < RATE_FLOOD_THRESHOLD - 1; i++) {
    spawn(sim, TRAFFIC_KINDS.ORDER_LOOKUP, "c1");
  }
  const tripped = sim.packets.filter(p => p.structural.blocked);
  assert.equal(tripped.length, 0,
    `benign traffic below the limit tripped the structural layer: ` +
    `${tripped.map(p => p.structural.rule)}`);
});

test("the rate window is per source, so two quiet sources are not one flood", () => {
  const sim = createSim({ seed: 31 });
  // Half the threshold from each of two sources, in the same window.
  for (let i = 0; i < RATE_FLOOD_THRESHOLD - 1; i++) {
    spawn(sim, TRAFFIC_KINDS.HTTP_GET, "c1");
    spawn(sim, TRAFFIC_KINDS.HTTP_GET, "c2");
  }
  const c1 = sim.packets.filter(p => p.sourceId === "c1");
  const c2 = sim.packets.filter(p => p.sourceId === "c2");
  assert.ok(c1.every(p => !p.structural.blocked), "c1 was flooded by c2's traffic");
  assert.ok(c2.every(p => !p.structural.blocked), "c2 was flooded by c1's traffic");
});

test("the rate verdict clears once the burst ages out of the window", () => {
  const sim = createSim({ seed: 37 });
  for (let i = 0; i < RATE_FLOOD_THRESHOLD; i++) spawn(sim, TRAFFIC_KINDS.HTTP_GET, "c1");
  const blockedNow = sim.packets.filter(p => p.structural.blocked).length;
  assert.ok(blockedNow > 0, "the burst did not trip the rate window at all");
  // Advance past the window with no traffic from c1.
  for (let i = 0; i < 80; i++) step(sim, 16);
  const later = spawn(sim, TRAFFIC_KINDS.HTTP_GET, "c1");
  assert.equal(later.structural.blocked, false,
    "the rate window never cleared, so a source is permanently condemned");
});

test("a benign-only stream never trips the structural layer", () => {
  // The false-positive guard for the structural tier specifically.
  for (const seed of [1, 2, 3, 5, 7, 11, 42, 99]) {
    const sim = createSim({ seed, config: { packetsPerSecond: 60 } });
    for (let i = 0; i < 900; i++) {
      step(sim, 16);
      for (const p of sim.packets) {
        if (!BENIGN_KINDS.includes(p.kind)) continue;
      }
    }
    const tripped = sim.packets.filter(
      p => BENIGN_KINDS.includes(p.kind) && p.structural.blocked,
    );
    assert.equal(tripped.length, 0,
      `seed ${seed}: ${tripped.length} benign packet(s) flagged structurally`);
  }
});

// ---- movement and pacing -------------------------------------------------------------------

test("arrivals are reported once per packet", () => {
  const sim = createSim({ seed: 41 });
  const seen = new Set();
  for (let i = 0; i < 900; i++) {
    step(sim, 16);
    for (const arrival of sim.arrivals) {
      assert.ok(!seen.has(arrival.id), `packet ${arrival.id} arrived twice`);
      seen.add(arrival.id);
    }
  }
  assert.ok(seen.size > 0, "no packet ever reached the guard");
});

test("a structurally blocked packet is decided at spawn and never queued", () => {
  // The interlock. A flood must cost the content tier nothing, because it is stopped
  // before any payload is read.
  const sim = createSim({ seed: 43 });
  let blocked = 0;
  for (let i = 0; i < 120; i++) {
    const packet = spawn(sim, TRAFFIC_KINDS.FLOOD, "a1");
    if (packet && packet.structural.blocked) blocked++;
  }
  assert.ok(blocked > 0, "a burst of 120 flood packets tripped nothing");
  assert.equal(sim.queue.length, 0, "structural blocks put something on the scan queue");
  for (let i = 0; i < 300; i++) step(sim, 16);
  assert.equal(sim.queue.length, 0);
  assert.equal(counters(sim).dropped, blocked, "a blocked packet was not counted as dropped");
});

test("a structurally blocked packet is still REPORTED", () => {
  // The counters and the screen have to agree. An earlier version counted a blocked packet
  // as dropped but never surfaced it, so a flood's entire result lived in the HUD and not
  // in the ledger — the page could not show the thing it exists to show.
  const sim = createSim({ seed: 43 });
  for (let i = 0; i < RATE_FLOOD_THRESHOLD + 6; i++) spawn(sim, TRAFFIC_KINDS.FLOOD, "a1");
  let reported = 0;
  for (let i = 0; i < 300; i++) {
    step(sim, 16);
    for (const packet of sim.structuralArrivals) {
      assert.ok(packet.response, "a reported structural packet carries no response");
      assert.ok(packet.structural.blocked, "a structural arrival was not actually blocked");
      reported++;
    }
  }
  assert.ok(reported > 0, "a blocked flood produced no reported arrivals");
  assert.equal(reported, counters(sim).dropped,
    `${reported} packets were reported but ${counters(sim).dropped} were counted`);
});

test("a structurally blocked packet is never routed to the server", () => {
  const sim = createSim({ seed: 43 });
  for (let i = 0; i < RATE_FLOOD_THRESHOLD + 6; i++) spawn(sim, TRAFFIC_KINDS.FLOOD, "a1");
  for (let i = 0; i < 300; i++) {
    step(sim, 16);
    for (const packet of sim.structuralArrivals) {
      assert.equal(packet.destination, "drop",
        "a structurally blocked packet was routed to the server");
    }
  }
});

test("the in-flight population stays bounded under a flood", () => {
  const sim = createSim({ seed: 47, config: { packetsPerSecond: 200 } });
  for (let i = 0; i < 2000; i++) {
    step(sim, 16);
    assert.ok(sim.packets.length <= MAX_INFLIGHT_PACKETS,
      `${sim.packets.length} packets in flight exceeds the cap`);
  }
});

test("counters add up to what was spawned", () => {
  for (const seed of [1, 3, 7, 13]) {
    const sim = createSim({ seed, config: { packetsPerSecond: 30 } });
    for (let i = 0; i < 600; i++) step(sim, 16);
    const c = counters(sim);
    const accounted = c.inspected + c.unavailable;
    assert.ok(accounted <= c.spawned,
      `seed ${seed}: counted ${accounted} outcomes against ${c.spawned} spawns`);
  }
});

test("pacing reports how much of the stream was actually inspected", () => {
  const sim = createSim({ seed: 53, config: { packetsPerSecond: 30 } });
  for (let i = 0; i < 600; i++) step(sim, 16);
  const p = pacing(sim);
  assert.equal(p.seen, counters(sim).spawned);
  assert.ok(p.ratio >= 0 && p.ratio <= 1, `ratio out of range: ${p.ratio}`);
});

// ---- the scan request contract --------------------------------------------------------------

test("buildScanRequest omits the path when there is none", () => {
  const sim = createSim({ seed: 59 });
  const packet = spawn(sim, TRAFFIC_KINDS.PII_EXFIL, "c1");
  const withPath = buildScanRequest(packet, "ciel_typed_v1", "typed");
  const without = buildScanRequest(packet, "english", null);
  assert.deepEqual(without, { message: packetText(packet), model: "english" });
  assert.deepEqual(withPath, { message: packetText(packet), model: "ciel_typed_v1", path: "typed" });
});

test("the text scanned is the text the ledger shows", () => {
  // Frozen onto the packet at spawn, so the two cannot disagree.
  const sim = createSim({ seed: 61 });
  const packet = spawn(sim, TRAFFIC_KINDS.KEY_LEAK, "c1");
  assert.equal(packetText(packet), packet.text);
  assert.equal(buildScanRequest(packet, "english").message, packet.text);
});
