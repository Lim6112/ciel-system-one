# laya-example

Demo of [laya](https://pypi.org/project/laya/): fast, non-autoregressive routing of
typed questions (choice / score / yes-no) over support-ticket-style state, using
pre-trained ModernBERT decision checkpoints. English input routes to the
`english` checkpoint, Hindi (and other languages) to `multilingual`, with an
explicit override for `typed-decisions`.

## Quickstart

```sh
uv sync
uv run python main.py
```

The three checkpoints (`english`, `multilingual`, `typed-decisions`) are
downloaded from Hugging Face on first use and cached in `~/.cache/huggingface`.

### Intel Macs (x86_64)

`uv sync` handles both architectures from one lockfile — the Intel-only caps in
`pyproject.toml` are marked on `platform_machine`, and both branches are declared
per package so `uv` forks rather than resolving a single version that satisfies
everything:

| | Intel (x86_64) | Apple Silicon (arm64) |
|---|---|---|
| torch | 2.2.2 | 2.11.0 |
| numpy | 1.26.4 | 2.4.6 |
| transformers | 4.57.6 | 5.18.0 |

Intel is capped because PyTorch stopped publishing macOS x86_64 wheels after
2.2.2, and that version forces two caps downstream: it is built against the
numpy 1.x C ABI (numpy 2.x fails with `_ARRAY_API not found`), and transformers 5.x
requires torch>=2.5, so on 2.2.2 it disables PyTorch and leaves `nn` undefined —
the `NameError: name 'nn' is not defined` that greets a fresh Intel checkout.

Two consequences worth knowing:

- **Python is 3.11 on both Macs** (`requires-python = ">=3.11,<3.12"`).
  transformers' ModernBERT module uses `@torch.compile`, and torch 2.2.2's Dynamo
  refuses Python 3.12+. The upper bound is global, not per-arch, so the Silicon
  Mac is held to 3.11 too even though torch 2.11 supports 3.12+. Per-arch lockfiles
  would recover that.
- **Intel runs on CPU.** MPS autocast needs torch>=2.4, so `laya_startup.py`'s
  `_default_device()` probes `torch.autocast` rather than asking
  `torch.backends.mps.is_available()` — on Intel those two disagree, the first
  says yes while the second raises. Expect ~440 ms per prediction rather than
  the ~35 ms an MPS machine gets.

Verified on Intel: `main.py` runs both checkpoints, 19 Python and 45 Node tests
pass. The arm64 column is from resolution, not execution. (The Node suite has since
grown — `node --test test_breakout_sim.mjs test_breakout_ontology.mjs test_ontology.mjs
test_fleet_sim.mjs test_penguin_sim.mjs test_penguin_commands.mjs` now runs 178
tests, all green, and `uv run pytest` runs 82.)

## Web UI

A browser UI for the same state / questions → result flow, backed by a
long-lived FastAPI server (the ~7s model preload happens once at startup).
The UI is branded **CJet**; the underlying decision engine is the laya library:

```sh
uv run uvicorn server:app
```

Then open http://localhost:8000. Pick one of twenty built-in examples from the
domain-grouped dropdown (support tickets in English/Korean, plus laya's
pre-tuned preset workflows — email triage, prompt guardrails, content
moderation, model routing — with one benign and one flagged example each) to
autofill both the state and the question cards, or enter the state as JSON and
build questions with the form (type `choice` / `score` / `noul`, instructions,
and per-option criteria rows). Optionally override the model checkpoint and
run: results render per question with probability bars and the routing
metadata (model, repo, reason, latency).

- `server.py` — FastAPI app; `POST /api/predict` validates the payload against
  laya's question schema with pydantic (choice → dict criteria, score → list,
  noul → none) and serves `static/index.html`
- `laya_startup.py` — shared startup used by both `main.py` and the server
  (offline hub + fast-build monkeypatch + preload; see below)
- `test_server.py` — stubbed-API tests, no model load:
  `uv run pytest test_server.py`
- `test_fleet_sim.mjs` — fleet simulation invariants, no model load:
  `node --test test_fleet_sim.mjs`
- `test_breakout_sim.mjs` / `test_breakout_ontology.mjs` — breakout invariants and the
  ontology's falsifiability tests: `node --test test_breakout_sim.mjs test_breakout_ontology.mjs`
- `test_penguin_sim.mjs` / `test_penguin_commands.mjs` — the penguin's world rules and the
  transcript→command mapping, no model load:
  `node --test test_penguin_sim.mjs test_penguin_commands.mjs`
- `asr.py` — local speech recognition (Moonshine tiny) for the penguin demo; `test_asr.py`
  covers WAV decoding and the offline-flag handling, no model load:
  `uv run pytest test_asr.py`

### laya plays Breakout

`static/breakout.html` (linked from the UI header) is an Atari-Breakout-style
canvas game where laya controls the paddle. laya reads text, not pixels: every
decision tick serializes the game state into a short text observation
(ball position relative to the paddle, direction, estimated time to paddle
level, bricks remaining) and asks one typed question —
`move: choice(left/stay/right)`. The gameplay screen shows the canvas, the
exact state text laya receives ("Laya's view"), and the decision with
probability bars, confidence and latency ("Laya's decision"). Toggle autoplay
off to play by arrow keys and compare; sliders tune the decision interval and
ball speed, and the model override selects the checkpoint.

It also carries an **Ontology** panel — the same TBox editor and ABox view as the
fleet demo — plus an **Ontology: on/off** button in the Controls row. One switch, one
principle: when the ontology is on it judges **and its verdicts control** (question
suppression, the stale-command re-aim, and the model's own observation regime); when it
is off it controls nothing. Declared rules: `Ball ⊑ ≥1 catchableBy` (a ball the physics
cannot recover is a fact rather than a mistake) and `Ball ⊑ ≥1 predictedPosition` (a
ball that will reach paddle height has a predictable landing point, wall reflections
priced in).

**Injecting that knowledge measurably improves performance.** With the ontology on, the
model's input is rewritten around the prediction — the side of the paddle is named from
the wall-folded *landing* point (0% wrong side, versus 12.4% for the naive
current-position observation) and the two absolute coordinates the gap was computed from
are stated. Closed-loop A/B against the real checkpoint, 60 s × 3 seeds × 2 batches:
**+8.3 score in both batches** (14.0 → 22.3 and 15.0 → 23.3), balls lost after an ask
2.7/3.0 → 1.0/1.3, provably lost 0 everywhere, zero extra model calls. Details and
honest decomposition in
[The same idea in Breakout](#the-same-idea-in-breakout--and-where-the-fleet-invariant-does-not-carry-over).

### laya runs the warehouse

`static/warehouse.html` (linked from the UI header) simulates a cargo house
and three delivery robots on crossing routes, each with a forward lidar. When a
robot's lidar sees its path blocked, it serializes the situation to **a flat JSON
state** and asks laya a question about the blocker. The screen shows the canvas
with lidar cones and route lines, per-robot lidar chips, "Laya's view" (the exact
JSON the blocked robot sent) and "Laya's decisions" (action bars, confidence,
latency, history, and a `decided · deferred` counter for the geometry gate). Hard
collision prevention is independent of laya — physics never lets robots overlap;
laya decides strategy. Toggle autoplay off to watch blocked robots stop forever:
the deadlock laya prevents.

**There are two blockers and they are two different problems**, so the demo asks
two different questions. A **storage rack** is static and never yields — waiting
is *provably* wrong, so `stay` is not offered at all and the question is a 2-way
`sidestep_direction` whose gold is the side with more room. Another **robot** can
yield, so it gets the 3-way `avoid_direction`, where `stay` means *I give way* (a
priority rule), not "the obstacle will clear". Measured over the recorded traffic,
62% of blocks are racks (110/178) — so most of this demo is static avoidance, not
multi-agent deadlock.

The sim lives in `static/warehouse_sim.js` — a pure, seeded, DOM-free module.
`warehouse.html` is only the DOM layer over it, and `eval_warehouse.mjs` imports
the same file, so the browser and the harness cannot drift apart.

#### The doc head answers this domain by surface form, not geometry

`model=ciel_decision_model_v1` (the doc/breakout head) collapsed to a single
action on warehouse states: **171 of 178** recorded decisions were `right`,
reproduced live at 8/8, and it stayed `right` with the option order reversed and
with the obstacle flipped from "to my left" to "to my right" — so it was neither a
tie-break nor reading the geometry. The cause is coverage, not tuning: neither
training JSONL contains a single occurrence of "warehouse", "storage rack",
"dock", "lidar", "aisle" or "blocked", and the head's only spatial-choice
training is Breakout paddle control — whose gold rule is the **opposite sign**
(move *toward* the ball; sidestep *away from* the blocker).

Prose-tuning could not fix it, which is worth recording because the two earlier
Breakout fixes (`8fc3dd5`, `ebfc76c`) were exactly that and did work — there the
prose *was* the head's training domain. Here, swapping the noun, changing the
letter case, or appending one field each flipped the answer; sweeping the gap
from 1 to 9999 px changed nothing, because a bge embedding is nearly insensitive
to the value of a number. The fix is a trained head, not a better sentence.

#### `ciel_warehouse_v1` — the warehouse head

`model=ciel_warehouse_v1` selects a head trained on this domain as its own
dynamic-head sub-path (`warehouse`) beside `doc` and `typed`, off the same bge-m3
backbone. Its corpus is synthetic and template-rendered — `states/warehouse.py` in
the decision_lab checkout, alongside `states/breakout.py` — so the gold is exact by
construction and no LLM is involved. The rules it follows are Breakout's: gold
comes from geometry, distractors are decorrelated from the gold, and templates with
no direction words at all exist to force a geometry→direction mapping rather than a
lexical shortcut.

The state is **JSON now, not prose**, and that is the point: the corpus renders the
same flat JSON the browser sends, so training and inference share one format. Both
prior incidents in this project were train-here/infer-there mismatches, so the
format is the contract rather than something to keep tuning. Two constraints follow
and are pinned by `test_warehouse_sim.mjs`:

- **Flat and scalar only.** `server.py` `str()`s each top-level value, so a nested
  object would go upstream as a Python `repr` the head has never seen.
- **At most 15 leaves.** The head prepends the whole text as field 0, and the
  chunker merges leaves only above `max_fields` — so staying under it keeps the
  decisive leaf (`obstacle_side`, or `gap_left`/`gap_right` for a rack) standing
  alone instead of blended into a neighbour's text.

#### The geometry gate, and the escalation trigger

Two safety layers sit outside the model, and neither pretends the model is in the
loop when it is not:

- **The gate.** The geometrically feasible side set is computed before asking, and
  an answer outside it is deferred to geometry *at any confidence* — a confident
  impossible answer is still impossible. Below `CONF_FLOOR` the model is deferred
  too, and the `decided · deferred` counter shows how often that happened.

  `CONF_FLOOR` is **0.9**, from the warehouse head's measured closed-loop p10 (0.995)
  rather than from a guess — and it is a *single* constant serving a dropdown that
  can select models an order of magnitude below it (the doc head's p50 is 0.57, the
  laya arms ~0.03). That is deliberate: those models answer this domain at chance or
  worse, so deferring them to geometry is the outcome the gate exists for. A floor
  reused across models does not mis-set itself, it silently disables itself, which is
  why the number is derived per model and the counter is on screen.
- **Escalation is now a stall timer, not a blocked timer.** The old trigger (5 s
  continuously blocked) could not fire: across the 178 recorded calls the longest
  block was **3 s**, so the 2-way branch was unreachable and `stay` was always on
  the menu. `STALL_MS` keys on *progress staleness* instead — no forward movement —
  which is the condition the demo actually cares about and holds for rack blocks too.

Re-derive the oracle numbers with `node eval_warehouse.mjs --model ciel_warehouse_v1`;
that per-side accuracy and confusion table is the evidence for the head, and the
`--model ciel_decision_model_v1` run is the before picture.

#### A rule that reads well and measures badly

The dock is where contention actually is — **54.4% of all blocked ticks** in a 60 s
run are spent inside a 90 px radius of the cargo house (185,889 of 341,911 across
seven seeds) — and "a docking station serves one vehicle" is a real domain
constraint that states cleanly what the fleet demo's seat limit states. So
`warehouse_sim.js` implements it: Tier 0, occupancy relative (whoever is inside
keeps it, everyone else heading in holds), no priority function needed because only
one robot can be the incumbent.

**It measured as a net loss and ships off.** Seven to nine seeds, 60 s each, the
geometry oracle as the policy, the rule as the only difference:

| | mean deliveries |
| --- | --- |
| rule off | **93.6** and **91.1** (two consecutive runs) |
| rule on | 55.0 and 48.0 |

Roughly 40% of throughput, at every radius swept (20/30/45/60/90 px; best case 46.1
at 45 px). It also failed to hold its own invariant — 1,800–2,800 ticks of two
occupants — because the hold parks a robot at the zone edge, where it still counts
as outside by its own test and still blocks the apron. Serialising the apron costs
more than the overlap it prevents: the zone is a bottleneck precisely because the
routes cross in it, so discarding the parallel crossing hurts more than the overlap
helps.

This is recorded rather than deleted because the negative result is the useful part,
and because a later attempt should not start from zero. What it would take: a radius
small enough to exclude the crossing apron (or a zone shaped as a segment of the
final approach rather than a disc), a hold position genuinely outside the zone *and*
off the routes others use, and most likely a reservation model rather than a mutex —
the win would be ordering arrivals, not excluding them. All of that is a bigger
change than the rule, and none of it is measured yet.

`test_warehouse_sim.mjs` pins the negative result: one test asserts the rule is off
by default (so a silent flip cannot regress the demo with nothing to explain it),
another checks the implementation still holds one occupant when deliberately
enabled with `dockRule: true`.

**Measured, closed-loop** (`node eval_warehouse.mjs`, service on the warehouse
sub-path, 60 scenarios):

| arm | accuracy vs geometry | prediction distribution | confidence p10 / p50 |
| --- | --- | --- | --- |
| `ciel_warehouse_v1` | **60/60 (100%)** | left 30 · right 30 | 0.995 / 0.998 |
| `ciel_decision_model_v1` (before) | 20/40 (50%) | **left 40** — collapsed | 0.522 / 0.568 |

The before arm is worth reading carefully: under the JSON state it collapsed to
`left` rather than the `right` from the prose era, i.e. a different constant, still
uncorrelated with geometry, still exactly chance. That is the shape of an
out-of-domain answer, and it is why the fix had to be a trained head rather than
another round of prose.

**Measured, in-distribution** (`python -m decision_lab eval` on the warehouse
config's `test_warehouse` split, 200 subsampled states, `results/report.md`):

| question | per-label accuracy | confusion |
| --- | --- | --- |
| `avoid_direction` | left / right / stay **100%** each | clean diagonal (34 / 29 / 36) |
| `sidestep_direction` | left / right **100%** each | clean diagonal (83 / 81) |

Mean 65.8% is the average over both questions plus the derived predicates, not a
per-question figure — the two choice questions are exact. The doc/breakout arm was
re-measured on the same run as the regression gate: task in-dist 0.9067 against the
0.924 baseline and breakout paddle 0.965 against 0.976, both inside the repository's
stated ±2pp gate and its ~3pp seed variance.

The 3-way robot branch is not sampled by the closed-loop run (see the limitation
below), so it was probed directly against the service instead and is correct in all
three cases: blocker right → `left` (0.997), blocker left → `right` (0.985),
head-on → `stay` (1.000).

**In the browser**, with `ciel_warehouse_v1` selected: decisions vary across all three
actions at ~100% confidence, `blockedNow` repeatedly returns to 0, and deliveries
accumulate — the partial gridlock the demo used to stall in is gone. The gate defers
nothing on this head (324 decided · 0 deferred over a run), which is the expected
result of a floor below the head's p10.

**One limitation to know when reading the oracle.** The harness drives the sim, and
it does not feed answers back into it — so the scenarios it samples are the ones the
sim reaches *unopposed*, which under-samples genuinely deadlocked multi-robot blocks
(a robot that is never told to wait never stalls). In the one run recorded so far,
40/40 sampled scenarios were rack blocks and the 2-way question; the 3-way robot
branch is exercised by the demo, not yet by the harness. Feeding decisions back into
`eval_warehouse.mjs` is the obvious next step, and until then its accuracy is a
statement about the rack branch.


### laya dispatches the bus fleet

`static/fleet.html` (linked from the UI header) simulates an autonomous bus
fleet on a 50×50 grid: passengers appear with a pickup and a destination, and
every new demand is dispatched by laya —
`assign: choice(bus_a/bus_b/bus_c)`. Buses drive Manhattan-style through an
ordered list of stops, so a bus can be part-way through several demands at once.
The screen shows the grid with each bus's remaining route drawn as a dashed
polyline and an onboard-count badge, per-bus status chips (`Bus C · 3/4 · drop
grid (39, 39) (26) · +4 stops`), "Laya's view" (the dispatch state sent) and
"Laya's decisions" (assignment bars, confidence, latency, history). Toggle laya
dispatch off to compare against a greedy nearest-bus heuristic; sliders tune
demand frequency and bus speed.

The simulation, the domain rules and the question text live in
`static/fleet_sim.js` — a pure, seeded, DOM-free module. `fleet.html` is only
the DOM layer over it, and `eval_rules.mjs` imports the same file, so the
browser and the A/B harness cannot drift apart.

#### Domain rules: where they go, and why

The interesting question this demo answers is **where a domain rule belongs in
a System-1 model**. The rules here — a bus may carry several demands at once,
and a demand near a bus's current route is nearly free to add — are not in the
weights. laya does not execute rules, it reads text, so a bare assertion
("a bus may carry 2 demands") changes nothing. The rules are injected in four
places, each doing a different job:

| Tier | Where | What it carries |
| --- | --- | --- |
| 0 | code, not text | hard constraints — filter the option set so a rule is *unviolable* rather than *likely* |
| 1 | per-option `criteria` | the rule's derived numbers, one per bus |
| 2 | question `instructions` | the policy, constant across decisions |
| 3 | first key of the state | scenario-level rules |
| 4 | fine-tuning | last resort, gated on measurement |

Tier 1 is the one that matters, because of how the checkpoint is wired:
`build_sequence()` puts each option's criteria immediately before its `[MASK]`,
and the decision is the hidden state *at that mask*. The criteria are the
scorer's literal input.

The arithmetic lives in JS, not the model. `detourCost()` is the multi-pickup
rule made computable — the classic VRP insertion heuristic: fewest extra blocks
for a bus to also serve this demand, by splicing its pickup and drop-off into the
route it is already driving. `pickupDetour()` prices the pickup alone, because
the two answer different questions: a demand can have its pickup right on the
route and its destination well past the end, which makes the pickup free and the
demand as a whole not. Each option's criteria reports both:

```
bus_b: "Bus B: pickup on its route, +0 blocks, +14 total; 2 of 4 seats free"
```

A bus that has no room is **dropped from the options entirely** rather than
described as unavailable. That is Tier 0 doing its job: a rule the model cannot
see an option for is unviolable, not merely discouraged.

The state also carries a `rules:` line, and it is the **first** key: laya's
`build_sequence()` truncates the state from the right (`st = st[:room]`,
`truncate_left` is never set by `Agent.system_one`), so anything at the tail of
a long state is dropped silently. At the current fleet size that is not yet
binding — a mid-episode state is ~272 tokens against a ~459-token budget on the
`english` checkpoint — but it is free insurance as the state grows.

#### Measuring it

```sh
uv run uvicorn server:app            # terminal 1
node eval_rules.mjs --n 200 --seed 7 # terminal 2
```

The harness generates scenarios *counterfactually*: a sim is driven forward by
a fixed policy and every decision tick is snapshotted, then every arm answers
the *same* snapshots differing only in question text. That isolates the decision
instead of letting one arm's assignments steer the other arm's future inputs.

> **The no-rules baseline was re-measured and the old number was wrong.** The arm
> reported below used to strip only the `rules:` key, which left the rule in two other
> places: `recommendation` ended with a sentence built from `detourCost()` and
> `isOnRoute()` — that sentence *is* the rule — and `fleet` was `busText()`, which
> reports seats and multi-stop plans and so presupposes the world the rule describes.
> "No rules" was quietly a rules baseline. `buildBaselineState()` is a separate builder
> with none of that in it, and the clean baseline is roughly **twice as bad** (19.25 →
> 38.33 blocks). The rules arm is unchanged, so the apparent improvement grew — and that
> growth is not a new result, it is an old baseline being corrected.

Over 200 scenarios, 195 of which had a real nearest-vs-least-detour conflict:

```
                              no-rules           rules  rules-nofilter          greedy
mean detour regret           38.33 blk        6.43 blk        6.23 blk       34.60 blk
picks least-detour               39.0%           79.0%           83.6%           48.7%
picks nearest                    65.5%           55.0%           60.0%           76.5%
capacity violations                 51               0               4              48
mean confidence                  0.057           0.103           0.051           0.000
head tokens (est)                   48             137             137               —
ms                                 81             136             141               —
```

Two things to read here. The **injection works** — regret against the
least-detour bus drops ~6× and the model follows the criteria. And the
**capacity filter (Tier 0) is doing real work**: `rules` never assigns to a bus
that has no room, while the unfiltered arm does so four times and the naive
nearest-bus dispatcher forty-eight. Omitting an infeasible option makes the rule
unviolable rather than merely likely.

Note also how close `no-rules` sits to `greedy` (38.33 vs 34.60). Out of the box,
laya is barely better than "send it to the closest bus" on a domain it was never
trained on — which is the whole reason the injection is doing something.

`no-rules` and `greedy` show *negative* nearest-regret, which is not a bug: the
optimum is restricted to buses that can take the work, and the naive dispatcher
picks the closest bus whether or not it has room.

### laya listens to the penguin

`static/penguin.html` (linked from the UI header) is a penguin on an ice floe that
takes spoken commands. You say `go`, `back`, `turn left` or `turn right`; a local
speech recogniser transcribes it; the transcript becomes a command; the penguin
walks, turns, eats fish, and stops at open water and at the edge of the floe. It is
the only demo here with an audio input, and the only one whose input is *speech*
rather than text typed into a box.

The screen shows the floe with the penguin, the fish it is collecting, the open
water it must not walk into, "What the model is told" (the exact state text), and
"Which command did that mean?" (the four command bars, confidence, latency, and a
history). A **rules** reader and a **laya** reader both answer that question — see
below, because the difference between them is the point.

#### Speech recognition: Moonshine tiny, and why not Whisper

`asr.py` wraps [Moonshine](https://github.com/moonshine-ai/moonshine) tiny, which
matters for one unglamorous reason: **it is already installed**.
`transformers>=5` ships `MoonshineForConditionalGeneration`, so this adds a model
but not a single package — `pyproject.toml` is unchanged. Against Whisper tiny on
the same speech it is both faster and more accurate (34 ms vs 277 ms on an M-series
Mac; 12.00% vs 12.81% WER), which for a command answered out loud is the whole
ballgame. Both are ~27M parameters.

| Arm | Checkpoint | Size | Licence |
| --- | --- | --- | --- |
| English | `UsefulSensors/moonshine-tiny` | ~110 MB | MIT |
| Korean | `UsefulSensors/moonshine-tiny-ko` | ~110 MB | **not MIT** (`other`) |

The Korean checkpoint is a separate release with a non-MIT licence — noted here
rather than buried, because redistributing this demo downstream is a real scenario.
Weights download on first use (`uv run uvicorn server:app` will pause once) and are
cached by `transformers` afterwards.

There is **no audio dependency at all** — no ffmpeg, librosa, soundfile, torchaudio
or `av`. The browser captures raw Float32 through an `AudioWorklet`, resamples to
16 kHz mono with an `OfflineAudioContext`, and assembles a PCM WAV in JavaScript; the
server only ever sees that one format, which the stdlib `wave` module reads.
`MediaRecorder` is deliberately avoided — it returns webm/opus in Chrome and
something else again in Safari, and decoding either server-side would mean shipping
ffmpeg.

One hazard is worth recording. `laya_startup.py` sets `HF_HUB_OFFLINE=1` so laya never
reaches for the network mid-prediction, and that flag would also stop the first
Moonshine download dead. `asr.py` lifts it around the load and restores the original
value afterwards (`_hub_online`), rather than clearing it process-wide — the flag is
there for a reason and it is not this module's to change. `test_asr.py` pins that
window, including that the value is restored verbatim and after an exception.

#### The state laya is asked to route

The question is `command: choice(go/back/turn_left/turn_right)`, and the state is the
transcript plus the penguin's four neighbours as prose:

```json
{
  "spoken instruction": "turn left please",
  "penguin is facing": "north",
  "ahead of the penguin": "a fish",
  "to the left of the penguin": "open water",
  "to the right of the penguin": "open ice",
  "behind the penguin": "open ice",
  "steps remaining": "28",
  "fish remaining": "5"
}
```

`buildState` in `static/penguin_sim.js` is a *formatter over* `measure()`, the same
one-value-two-readers rule the other sims follow — so the sentence the model reads
and the value the tests assert over cannot drift. The neighbours are named by
`describeCell`, which keeps "open water" and "the edge of the floe" distinct; a test
asserts they stay distinct, since a blocked move is a different sentence depending on
which one stopped it.

#### Two readers, and what the difference shows

| Reader | What it is | When to use it |
| --- | --- | --- |
| `rules` (default) | keyword table in `static/penguin_commands.js` — no model, no network | always works, deterministic, replayable |
| `laya` | the transcript + the floe sent to `/api/predict` | to watch a model route the same sentence |

Measured on the four commands, the `laya` reader is good in English and bad in
Korean. Both are reported here as measured, not as designed:

| Transcript | Result | Winner probability |
| --- | --- | --- |
| `go forward` | `go` | 0.92 |
| `go back` | `back` | 0.80 |
| `turn left` | `turn_left` | 0.70 |
| `turn right` | `turn_right` | 0.88 |
| `앞으로` / `뒤로` | `go` (both) | 0.49 / 0.52 |
| `왼쪽으로 돌아` / `오른쪽으로 돌아` | `go` (both) | 0.55 / 0.53 |

Korean collapses to `go` — a confident wrong answer rather than an error. The cause
is upstream of the penguin: laya's language detector reports Korean text as
`script: latin`, so `auto` hands it to the *English* checkpoint, and every Korean
command comes back `go`. The `multilingual` checkpoint does not fix it either (it
answers `back` for all four, in both languages). The Korean **recognition** is fine —
`moonshine-tiny-ko` transcribes 앞으로, 왼쪽으로 돌아 and 오른쪽으로 돌아 correctly —
so the failure is entirely in the routing step. The page shows a warning when Korean
and a model reader are both selected, rather than hiding it or quietly disabling the
arm.

This is the same class of problem the breakout notes describe from the other side: a
head asked about prose outside its training distribution can answer confidently and
wrongly, and the honest response is to measure it, say so, and keep a deterministic
path available. It is also why `rules` is the default and the fallback — when laya is
unavailable the page falls back to it *loudly*, recording the failure in the history
instead of going quiet.

#### Running it

```bash
uv run uvicorn server:app          # first run downloads ~110 MB of Moonshine
# open http://localhost:8000/penguin.html
```

No microphone? Press <kbd>G</kbd>/<kbd>B</kbd>/<kbd>L</kbd>/<kbd>R</kbd> (or the arrow
keys), or use the four buttons. They drive the identical code path, so a scenario
replays exactly. To exercise the endpoint directly:

```bash
curl -s -X POST --data-binary @sample.wav \
  'http://localhost:8000/api/transcribe?lang=en' | python3 -m json.tool
```

The recogniser wants 16 kHz mono 16-bit PCM WAV and returns 400 for anything else
rather than guessing; `afconvert` or `ffmpeg -ar 16000 -ac 1` will produce one.

#### Limitations

- **Korean does not route through laya** (table above). Use `rules` for Korean.
- **Isolated single words transcribe worse than phrases.** Measured with macOS `say`:
  `turn left` as one word came back `"And they predict."`, while `"Please go back to
  the start."` came back near-correct and real recorded speech was exact. A 27M model
  on a half-second clip has very little to go on. Speaking a short phrase works
  noticeably better than barking one word.
- **Not streaming.** Recording is per-utterance — click, speak, click — with a 25 s
  ceiling. No wake word.
- **No ontology panel**, unlike breakout and fleet. The interesting axis here is
  speech→command, and an ontology would be a third thing competing for the screen.
- **The blocked step costs a step.** Deliberate — if walking into water were free,
  what is ahead would not matter and asking the model about it would be theatre.

#### ciel_decision_model_v1 — a local dynamic-head backend

`model: "ciel_decision_model_v1"` swaps in a dynamic decision head served over
HTTP by a local "decision_lab" service (`ciel.py` is the adapter, built the same
way as `jev.py`). The service answers caller-defined questions about a text state
in one forward pass — no retraining or restart for new question types — and needs
no API key and costs nothing.

```sh
# start the service (separate checkout, needs models/head_dynamic.pt)
uv run python -m decision_lab ui
curl -s http://127.0.0.1:8000/api/status      # {"ready":true} before use
```

The adapter URL is configurable: `DECISION_LAB_URL` defaults to
`http://127.0.0.1:8000`. Note the **port conflict**: `uv run uvicorn server:app`
also defaults to 8000, so run one of them elsewhere — e.g.
`uv run uvicorn server:app --port 8001`. The same variable is what the `/lab` relay
dials (below), via the same helper — so one setting moves both. Errors surface as HTTP 502
with the upstream body in `detail` (the service reports 503 while loading and 400 when
`models/head_dynamic.pt` is missing/untrained). State is sent as the bare field
values space-joined into one line, in insertion order — the head was trained on
prose, and "key: value"-style prefixes or newlines between fields measurably
flip correct answers to garbage (verified against the live service). This also
constrains the state prose itself: `breakout_sim.js`'s `buildState` phrases the
bricks line and the centered case (`offset N px`) the way the training prose
does, for the same reason. Choice options come from the criteria keys, score
levels from the criteria list.

#### Labeling that traffic — the service's own UI, embedded

`ciel.py` asks the service to log every call it makes (`X-Decision-Lab-Log: 1`,
`ciel.py:200`), and the service keeps those calls in `data/traffic/` when its own
`web.log_traffic` is on. Labeling them is what turns real traffic into training
data, and that labeling UI belongs to the service, not to this repo — so
`static/decision_lab.html` **embeds** it rather than reimplementing it:

```sh
cd decision-qwen-decision-head-casual-model && uv run python -m decision_lab ui   # :8000
uv run uvicorn server:app --port 8001                                            # this repo
# open http://localhost:8001/decision_lab.html — or follow the 🧪 link on the main UI
```

The page iframes the service and opens it on its **Label** view (`?tab=label`, which
the service's `app.js` honors). Nothing is copied, so the two frontends cannot drift.

**The frame is served same-origin, through this repo.** `server.py` relays `/lab` to
the service (`decision_lab_proxy.py`), and **Source** defaults to that relay. This is
the part that matters in practice: the service binds loopback and its port is
firewalled, so a browser anywhere else cannot reach `:8000` directly — and when this
app is served over HTTPS, framing a plain-HTTP origin is blocked outright regardless
of the firewall. Relaying means the browser only ever talks to this server.

The relay reads its destination from `ciel.base_url()`, the same reader `ciel.py`
uses, so `DECISION_LAB_URL` moves both the adapter and the embedded UI together and
they cannot disagree. Upstream errors pass through with their status and body, so the
page shows the service's own message ("no pending items to label") rather than a
generic failure; an unreachable service is a 502 naming the address it tried. Because
the probe is now same-origin, the reachability dot reads `/api/status` for real and
reports **ready**, not merely *reachable*.

One caveat is left, and it is the **port conflict above**: the relay's default
destination honors `DECISION_LAB_URL`, so a service running elsewhere needs that
variable set (or **Source** / `?url=` pointing at its own `/lab`). A cross-origin
**Source** is still subject to the browser's mixed-content rules — that is the
browser's call and the relay does not pretend otherwise.

#### Does it transfer? laya vs Jev

Everything above rests on one model, and the obvious objection is that reading
criteria off an option is specific to how laya's scorer reads a `[MASK]`. That is
testable, because **Jev** (TypeSafe, `typesafe/jev-1.13`) has the same interface:
the same typed questions over the same state, answering with the same
`{type, choice, probabilities, confidence}`. `server.py` takes `model: "jev"` and
swaps the backend; `jev.py` normalises the response into laya's envelope so no
client has a branch on who answered.

```sh
node eval_rules.mjs --compare --n 200 --seed 7     # both models, both arms
node eval_rules.mjs --model jev --n 200            # one model
```

200 scenarios, 195 conflicting. Same scenarios, same scoring, both models:

| | arm | detour regret | picks least-detour | capacity viol. | $ / decision | ms |
|---|---|---|---|---|---|---|
| laya | no-rules | 38.33 blk | 39.0% | 51 | $0 local | 81 |
| laya | rules | 6.43 blk | 79.0% | **0** | $0 local | 136 |
| Jev | no-rules | 35.18 blk | 45.5% | 47 | $0.000022 | 276 |
| Jev | rules | **1.54 blk** | **91.0%** | **0** | $0.000032 | 269 |

**It transfers, and transfers better.** Both models are close to useless on this
domain out of the box — laya's no-rules arm (38.33) is *worse than the naive
nearest-bus dispatcher* (34.60), and Jev's (35.18) is no better. Neither has
priors about a bus fleet. With the criteria injected, laya drops 6× and Jev drops
**23×**, landing at 1.54 blocks — within ~2% of the cost-optimal choice on a
scenario set neither model was trained for. So the technique is a property of
System-1 models, not of laya, which is the load-bearing claim in the IR section
below.

Three things worth separating out:

- **The capacity filter behaves differently by model.** With the option removed,
  both models record 0 violations. Without the filter, laya commits 4 and Jev
  commits **0** — Jev reads "full, no room" in the criteria and declines on its
  own. Tier 0 is what makes the rule unviolable for a weaker reader; a stronger
  one can be told. `rules-nofilter` on Jev (0.98 blk) is in fact slightly *better*
  than filtered (1.54) — with all three options visible it picks better among the
  feasible ones, so on Jev the filter costs a little and buys nothing.
- **Confidence is not comparable across models.** laya's rules arm sits at
  p50 0.033 because `temperature_by_options["choice:3-5"] = 1.76` flattens its
  logits; Jev reports p50 0.880 on the same question. The `CONF_FLOOR` tuned on
  laya (0.002) is effectively a no-op for Jev — every Jev decision clears it. A
  gate is a per-model constant, and moving models means re-deriving it.
- **Cost and latency.** Jev is $0.000032 a decision at this question size
  ($0.042/1M input tokens) — about six-tenths of a cent for 200 decisions. The
  real difference is that it is a network call: 269 ms against laya's 136 ms, and
  laya's marginal cost is zero because it runs locally.

Note that this comparison measures *how well each model uses what it is given*,
not domain knowledge: the no-rules arm still contains a nearest-bus recommendation
in prose, by design. Both models get the same hint.

#### The two worlds, and which one the rule is for

`fleet.html` has a **Multi-pickup physics** toggle, and the harness runs the
closed loop in both. Same rules, same cost function, different world.

Closed loop, 200 s per policy, one demand every 1 s, 3 seeds. Cells are
"delivered / blocks per demand":

```
                                     rules                greedy
----------------------------------------------------------------
multi-pickup on               189.7 / 36.0          189.3 / 38.0
multi-pickup off              133.0 / 40.2          127.3 / 39.9
```

**Multi-pickup is worth +43% throughput** to the rules dispatcher and +49% to
greedy. That is the headline, and it is the capability the rule was written for.

The detour rule itself is a smaller, different win: **~5% less distance driven**
(36.0 vs 38.0 blocks per demand) with multi-pickup, and no throughput gain at all
— at one demand per second both policies are capacity-saturated, so they deliver
the same 189. Without multi-pickup it is worth +4% throughput and nothing on
distance. Reported as measured, because the earlier version of this README
claimed the rule "did not pay off" and the real reason was that the simulator
could not multi-pickup at all.

#### Multi-pickup physics

The toggle switches the simulator between two worlds on the *same* route
representation, and the cost model switches with it.

**On** — a bus holds an ordered `stops` list and carries up to `RULES.seats` (4)
at once. A new demand is spliced in at the cheapest feasible position, with the
drop-off always after its own pickup. Buses carry a real seat limit, so a bus
with no room is **removed from the question's options** and the demand waits for
one to free up (counted in the "No seat" HUD field). The bus draws its remaining
route and an onboard-count badge so the plan is visible.

**Off** — a bus carries exactly one demand and queues the rest, which is the
original serial behaviour, preserved exactly.

`detourCost()` and `isOnRoute()` branch on the mode, and that branch is the
point rather than a detail: **a pickup on your route is only cheap if you can
actually stop for it.** In serial mode `detourCost` is the full serial cost and
`isOnRoute` is false for any committed bus, so the same rule honestly reports no
advantage in a world that cannot deliver on it. A test asserts that no committed
serial bus ever reports a pickup as on-route.

The planning horizon reuses `seats` as "how many more passengers a bus commits
to". It is not a second capacity — it is what stops a bus planning forty
orders — and it also keeps the O(n²) insertion scan cheap.

#### What the rewrite actually fixed

The earlier version priced buses with an insertion heuristic while the physics
served one job at a time — two different models of the same bus, and the gap
between them is what the closed loop kept catching. A bus now holds an ordered
`stops` list; the physics walks it and every cost function reads it, so the
number quoted in the question text is the number the bus drives. A test asserts
that identity directly (`routeLength(after) - routeLength(before) === delta`).

Three real bugs the work surfaced, all now covered by tests:

- **A passenger could be assigned twice.** `dispatch()` awaits the prediction, so
  a passenger still in `waiting` was picked up a second time by the retry loop,
  and its drop-off was counted twice — visible in the browser as *delivered 50,
  demands 30*. Fixed with a `dispatching` guard; the `delivered ≤ demands`
  invariant now pins it.
- **The route grew without bound.** Two stops per demand, against demands
  arriving faster than three buses can serve them, and the insertion scan blew
  up — a real hang, not a slow path. Fixed by `planHorizon`.
- **`busText` crashed in serial mode** on a bus with a queue but no active job,
  which is a real state: it has just dropped its last passenger and has not been
  handed the next one yet.

One measurement bug is worth recording too, because it nearly produced a false
negative: an early version of the throughput comparison drove the new demands
through one policy and the retry queue through another, so at low load — where
nothing ever waits — the detour policy was never actually exercised and the two
arms came out identical. The browser's default demand rate hid this as well: at
3 s or more every bus is idle on arrival and any policy collapses to
nearest-bus, so the slider now runs 0.5–4 s.

#### Confidence gating

`fleet.html` falls back to the deterministic cost model when laya's confidence
is below `CONF_FLOOR`, and shows a **decided / deferred** counter in the HUD —
because a gate silently caps how often the model actually decides, and a gate
tuned until laya "always agrees" has removed laya from the loop.

The floor is set from the measured distribution, not guessed. Confidence on this
workload is heavily compressed near zero (p10 0.004, p50 0.041, p95 1.000),
because the checkpoint ships `temperature_by_options["choice:3-5"] = 1.76`,
which softens the logits, and a near-uniform 3-way distribution has low entropy
confidence. An earlier guess of 0.45 deferred **100%** of decisions — the rules
were in the prompt and laya never got to use them.

`CONF_FLOOR` is **0.002**, the measured 10th percentile of the rules arm's confidence
over 200 scenarios (p10 0.002, p50 0.033, p95 1.000). It defers roughly a tenth of
decisions. An intermediate value of 0.013 sat near the 20th percentile and deferred twice
what the comment beside it claimed, which is how it went unnoticed — the number and the
claim now agree. Re-derive it with `node eval_rules.mjs --n 200`; the distribution moves
with the criteria text.

The p95 of 1.000 is the capacity filter showing through — when only one bus has room,
the model is not choosing, and the HUD counter is what makes that visible rather than an
accident.

**The floor is per-model, and this is the trap worth knowing about.** It is
calibrated to laya's distribution (p50 0.033), which is compressed near zero by the
checkpoint's own temperature. Jev reports p50 0.880 for the same question, so the
same 0.002 defers nothing at all — selecting `jev` in the browser defers 0 of 27
decisions, while laya defers about a fifth. Moving models means re-deriving this
constant from that model's percentiles, and reusing a floor across models is not
neutral: it silently disables the gate rather than mis-setting it.

`act_probability` is reported but deliberately *not* gated on: measured at
1.000 across every percentile here, so a floor on it would be dead code. Jev does
not return the field at all, which the `?? 1` in `askCJet` absorbs.

#### The ontology, and the enforce toggle

`static/fleet.html` has an **Ontology** panel (left side) with a live TBox
editor, an ABox view of what the reasoner currently believes, and one checkbox:
**enforce Tier 0 with the ontology**.

**Tier 0** is the hard-constraint layer — here, the seat limit. Everything else in
this demo (detour distance, nearest-bus) is a *preference*. Tier 0 can be
answered two ways:

- the hand-written `capacityOk` — an integer comparison, `onboard.size < seats`.
  This is the default and what the browser runs.
- the TBox/ABox reasoner in `static/ontology.js` — axioms are *declared*
  ("a bus may not exceed the seat limit") and a reasoner decides whether an
  assignment is consistent with them.

The toggle swaps the predicate. Nothing else changes: `ontology.js` never imports
`fleet_sim.js`, so the dependency is one-way and the two cannot form a cycle.

**Turning it on should not change any decision. If it does, that is a bug.** That
is the whole point. The ontology is a few hundred lines of reasoner replacing a
one-line integer check, so "do the two reach the same answer?" is the evidence
that the declared axioms faithfully reimplement the code they replace — a
hand-rolled `if` cannot be wrong in a way you can detect, but a declared axiom set
can be compared against the thing it displaces.

`node eval_rules.mjs` runs exactly that comparison, deterministically:

```
600 bus/demand pairs compared, 0 disagreements
```

**That 600 is weaker than it looks, and the panel says so.** `insertStops` returns
`null` at the planning horizon *before* consulting the seat predicate — either
implementation's. Re-measured: of the 600 pairs, **221 were refused by both**
layers without the predicate ever being called, so the seat check actually fired
on **379**. The equivalence is real but rests on fewer comparisons than the
headline suggests.

The browser demo is a *worse* place to check this than the harness, and the toggle's
own note explains why: the demo runs on wall-clock, so two runs see different
demand counts. Counting deliveries on screen is not that check. What is visible
there is that the two behave the same — the harness is where it is settled.

The stronger evidence is `test_ontology.mjs`, which compares the two predicates
directly over 5,000+ candidate routes with the horizon bypassed, and passes.

Editing the TBox (e.g. dropping the seat limit 4 → 2) restarts the simulator on
the same seed, because routes planned under the old limit would otherwise linger.
`checkTBox()` reports satisfiability live, so a contradictory axiom set is visible
before it silently changes dispatch.

##### What this ontology is not

The TBox is data, but the reasoner is **not** data-driven, and the difference is
worth being precise about because the demo's own framing invites the wrong reading.

`reason()` dispatches on an axiom's `form`, and only four forms have code behind
them — `IMPLEMENTED_FORMS` is that list. Everything else about an axiom *is* data:
`subject`, `property`, and `value` are read from the TBox, which is why dropping the
seat limit 4 → 2 changes dispatch with no code change at all. But the **kind** of
constraint is code. Adding a genuinely new kind — "a bus may not drift more than
10 cells from its depot" — is not a TBox edit.

Before this was enforced, such an axiom passed `checkTBox()` as coherent, appeared
in the editor, and was silently never evaluated: the self-check reported a rule as
enforced while it was inert. `checkTBox()` now rejects any axiom whose `form` is not
in `IMPLEMENTED_FORMS`, and the editor's form dropdown is derived from that same
list rather than restating it. So the failure mode is gone, but the **cost** is
still there — a new form requires code in `reason()` *and* an entry in the list, and
the axiom is refused until both exist. This is fail-fast, not extensibility.

Two related couplings remain, both deliberate:

- **Three axiom ids are hardcoded** — `axiom("bus-capacity")`,
  `axiom("demand-must-be-served")`, `axiom("passenger-single-state")`. Only
  `range` is evaluated by a generic loop over the TBox. So the *value* of a
  capacity axiom is data; the *connection* between an id and its check is not.
- **Tier 1 is not in the ontology at all.** `reason()` takes `detour` as an
  injected extension function, because insertion cost is optimisation rather than
  logic. A violation the reasoner reports is always about Tier 0.

A real OWL 2 RL engine (`owlrl`) was evaluated and rejected. Two findings, one of
which is decisive and one of which is not.

**It cannot run here at all.** `owlrl` is Python-only with no JS build, and
`canServe` is an ES module shared by three consumers — `fleet.html`, `test_ontology.mjs`
and `eval_rules.mjs`. Moving it server-side would delete the TBox editor, the ABox
panel and the enforce toggle from the browser demo, and push both test suites over
HTTP. That settles it on its own.

**It is also slower, though not fatally so.** Measured against this repo's actual
predicate (`owlrl` 7.6.2 / `rdflib` 7.6.0, in-process, minimal ABox of one bus and
four passengers):

| | per check | vs. now |
|---|---|---|
| this reasoner | 3.4–4.2 µs | — |
| `owlrl`, ABox rebuilt each call | 23–25 ms | ~6,000× |
| `owlrl`, graph reused, closure re-run | 11–12 ms | ~3,000× |
| `owlrl`, closure on an already-settled graph | 10–12 ms | ~2,500× |

The predicate runs 3.00× per demand (measured over 37 demand ticks, 111 calls), so
`owlrl` would add ~70 ms per demand — 2% CPU at the slowest demand rate, 28% at the
fastest. Slow, but not disqualifying on its own. The decisive number is the last row:
`expand()` is forward-chaining and re-runs the whole rule set on every call, so even
with the graph kept alive and nothing new to infer it still costs ~10 ms. About half
the cost is avoidable ABox reconstruction; the other half is structural.

That second half is the part worth noting, because it is what this module is already
built to avoid. The TBox is a parsed module constant rather than re-parsed turtle; the
ABox is a hand-built `Map` rather than an RDF graph; and `reason()` is a single pass
over the axioms rather than a fixpoint. All three exist to keep those two costs at
zero. An OWL-RL adoption would reintroduce both.

Note also that the rebuild figure is a *floor*: it was measured on a minimal ABox, and
inference cost scales with graph size, so the real demo's graph would be larger.

**What the ontology actually buys here**, stated without overclaiming: for a seat
limit, one line of `if` is sufficient and the reasoner is 300 lines of overkill.
What it adds is that the rule is *declared*, so it can be contradicted
(`checkTBox` catches `≥6` against `≤4`), it can name what it caught
(`bus-capacity :: carries 5 passengers, at most 4 allowed`), and it can be
compared against the code it replaces. A hand-written `if` cannot be wrong in any
way you can detect. The tradeoff only starts paying if rules multiply far enough
to interfere with each other — and this demo is not there yet.

#### The same idea in Breakout — and where the fleet invariant does *not* carry over

`static/breakout.html` now carries an Ontology panel too, with the same TBox editor and
ABox view — enforcement lives on the **Ontology: on/off** button in the Controls row
rather than a panel checkbox, one switch with one principle: when the ontology judges,
its verdicts control; when it is off, it controls nothing. The **TBox is data, but the reasoner is code** and the axiom
**ids are hardcoded** — so is the "Tier 1 arrives as an injected extension" split. What is
new is the constraint: reachability.

`Ball ⊑ ≥1 catchableBy`. A descending ball below the brick field that no `left/stay/right`
reaches is a violation, not a wrong answer. The verdict is computed by forward-simulating
the physics (`rolloutReachable` in `breakout_sim.js`) and injected as an extension, exactly
as fleet injects `detour` — cost is not logic.

**Why not the obvious arithmetic.** "time to paddle × paddle speed ≥ gap" is what you would
write, and it is wrong. Four closed forms were measured against the rollout over 40,000
states; the best agreed **83%** of the time and every residual error ran the same way —
declaring balls uncatchable that the physics catches. That is precisely the failure this
feature exists to fix, because it would exculpate the model on balls it genuinely lost. The
rollout costs **~9 µs typical, ~81 µs worst case** against a decision every 150–500 ms.

**A bug the tests caught, worth recording.** The rollout first ran for a fixed 2.5 s. The
brute-force sweep found **1136 of 21000** balls declared unreachable that the physics
caught — every one a slow ball. A paddle bounce can leave the ball descending at ~50 px/s,
and the ~210 px to the paddle then takes over 4 seconds. The horizon is now derived from
`ball.vy` rather than fixed. That number is the reason the sweep forward-simulates the *real*
`step()` rather than re-deriving the same algebra twice.

```
21000 states compared against the real step(), 0 false positives, 22 false negatives
```

**The invariant had to be restated, because the fleet one is false here.** Fleet's is
*"turning it on should not change any decision — if it does, that is a bug."* That holds
there by construction: the ontology filters an option set a caller already assembled. In
breakout it **cannot** hold. Once the ball is provably gone, `left/stay/right` has no correct
answer, so the option set *is* the question — gating it necessarily changes what is asked.
Claiming otherwise would be false. What replaces it:

> **Clause 1 — equivalence (inherited).** The reasoner's feasibility answer and the
> hand-written arithmetic agree wherever both apply.
>
> **Clause 2 — soundness of the gate (new).** Enforcement must never suppress a question
> whose answer could have changed the outcome. A false "unreachable" is the costly error; a
> false "catchable" merely wastes one ask. `CATCH_TOL_PX` is therefore non-negative by
> construction and asserted as such, so it cannot be "tidied" negative later.

Clause 2 is the one the 21,000-state sweep enforces, and it is one-directional on purpose.
Enforce **defaults to on** — the "Ontology: on/off" button in the Controls row is the single
switch, and when it is off the ontology controls nothing (the observation reverts to the
naive current-position baseline below, questions always go out, the stale-command re-aim
stops).

**Above the bricks the ontology is silent.** A descending ball above `y = 121` can be
deflected before it reaches the paddle, so no sound verdict exists. The decidable band is
`ball.y ∈ (121, 331)`; outside it the panel renders "no verdict", never a violation. A
verdict is about the geometry *at that tick*, but below the brick field no new information
can change the trajectory — a ball declared unreachable stays unreachable.

**What it does, measured.** The gate alone (suppressing hopeless questions) remains worth
exactly zero on this demo's controller — unreachable balls are rare (1.7%–6.7% of
questions, and that figure is a property of the controller, not of Breakout). What the
ontology now adds is the knowledge, and the knowledge pays. The two HUD counters still
split the loss into *provably lost* and *lost after a catchable ask* — the second is the
model's real failure rate, and it is the honest form of "the demo stops blaming the model
for physics-impossible catches".

**`node eval_breakout.mjs` is the on/off A/B.** The offline table is still the first look —
no server, no spend, the gate and the physics are pure functions — but since the observation
regime split (below), the closed-loop score IS the headline (measured further down):

```
600 decision ticks sampled (seed 11, 0.25s cadence)
  ontology-off: 600 asked · ontology-on: 560 asked, 40 suppressed (6.7%)
  suppressed but actually catchable : 0   <- must be 0; this is clause 2
  asked but hopeless                : 0
  closed form disagrees with the rollout on 12.8% of states
```

**The suppression rate is low here, and the reason is the demo's controller — not the game.**
An earlier version of this section said the opposite, and a sweep of the aim bias says that
was wrong. Suppression rate against controller quality (600 decidable states, seed 7, the
real `rolloutReachable`):

```
aim bias    unreachable    suppression
    0 px            0            0.0%
   40 px            3            0.5%
   80 px           54            9.0%
  120 px           90           15.0%
  160 px          144           24.0%
  200 px          192           32.0%
  240 px          206           34.3%
```

The paddle crosses at 200 px/s while the ball falls the ~210 px in 1.1–3 s, so it recovers
from most bad positions. A hopeless ball needs the paddle already committed the wrong way —
and the demo's controller is mostly not, so the gate has little to close. **The 1.7%–6.7%
quoted above is this curve's far left, i.e. a statement about the demo policy rather than
about Breakout.** At a genuinely bad aim bias the gate suppresses a third of what it is asked.

One consequence still cuts against reading the GATE as a performance win:

* **Suppression is not free.** A suppressed ball gets no command, so the paddle holds still
  and the cost lands on the *next* ball. Over the 192 suppressed balls at 200 px of bias, the
  recovery distance the next ball inherits is a median of 124 px and a p90 of 267 px; 74% of
  them need more than 80 px of travel to get back. The harness's "balls lost should be
  ~unchanged" therefore means the gate's cost roughly cancels its benefit, not that there is
  a benefit.

The gate's own value in this example is presentational and structural: it splits *provably
lost* from *lost after a catchable ask*, so the demo stops attributing physics-impossible
catches to the model, and it is the second domain proving the shared TBox/ABox core. **The
gate itself is not a speedup — the knowledge injection below is.**

Two measurement traps this harness hit and now guards against in its own comments, because
both produced a clean sheet of zeros that meant nothing: generating scenarios with a
**perfect** paddle (every ball trivially catchable, nothing to suppress), and an aim bias too
small to matter (unreachable rate 0.0% below ~60 px of bias).

**Reproducing the measurements below.** Three scripts, no model server and no spend — the gate
and the physics are pure functions, so every number in this section comes from the same code
path the browser runs:

```
node tools/bias_sweep.mjs      # the suppression table, and what suppression costs
node tools/stale_command.mjs   # the wall-bounce fix, per-lag and per-arm
node tools/landing_accuracy.mjs # how exact the landing point handed to the model is
```

Each script's header records the trap that produced a wrong number the first time, since the
mistakes are more reusable than the results.

**The observation is now the knowledge, in two regimes.** `buildState()` once compared the
paddle to the ball's *current* x; the ball drifts and reflects on the way down, so over
80,000 decidable states that named the wrong side **12.4%** of the time. The fix took that
to **0** by construction, and both the prose and the reasoner read the same `measure()` so
the number the model sees and the number reasoned over cannot diverge. The master switch
now splits the observation in two:

- **Ontology on** — the side is named from the wall-folded **landing point** (the
  `ball-predicted-at-paddle-level` axiom's output) and the two absolute x positions are
  stated: *"It will land at x≈249, and the paddle is at x≈100."* Above the decidable band
  (rising, above the bricks) there is no prediction, so the prose falls back to the current
  position — no claim where the ontology has none.
- **Ontology off** — the **naive current-position** observation, deliberately resurrected
  as the contrast arm: it names the wrong side ~12.4% of the time, measured on the actual
  prose over 4,000 states. The ON regime names it 0 times on the same states. That test
  breaks if the two regimes ever stop disagreeing — the contrast is the demo.

Two intermediate forms were measured against the real model and rejected: coordinates
appended alongside the gap prose changed **0 of ~2,600 answers** (the model keys on the gap
phrase), and coordinates *without* the prose collapsed the score **19.3 → 6.3** — this
checkpoint derives nothing from raw coordinates, so the prose stays and the coordinates ride
alongside.

**A wall-bounce fix: the one thing here that pays.** The paddle visibly misplays
after the ball reflects off a side wall, and the reading is that the demo should re-aim the
instant `vx` flips — a *stale-command* axiom, `Command ⊑ ≥1 validUntil`, a temporal validity
condition rather than a reachability one, and it lives in the TBox as an independent axis. The
premise checks out: with a 0.25 s hold, **38.8%** of in-command wall reflections leave the held
command pointing at the wrong side.

**The axiom is over the SIDE of the paddle, not the landing coordinate** — and that is not a
wording choice. The obvious statement, `validUntil(x_next ≠ x_goal)`, is built on a premise that
turns out to be false, in a way that fails silently. `buildLandingX` folds the trajectory through
`reflectX`, so the *predicted* landing point is continuous across a reflection by construction:
measured over 12 reflections observed mid-descent, the issued and current landing points agreed
to within 0.7 px every time. A coordinate comparison therefore detects nothing, forever, while
looking exactly like a working detector that simply never fires.

What the reflection invalidates is the side the ball will *arrive* on — which is what the
paddle drives toward and what the model's answer named. So `commandIsStale` compares the sign of
the gap at issue time against the sign now, and only after `step()` has recorded that `vx`
actually reversed (a paddle bounce reverses `vx` too, and is not this event). Both halves are
tested against real physics: `the landing coordinate does NOT move at a bounce` pins the premise
that makes the sign the right thing to compare, so if the fold in `reflectX` ever changes, the
test says so instead of the axiom quietly becoming a no-op.

Against the **real checkpoint**, the gain holds — and with the two-regime observation it is
now a measured performance claim, not a bookkeeping one. `eval_breakout.mjs --loop` drives
four arms — the naive baseline (`ontology-off`: current-position observation, every ball
asked), the prediction regime with the gate (`ontology-on`), and the same two with the
stale-command re-aim — 60 s per seed, two independent batches:

```
                              ontology-off      ontology-on            reaim   ontology-on+reaim
batch 1 (seeds 11-13)
questions asked                         166              241              169                241
lost after an ask                       2.7              1.3              2.7                1.0
provably lost                           0.0              0.0              0.0                0.0
score                                   14.0             19.3             13.7               22.3
batch 2 (seeds 21-23)
questions asked                         173              196              183                241
lost after an ask                       3.0              2.3              3.0                1.3
provably lost                           0.0              0.0              0.0                0.0
score                                   15.0             17.3             15.0               23.3
```

**The button comparison is `ontology-off` vs `ontology-on+reaim`: +8.3 in both batches**
(14.0 → 22.3 and 15.0 → 23.3), with balls lost after an ask 2.7 → 1.0 and 3.0 → 1.3 — the
naive-baseline runs lose so many balls the game ends early (166/173 asks against 241),
which is also why their api-call counts are lower. Four things in the table are the point:

* **The gain is the knowledge, not the gate.** Suppression fired 0–3 times across both
  batches (the demo's controller is too good for the gate to close — the bias sweep below
  predicts exactly this). The +8.3 comes from the observation regime: the model is told
  where the ball will land instead of where it is.
* **The ON columns reproduce the historical baseline exactly** (19.3/22.3, 17.3/23.3) — the
  prediction regime is the prose the demo always sent, plus the coordinates. So the +8.3 is
  attributable to the OFF arm's naive observation, not to ON getting faster.
* **The re-aim arm alone is worth nothing now** (13.7, 15.0 — level with the baseline). The
  stale-command fix re-derives toward the *predicted* landing point; against the naive
  observation it churns instead of catching. In the earlier prose-only baseline the same fix
  won 3 of 3 seeds for +9 net — the fix and the prediction compose, and the pair is what the
  button turns on.
* **It is free.** The prediction and the re-aim both re-derive from `measure()` over the
  live sim; no extra model calls in either direction.

**Three seeds per batch is not many.** +8.3 held across two independent batches, but the
per-seed spread is wide enough that the confidence interval is not tight. Seeds default to
11–13 so the tables reproduce; widen them before trusting the magnitude:

```
uv run uvicorn server:app                                   # terminal 1
node eval_breakout.mjs --n 40 --seed 3 --loop --loopsecs 60
```

Roughly 20 minutes for 40 seeds across the full four arms — the runtime is dominated by real
model calls, one per question. Raise `--loopsecs` for a tighter interval or narrow `--n` to
iterate; the `wall bounces seen` row tells you immediately whether a run had enough
opportunities for the fix to be able to matter, and a run reporting a small bounce count cannot
support a conclusion either way.

**The axiom is now built and wired into the demo.** `command-stale-on-bounce` is in `TBOX`, and
`breakout.html` enforces it every frame in `frame()`:

```js
if (axiom("command-stale-on-bounce") && commandIsStale(sim)) {
  const m = measure(sim);
  const dir = m.reachable === false ? 0 : (Math.abs(m.gap) <= 1 ? 0 : Math.sign(m.gap));
  issueCommand(sim, dir);
}
```

It is gated on the master switch **and** the axiom being present — turn the Ontology button
off, or delete `command-stale-on-bounce` in the TBox editor, and the correction stops, with
`unasserted` reporting the deletion. That is the same hole commit `a5149d0` closed for the
reachability axiom, kept closed here: an axiom that is displayed as enforced and enforced by
nothing is the failure mode worth designing against.

**A correction to what this section claimed before, from a bug worth remembering.** An earlier
version here reported the upper bound as 0 and the fix as rejected. That was wrong: the harness
detected the reflection by comparing `vx` *before* the `step()` that produces it, so the branch
never fired and all three arms returned identical numbers — which read exactly like "the fix
does nothing." The three-way tie was the tell. **A tie between arms that should differ is a
broken instrument, not a null result**; compare the arms before believing their agreement. The
same harness also first reported the fix as a large regression (491 → 330) by resetting the
command expiry at the bounce, holding a stale command past its deadline — clamping to the
original expiry is what makes it work at all.

**The "no version of this change helps" argument was sound reasoning on a broken premise.** It
rested on a reflection never making a catchable ball uncatchable — true, and irrelevant. The
mechanism is not that the bounce costs a catch; it is that the paddle was already committed the
wrong way *before* the reflection and stayed committed after it.

A related intuition is worth killing too, since it is the natural next guess: re-deriving the
command every physics step instead of on a 0.25 s cadence scores **202 against the cadence's
380**. `aimTarget` flips sign as the paddle crosses the landing point, so continuous
re-aiming oscillates. In this game "update more often" is the wrong direction — the fix reacts
to the rare *event*, not a shorter clock.

**What remains.** The prediction the model is shown is already exact —
over 334 asked-and-catchable states the gap between the reported `landingX` and where the ball
is actually caught is a median of **0.2 px** (p90 0.2, max 3.7), and states where a reflection
intervened are *not* worse (0.1 px mean) than states where none did. `buildLandingX` expands
not-yet-happened reflections through the `reflectX` triangle wave, so a predicted bounce is
already priced in. What is left is the cadence and the swing past the target. Both candidate
fixes — a shorter ask interval (4× the model calls) or a monotone target band in
`buildState()` to damp the oscillation — are changes to the observation and the demo's cadence,
not to the ontology. Neither was built.

**Two modules, one shared core.** `breakout_ontology.js` is separate from `ontology.js`
rather than a generalisation of it: the fleet reasoner has three hardcoded axiom ids the
README names as a deliberate coupling, and parameterising them is a rewrite of the thing
`test_ontology.mjs` protects. What was genuinely shared — `checkTBox`, `describeAxiom`,
`addAxiom`, `removeAxiom`, `resetTBox`, and the editor itself — moved to
`static/ontology_core.js` as `mountTBoxEditor({ root, adapter })`. `ontology.js` re-exports
the moved names, so `test_ontology.mjs` and `eval_rules.mjs` are untouched and still pass.

**`IMPLEMENTED_FORMS` deliberately did not move.** Both domains happen to implement the same
four forms, so a shared list is the obvious tidy-up and it would re-open the hole commit
`a5149d0` closed: a form fleet's `reason()` handles would pass breakout's `checkTBox` and then
be enforced by nothing. Each domain exports its own, and `checkTBox` takes it as an argument.

**Browser verification.** The panel was checked in headless Chromium, not just in node: panel
mounts with the declared axioms (7 today) and a satisfiable TBox, the toggle flips without
error, the ABox
populates with a live verdict, duplicate axioms are refused *by name*
("ball-must-be-catchable already asserts that"), and add/delete/reset all repaint. Two bugs
were caught only there — a stray `</content>` left in the spliced script, and a stale axiom
list after an edit — neither of which any node test could see.

#### Fine-tuning

Not done, and the measurement says it should stay that way. `laya` ships no
training code, though `common.py` does expose `build_model`, `collate_items`,
`proper_reward` and `td_lambda_targets`, and the config is named
`rl_agent_config.json` with a `training` block — so a finetune means writing a
loop against `DecisionModel` ourselves, with the simulator as both reward oracle
and expert. Two costs to name up front: 421M parameters on MPS iterates slowly,
and narrow fine-tuning of a third-party checkpoint tends to wreck the
calibration that `confidence` currently provides — which would also break the
gate above.

The gate was: finetune only if the rules arm left regret materially above the
floor on scenarios where the rule is clearly correct. It does not — regret falls
38.33 → 6.43 blocks on laya and 35.18 → 1.54 on Jev, without touching a weight.
The remaining gap is not something the model fails to learn; it is the ~5% of
distance the routing objective can still give back, and a finetune would be
paying 421M parameters of training to chase it. The cross-model result is what
settles this: if the technique were a property of the checkpoint rather than of
System-1 models in general, Jev would not have improved 23× without training.

## IR 자료

> 이 섹션은 위 기술 작업에서 **실제로 측정된 값만** 재구성한 요약입니다.
> 시장 규모, 고객, 매출, 도입 사례는 이 저장소에 근거가 없어 기재하지 않았습니다.
> 본 저장소는 시뮬레이션 데모이며, 프로덕션 검증 전입니다.

### 한 줄 요약

**업무 도메인 규칙을 AI 의사결정 모델에 재학습 없이 반영할 수 있고, 그것이 실제로
판단을 바꾸며 우회로 비용을 줄였음을 측정했습니다.** 같은 규칙·같은 비용 함수로,
운영 방식이 그 규칙을 수용하는지에 따라 가치가 사라지거나 나타납니다.

### 무엇을 만들었는가

좌석·픽업 순서·다중 승객 등 **업무에 고유한 제약과 지식**을 가진 의사결정 문제에
범용 사전학습 체크포인트(421M 파라미터)를 그대로 연결했습니다. 모델은 1회
비-autoregressive 추론으로 판단하며, 추론 속도는 판단당 약 150ms입니다.

핵심은 규칙을 "가르쳐서" 넣지 않는다는 점입니다. 규칙에서 파생되는 **수치**(이
버스에 추가되는 블록 수, 남은 좌석)를 계산해 판단 옵션에 붙이면, 모델이 그 수치를
따릅니다. 모델을 고쳐-tune할 필요가 없습니다.

### 측정된 결과

**규칙이 모델에 도달하는가** (200개 시나리오 중 195개가 실제 충돌 케이스)

| 지표 | 규칙 없음 | 규칙 반영 |
|---|---|---|
| 최소우회 regret | 38.33 블록 | **6.43 블록** |
| 최소우회 버스 선택률 | 39.0% | **79.0%** |

**모델이 바뀌어도 같은 결과인가** — laya(CJet)와 Jev(TypeSafe) 동일 인터페이스 대조

| | 규칙 없음 | 규칙 반영 | 판단당 비용 | 지연 |
|---|---|---|---|---|
| **laya (CJet)** | 38.33 블록 | 6.43 블록 | $0 (로컬) | 136 ms |
| **Jev** | 35.18 블록 | **1.54 블록** | $0.000032 | 269 ms |

두 모델 모두 **규칙 없이는 쓸모가 없습니다** — laya의 no-rules(38.33)는 단순 최근접
휴리스틱(34.60)보다 오히려 나쁩니다. 규칙을 넣으면 laya는 6배, Jev는 **23배**
개선되고 1.54 블록까지 내려갑니다. 즉 이 기법은 laya의 `[MASK]` 구조 특유한
것이 아니라 **System-1 모델 일반의 성질**입니다. IR 주장에서 가장 무게가 실리는
부분이 이겁니다.

**하드 제약(좌석)이 실제로 지켜지는가**

| | 규칙 없음 | 규칙 반영 | 규칙 반영(필터 미적용) | 단순 최근접 |
|---|---|---|---|---|
| laya 좌석 초과 배정 | 51건 | **0건** | 4건 | 48건 |
| Jev 좌석 초과 배정 | 47건 | **0건** | **0건** | 48건 |

좌석이 없는 버스를 **선택지에서 삭제**했기 때문에 위반이 0입니다. 규칙을
"설명"한 것이 아니라 "제거"해서, 모델이 보지 못한 선택지를 고를 수 없게 한
결과입니다.

다만 **모델에 따라 필터의 가치가 다릅니다.** 필터 없이 criteria 문장만으로
알려주면 laya는 4건 위반하지만 Jev는 **0건**입니다 — Jev는 "full, no room"을
읽고 스스로 거절합니다. Tier 0는 읽는 능력이 약한 모델을 위해 규칙을
"불가능하게" 만드는 장치이며, 잘 읽는 모델에게는 필터가 아무것도 보장하지
않습니다(사실 Jev에서는 필터가 오히려 regret을 0.98 → 1.54로 조금 악화시킵니다).

> **이전 수치는 정정되었습니다.** 위 "규칙 없음" 수치는 당초 `rules:` 키만 지우고
> 다시 측정한 값이었고, 그 상태에는 우리 규칙이 다른 두 경로로 남아 있었습니다
> (`recommendation` 문장에 `detourCost()`로 만든 문장이 포함됨). 깨끗하게 다시 만든
> 기준선은 약 2배 나쁩니다(19.25 → 38.33). 규칙 arm는 그대로이므로 개선 폭이
> 커져 보이지만, **새 결과가 아니라 이전 기준선의 오류가 수정된 것**입니다.

**운영 방식이 바뀔 때** (시뮬레이션, 정책당 200초, 3시드)

```
                              rules              greedy
multi-pickup on        189.7 / 36.0        189.3 / 38.0
multi-pickup off       133.0 / 40.2        127.3 / 39.9
```

- 다중 탑승이 가능한 운행일 때: 처리량 **+43%**, 주행거리 **−5%**
- 직렬(1명씩)일 때: 규칙의 이득이 **0으로 사라짐**

**고급 지식 주입 — 패들 제어 (Breakout, 실측)**

온톨로지가 선언한 지식 — 벽 반사까지 반영한 **착륙 지점 예측** (`Ball ⊑ ≥1 predictedPosition`)
— 을 모델의 입력에 주입했을 때의 성능 변화입니다. 대조군은 의도적으로 naive 관측(공의 **현재
위치** 기준, 하강 중 옆면을 12.4%의 확률로 반대로 말함)이고, 실험군은 착륙 지점 기준 판정에
절대 좌표("x≈249에 떨어지고, 패들은 x≈100")를 함께 줍니다.

| | naive 관측 (온톨로지 off) | 예측 관측 (온톨로지 on) |
|---|---|---|
| score (배치 1, 시드 11–13) | 14.0 | **22.3 (+8.3)** |
| score (배치 2, 시드 21–23) | 15.0 | **23.3 (+8.3)** |
| 질문 후 잃은 공 | 2.7 / 3.0 | **1.0 / 1.3** |
| 잘못된 방향 지명 | 12.4% | **0%** (4,000 상태 실측) |
| 추가 모델 호출 | — | **0회** |

두 배치 모두 +8.3으로 일관됩니다. naive 관측에서는 라이브 3개 중 평균 2.7~3.0개를 잃어
게임이 조기 종료됐고(질문 166/173회), 예측 관측에서는 60초 내내 플레이했습니다(241회).
억제 게이트는 이 실행에서 0~3회 발동에 그쳤으므로 **개선은 게이트가 아니라 주입된 지식
(착륙 예측)의 효과**입니다. 온톨로지 off 시 나쁜 관측이 되는 것은 대조를 보여주기 위한
의도된 설계입니다. 재현: `node eval_breakout.mjs --n 3 --seed 11 --loop --loopsecs 60`.

### 해석: 이 실험에서 가장 값진 발견

**1. AI가 못 한 것이 아니라, 운영 방식이 규칙을 받쳐주지 않았던 것입니다.**

규칙 주입 직후 측정한 결과는 이렇았습니다 — 규칙은 모델에 잘 들어갔고
우회로 regret이 크게 줄었는데, 실제 처리량은 오히려 **적었습니다**(76.0 vs 78.3).
원인은 AI가 아니라 **버스가 구조상 한 명씩만 태울 수 있었던 것**이었습니다. 곧,
"도메인 규칙을 넣어라"가 아니라 **"그 규칙이 성립할 수 있는 운영 모델부터 갖춰라"**
는 순서를 말합니다.

**2. 기법은 특정 모델의 구조가 아니라 System-1 모델 일반의 성질입니다.**

동일 인터페이스를 가진 다른 System-1 모델(Jev)로 반사실 A/B를 돌렸을 때,
Jev는 규칙 없이는 laya와 마찬가지로 쓸모없었고(35.18 vs laya 38.33, 둘 다 단순
최근접 34.60 수준), 규칙 주입 시 **23배** 개선되며 1.54 블록까지 내려갔습니다.
laya의 `[MASK]` 스코어러 구조에 특화된 트릭이었다면 Jev에서는 작동하지 않아야
했습니다. 즉 특정 벤더 선택이 아니라 **방법론**을 사는 것이고, 모델 교체 시
기존 주입 코드가 그대로 유지됩니다.

### 조건과 비용

- 모델 크기·지연: 421M 파라미터, 판단당 ~150ms (기준 입력 대비 +40ms).
  비교 대상 Jev는 269ms, 판단당 $0.000032 (판단 200회 ≈ 0.6센트)
- 학습 비용 0: 파인튜닝 미수행. 규칙 반영만으로 판단이 바뀌며, 파인튜닝 시
  calibration이 손상되어 신뢰도 게이트가 무력화될 위험이 있음
- 재현성: 시뮬레이션이 시드 고정·결정론적이며, 같은 시나리오에서 동일 재현
- **모델 교체 시 신뢰도 게이트는 재파생해야 합니다.** laya의 신뢰도는
  p50 0.033, Jev는 같은 질문에 p50 0.880입니다. laya에 맞춰 잡은 임계값
  (0.002)을 그대로 쓰면 Jev에서는 **무효**가 되어 27건 중 0건만 게이트됩니다.
  임계값은 모델별 상수이고, 모델을 바꾸면 재측정 없이 그대로 두면 게이트가
  조용히 꺼집니다.

### 한계 — 주장 범위를 넘어서는 부분

- **시뮬레이션 결과이며 운영 데이터가 아닙니다.** 실운영에서의 효과는 별도
  검증 대상입니다.
- 규칙 주입의 효과(−5% 주행거리)는 처리량 개선이 아니라 **효율 개선**입니다.
  다중 탑승의 +43%는 AI 기여가 아니라 **운영 방식 변경의 효과**입니다. 두 값을
  혼동하지 않는 것이 중요합니다.
- 강제 제약(좌석)이 있는 문제는 코드로 처리했습니다. 프롬프트에 "이러면 안 된다"고
  쓰는 것으로는 보장되지 않기 때문입니다. 다만 **모델이criteria를 제대로 읽으면
  필터 없이도 0건**이었습니다(Jev). 즉 코드로 제거하는 것은 "안전한 기본값"이지
  유일한 방법은 아닙니다. 자연어로 표현되는 **선호**(soft preference)만 모델에
  맡기는 경계가 명확하지 않은 영역은 남아 있습니다.
- **Jev 비교는 단일 스냅샷(1.13-20260917) 1개입니다.** 버전 간 차이가 이번
  격차에 기여했는지 알 수 없습니다. 시나리오·판정 코드는 동일하나 200회 1회
  실행이므로 seed 재현성은 있으나 통계적 표본은 아닙니다.
- 규모 검증 없음: 버스 3대, 격자 50×50, 421M 모델 기준. 파라미터 스케일업과
  동시 에이전트 수 증가에 대한 검증은 없습니다.

### 다음 검증 과제

1. 실제 운영 데이터로 동일 프로토콜(반사실 A/B + closed loop) 재실행
2. 파인튜닝 게이트: 규칙 반영 후 regret이 기준선을 크게 넘을 때만 재고려 (현재
   38.33 → 6.43으로 통과하지 않음 → **파인튜닝하지 않음**)
3. 언어 분기: 현재 영어 체크포인트 기준이며, 다국어 입력에서의 규칙 유효성 미검증
4. 모델 3종 이상으로 확대: 현 결론은 2개 모델(421M 로컬 / 원격 API)이며,
   Jev 스냅샷 변경 후 재측정

## Sensor anomaly detection

`sensor_anomaly_detection.py` is a standalone, stdlib-only script (no laya
dependency, nothing new to install) that demonstrates statistical anomaly
detection on synthetic sensor data:

```sh
uv run python sensor_anomaly_detection.py
```

### What it does

1. Generates a deterministic temperature + humidity feed (500 samples each,
   5-minute interval, periodic signal + Gaussian noise, `SEED=42`) and injects
   known faults: spike, out-of-range, flatline (stuck sensor), and transient
   drift (ramp up + ramp down).
2. Runs six detectors and merges their flags per reading:
   - `range` — raw value outside the sensor's physical limits
   - `global_z` / `rolling_z` — z-score vs whole series / trailing window, on
     detrended residuals (reading minus the expected periodic signal)
   - `rate` — jump between consecutive readings over the rate limit
   - `flatline` — N consecutive identical readings
   - `level_shift` — median of a trailing window vs the preceding window
     (catches drift; medians keep a single spike from masquerading as a shift)
3. Prints flagged readings (severity = number of agreeing detectors) and
   precision/recall against the injected ground truth.

### Results (SEED=42)

| Sensor       | Precision | Recall | Notes                                            |
| ------------ | --------- | ------ | ------------------------------------------------ |
| temperature  | 80%       | 100%   | spike and out-of-range fully caught              |
| humidity     | 89%       | 70%    | flatline 100%; drift caught ~2/3 through its ramp |

Slow ramps are the hardest shape for windowed statistics: a trailing-window
z-score mathematically caps at √12/2 ≈ 1.73σ on any linear ramp (the slope
cancels), which is why drift needs the dedicated level-shift detector — and
even then it only fires once ~14 of its last 20 samples are inside the ramp.
The residual false positives are statistical noise tails plus documented
"settling" flags while window detectors clear an anomaly's aftermath.

### Raising precision

Three levers, most effective first:

1. **Require detector agreement** — keep only readings flagged by ≥2 detectors
   (`if len(methods) >= MIN_AGREEING_DETECTORS`). On this data all false
   positives are single-detector flags.
2. **Raise thresholds** (`Z_THRESHOLD`, `LEVEL_SHIFT_THRESHOLD`) — fewer false
   alarms, more missed events.
3. **Confirm-before-alert** — require 2–3 consecutive flags before alerting
   (the usual pattern for live streams).

### Note on laya / Jev

This script deliberately does **not** use laya: its checkpoints are trained on
support-ticket text, so numeric sensor data is out-of-domain. The sensible
architecture is statistics for detection and a System One model (laya, or
TypeSafe AI's frontier [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev))
as a downstream judgment layer — feeding detector evidence as state and asking
typed questions (fault type = choice, urgency = score, needs-maintenance = yes-no).

## Startup performance

Stock `laya` + `Router(preload=True)` spent ~100s before the first prediction on
an Apple-silicon (MPS) machine. Two independent causes, both worked around in
`main.py` without touching the installed package (so the patch survives
`uv sync` / reinstalls):

| Phase                          | Before  | After   |
| ------------------------------ | ------- | ------- |
| Hub file checks (per run)      | ~1s     | 0s      |
| Model build (per checkpoint)   | ~29s    | ~0.1s   |
| Weight load + MPS transfer     | ~2s     | ~2s     |
| **Total (`uv run python main.py`)** | **~100s** | **~7s** |

Predictions are bit-identical before and after the patch (verified on both the
English and Hindi routing paths).

### Cause 1: per-run hub checks

Every run, `Agent.__init__` calls `snapshot_download()`, which performs a
network round-trip to Hugging Face to validate each cached file — even when
everything is already in the local cache. The `Fetching 5 files` progress bars
appear on every run while transferring `0.00B`.

**Fix:** set `HF_HUB_OFFLINE=1` before importing `laya` (`main.py:5`). Skipped
entirely when checkpoints are cached; unset it (or don't set it) when a newer
checkpoint must be pulled.

### Cause 2: wasted random initialization

`laya.common.build_model()` constructs the encoder with
`AutoModel.from_config()`, which *randomly initializes all ~400M parameters* —
about 29s per checkpoint. `Agent.__init__` then immediately calls
`load_state_dict(strict=True)` and overwrites every one of those tensors with
the checkpoint weights. That's ~90s of pure waste across the three checkpoints.

**Fix:** `main.py` replaces `laya.agent.build_model` with `_fast_build_model()`
(`main.py:14-29`), which wraps the same construction in transformers'
`no_init_weights()` context manager. Parameter tensors are still allocated (so
shapes and the strict state load are unchanged), but the expensive random fills
are skipped: ~29s → ~0.1s per checkpoint.

#### Why not a meta-device build?

A `torch.device("meta")` build is even faster, but `to_empty()` leaves
**non-persistent buffers** uninitialized — ModernBERT registers its RoPE
`inv_freq` buffers with `persistent=False`, so they are not part of the
checkpoint `state_dict` and would contain garbage. This silently corrupts
position embeddings and changes predictions (the English routing flipped from
`billing` to `sales` when tested). `no_init_weights()` skips only the parameter
initializers, so those buffers are computed normally.

#### What the monkeypatch does and doesn't change

- Replaces only the encoder/`DecisionModel` constructor. Everything else in the
  load path (checkpoint download, `_verify_compatibility()` architecture checks,
  strict `load_state_dict`, temperature clamping, device placement) is untouched.
- `no_init_weights()` is a private-ish API (`transformers.initialization`); if a
  future transformers release moves it, the patch fails loudly at import time
  rather than producing wrong results.
- The patch is applied before `from laya import Router` so every checkpoint
  `Router` loads uses it.

### Remaining startup cost

The residual ~7s is mostly unavoidable: importing `torch`/`transformers`
(~1s) and moving ~1.16B parameters onto the MPS device (~2s per checkpoint
round of allocation + transfer). If startup ever matters more than first-token
latency, run the router as a long-lived server process and preload once.
## 부록 A. CJet 대 Jev — 전이성 실험의 보존 기록

> 원래 독립 문서(`cjet-jev-compare.md`)였던 것을 README에 통합했습니다. 같은 실험의
> 본문 기록은 [Does it transfer? laya vs Jev](#does-it-transfer-laya-vs-jev)에 있고,
> 중복되는 표·해석·기준선 정정은 본문으로 통합했습니다. 여기는 본문에 없는 기록만
> 남긴 축약본입니다. 모든 수치는 `node eval_rules.mjs --compare --n 200 --seed 7`로
> 재현됩니다.

### 인터페이스 동일성 — 비교가 공정한 이유

Jev(TypeSafe, `typesafe/jev-1.13`, OpenRouter Decisions API 경유)는 laya와 동일한 타입
질문 인터페이스를 받는 System-1 의사결정 모델입니다.

| | laya | Jev |
|---|---|---|
| 질문 | `{type, instructions, criteria}` | 동일 |
| choice criteria | `{key: description}` | 동일 |
| score criteria | `[label, ...]` | 동일 |
| 응답 | `{type, choice, probabilities, confidence}` | 동일 |
| 추가 필드 | `action.act_probability` | 없음 |
| 사용량 | `input_tokens` | `input_tokens`, `output_tokens`, **`cost` (USD)** |

`server.py`는 `model: "jev"`를 받아 백엔드를 교체하고, `jev.py`는 응답을 laya의
envelope 형태로 정규화합니다. 그래서 어떤 클라이언트도 "누가 답했는지" 분기하지
않습니다. 동일한 state, 동일한 질문, 동일한 채점 — 바뀐 것은 어느 모델이 답했는지뿐입니다.

### 8-arm 전체 표와 충돌-제한 수치

전체 200개 시나리오 — 본문의 4-row 표에 두 arm을 더해 전부:

| 모델 | arm | 평균 regret | 최소우회 선택률 | 좌석 위반 | 판단당 비용 | ms |
|---|---|---|---|---|---|---|
| laya | no-rules | 38.33 blk | 39.0% | 51 | $0 (로컬) | 81 |
| laya | rules | 6.43 blk | 79.0% | **0** | $0 (로컬) | 136 |
| laya | rules-nofilter | 6.23 blk | 83.6% | 4 | $0 (로컬) | 141 |
| laya | greedy | 34.60 blk | 48.7% | 48 | — | — |
| Jev | no-rules | 35.18 blk | 45.5% | 47 | $0.000022 | 276 |
| Jev | rules | **1.54 blk** | **91.0%** | **0** | $0.000032 | 269 |
| Jev | rules-nofilter | 0.98 blk | 92.5% | 0 | $0.000032 | 262 |
| Jev | greedy | 34.60 blk | 48.0% | 48 | — | — |

충돌하는 195개로 제한해도 순위는 그대로입니다 — laya는 38.83 → 6.60 blk, 39.5% → 78.5%,
Jev는 35.61 → **1.55** blk, 46.2% → **91.3%**입니다.

### 신뢰도 백분위 (모델별)

동일한 질문에 대해:

| 모델 | p10 | p50 | p95 |
|---|---|---|---|
| laya | 0.002 | 0.033 | 1.000 |
| Jev | 0.420 | 0.880 | 1.000 |

laya에 맞춘 `CONF_FLOOR = 0.002`를 Jev에 쓰면 27건 중 **0건**만 게이트됩니다 —
오발생도 경고도 없이 게이트가 조용히 게이트를 멈춥니다. 모델이나 질문 텍스트가 바뀔
때마다 calibration 상수를 다시 파생해야 합니다. `act_probability`는 두 모델 모두 모든
백분위에서 1.000이므로 게이트하지 않습니다.

### 이 부록의 한계

- **스냅샷 1개, 실행 1회.** Jev `1.13-20260917`, 시나리오 200개, 단일 패스. 시드는
  재현 가능하지만 큰 표본이 아니며, 버전 간 차이는 측정하지 않았습니다.
- **이 비교가 재는 것은 도메인 지식이 아니라 정보 사용량입니다.** no-rules arm에도
  설계 의도로 최근접 버스 추천이 산문으로 들어 있습니다 — 두 모델이 동일한 힌트를 받고,
  두 arm을 가르는 것은 규칙에서 파생된 것뿐입니다.
- **모델 2개는 조명이지 조사 결과가 아닙니다.** 주장은 "laya 밖에서도 전이된다"까지이며,
  "System-1 모델 일반에 통한다"까지 확장하려면 백엔드가 더 필요합니다.
- **시뮬레이션이며 운영 데이터가 아닙니다.**

## 부록 B. CJet 기술 자료 — 고유 내용의 보존 기록

> 원래 독립 문서(`cjet-merit.md`)였던 것을 README에 통합했습니다. 규칙 계층 표, Jev
> 비교, 비용·속도, 수치 요약은 본문과 [IR 자료](#ir-자료)로 통합되어 있으므로 여기서
> 반복하지 않습니다. 남기는 것: 본문에 없는 인사이트와 논증만입니다.

### 계획 한계(over-commitment)는 좌석과 별개의 제약이다

"이 버스는 최대 4명까지 태울 수 있다"는 제약과 별개로, 이 시뮬레이터를 실제로 제한하는
것은 대개 **계획 한계**입니다. 버스는 `planHorizon = 2 × seats` 정거장까지만 경로에
약속할 수 있고, 이미 그만큼 약속했다면 좌석이 남아 있어도 새 수요를 거부합니다.
17,243건의 거절 사유를 분류한 결과 **100%가 계획 한계**였고 좌석 제한이 단독 원인이 된
경우는 **0건**이었습니다 — `insertStops`는 좌석 predicate(hand-written이든 온톨로지든)를
부르기 전에 horizon에서 먼저 끊기기 때문입니다. 그래서 UI 필드 이름이 `No room`이 아니라
**Over-committed**입니다. 두 제약은 별개이며, 이 시뮬레이터의 상한은 후자입니다. ^[inferred]

수용할 수 없는 버스를 선택한 건수(좌석이든 계획 한계든):

| | 규칙 없음 | 규칙 반영 | 필터 미적용 | greedy |
|---|---|---|---|---|
| CJet | 51건 | **0건** | 4건 | 48건 |
| Jev | 47건 | **0건** | 0건 | 48건 |

### 규칙 계층 ↔ 온톨로지 대응

**이 계층 구조는 온톨로지 계층과 대응하며, TBox/ABox reasoner로 구현되어 있습니다.**

| 규칙 계층 | 온톨로지 대응 | 성격 |
|---|---|---|
| Tier 0 — 하드 제약 | **OWL cardinality / restriction axiom** | 일관성 검증 — 위반 시 추론 불가 |
| Tier 1 — 파생 수치 | **reasoner 또는 SPARQL 질의 결과** | 결정론적 계산 |
| Tier 2 — 정책 | 온톨로지에 내장된 guidance | |
| state | **ABox** (개인에 대한 assertion) | |
| question | **질의** | |

가장 중요한 대응은 **Tier 0 ↔ OWL axiom**입니다. 좌석 제약은 사실상
`Bus hasAtMost 4 Passenger`라는 cardinality axiom입니다. 구조적 이점:

1. **규칙의 출처가 코드가 아니라 형식적 규준이 됩니다** — reasoner가 건전성을 검증합니다.
2. **"위반 불가"가 단일 규칙이 아니라 전체 일관성 검사로 확장됩니다** — 규칙 집합끼리
   모순되는 경우를 사전에 잡습니다(`checkTBox()`).
3. **파생 수치가 선언적으로 파생됩니다** — 단, Tier 1의 수치는 추론이 아니라 **확장
   함수**입니다: 거리는 최적화이지 논리가 아니며, reasoner는 등록된 함수를 이름으로
   호출하고 시뮬레이터를 import하지 않습니다(순환 방지). **구조와 제약은 온톨로지가,
   수는 그 온톨로지가 부를 수 있는 함수가 줍니다.**

측정 (`node eval_rules.mjs --n 200 --seed 7`):

| 측정 | 결과 |
|---|---|
| TBox 자기 검사 | satisfiable |
| hand-written 대비 **불일치** | **600쌍 중 0건** (단, 아래 단서) |
| `ontology` arm regret | 6.43 blk — `rules` arm과 **완전히 동일** |
| 추론 비용 | 경로 검사당 **1.19 µs** (판단 136,000 µs의 0.0009%) |

`ontology` arm이 `rules` arm과 숫자 단위로 같다는 것이 핵심입니다 — 모델 입력은 동일하고
**제약 계층만 다르므로**, 결과가 같다는 것은 TBox가 hand-written 로직을 정확히 재현한다는
뜻입니다. 대응표가 아니라 측정입니다.

**600쌍이라는 숫자는 두 기록이 서로 다르게 집계합니다 — 그 불일치 자체가 정보입니다.**
본문 기록은 "600쌍 중 221쌍이 양쪽 predicate 호출 없이 거절 → 좌석 검사는 379쌍에서
발동"이고, 이 부록의 원 문서 기록은 "24,000회 호출 중 71.8%가 horizon에서 먼저 끊김 →
발동 약 170쌍"입니다. predicate가 실제로 발동하는 비율은 시나리오 구성(multi-pickup
모드 등)에 따라 크게 달라진다는 뜻이고, 그래서 **등가성의 주된 증거는 어느 하네스
숫자도 아니라 `test_ontology.mjs`의 후보 경로 5,000개 이상 직접 비교 테스트**입니다.
이 테스트가 통과합니다.

브라우저(`static/fleet.html` 좌측 Ontology 패널)에서 TBox를 실시간 편집할 수 있고,
편집은 같은 시드로 시뮬레이터를 재시작합니다 — 도중에 제한을 바꾸면 옛 제한으로 계획된
경로가 남기 때문입니다. 브라우저에서 정확한 등가성은 증명할 수 없습니다(데모가
wall-clock으로 돌아 두 실행의 수요 수가 다릅니다). 결정적 검사는 하네스와 테스트가
하고, 브라우저는 "두 경로가 똑같이 동작한다"를 보여줍니다.

### 실패 모드와 운영 특성 — 감사 가능한 결정 엔진의 성질

- **실패가 유계(bounded)입니다.** 자기회귀 생성 모델과 달리 라벨을 **선택**합니다.
  잘못되면 틀린 버스가 지정될 뿐, 존재하지 않는 버스를 지어내지 않습니다. 실패가 항상
  감사 가능한 형태로 나타나는 결정적 안전 속성입니다.
- **확률 분포를 반환합니다.** 라벨만이 아니라 옵션별 확률을 반환하므로, 신뢰도 게이트로
  판단을 보류하거나, 상위 k개를 유지해 사람에게 에스컬레이션하거나, 보정된 분포로
  임계값을 설정할 수 있습니다.
- **감사 추적.** 모든 판단 응답에 라우팅 근거(`routing.reason`), 옵션별 확률, 신뢰도,
  지연, 사용된 입력이 붙습니다. "왜 이 버스인가"에 항상 답할 수 있습니다.
- **결정론 · 재현성.** 시드 고정으로 동일 입력 → 동일 출력. 시뮬레이션 불변식 테스트와
  API 테스트가 회귀를 잡습니다.
- **데이터 비외주.** 입력 state가 외부로 나가지 않습니다 — 로컬 실행이 충족하는 조건.
- **우아한 성능 저하.** 신뢰도가 임계값 아래로 떨어지면 결정론적 비용 모델로 자동
  위임하고, 어느 판단이 모델 결정이고 어느 판단이 위임인지 **decided/deferred
  카운터로 노출**합니다. 위임을 숨기지 않는 설계입니다.
- **벤더 비종속.** 모델 교체 시 주입 코드는 그대로 유지됩니다(Jev로 실증). 하드 제약은
  텍스트가 아니라 코드에 있으므로, **어떤 모델로 교체해도 제약은 강제됩니다.**

### 이 부록의 한계 (본문에 없는 항목)

- **규칙이 판단을 지배합니다.** 파생 수치를 넣으면 모델이 그 수치를 따릅니다. 규칙이
  잘못되면 결과도 잘못된다는 뜻이고, 규칙의 정확성이 시스템의 상한입니다. Tier 0의
  코드 강제가 이 위험을 줄입니다.
- **온톨로지 ↔ 규칙 계층 대응의 일부는 설계 서술입니다.** 구현·측정된 것은 위 표 아래
  측정 절이고, "SPARQL 질의 결과" 같은 항목은 실제 배치에서의 형태를 말하는 것이지 이
  저장소의 구현이 아닙니다.
- **자연어 선호(soft preference)의 경계는 불명확합니다.** 강제 제약은 코드로 처리하지만
  "우선 이쪽" 류는 모델 판단에 의존하며, 모델별로 필터의 가치가 달랐습니다(laya는
  필터 없이 4건 위반, Jev는 0건 — Jev에서 필터는 오히려 regret을 0.98 → 1.54로
  악화시켰습니다).

### 부록 B-1. 온톨로지 파인튜닝의 위험성 — 채택하지 않은 경로의 기록

**온톨로지 경로를 파인튜닝과 결합하는 구성은 위험합니다.** 가장 큰 위험은 퍼포먼스
하락이 아니라 **구조적 후퇴**입니다.

**가장 날카로운 위험: 닫힌 고리.** 온톨로지가 라벨을 생성하고 그 라벨로 파인튜닝한 뒤
추론 시 온톨로지 파생 수치를 다시 모델에게 주면, 모델이 하는 일은 **reasoner를 흉내
내는 것**이 됩니다 — 이미 보유한 함수를 421M 파라미터로 재현하는 셈이며, 동시에 세
가지를 잃습니다:

1. **reasoner의 오류가 학습 불가능한 오답이 됩니다.** 모델이 reasoner와 정확히
   일치하도록 학습되므로, 틀린 reasoner 출력은 탐지할 수 없습니다 — 정답처럼 보입니다.
2. **온톨로지 변경마다 재학습이 필요합니다.** 지금은 코드 한 줄입니다. 파인튜닝하면
   선언적 규칙이 가중치 아티팩트로 굳어집니다.
3. **본 설계의 핵심 장점이 사라집니다.** 규칙을 추론 시점에 편집 가능하게 둔 것이
   벤더 비종속과 파인튜닝 회피의 근거였습니다.

**직접 측정하지 않은 성능 하락 경로 (추론이지만 메커니즘은 구체적):**

- **입력 분포가 좁아지면 본래 능력을 잃습니다.** `english` 체크포인트는 산문을 읽도록
  학습됐습니다. 온톨로지 파생 구조 수치로 파인튜닝하면 산문 읽기를 잃을 수 있고, 그
  순간 Tier 3(state의 `rules:` 줄)이 무력화됩니다.
- **온톨로지 변경에 더 취약해집니다.** 파인튜닝된 모델은 특정 criteria 레이아웃에
  적응하고, criteria를 바꾸는 것 자체가 out-of-distribution이 됩니다.
- **벤더 선택지가 사라집니다.** "주입이 System-1 모델 일반의 성질"이라는 결과는 정확히
  파인튜닝하지 않았기 때문에 얻은 것입니다.

**이 저장소의 관련 증거:** (1) 학습되지 않은 영역에서 모델은 아무것도 안 하는 것보다
나쁩니다 — no-rules laya(38.33)는 단순 최근접(34.60)보다 나쁩니다. (2) 신뢰도 게이트가
조용히 꺼지는 법을 이미 겪었습니다 — 임계값 0.45가 판단 100%를 기각했고 파이프라인은
멀쩡해 보였습니다.

**판단 기준** (파인튜닝 검토 시 순서대로): 원 모델이 트릭 베이스라인보다 낮은가(→ 모델
문제, 정당) / 규칙 주입만으로 트릭 베이스라인을 크게 넘었는가(→ 정보 문제, **여기서
멈춤**) / 잔여 구간이 온톨로지 미커버인가(→ 잔여 학습 정당). 이 프로젝트는 두 번째에서
멈추는 케이스였습니다 — 34.60 → 6.43은 주입만으로 달성됐습니다.

**파인튜닝이 정당한 경우**는 온톨로지가 결정론적 계산을 처리하고 남은 구간이 "이
상황에서는 A보다 B가 낫다"같이 수식으로 도출되지 않는 **학습이 필요한 판단**일 때입니다.
벤더도 이 관행을 합니다 — `typed-decisions` 체크포인트가 합성 워크플로 4종에
파인튜닝된 정확히 그것입니다.

**측정하지 않은 것:** 파인튜닝을 실행한 적이 없습니다. 위 하락 경로는 전부 추론입니다.
laya는 `proper_reward`와 `td_lambda_targets`를 노출하고 설정이 `rl_agent_config.json`이므로
실행은 가능합니다 — 시뮬레이터가 정확한 reward oracle이자 expert이므로, 온톨로지 파생
라벨로 파인튜닝한 체크포인트를 같은 하네스로 재측정하면 "어디까지 올라가며 어떤 부작용이
생기는가"에 숫자로 답할 수 있습니다. 이 기록은 측정하지 않은 경로를 채택하지 않은 이유를
남기는 것이며, 파인튜닝이 불가능하다는 주장이 아닙니다.
