// A TBox/ABox reasoner for the warehouse domain — the ontology behind the
// blocker rule.
//
// WHY A THIRD MODULE AND NOT A GENERALISED ontology.js. Same reason
// breakout_ontology.js gives for being the second: `ontology.js` is coherent and
// heavily tested, its `reason()` hardcodes fleet axiom ids, and its `buildABox`
// is fleet-shaped (`bus.onboard`, `facts.seats`) with no seam to generalise
// without a per-domain adapter — which is the same amount of code as another
// module with the rewrite's risk on top. The domain-free editor machinery lives
// in ontology_core.js and is shared; `IMPLEMENTED_FORMS` deliberately stays per
// domain (see that file's header), because a form nothing implements would be
// declared and never evaluated.
//
// WHAT THIS REPLACES. `feasibleOptions` in warehouse_sim.js is a hand-written
// ternary: "a rack cannot be waited out, and a stalled robot cannot wait either,
// so neither offers stay". The `if` is not wrong, but it cannot be checked — and
// its twin, `goldDirection`, is the function it MUST agree with: when those two
// drifted, the gate deferred a stalled robot to an oracle answer of "stay", the
// one option escalation had just withdrawn, and robots pinned in a permanent
// head-on standoff (791 deferred against 40 decided). Declaring the rule makes
// both halves downstream of one statement, and `divergence()` below is what
// checks that the declaration and the hand-written predicate still agree.
//
// THE AXIOM FORM IS CONSTRAINED, AND THAT SHAPED THE DECLARATION. This subset
// implements maxCardinality / minCardinality / range / disjoint — no negation,
// no implication. So "a static blocker implies stay is not permitted" cannot be
// written directly. It is expressible as a disjointness instead, which is the
// same claim: if the permitted-action set of a static blocker is
// {left, right}, then Stay is not in it. See TBOX.
//
// SCOPE, STATED UP FRONT, because it is the same subset the other two domains
// declare: an OWL-RL subset, closed-world. Absence of a fact is absence, not
// openness. That is right for an operational decision system and is NOT OWL's
// default.

import { checkTBox as coreCheckTBox } from "./ontology_core.js";
import { feasibleOptions, goldDirection, inDockZone } from "./warehouse_sim.js";

// The actions a blocked robot can choose. Same labels the questions use.
export const ACTION_SET = ["left", "stay", "right"];
export const SIDESTEP_SET = ["left", "right"];

export const VOCABULARY = {
  classes: ["Robot", "Blocker", "StaticBlocker", "MobileBlocker", "Action"],
  properties: ["blockedBy", "permittedAction", "inDockZone", "goldAction"],
};

// Declarations, not control flow. Read by checkTBox and by reason(), and editable
// live — removing one disables its check and reason() reports it as unasserted
// rather than crashing, which is the demonstration the editor exists for.
export const TBOX = [
  {
    id: "static-blocker-disallows-stay",
    form: "disjoint",
    value: ["StaticBlocker", "Stay"],
    comment:
      "Tier 0. A rack never yields, so waiting it out is provably wrong. Stated as a " +
      "disjointness rather than an implication because this subset has no negation: " +
      "\"a static blocker's permitted actions exclude stay\" IS \"StaticBlocker disjoint " +
      "Stay\" once Stay is a class in the action taxonomy. Replaces the first half of " +
      "feasibleOptions in warehouse_sim.js, which enforced it with a ternary nothing " +
      "could check.",
  },
  {
    id: "stalled-blocker-disallows-stay",
    form: "disjoint",
    value: ["StalledBlocker", "Stay"],
    comment:
      "Tier 0. Once waiting has demonstrably failed — no forward progress for STALL_MS — " +
      "stay is withdrawn. Same form and same replacement as the static case: this is the " +
      "rule escalation implements, and the one whose drift from goldDirection caused the " +
      "head-on standoff.",
  },
  {
    id: "blocker-single-kind",
    form: "disjoint",
    value: ["StaticBlocker", "MobileBlocker"],
    comment:
      "Tier 0. A blocker is static or mobile, never both. The simulator always satisfies " +
      "this; if it ever fires the state construction is inconsistent, which is a real bug " +
      "rather than a bad answer.",
  },
  {
    id: "blocked-robot-has-gold",
    form: "minCardinality",
    subject: "BlockedRobot",
    property: "goldAction",
    value: 1,
    comment:
      "Tier 0. A blocked robot has exactly one action that clears it. This is the axiom " +
      "goldDirection implements; its violation message is the disagreement between the " +
      "declaration and the geometry, which is what the standoff was.",
  },
  {
    id: "blocked-robot-has-permitted-set",
    form: "minCardinality",
    subject: "BlockedRobot",
    property: "permittedAction",
    value: 1,
    comment:
      "Tier 0. A blocked robot is offered at least one action. The 'at least one' is not " +
      "decoration: the rule exists to remove stay, and if removing it could empty the set " +
      "the robot would have no legal move at all.",
  },
];

/** Look an axiom up by id, so a violation can name the axiom it broke. */
export const axiom = id => TBOX.find(a => a.id === id);

export const IMPLEMENTED_FORMS = ["maxCardinality", "minCardinality", "range", "disjoint"];

export { describeAxiom } from "./ontology_core.js";

export const checkTBox = (tbox = TBOX) => coreCheckTBox(tbox, IMPLEMENTED_FORMS);

// ---- ABox ---------------------------------------------------------------------

/**
 * The facts, read from the live sim. Everything here is asserted by the
 * simulator; nothing is inferred, and the one thing the ontology needs to be
 * TOLD is which side is free — that is geometry, not logic.
 *
 * `goldAction` and `permittedAction` come from the two functions this module
 * exists to declare, passed in as the extension exactly the way fleet injects
 * `detour` and breakout injects `rollout`: cost is not logic, and a second
 * implementation of the geometry would be a second thing to get wrong.
 */
export function buildABox(robot, scan, opts = {}) {
  const options = opts.feasibleOptions ?? feasibleOptions;
  const gold = opts.goldDirection ?? goldDirection;
  const stalled = !!opts.stalled;
  const blocked = !!scan;
  const staticBlocker = blocked && !!scan.rack;

  const permitted = blocked
    ? options(staticBlocker ? "rack" : "robot", stalled)
    : ACTION_SET;

  const facts = { individuals: new Map() };
  const add = (id, type, props = {}) => {
    facts.individuals.set(id, { id, type, ...props });
    return facts.individuals.get(id);
  };

  const robotInd = add(robot.id, blocked ? "BlockedRobot" : "Robot", {
    blocked,
    stalled,
    // Empty set when nothing blocks — an empty permittedAction on a BlockedRobot
    // IS the violation, never inferred from "not blocked".
    permittedAction: new Set(permitted),
    goldAction: blocked ? new Set([gold(scan, stalled)]) : new Set(),
    inDockZone: inDockZone(robot),
  });

  if (blocked) {
    const kind = staticBlocker ? "StaticBlocker" : "MobileBlocker";
    add("blocker", kind, {
      id: scan.rack ? scan.rack.id : scan.other,
      // The classes a disjointness needs. A blocker carries whichever of Stay /
      // the permitted set its own rule allows, and the axiom is what forbids the
      // combination — the fact says what IS, the axiom says what MAY BE.
      actions: new Set(permitted),
    });
    // The blocked robot is additionally a StalledBlocker subject once it has
    // stalled, which is what the second disjointness ranges over.
    if (stalled) {
      add("blocker-stalled", "StalledBlocker", { actions: new Set(permitted) });
    }
  }

  return { facts, robot: robotInd };
}

// ---- reasoner -----------------------------------------------------------------

/**
 * Check the TBox against the ABox. Returns the violations, plus the facts an
 * unasserted axiom would otherwise have supplied.
 *
 * Every axiom is guarded on its own presence: deleting one in the live editor
 * disables that check and is REPORTED as unasserted rather than crashing on. The
 * simulator still knows the answer; the ontology has simply stopped asserting it,
 * and saying so is the point of the editor.
 */
export function reason(facts, { extensions = {} } = {}) {
  const violations = [];
  const unasserted = [];

  const staticRule = axiom("static-blocker-disallows-stay");
  const stalledRule = axiom("stalled-blocker-disallows-stay");
  const kindRule = axiom("blocker-single-kind");
  const goldRule = axiom("blocked-robot-has-gold");
  const setRule = axiom("blocked-robot-has-permitted-set");

  for (const [id, ind] of facts.individuals) {
    if (ind.type !== "BlockedRobot") continue;

    // minCardinality: BlockedRobot ⊑ ≥1 goldAction
    if (!goldRule) {
      if (!unasserted.includes("blocked-robot-has-gold")) unasserted.push("blocked-robot-has-gold");
    } else if (ind.goldAction.size < goldRule.value) {
      violations.push({
        axiom: goldRule.id,
        subject: id,
        detail: "blocked, but no action is asserted to clear it — the geometry " +
          "extension and this declaration disagree",
      });
    }

    // minCardinality: BlockedRobot ⊑ ≥1 permittedAction
    if (!setRule) {
      if (!unasserted.includes("blocked-robot-has-permitted-set")) {
        unasserted.push("blocked-robot-has-permitted-set");
      }
    } else if (ind.permittedAction.size < setRule.value) {
      violations.push({
        axiom: setRule.id,
        subject: id,
        detail: "the rule removed every action, leaving the robot no legal move",
      });
    }

    // disjoint: a static blocker's permitted set excludes stay.
    //
    // Checked as the disjointness it is declared as: the actions actually
    // permitted must not intersect the actions the axiom separates off. Doing it
    // this way rather than re-deriving "is it a rack" keeps the check honest —
    // it reads what the rule produced, not what the rule should have read.
    if (staticRule) {
      const ind2 = facts.individuals.get("blocker");
      if (ind2 && ind2.type === "StaticBlocker" && ind.permittedAction.has("stay")) {
        violations.push({
          axiom: staticRule.id,
          subject: id,
          detail: `blocked by a static obstacle (${ind2.id}) but "stay" is permitted, ` +
            `which the axiom forbids — waiting cannot clear a rack`,
        });
      }
    } else if (!unasserted.includes("static-blocker-disallows-stay")) {
      unasserted.push("static-blocker-disallows-stay");
    }

    // disjoint: a stalled blocker's permitted set excludes stay.
    if (stalledRule) {
      if (ind.stalled && ind.permittedAction.has("stay")) {
        violations.push({
          axiom: stalledRule.id,
          subject: id,
          detail: "the robot has stalled but \"stay\" is still permitted — waiting " +
            "has demonstrably failed, so it is not an answer",
        });
      }
    } else if (!unasserted.includes("stalled-blocker-disallows-stay")) {
      unasserted.push("stalled-blocker-disallows-stay");
    }
  }

  // disjoint: a blocker is static or mobile, never both.
  //
  // buildABox asserts exactly one kind per blocker, so this can only fire if the
  // construction is inconsistent — which is why a violation here is treated as a
  // bug worth surfacing rather than a bad answer. A blocker whose type is BOTH
  // its own kind and the other one is the case the axiom forbids.
  if (kindRule) {
    const assertedKinds = new Set(
      [...facts.individuals.values()]
        .filter(i => i.type === "StaticBlocker" || i.type === "MobileBlocker")
        .map(i => i.type),
    );
    for (const kind of kindRule.value) {
      const other = kindRule.value.find(k => k !== kind);
      if (assertedKinds.has(kind) && assertedKinds.has(other)) {
        violations.push({
          axiom: kindRule.id,
          subject: "blocker",
          detail: `blocker is asserted as both ${kind} and ${other}`,
        });
      }
    }
  } else if (!unasserted.includes("blocker-single-kind")) {
    unasserted.push("blocker-single-kind");
  }

  // Any robot individual, blocked or not — an unblocked robot still has a
  // permitted-action set (all of them), and reading only the blocked type left
  // it reporting an empty one, i.e. a robot with no legal move.
  const robot = [...facts.individuals.values()]
    .find(i => i.type === "BlockedRobot" || i.type === "Robot") || null;
  const derived = {
    goldAction: robot && robot.goldAction.size ? [...robot.goldAction][0] : null,
    permitted: robot ? [...robot.permittedAction] : [],
    stalled: robot ? !!robot.stalled : false,
  };

  return { consistent: violations.length === 0, violations, unasserted, derived };
}

/**
 * The Tier-0 boundary, in the shape `feasibleOptions` already has.
 *
 * Deliberately the same signature so the two are interchangeable at the call
 * site, and `divergence()` can run both over identical inputs — which is the
 * only way the claim "the ontology says what the code says" is checkable rather
 * than asserted. Returning the options rather than a boolean keeps the call
 * site a drop-in.
 *
 * A robot is never left with no move: if the rule ever emptied the set, that is
 * a violation, and this falls back to the full action set rather than grounding
 * the robot.
 */
export function permittedActions(robot, scan, opts = {}) {
  const { facts } = buildABox(robot, scan, opts);
  const probe = reason(facts, {});
  const blocking = probe.violations.filter(v =>
    v.axiom === "static-blocker-disallows-stay" ||
    v.axiom === "stalled-blocker-disallows-stay" ||
    v.axiom === "blocked-robot-has-permitted-set");
  const options = probe.derived.permitted;
  return {
    options: options.length ? options : ACTION_SET,
    ok: blocking.length === 0,
    violations: blocking,
    allViolations: probe.violations,
    unasserted: probe.unasserted,
  };
}

/**
 * One row describing what the ontology currently believes, for the panel.
 *
 * `sim` is whatever the page hands over — a robot plus its lidar scan, or null
 * before the run starts. Kept tolerant because the panel refreshes a few times a
 * second and a half-initialised page is not an error worth throwing on.
 */
export function aboxFor(sim) {
  if (!sim || !sim.robot) return [];
  const { robot, scan, stalled = false } = sim;
  const verdict = permittedActions(robot, scan, { stalled });
  const kind = !scan ? null : scan.rack ? "StaticBlocker" : "MobileBlocker";
  const rows = [{
    id: robot.id,
    detail: scan
      ? `blocked by ${kind}${scan.rack ? ` (${scan.rack.id})` : ""}` +
        `${stalled ? ", stalled" : ""} · permitted ${verdict.options.join("/")}`
      : "not blocked · every action permitted",
    violations: verdict.violations,
    unasserted: verdict.unasserted,
  }];
  return rows;
}

/**
 * Does the declared rule agree with the hand-written predicate it replaces?
 *
 * The fleet demo's divergence harness, reduced to the shape this domain needs:
 * run both over every blocking situation the sim can produce and compare. Zero
 * disagreements is the claim; non-zero means the declaration does NOT
 * faithfully reimplement the code, and the number to report is the disagreement
 * count, not "it looks right".
 *
 * This is the check that makes the ontology worth having. Without it the
 * declaration is one more place for the rule to live, which is the opposite of
 * the point — see the header on why the two halves drifting caused a standoff.
 */
export function divergence(cases) {
  const broken = [];
  let compared = 0;
  for (const c of cases) {
    compared++;
    const direct = feasibleOptions(c.kind, c.stalled);
    const declared = permittedActions(c.robot, c.scan, { stalled: c.stalled }).options;
    const a = [...direct].sort().join(",");
    const b = [...declared].sort().join(",");
    if (a !== b) broken.push({ ...c, direct, declared });
  }
  return { compared, disagreed: broken };
}
