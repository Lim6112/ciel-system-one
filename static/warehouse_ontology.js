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
import {
  feasibleOptions,
  goldDirection,
  headOnRole,
  inDockZone,
} from "./warehouse_sim.js";

// The actions a blocked robot can choose. Same labels the questions use.
export const ACTION_SET = ["left", "stay", "right"];
export const SIDESTEP_SET = ["left", "right"];

// `backward` is NOT in ACTION_SET, and that is the point of the rule below rather
// than an omission. It is a member of the action taxonomy -- a robot can do it, so it
// is a class here -- but it is not part of the everyday choice, because reversing
// gives up route the robot has already driven. It appears only where the other three
// have been tried and failed, which is what the deadlock axiom states.
export const ESCAPE_SET = ["backward"];
// NO `stay`. A deadlocked robot is stalled by construction -- DEADLOCK_MS (8 s) is
// longer than STALL_MS (4 s), so anything that can be deadlocked has already had
// `stay` withdrawn -- and leaving it in produced a set that violated the
// stalled-blocker axiom on every single deadlocked state. The first version did
// include it, and the reason it looked right is that `stay` reads like a harmless
// extra option; it is not, it is the option the whole escalation exists to remove.
export const DEADLOCK_SET = ["backward", "left", "right"];

export const VOCABULARY = {
  classes: ["Robot", "Blocker", "StaticBlocker", "MobileBlocker", "Action",
            "DeadlockedRobot", "Backward"],
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
    id: "one-side-advances",
    form: "disjoint",
    value: ["AdvancingRobot", "YieldingRobot"],
    comment:
      "Tier 0. In a dead-ahead meeting exactly one robot advances and the other yields. " +
      "The dead band has no side to read (the lateral offset is zero), so without this " +
      "both robots were told to wait and the pair burned STALL_MS 4 s doing nothing -- " +
      "measured as ~4.4 s head-on resolution against ~0.22 s for every other block. The " +
      "election is by robot id: both robots compare their own id against the other's, so " +
      "exactly one concludes \"advance\" and neither needs shared state. A " +
      "first-detected (blockedSince) election was measured worse and replaced -- it is " +
      "each robot's own history rather than a property of the meeting, so a stale value " +
      "elects the wrong side and flipped the role mid-encounter in ~6% of cases (p90 " +
      "1,376 ms and 704 encounters over 4 s, against 672 ms and zero). Stated as a " +
      "disjointness because this subset has no implication -- \"one advances\" IS " +
      "\"AdvancingRobot disjoint YieldingRobot\".",
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
  {
    id: "deadlocked-robot-may-reverse",
    form: "range",
    subject: "DeadlockedRobot",
    property: "permittedAction",
    value: ["backward", "left", "right"],
    // The wording the state carries, kept ON the axiom rather than in the simulator.
    // `buildState` writes this into the doc it sends the model, so deleting this
    // axiom removes the sentence as well as the permission — otherwise the model
    // would keep being told the robot is wedged while the ontology had stopped
    // saying so, which is one rule living in two places.
    //
    // IT SAYS WHAT IS WRONG, NOT WHAT TO DO. The leaf reports that the robot is
    // wedged and for how long; it does not name an action. This document describes a
    // scene, and the answer has to come from the criteria: naming the move here would
    // make the model's confidence look like judgement when it was dictation, and
    // would make the page's agreement counter measure the prompt instead of the head.
    //
    // An earlier version read "wedged, no progress for 8s; sidestepping has not freed
    // me", and it was shortened on the theory that mentioning sidestepping pulled the
    // head toward the `left`/`right` criteria, which also say "sidestep". THAT THEORY
    // IS NOT SUPPORTED, and neither is any strong claim about this leaf's wording:
    // measured against the live service, a short phrasing and a long explanatory one
    // both answer `backward` at the SAME confidence (0.94), so the text here is not
    // what moves the answer. The shorter sentence is kept because it is shorter.
    //
    // What DOES decide it is the sub-path, and -- more sharply -- whether
    // `obstacle_side` is present. Both are recorded on this axiom because a `left`
    // here is otherwise very hard to attribute.
    stateLeaf: { deadlock: "wedged, no progress for 8s" },
    comment:
      "Tier 1. A deadlocked robot -- stuck AND blocked for DEADLOCK_MS (8 s) -- is " +
      "permitted to reverse, which no other state permits. This is the only axiom here " +
      "that ADDS an action rather than removing one, so it is stated as a range rather " +
      "than a disjointness: the other rules say what a robot may not do, and this one " +
      "says what only this situation allows. `backward` is deliberately absent from " +
      "ACTION_SET so that it cannot appear in the everyday choice; the range is what " +
      "admits it. Reversing is not free -- it gives up route already driven -- which is " +
      "why it is gated on the two clocks rather than offered whenever a robot is " +
      "blocked. " +
      // The measured failure mode, recorded where someone debugging a wrong answer will
      // actually be reading. It is NOT the observation text, which was the first
      // suspect and a dead end: both wordings answer `backward`.
      "A `left` HERE HAS TWO CAUSES AND THE STATE TEXT IS NEITHER OF THEM. Measured " +
      "against the live service on real wedged states. " +
      "FIRST, `obstacle_side`: present, this head answers `left` at confidence 1.00 " +
      "with `backward` at 0.00; absent, the same state answers `backward` at 0.90. It " +
      "is the single field that decides the answer -- removing any other one changes " +
      "nothing -- and buildState therefore withholds it, and skips the head-on side " +
      "rewrite, whenever `deadlocked` is set. " +
      "SECOND, the sub-path: warehouse gives backward 0.81/0.92/0.99/0.77 over four " +
      "state shapes, the service default 0.58/0.69/0.59/0.49, and the `typed` " +
      "sub-path left or right with backward never above 0.33. The `typed` head has no " +
      "warehouse training, so it cannot answer the question at all -- and since it " +
      "always returns one of the options it was sent, that failure looks like a " +
      "considered `left`. `ciel_typed_v1` in the Model dropdown reaches it. " +
      "So if a deadlock answers `left`: check the dropdown, then check that " +
      "`obstacle_side` is absent -- and do not start with the observation text, which " +
      "was measured to make no difference (0.94 either way).",
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
  const roleOf = opts.headOnRole ?? headOnRole;

  const stalled = !!opts.stalled;
  const blocked = !!scan;
  const staticBlocker = blocked && !!scan.rack;
  // NOT gated on `blocked`, and that is the point rather than an oversight. A robot
  // can be wedged with NOTHING in its lidar cone -- the hard collision check stops it
  // at MIN_SEP while the blocker sits off to the side, which measured at 53.6% of all
  // stuck-in-contact ticks. Gating this on `blocked` made the escape unreachable in
  // the majority of the stalls it was added for, and the failure was quiet: the state
  // was typed `Robot` with nothing to report, so nothing asked a question and no
  // action could be chosen.
  const deadlocked = !!opts.deadlocked;

  // The head-on role, so both halves of the rule see the same election. Computed
  // here rather than passed in, because the whole point is that the two robots
  // reach it independently from `blockedSince`.
  const role = blocked && !staticBlocker ? roleOf(robot, scan) : null;

  // The escape set is READ OFF THE AXIOM, and this is the one place in the file where
  // an axiom is the SOURCE of behaviour rather than a check on it. Every other rule
  // here works the other way round: `feasibleOptions` computes the set and the axiom
  // verifies the computation, so deleting one stops the check but leaves the options
  // alone. That is fine for a rule whose implementation is a ternary; it is not fine
  // for this one, where the whole capability is "the ontology admits it". A literal
  // `DEADLOCK_SET` here meant the axiom's `value` was only ever compared against the
  // set the code already used, so removing the declaration in the live editor left
  // the escape fully working and the declaration was decoration.
  //
  // WITH NO AXIOM THERE IS NO ESCAPE, so the robot falls back to the ordinary set
  // rather than to a hardcoded escape. Crashing is still not an option and `reason()`
  // still reports the axiom as unasserted -- but the capability goes with the
  // declaration, because that is what makes the declaration load-bearing.
  const escapeRule = axiom("deadlocked-robot-may-reverse");
  const escapeSet = (escapeRule && Array.isArray(escapeRule.value) && escapeRule.value.length)
    ? escapeRule.value : null;

  const ordinary = blocked
    ? options(staticBlocker ? "rack" : "robot", stalled, role)
    : ACTION_SET;
  const permitted = (deadlocked && escapeSet) ? escapeSet : ordinary;

  const facts = { individuals: new Map() };
  const add = (id, type, props = {}) => {
    facts.individuals.set(id, { id, type, ...props });
    return facts.individuals.get(id);
  };

  const robotInd = add(robot.id, deadlocked ? "DeadlockedRobot"
    : blocked ? "BlockedRobot" : "Robot", {
    blocked,
    stalled,
    deadlocked,
    // Empty set when nothing blocks — an empty permittedAction on a BlockedRobot
    // IS the violation, never inferred from "not blocked".
    permittedAction: new Set(permitted),
    goldAction: blocked ? new Set([gold(scan, stalled, role, deadlocked)]) : new Set(),
    // Recorded on the fact so the reasoner and the panel can both say which
    // robot the priority rule elected, rather than recomputing it.
    headOnRole: role,
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
  const escapeRule = axiom("deadlocked-robot-may-reverse");

  for (const [id, ind] of facts.individuals) {
    if (ind.type !== "BlockedRobot" && ind.type !== "DeadlockedRobot") continue;

    // range: DeadlockedRobot ⊑ permittedAction ∈ the escape set.
    //
    // Checked in both directions, because this axiom is the file's only one that
    // GRANTS an action rather than withdrawing one, and a grant has two ways to be
    // wrong. A deadlocked robot that cannot reverse is stuck by construction; a robot
    // that CAN reverse while not deadlocked has been handed a way to give up route it
    // should not give up, which is the failure this gating exists to prevent. The
    // second direction is also what keeps `backward` from leaking into the everyday
    // option set through some later edit.
    if (!escapeRule) {
      if (!unasserted.includes("deadlocked-robot-may-reverse")) {
        unasserted.push("deadlocked-robot-may-reverse");
      }
    } else {
      const allowed = new Set(escapeRule.value);
      for (const action of ind.permittedAction) {
        if (action !== "backward") continue;
        if (!ind.deadlocked) {
          violations.push({
            axiom: escapeRule.id,
            subject: id,
            detail: "permitted to reverse while not deadlocked — reversing gives up " +
              "route, so it is only legal once the other moves have failed",
          });
        }
      }
      if (ind.deadlocked && !ind.permittedAction.has("backward")) {
        violations.push({
          axiom: escapeRule.id,
          subject: id,
          detail: "deadlocked but not permitted to reverse, so it has no move that " +
            "changes the geometry — this is the deadlock the axiom exists to break",
        });
      }
      for (const action of ind.permittedAction) {
        if (!allowed.has(action)) {
          violations.push({
            axiom: escapeRule.id,
            subject: id,
            detail: `permitted action ${action} is outside the deadlock range`,
          });
        }
      }
    }

    // minCardinality: BlockedRobot ⊑ ≥1 goldAction
    //
    // This is a claim about GEOMETRY, so it only applies to a robot that has a
    // blocker to compute geometry from. A deadlocked robot can have nothing in its
    // cone at all, and asking "which action clears it" of a robot that can see
    // nothing is a question with no answer rather than a missing one — reporting it
    // as a violation would make the escape axiom's own state permanently inconsistent.
    if (!goldRule) {
      if (!unasserted.includes("blocked-robot-has-gold")) unasserted.push("blocked-robot-has-gold");
    } else if (ind.blocked && ind.goldAction.size < goldRule.value) {
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

  // disjoint: one side advances in a dead-ahead meeting, never both.
  //
  // The election lives in the simulator (headOnRole reads `blockedSince`), and
  // this checks its outcome rather than re-deriving it — the same split the
  // reachability axioms use, where the verdict is computed and the axiom declares
  // that it exists. A dead-ahead robot whose role came back `advance` must not
  // also be holding, and a `yield` must not be moving; if the two roles are ever
  // both asserted, or neither, the priority rule has stopped electing.
  const sideRule = axiom("one-side-advances");
  if (sideRule) {
    const roles = [...facts.individuals.values()]
      .filter(i => i.type === "BlockedRobot" && i.headOnRole)
      .map(i => i.headOnRole);
    const advancing = roles.filter(r => r === "advance").length;
    const yielding = roles.filter(r => r === "yield").length;
    // Both robots are in the same ABox only when a caller builds it for the pair;
    // for a single robot's facts this is satisfied trivially, which is why the
    // check is written as an impossibility rather than a counting requirement.
    if (advancing > 0 && yielding > 0) {
      // Not a violation: that is the healthy pair. Recorded for the panel.
    }
    const robotInd = [...facts.individuals.values()]
      .find(i => i.type === "BlockedRobot");
    if (robotInd && robotInd.headOnRole === "advance" &&
        robotInd.holding === true) {
      violations.push({
        axiom: sideRule.id, subject: robotInd.id,
        detail: "elected to advance but is holding — the priority rule and the " +
          "motion it is supposed to cause disagree",
      });
    }
    if (robotInd && robotInd.headOnRole === "yield" &&
        robotInd.permittedAction.has("stay") === false) {
      violations.push({
        axiom: sideRule.id, subject: robotInd.id,
        detail: "elected to yield but \"stay\" is not permitted — the robot has been " +
          "told to move out of a meeting it was elected to concede",
      });
    }
  } else if (!unasserted.includes("one-side-advances")) {
    unasserted.push("one-side-advances");
  }

  // Any robot individual, blocked or not — an unblocked robot still has a
  // permitted-action set (all of them), and reading only the blocked type left
  // it reporting an empty one, i.e. a robot with no legal move.
  //
  // DeadlockedRobot is listed too, and leaving it out was a real bug rather than an
  // oversight of style: a deadlocked robot is typed `DeadlockedRobot` INSTEAD of
  // `BlockedRobot`, so this lookup found nothing, `derived.permitted` came back EMPTY,
  // and `permittedActions` then fell back to the plain ACTION_SET. The escape was
  // admitted by buildABox and silently dropped by the reasoner, so the option set for
  // a wedged robot was exactly the one that had already failed. Caught by asserting
  // that a deadlocked robot's set contains `backward`.
  const robot = [...facts.individuals.values()]
    .find(i => i.type === "BlockedRobot" || i.type === "DeadlockedRobot" || i.type === "Robot")
    || null;
  const derived = {
    goldAction: robot && robot.goldAction.size ? [...robot.goldAction][0] : null,
    permitted: robot ? [...robot.permittedAction] : [],
    stalled: robot ? !!robot.stalled : false,
    // Surfaced so a caller asking the question uses the SAME election the reasoner
    // used, rather than recomputing it (or worse, not passing one at all). The page
    // did the latter: it built the option set through here, where the role is
    // applied, and then computed the displayed gold with no role — so the two halves
    // of this one rule were derived independently in the same function call, which
    // is exactly the arrangement that produced the head-on stand-off this election
    // was added to fix. Both paths happened to agree at HEAD (an advancing robot
    // stalls soon after, and the no-role branch then answers a sidestep too), which
    // is why it was invisible: a latent bug, not an active one.
    headOnRole: robot ? robot.headOnRole ?? null : null,
    // The deadlock flag, so a caller can build the right question from what the
    // reasoner already decided rather than re-deriving the 8 s gate.
    deadlocked: robot ? !!robot.deadlocked : false,
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
    v.axiom === "deadlocked-robot-may-reverse" ||
    v.axiom === "blocked-robot-has-permitted-set");
  const options = probe.derived.permitted;
  const escapeRule = axiom("deadlocked-robot-may-reverse");
  return {
    options: options.length ? options : ACTION_SET,
    // Carried out so a caller can compute its own gold with the election this
    // reasoner applied, instead of deriving the two halves of the rule separately.
    headOnRole: probe.derived.headOnRole ?? null,
    // And the deadlock flag, for the same reason: a caller that wants to build its
    // question must not have to re-derive the gating the ontology just applied.
    deadlocked: probe.derived.deadlocked ?? false,
    // THE SENTENCE THE STATE CARRIES, taken off the axiom that admitted the escape
    // rather than written into the simulator. Empty once the axiom is gone, so
    // deleting it in the live editor stops the observation from claiming the robot
    // is wedged at the same moment it stops permitting the escape -- see `stateLeaf`.
    stateLeaf: (probe.derived.deadlocked && escapeRule && escapeRule.stateLeaf)
      ? escapeRule.stateLeaf : {},
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
  const { robot, scan, stalled = false, deadlocked = false } = sim;
  const verdict = permittedActions(robot, scan, { stalled, deadlocked });
  const kind = !scan ? null : scan.rack ? "StaticBlocker" : "MobileBlocker";
  const rows = [{
    id: robot.id,
    detail: scan
      ? `blocked by ${kind}${scan.rack ? ` (${scan.rack.id})` : ""}` +
        `${stalled ? ", stalled" : ""}${deadlocked ? ", DEADLOCKED" : ""}` +
        ` · permitted ${verdict.options.join("/")}`
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
    // The head-on role is part of the rule now, so it has to be part of the
    // comparison. Omitting it was a real false negative: the ontology computed a
    // role and withheld "stay" for the advancing robot, the direct call did not,
    // and the two disagreed 2,476 times — the check was right and the harness was
    // comparing against an older signature.
    const role = c.kind === "robot" ? headOnRole(c.robot, c.scan) : null;
    const direct = feasibleOptions(c.kind, c.stalled, role);
    const declared = permittedActions(c.robot, c.scan, { stalled: c.stalled }).options;
    const a = [...direct].sort().join(",");
    const b = [...declared].sort().join(",");
    if (a !== b) broken.push({ ...c, role, direct, declared });
  }
  return { compared, disagreed: broken };
}
