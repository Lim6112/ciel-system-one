// The warehouse ontology panel — a domain adapter over the shared editor in
// ontology_core.js, the same structure as breakout_panel.js.
//
// WHAT DIFFERS FROM THE OTHER TWO DOMAINS, and it is the honest part: on fleet,
// turning the ontology on must not change any decision, and that holds by
// construction. Here it CANNOT hold, and neither can breakout's version of the
// claim. The rule removes an option from the menu — that IS a behaviour change,
// by design, because the whole point is that "wait" is not an answer in front of
// an obstacle that never moves.
//
// So the note below states the weaker, checkable thing instead: the declared rule
// and the hand-written predicate agree, over every blocking situation the sim
// produces. That is the claim this domain can actually keep, and
// `node --test test_warehouse_ontology.mjs` is what holds it.
//
// ENFORCEMENT IS NOT A SWITCH HERE. Unlike breakout (an "Ontology: on/off" button)
// or fleet (a panel checkbox), the warehouse rule is not optional: the option set
// is what the sample shown uses, and the geometry gate is the switch this page
// already has. A second, competing switch would suggest the rule can be off on a
// page whose whole subject is the rule.
//
// THE DOCK-ZONE AXIOM IS DELIBERATELY ABSENT. It is implemented in the simulator
// and measured as a net loss — roughly 40% of throughput — so it ships off, and
// declaring it here would advertise a rule the demo does not enforce. The finding
// is recorded in warehouse_sim.js above `DOCK_ZONE_RADIUS`.

import { TBOX, VOCABULARY, IMPLEMENTED_FORMS, aboxFor, permittedActions } from "./warehouse_ontology.js";
import { mountTBoxEditor } from "./ontology_core.js";

// Snapshot at construction time, before the panel is ever mounted, so a stray
// edit during a session is not what `reset` restores you to.
const PRISTINE = TBOX.map(a => ({ ...a, value: Array.isArray(a.value) ? [...a.value] : a.value }));

export function mountWarehousePanel({ root, getSim, onChange }) {
  return mountTBoxEditor({
    root,
    adapter: {
      tbox: TBOX,
      pristine: PRISTINE,
      implementedForms: IMPLEMENTED_FORMS,
      vocabulary: VOCABULARY,
      getSim,
      aboxFor,
      notes: sim => {
        if (!sim || !sim.robot) {
          return [{ ok: true, text: "no blocked robot yet — start the run" }];
        }
        const { robot, scan, stalled } = sim;
        if (!scan) {
          return [{ ok: true, text: `${robot.id} is not blocked, so every action is permitted` }];
        }
        const kind = scan.rack ? "static obstacle" : "another robot";
        const verdict = permittedActions(robot, scan, { stalled });
        return [
          {
            ok: true,
            text: `${robot.id} is blocked by a ${kind}; permitted: ` +
              `${verdict.options.join(" / ")}` +
              (stalled ? " (stalled, so wait is withdrawn)" : "") +
              (scan.rack ? " (static, so wait was never offered)" : ""),
          },
          ...verdict.violations.map(v => ({ ok: false, text: `${v.axiom}: ${v.detail}` })),
        ];
      },
      note: "The declared rule is what decides which directions this page offers: a static " +
            "obstacle never yields and a stalled robot has already been shown that waiting " +
            "fails, so \"wait\" is removed from the menu in both cases rather than " +
            "discouraged in the prompt. Removing an axiom here disables its check and the " +
            "panel says so — the geometry still knows the answer, the ontology has simply " +
            "stopped asserting it. This domain cannot claim that enforcement leaves " +
            "decisions unchanged, because removing an option IS the change; what it can " +
            "claim, and what `node --test test_warehouse_ontology.mjs` checks over every " +
            "blocking situation the simulator produces, is that the declared rule and the " +
            "hand-written predicate it replaced agree. That agreement is the point: the two " +
            "halves of this rule once lived in separate functions, drifted, and pinned " +
            "robots in a permanent head-on standoff.",
      onChange,
    },
  });
}

/** The permitted action set, as the decision loop sees it. */
export function permittedFor(robot, scan, stalled) {
  const verdict = permittedActions(robot, scan, { stalled });
  return {
    options: verdict.options.length ? verdict.options : ["left", "stay", "right"],
    // The election the reasoner actually applied, so the caller's displayed gold is
    // computed with the same role rather than a second, independent guess. See the
    // note on `derived.headOnRole` in warehouse_ontology.js.
    role: verdict.headOnRole ?? null,
  };
}
