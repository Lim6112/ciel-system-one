// The domain-free half of the ontology layer.
//
// WHY THIS FILE EXISTS. The TBox editor and the axiom forms are not fleet-specific —
// `checkTBox` never mentions Bus, `describeAxiom` only reads `a.form`, and `addAxiom`
// compares on shape. What IS domain-specific is which forms a given reasoner implements
// and where an axiom's value has a second home (`RULES.seats`). Those become parameters
// here, so the second demo gets an editor instead of a copy of one.
//
// WHAT DELIBERATELY DID NOT MOVE: `IMPLEMENTED_FORMS`. It looks like the most reusable
// thing in the file — it is the same four strings in both domains — and making it global
// would re-open exactly the hole commit a5149d0 closed. That commit made `checkTBox`
// REJECT an axiom whose form nothing implements, because an axiom that is declared but
// never evaluated reads as enforced and is inert. A shared list would let a form that
// fleet's `reason()` implements be accepted by breakout's `checkTBox` and then enforced
// by nothing. Each domain passes its own list, and the whole point survives.
//
// `ontology.js` re-exports everything moved out of it, so `test_ontology.mjs` and
// `eval_rules.mjs` import from the same place they always did.

// ---- axiom forms ---------------------------------------------------------------
//
// `disjoint` is implemented and checked everywhere but deliberately not OFFERED: it takes
// a list of classes rather than a single value, and the editor's value field is built for
// the other three. Offering it would let an editor author save an axiom with no way to
// read it back.

export const FORM_SPECS = [
  { value: "maxCardinality", label: "at most (maxCardinality)", numeric: true },
  { value: "minCardinality", label: "at least (minCardinality)", numeric: true },
  { value: "range", label: "values are of class (range)", numeric: false },
];

/** The forms a given domain's editor should offer, derived from what IT implements. */
export const offerableForms = implementedForms =>
  FORM_SPECS.filter(f => implementedForms.includes(f.value));

/** A readable rendering of one axiom, for the list and for the TBox self-check text. */
export function describeAxiom(a) {
  if (!a) return "(removed)";
  if (a.form === "maxCardinality") return `${a.subject} ⊑ ≤${a.value} ${a.property}`;
  if (a.form === "minCardinality") return `${a.subject} ⊑ ≥${a.value} ${a.property}`;
  if (a.form === "range") return `${a.subject}.${a.property} ⊑ ${a.value}`;
  if (a.form === "disjoint") return `${a.value.join(" ⊥ ")}`;
  return JSON.stringify(a);
}

// ---- TBox self-check -----------------------------------------------------------

/**
 * Is the axiom set itself coherent? Reports classes it cannot satisfy.
 *
 * Returns {satisfiable: bool, unsatisfiable: [{id, reason}]}.
 *
 * `implementedForms` is passed in rather than imported because it is per-domain — see the
 * header. Passing the wrong list here would make this function certify a TBox as coherent
 * while some of its axioms are never consulted, which is the one failure a self-check must
 * not have.
 */
export function checkTBox(tbox, implementedForms) {
  const unsatisfiable = [];

  // An axiom in a form nothing implements. Reported first, because it is the failure that
  // hides the others: an unimplemented axiom is not merely unevaluated, it makes the whole
  // set untrustworthy — `satisfiable: true` here would be a claim that the declared rules
  // hold, when some of them are never consulted. Reject rather than warn, so the live
  // editor cannot save a TBox that reads as enforced and is not.
  for (const a of tbox) {
    if (implementedForms.includes(a.form)) continue;
    unsatisfiable.push({
      id: a.id,
      reason: `form "${a.form}" is not implemented: no reasoner implements it, so this ` +
              `axiom would be declared and never evaluated. Add it to this domain's ` +
              `IMPLEMENTED_FORMS and give its reason() a case for it.`,
    });
  }

  // Contradictory cardinality on the same (subject, property) pair.
  const bySlot = new Map();
  for (const a of tbox) {
    if (a.form !== "maxCardinality" && a.form !== "minCardinality") continue;
    const slot = `${a.subject}.${a.property}`;
    if (!bySlot.has(slot)) bySlot.set(slot, []);
    bySlot.get(slot).push(a);
  }
  for (const [slot, group] of bySlot) {
    const maxes = group.filter(a => a.form === "maxCardinality");
    const mins = group.filter(a => a.form === "minCardinality");
    const hi = mins.length ? Math.max(...mins.map(a => a.value)) : -Infinity;
    const lo = maxes.length ? Math.min(...maxes.map(a => a.value)) : Infinity;
    if (hi > lo) {
      unsatisfiable.push({
        id: group.map(a => a.id).join(" + "),
        reason: `${slot}: requires at least ${hi} and at most ${lo} — no value satisfies both`,
      });
    }
  }

  // A class declared disjoint from itself, or from a class it must be a member of.
  for (const a of tbox) {
    if (a.form === "disjoint") {
      const seen = new Set();
      for (const c of a.value) {
        if (seen.has(c)) {
          unsatisfiable.push({ id: a.id, reason: `disjoint axiom lists ${c} twice` });
        }
        seen.add(c);
      }
    }
  }

  return { satisfiable: unsatisfiable.length === 0, unsatisfiable };
}

// ---- TBox mutation -------------------------------------------------------------

/**
 * Add an axiom to `tbox`. Returns {ok, id} or {ok: false, error} — it never throws,
 * because the caller is an editor input and the message is rendered next to the button.
 */
export function addAxiom(tbox, { form, subject, property, value }, implementedForms) {
  // Compared on form+subject+property, not on id: the shipped axioms carry hand-written
  // ids ("bus-capacity", not "maxCardinality-Bus-onboardPassenger"), so an id comparison
  // would let a semantic duplicate straight through.
  const clash = tbox.find(a =>
    a.form === form && a.subject === subject && a.property === property);
  if (clash) return { ok: false, error: `${clash.id} already asserts that` };
  const numeric = offerableForms(implementedForms).find(f => f.value === form)?.numeric;
  const parsed = numeric ? Math.floor(Number(value)) : value;
  if (parsed === undefined || parsed === null || (numeric && Number.isNaN(parsed))) {
    return { ok: false, error: "that value is not valid for this axiom form" };
  }
  const id = `${form}-${subject}-${property}`;
  tbox.push({ id, form, subject, property, value: parsed });
  return { ok: true, id };
}

export function removeAxiom(tbox, id) {
  const i = tbox.findIndex(a => a.id === id);
  if (i < 0) return { ok: false, error: "no such axiom" };
  tbox.splice(i, 1);
  return { ok: true, id };
}

/**
 * Restore `tbox` to `pristine`.
 *
 * `onReset` is where a domain restores the value it keeps OUTSIDE the TBox (fleet's
 * `RULES.seats`). Without it, an axiom edit that changed a configured value would leave
 * that value stuck after a reset — the TBox would say 4 and the simulator 2, and the
 * reasoner's own mismatch violation would fire on every decision.
 */
export function resetTBox(tbox, pristine, onReset) {
  tbox.splice(0, tbox.length, ...pristine.map(a => ({ ...a })));
  onReset?.();
  return { ok: true };
}

/** Consistency of the axiom set, plus whether the declared axioms are all present. */
export function tboxStatus(tbox, pristine, implementedForms) {
  const check = checkTBox(tbox, implementedForms);
  const missing = pristine.filter(p => !tbox.some(a => a.id === p.id))
    .map(p => ({ id: p.id, text: describeAxiom(p) }));
  return { ...check, missing };
}

// ---- DOM -----------------------------------------------------------------------

export const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  // `append(null)` inserts the string "null" rather than skipping, and the conditional
  // children below are all `cond ? node : null`.
  for (const k of kids) { if (k !== null && k !== undefined) node.append(k); }
  return node;
};

// ---- the editor ----------------------------------------------------------------

/**
 * Mount the TBox editor into `root`, for whichever domain `adapter` describes.
 *
 * `adapter` supplies everything domain-specific, so this function contains no fleet
 * vocabulary and no seat logic:
 *
 *   tbox              the live, mutable TBox array
 *   implementedForms  what THIS domain's reason() evaluates (see the header)
 *   vocabulary        {classes, properties} for the add-an-axiom dropdowns
 *   aboxFor(sim)      -> [{id, detail, violations, unasserted}] — the live belief view
 *   notes(sim)        -> [{ok, text}] — domain lines shown under the consistency check
 *                        (fleet's seat/TBox agreement check; breakout's decidable band)
 *   enforceChecked(sim) / setEnforce(sim, on)
 *                     OPTIONAL — omit both and no toggle is rendered, for domains
 *                     (breakout) whose enforcement switch lives outside the panel.
 *   note              the paragraph explaining what enforcement does and does not promise
 *   labels            OPTIONAL — overrides for this file's own chrome, so a page can be
 *                     presented in another language without forking the editor. Anything
 *                     omitted falls back to the English in DEFAULT_LABELS, which is what
 *                     every existing domain gets by not passing the field at all.
 *   onChange(what)    called after every mutation, and after the enforce toggle
 *
 * The editor is created ONCE and mutated in place, so `rerender()` never rebuilds the
 * `aboxHost` — that is what lets a timer repaint the ABox without stealing focus from a
 * half-typed value. `adapter.pristine` must be captured when the adapter is BUILT, not at
 * module load, or the second domain's snapshot captures the first one's TBox.
 */

/**
 * The editor's own chrome, in English.
 *
 * Per-domain so a page can be another language without a second editor (see `labels`
 * above). The domain's own words — axiom comments, ABox rows, the closing note — are not
 * here: those come from the adapter already, because only the domain knows them.
 */
export const DEFAULT_LABELS = {
  tboxHeading: n => `Ontology — TBox (${n} axioms)`,
  aboxHeading: "Ontology — ABox (what it currently believes)",
  consistencyOk: "consistency: satisfiable — the axiom set is coherent",
  consistencyBad: details => "UNSATISFIABLE: " + details,
  notAsserted: (ids, texts) =>
    `not asserted: ${ids} — the ontology no longer enforces ${texts} ` +
    `(shown as declared by default)`,
  declared: "declared",
  add: "add",
  reset: "reset to declared defaults",
  removeAxiom: "remove this axiom",
  violationSuffix: n => ` · ${n} violation(s)`,
  unassertedSuffix: " · unasserted",
};

export function mountTBoxEditor({ root, adapter }) {
  const { tbox, implementedForms, vocabulary, aboxFor, notes, enforceChecked, setEnforce,
          note, onChange } = adapter;
  const labels = { ...DEFAULT_LABELS, ...(adapter.labels ?? {}) };
  const pristine = adapter.pristine ?? tbox.map(a => ({ ...a }));
  const forms = offerableForms(implementedForms);
  const aboxHost = el("div", { className: "abox" });

  /** Repaint only the ABox. Cheap enough to call a few times a second. */
  const refreshABox = () => {
    const rows = aboxFor(adapter.getSim());
    aboxHost.replaceChildren(...rows.map(b => el("div", { className: "abrow" },
      el("strong", { textContent: b.id }),
      ` ${b.detail}`,
      b.violations.length
        ? el("span", { className: "bad", textContent: labels.violationSuffix(b.violations.length) })
        : null,
      b.unasserted.length
        ? el("span", { className: "warn", textContent: labels.unassertedSuffix })
        : null,
    )));
  };

  const rerender = () => {
    const sim = adapter.getSim();
    const status = tboxStatus(tbox, pristine, implementedForms);

    root.replaceChildren();

    // --- declared axioms
    const list = el("div", { className: "axioms" });
    for (const a of [...tbox]) {
      const numeric = forms.some(f => f.value === a.form && f.numeric);
      // A disjoint axiom's value is its class list, which `describeAxiom` already
      // renders — a second, truncated copy of it in the value column helps nobody.
      const value = a.form === "disjoint"
        ? el("span", { className: "axnone", textContent: labels.declared })
        : numeric
        ? el("input", { type: "number", min: "0", value: String(a.value), className: "axval" })
        : el("span", { className: "axval axfixed", textContent: String(a.value) });
      if (numeric) {
        value.addEventListener("change", () => {
          // Most axioms are declared once and read from here. The domain gets first
          // refusal via `onValueChange`, because some of them have a second home.
          const n = Math.max(0, Math.floor(Number(value.value)));
          if (adapter.onValueChange?.(a, n) === false) value.value = String(a.value);
          else a.value = n;
          onChange(a.id);
        });
      }
      list.append(el("div", { className: "axiom" },
        el("code", { className: "axtext", textContent: describeAxiom(a) }),
        value,
        el("button", {
          className: "axdel", title: labels.removeAxiom, textContent: "×",
          onclick: () => { removeAxiom(tbox, a.id); onChange(a.id); },
        }),
        a.comment ? el("div", { className: "axnote", textContent: a.comment }) : null,
      ));
    }
    root.append(el("h2", { textContent: labels.tboxHeading(tbox.length) }), list);

    if (status.missing.length) {
      root.append(el("div", { className: "warn" },
        labels.notAsserted(
          status.missing.map(m => m.id).join(", "),
          status.missing.map(m => m.text).join("; "),
        )));
    }
    root.append(el("div", { className: status.satisfiable ? "ok" : "bad" },
      status.satisfiable
        ? labels.consistencyOk
        : labels.consistencyBad(status.unsatisfiable.map(u => `${u.id} (${u.reason})`).join("; "))));

    for (const n of notes(sim)) {
      root.append(el("div", { className: n.ok ? "ok" : "bad", textContent: n.text }));
    }

    // --- add an axiom
    const sel = (options, value) => {
      const s = el("select");
      for (const o of options) {
        const opt = typeof o === "string" ? { value: o, label: o } : o;
        s.append(el("option", { value: opt.value, textContent: opt.label, selected: opt.value === value }));
      }
      return s;
    };
    const fSel = sel(forms, "minCardinality");
    const sSel = sel(vocabulary.classes, vocabulary.classes[0]);
    const pSel = sel(vocabulary.properties, vocabulary.properties[0]);
    const vIn = el("input", { type: "number", min: "0", value: "6", className: "axval" });
    const syncValueForForm = () => {
      const f = forms.find(x => x.value === fSel.value);
      if (f.numeric) { vIn.type = "number"; vIn.className = "axval"; }
      else { vIn.type = "text"; vIn.className = "axval axfixed"; vIn.value = vocabulary.classes[0]; }
    };
    fSel.addEventListener("change", syncValueForForm);
    const msg = el("span", { className: "axmsg" });
    root.append(el("div", { className: "axadd" },
      el("span", { className: "axaddlbl", textContent: labels.add }),
      fSel, sSel, pSel, vIn,
      el("button", {
        className: "axbtn", textContent: "+",
        onclick: () => {
          const r = addAxiom(tbox, {
            form: fSel.value, subject: sSel.value, property: pSel.value,
            value: forms.find(x => x.value === fSel.value)?.numeric ? Number(vIn.value) : vIn.value,
          }, implementedForms);
          msg.textContent = r.ok ? `added ${r.id}` : r.error;
          if (r.ok) onChange(r.id);
        },
      }),
      msg,
    ));

    // --- what the ontology currently believes
    // aboxHost is created once and reused across every rerender, so it can be refreshed
    // on a timer without touching the axiom rows — rebuilding the whole panel on a tick
    // would steal focus from a value input mid-edit.
    root.append(el("h2", { textContent: labels.aboxHeading }), aboxHost);
    refreshABox();

    // --- enforcement toggle, ONLY when the domain has one. Breakout drives enforcement
    // from a button in its own Controls row instead, and an adapter without
    // `enforceChecked` must not render a second, competing switch here.
    if (enforceChecked) {
      const toggle = el("input", { type: "checkbox" });
      toggle.checked = !!enforceChecked(sim);
      toggle.addEventListener("change", () => {
        setEnforce(sim, toggle.checked);
        onChange(toggle.checked ? "enforce" : "unenforce");
      });
      root.append(el("div", { className: "axenforce" }, toggle,
        el("label", { textContent: adapter.enforceLabel })));
    }
    root.append(el("div", { className: "axnote", textContent: note }));

    root.append(el("div", { className: "axreset" },
      el("button", { className: "axbtn", textContent: labels.reset,
                     onclick: () => { resetTBox(tbox, pristine, adapter.onReset); onChange("reset"); } })));
  };

  rerender();
  return { rerender, refreshABox };
}