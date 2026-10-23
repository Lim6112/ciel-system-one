// The traffic defence panel — a domain adapter over the shared editor in
// ontology_core.js, the same structure as warehouse_panel.js and breakout_panel.js.
//
// ENFORCEMENT IS NOT A SWITCH HERE, for the reason warehouse_panel.js gives: the rule
// removes responses from the permitted set, and removing a response IS the behaviour
// change. An "Ontology: on/off" button beside it would suggest the rule is optional on a
// page whose whole subject is that the rule decides. The live TBox editor is the switch —
// deleting an axiom disables its check and the panel reports it as unasserted, and the
// next packet meets a different rule set.
//
// THE ABox HERE IS A PASS-THROUGH, and that is deliberate rather than lazy. The panel is
// refreshed a few times a second while the sim runs, so it reads whatever passage the
// page last handed over. It does no reasoning of its own — `aboxFor` in the ontology does
// that — because a panel that re-derived the verdict would be a second home for the rule,
// which is the failure this whole layer exists to prevent.

import { TBOX, VOCABULARY, IMPLEMENTED_FORMS, aboxFor, permittedResponses } from "./traffic_ontology.js";
import { mountTBoxEditor } from "./ontology_core.js";

// Snapshot at construction time, before the panel is ever mounted, so a stray edit during
// a session is not what `reset` restores you to.
const PRISTINE = TBOX.map(a => ({
  ...a,
  value: Array.isArray(a.value) ? [...a.value] : a.value,
}));

export function mountTrafficPanel({ root, getSim, onChange }) {
  return mountTBoxEditor({
    root,
    adapter: {
      tbox: TBOX,
      pristine: PRISTINE,
      implementedForms: IMPLEMENTED_FORMS,
      vocabulary: VOCABULARY,
      getSim,
      aboxFor,
      // Only the VIOLATIONS are reported here, and that is a deliberate narrowing.
      //
      // The ABox row above already names the fired classes, the permitted set and the
      // route, so a note restating them was a second copy of the same sentence — and the
      // two live on different clocks, so they could disagree on screen ("nothing has
      // fired" beside a row naming ConfirmedPII). One line, one home: the ABox describes
      // the passage, and this reports only what the ABox cannot — that a declared axiom
      // and the flow it judged have come apart.
      notes: passage => {
        if (!passage) {
          return [{ ok: true, text: "아직 가드에 도달한 패킷이 없습니다 — 스트림을 시작하세요" }];
        }
        return permittedResponses(passage).violations
          .map(v => ({ ok: false, text: `${v.axiom}: ${v.detail}` }));
      },
      note: "선언된 규칙이 패킷의 운명을 정한다. 속도 윈도우를 넘긴 소스, 포트를 훑은 " +
            "범위, 확인된 자격증명, 아무도 검사할 수 없었던 페이로드 — 이 모두는 모델에 " +
            "묻기도 전에 대응 집합에서 \"allow\"가 철회된다. 여기서 공리를 제거하면 그 " +
            "검사가 꺼지고 패널이 그 사실을 알린다. 패킷은 여전히 가드를 만나며, " +
            "온톨로지가 그 규칙을 더 이상 주장하지 않을 뿐이다. " +
            "이 도메인은 시행이 결정을 바꾸지 않는다고 주장할 수 없다 — 대응을 철회하는 " +
            "것이 바로 그 변경이기 때문이다. 대신 주장할 수 있는 것, 그리고 " +
            "`node --test test_traffic_ontology.mjs`가 시뮬레이터가 만드는 모든 passage " +
            "형태에 대해 검사하는 것은, 선언된 규칙과 그것이 대체한 손으로 쓴 술어가 " +
            "일치한다는 것이다. 그 일치가 핵심이다 — 그 둘은 warehouse 도메인에서 서로 " +
            "다른 함수에 살다가 어긋났고, 로봇들을 대치 상태에 가두었다.",
      labels: {
        tboxHeading: n => `온톨로지 — TBox (공리 ${n}개)`,
        aboxHeading: "온톨로지 — ABox (현재 믿고 있는 것)",
        consistencyOk: "일관성: 만족 가능 — 공리 집합이 정합적입니다",
        consistencyBad: details => "만족 불가능: " + details,
        notAsserted: (ids, texts) =>
          `주장되지 않음: ${ids} — 온톨로지가 더 이상 ${texts} 을(를) 강제하지 않습니다 ` +
          `(기본값으로 선언된 것으로 표시)`,
        declared: "선언됨",
        add: "추가",
        reset: "선언된 기본값으로 되돌리기",
        removeAxiom: "이 공리 제거",
        violationSuffix: n => ` · 위반 ${n}건`,
        unassertedSuffix: " · 미주장",
      },
      onChange,
    },
  });
}

/** The permitted response set, as the decision loop sees it. */
export function permittedFor(passage) {
  const verdict = permittedResponses(passage);
  return {
    options: verdict.options,
    undecided: verdict.undecided,
    route: verdict.route,
    allowed: verdict.allowed,
    structuralType: verdict.structuralType,
    contentType: verdict.contentType,
    violations: verdict.violations,
  };
}
