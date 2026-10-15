// Turning a transcript into one of four commands, without a model.
//
// WHY THIS EXISTS ALONGSIDE THE MODEL. The demo has two arms, and this is the one that
// always works. A keyword match needs no weights, no download, and no network, so the
// page does something useful on a cold laptop and the tests can pin the mapping down.
// The `laya` arm in penguin.html asks a model to route the same transcript; when it is
// unavailable, the page falls back here rather than going quiet.
//
// NOTHING HERE TOUCHES THE DOM, and nothing here loads a model. `parseCommand` is a pure
// string function, which is what makes the whole mapping testable in node.

// ---- the tables ---------------------------------------------------------------------
//
// ORDER IS THE ALGORITHM. Evaluation is first-match-wins across the CONCATENATION of
// these arrays, so a more specific phrase must appear before a more general one or it can
// never fire. The cases that actually bite:
//
//   "turn left" must beat "left"      -> otherwise EN "turn left" still resolves left,
//                                        which hides the bug; the Korean side is where
//                                        it shows (see below)
//   "뒤로" must beat "뒤"              -> "뒤" alone is a substring of "뒤로"
//   "우회전" must beat "회전"          -> "회전" alone means a turn with no direction
//
// So the four groups below are ordered most-specific-first WITHIN each command, and the
// commands themselves are ordered turn-before-translation so a direction word present in
// a turn phrase is consumed by the turn rule.

const EN = [
  { cmd: "turn_left", words: ["turn left", "rotate left", "left turn", "to the left"] },
  { cmd: "turn_right", words: ["turn right", "rotate right", "right turn", "to the right"] },
  { cmd: "back", words: ["go back", "backward", "backwards", "reverse", "back up", "back"] },
  { cmd: "go", words: ["go forward", "forward", "straight", "ahead", "go", "walk", "move"] },
  // Bare direction words last: "left" / "right" only mean a turn once every phrase that
  // needed the longer form has had its chance.
  { cmd: "turn_left", words: ["left"] },
  { cmd: "turn_right", words: ["right"] },
];

const KO = [
  { cmd: "turn_left", words: ["왼쪽으로 돌아", "왼쪽으로", "좌회전", "왼쪽"] },
  { cmd: "turn_right", words: ["오른쪽으로 돌아", "오른쪽으로", "우회전", "오른쪽"] },
  { cmd: "back", words: ["뒤로 가", "뒤로", "후진", "뒷걸음"] },
  { cmd: "go", words: ["앞으로 가", "앞으로", "전진", "가", "걸어"] },
];

// English and Korean are matched TOGETHER rather than by a declared language, because a
// Korean speaker says "go" and an English speaker's transcript of "왼쪽" is still "왼쪽".
// Guessing the language from the transcript would add a failure mode for nothing.
const RULES = [...KO, ...EN];

// ---- normalisation -------------------------------------------------------------------

/**
 * Fold a transcript into the form the tables are written in.
 *
 * ASR output is not clean: it arrives capitalised, punctuated, and sometimes with a
 * trailing space or a period. "Turn left." and "turn left" are the same instruction and
 * must not depend on the recogniser's punctuation habits.
 */
export function normalize(transcript) {
  return String(transcript ?? "")
    .toLowerCase()
    .replace(/[.,!?;:"'`~…·]/g, " ")   // punctuation -> space, so it cannot glue words
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * Pick the command a transcript names, or null.
 *
 * `matched` returns the exact phrase that fired, which the HUD shows — when a read is
 * wrong it is usually wrong because a different keyword fired, and hiding which one
 * makes that impossible to see.
 *
 * `ambiguous` is true when more than one DISTINCT command matched. That is not treated
 * as an error: first-match-wins already resolved it, and the flag exists so the page can
 * surface a genuinely contradictory utterance ("go back and turn left") instead of
 * silently obeying half of it.
 */
export function parseCommand(transcript) {
  const text = normalize(transcript);
  if (!text) return { cmd: null, matched: null, ambiguous: false };

  let hit = null;
  const commands = new Set();
  for (const rule of RULES) {
    for (const word of rule.words) {
      if (!text.includes(word)) continue;
      commands.add(rule.cmd);
      if (!hit) hit = { cmd: rule.cmd, matched: word };
      break;                                  // one word per rule; first is most specific
    }
  }

  if (!hit) return { cmd: null, matched: null, ambiguous: false };
  return { cmd: hit.cmd, matched: hit.matched, ambiguous: commands.size > 1 };
}

/** Human label for a command, for the HUD. */
export function describeCommand(cmd) {
  switch (cmd) {
    case "go": return "go — one step forward";
    case "back": return "back — one step backward";
    case "turn_left": return "turn left — 90° in place";
    case "turn_right": return "turn right — 90° in place";
    default: return "no command recognised";
  }
}
