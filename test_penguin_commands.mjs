// node --test test_penguin_commands.mjs
//
// The mapping from a transcript to one of four commands. This is pure string work, so
// everything here is a real assertion about the shipped behaviour — no model, no network,
// no browser.
//
// The interesting cases are not the happy path. They are the ones where a naive
// substring scan silently picks the wrong command, which is the failure mode this whole
// file exists to prevent.

import { test } from "node:test";
import assert from "node:assert/strict";

import { parseCommand, normalize, describeCommand } from "./static/penguin_commands.js";

const cmd = t => parseCommand(t).cmd;

// ---- the four commands, both languages ------------------------------------------

test("English phrases map to the four commands", () => {
  assert.equal(cmd("go"), "go");
  assert.equal(cmd("go forward"), "go");
  assert.equal(cmd("walk"), "go");
  assert.equal(cmd("back"), "back");
  assert.equal(cmd("go back"), "back");
  assert.equal(cmd("reverse"), "back");
  assert.equal(cmd("turn left"), "turn_left");
  assert.equal(cmd("left"), "turn_left");
  assert.equal(cmd("turn right"), "turn_right");
  assert.equal(cmd("right"), "turn_right");
});

test("Korean phrases map to the four commands", () => {
  assert.equal(cmd("앞으로"), "go");
  assert.equal(cmd("앞으로 가"), "go");
  assert.equal(cmd("전진"), "go");
  assert.equal(cmd("뒤로"), "back");
  assert.equal(cmd("뒤로 가"), "back");
  assert.equal(cmd("후진"), "back");
  assert.equal(cmd("왼쪽"), "turn_left");
  assert.equal(cmd("좌회전"), "turn_left");
  assert.equal(cmd("오른쪽"), "turn_right");
  assert.equal(cmd("우회전"), "turn_right");
});

// ---- the traps -------------------------------------------------------------------
//
// Each of these passes a naive implementation that scans a flat word list.

test("'go back' is back, not go — the longer phrase wins", () => {
  // If "go" were tried before "back", this would resolve to go and the penguin would
  // walk into whatever the speaker was trying to reverse away from.
  assert.equal(cmd("go back"), "back");
  assert.equal(cmd("뒤로 가"), "back");
});

test("'turn left' is a turn, and is not mistaken for right", () => {
  const left = parseCommand("turn left");
  assert.equal(left.cmd, "turn_left");
  assert.equal(left.ambiguous, false, "'turn left' matched something else too");
});

test("'회전' without a direction is not silently resolved to a side", () => {
  // "회전" alone means "rotate" with no direction. It matches no rule, and that has to be
  // null rather than a coin flip.
  assert.equal(cmd("회전"), null);
});

test("a direction word inside a turn phrase is consumed by the turn rule", () => {
  // "to the left" contains "left". Both map to turn_left, so this is really asserting the
  // phrase table had its chance first; if a future edit let the bare word win, the result
  // would still be turn_left and only the `matched` field would show the drift.
  const p = parseCommand("to the left");
  assert.equal(p.cmd, "turn_left");
  assert.equal(p.matched, "to the left");
});

test("English 'left'/'right' as bare words are turns, not translations", () => {
  // There is no side-step command in this world — left/right can only be turns. Worth
  // pinning because it is a design choice, not an accident of the table.
  assert.equal(cmd("left"), "turn_left");
  assert.equal(cmd("right"), "turn_right");
});

// ---- normalisation ----------------------------------------------------------------

test("case and punctuation do not change the outcome", () => {
  assert.equal(cmd("Go."), "go");
  assert.equal(cmd("TURN LEFT!"), "turn_left");
  assert.equal(cmd("  Go   back  "), "back");
  assert.equal(cmd("왼쪽으로."), "turn_left");
});

test("normalize folds case, punctuation and runs of whitespace", () => {
  assert.equal(normalize("  Turn,  LEFT!  "), "turn left");
  assert.equal(normalize(null), "");
  assert.equal(normalize(undefined), "");
});

test("an empty or unmatched transcript is null, not a guess", () => {
  assert.equal(cmd(""), null);
  assert.equal(cmd("   "), null);
  assert.equal(cmd("the weather is nice today"), null);
  assert.equal(cmd("안녕하세요"), null);
});

// ---- ambiguity, surfaced not hidden -----------------------------------------------

test("a contradictory utterance still resolves but is flagged ambiguous", () => {
  const p = parseCommand("go back and turn left");
  assert.ok(p.cmd, "should still pick a command");
  assert.equal(p.ambiguous, true);
});

test("an unambiguous utterance is not flagged", () => {
  assert.equal(parseCommand("go forward").ambiguous, false);
});

// ---- the HUD label ------------------------------------------------------------------

test("describeCommand names every command and the null case", () => {
  for (const c of ["go", "back", "turn_left", "turn_right"]) {
    assert.match(describeCommand(c), /\w/, `no label for ${c}`);
  }
  assert.notEqual(describeCommand(null), describeCommand("go"));
});
