"""Run one guarded turn from the command line and print what the guard did.

The same job as main.py in this repo: a runnable script that demonstrates the thing the
README describes, so the demo can be seen without starting a server.

    uv run python guard_cli.py "Where is my order A1?"
    uv run python guard_cli.py --model english "My SSN is 123-45-6789"
    uv run python guard_cli.py --model ciel_decision_model_v1 --path doc "My SSN is 123-45-6789"

The last two are the interesting pair: the same prompt, checked by two different arms,
scoring differently. That is the finding this example exists to show.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import guard_agent
import guard_pii


def _print_arms(indent: str = "  ") -> None:
    """The available arms and what each one is."""
    for arm in guard_pii.ARMS.values():
        kind = "sub-path" if arm.kind == "lab" else "kind"
        extra = f", {kind} {arm.path}" if arm.path else ""
        print(f"{indent}{arm.key} — {arm.label} ({arm.kind}{extra})")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("message", nargs="?", help="the message to send")
    parser.add_argument(
        "--model",
        default="english",
        help="which arm checks it: " + ", ".join(guard_pii.ARMS),
    )
    parser.add_argument(
        "--path",
        default=None,
        help="which decision_lab sub-path, for the lab arms",
    )
    parser.add_argument(
        "--arms",
        action="store_true",
        help="list the arms and exit",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the raw response instead of the trace",
    )
    args = parser.parse_args(argv)

    if args.arms:
        _print_arms()
        return 0

    if not args.message:
        parser.error("a message is required (or use --arms to list the arms)")

    try:
        arm = guard_pii.resolve_arm(args.model, args.path)
    except guard_pii.ArmError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"arm     : {arm.key} ({arm.kind}{', ' + arm.path if arm.path else ''})")
    print(f"chat mdl: {os.environ.get(guard_agent.MODEL_ENV, guard_agent.DEFAULT_MODEL)}")
    # The message is NOT echoed. The whole point of the example is a guard that keeps
    # secrets out of logs, and a CLI that prints the raw prompt it was asked to guard
    # would be the one place the secret lands in plain text.
    print(f"message : ({len(args.message)} chars, not echoed)")
    print()

    try:
        result = guard_agent.chat(args.message, arm=arm)
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print("guard trace:")
    for event in result.get("guard_events") or []:
        status = "BLOCKED" if event.get("blocked") else "passed"
        stage = event.get("stage")
        suffix = f" — {event['reason']}" if event.get("reason") else ""
        print(f"  [{status:7}] {stage}{suffix}")
        for finding in event.get("findings") or []:
            print(
                f"            {finding['detector']}/{finding['category']} "
                f"score={finding['score']:.3f} {finding['detail']}"
            )
    print()
    print("answer  :", result.get("answer"))
    print("blocked :", result.get("blocked"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
