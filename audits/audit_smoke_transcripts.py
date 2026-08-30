#!/usr/bin/env python3
"""Summarize one run tag's saved LLM transcripts from JSONL on stdin."""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_tag")
    parser.add_argument("--show-indented", action="store_true")
    args = parser.parse_args()

    rows = []
    for line in sys.stdin:
        if args.run_tag not in line:
            continue
        rows.append(json.loads(line))

    methods: collections.Counter[str] = collections.Counter()
    first_definition_indent: collections.Counter[int] = collections.Counter()
    old_icl = correct_icl = failed = 0
    for row in rows:
        namespace = str(row.get("namespace") or "")
        parts = namespace.split("/")
        methods[parts[2] if len(parts) > 2 else "?"] += 1
        messages = row.get("messages") or []
        contents = [str(message.get("content") or "") for message in messages]
        old_icl += any("  def example_name" in content for content in contents)
        correct_icl += any("def example_name(x):\n    return x" in content for content in contents)
        failed += bool(row.get("failed"))

        completion = str(row.get("completion") or "")
        match = re.search(r"(?m)^([ \t]*)(?:async +def|def|class) +", completion)
        if match:
            width = len(match.group(1).replace("\t", "    "))
            first_definition_indent[width] += 1
            if args.show_indented and width:
                snippet = completion[max(0, match.start() - 120):match.start() + 500]
                print(f"INDENTED namespace={namespace} request_id={row.get('request_id')} width={width}")
                print(snippet.replace("\n", "\\n"))

    print(f"rows={len(rows)}")
    print(f"methods={dict(sorted(methods.items()))}")
    print(f"old_indented_icl={old_icl}")
    print(f"correct_top_level_icl={correct_icl}")
    print(f"failed_transcripts={failed}")
    print(f"first_definition_indent={dict(sorted(first_definition_indent.items()))}")


if __name__ == "__main__":
    main()
