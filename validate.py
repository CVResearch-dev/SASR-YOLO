#!/usr/bin/env python3
"""Pre-acceptance preview: evaluation is disabled; no metrics are generated."""
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    message = (
        "PRE-ACCEPTANCE PLACEHOLDER — NOT RUNNABLE.\n"
        "Numerical evaluation settings are zeroed placeholders, not the paper "
        "protocol. Evaluation is disabled and no metrics are generated.\n"
        "Use the complete release when it becomes available."
    )
    print(message, file=sys.stdout if any(a in ("-h", "--help") for a in args) else sys.stderr)
    return 0 if any(a in ("-h", "--help") for a in args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
