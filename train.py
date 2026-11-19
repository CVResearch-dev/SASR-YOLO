#!/usr/bin/env python3
"""Pre-acceptance preview: training is disabled; all hyperparameters are placeholders."""
import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    message = (
        "PRE-ACCEPTANCE PLACEHOLDER — NOT RUNNABLE.\n"
        "Numerical hyperparameters have deliberately been set to zero; they "
        "are not the paper's experimental settings. Training is disabled.\n"
        "The authors will release the complete configuration separately."
    )
    print(message, file=sys.stdout if any(a in ("-h", "--help") for a in args) else sys.stderr)
    return 0 if any(a in ("-h", "--help") for a in args) else 2


if __name__ == "__main__":
    raise SystemExit(main())
