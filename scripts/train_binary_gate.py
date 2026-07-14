"""Formal Phase-2 entry point for the frozen-delta binary channel gate.

The implementation is shared with the learnability validation so the
validated and deployed decision semantics cannot silently diverge.
"""

from __future__ import annotations

from experiment_gate_learnability import main


if __name__ == "__main__":
    main()
