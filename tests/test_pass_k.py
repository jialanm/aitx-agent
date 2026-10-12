"""pass^k from repeated runs: the unbiased estimator and its endpoints."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location("pass_k", Path(__file__).parent.parent / "scripts" / "pass_k.py")
pass_k = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(pass_k)


def test_curve_endpoints_and_middle():
    # Q1 always right, Q2 never, Q3 right in 2 of 4 runs.
    correct = {"Q1": [True] * 4, "Q2": [False] * 4, "Q3": [True, False, True, False]}

    curve = pass_k.pass_k_curve(correct)

    assert curve[0] == (1 + 0 + 0.5) / 3                      # pass^1 is mean accuracy
    assert curve[3] == (1 + 0 + 0) / 3                        # pass^4: only Q1 is right every time
    assert abs(curve[1] - (1 + 0 + 1 / 6) / 3) < 1e-9         # pass^2 for Q3: C(2,2)/C(4,2) = 1/6


def test_a_perfectly_consistent_agent_has_a_flat_curve():
    correct = {"Q1": [True] * 5, "Q2": [False] * 5}
    assert pass_k.pass_k_curve(correct) == [0.5] * 5
