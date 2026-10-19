"""Exact scalar-fusion feasibility for frozen IEEE-754 binary64 distances.

All decisions use integer arithmetic.  A common power-of-two denominator
represents the input distances exactly; it cancels from every comparison.
Reported endpoint fractions are reduced and use decimal strings so that a JSON
consumer cannot silently round their integers.  Float endpoint values and
widths are display aids only and never determine feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from numbers import Integral
from typing import Iterable

import numpy as np


def _validate(dr, dt, positive, tie_keys):
    r, t = np.asarray(dr), np.asarray(dt)
    p, keys = np.asarray(positive), np.asarray(tie_keys)
    if r.dtype != np.float64 or t.dtype != np.float64:
        raise ValueError("dr and dt must be float64 arrays")
    if any(a.ndim != 1 for a in (r, t, p, keys)):
        raise ValueError("all inputs must be one-dimensional")
    if not (len(r) == len(t) == len(p) == len(keys)) or len(r) == 0:
        raise ValueError("inputs must have the same nonzero length")
    if p.dtype != np.bool_:
        raise ValueError("positive must have boolean dtype")
    if keys.dtype.kind not in "iu" or len(np.unique(keys)) != len(keys):
        raise ValueError("tie_keys must contain unique integers")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(t)):
        raise ValueError("distance arrays must be finite")
    return r, t, p, keys


def _integer_scores(r: np.ndarray, t: np.ndarray):
    ratios_r = [float(x).as_integer_ratio() for x in r]
    ratios_t = [float(x).as_integer_ratio() for x in t]
    exponent = max(d.bit_length() - 1 for _, d in ratios_r + ratios_t)
    ri = [n << (exponent - (d.bit_length() - 1)) for n, d in ratios_r]
    ti = [n << (exponent - (d.bit_length() - 1)) for n, d in ratios_t]
    return ri, ti


def _ratio(n: int, d: int):
    if d < 0:
        n, d = -n, -d
    divisor = gcd(n, d)
    return n // divisor, d // divisor


def _fraction_json(n: int, d: int):
    n, d = _ratio(n, d)
    try:
        display_value = float(Fraction(n, d))
    except OverflowError:
        # Infeasibility certificates can have active bounds far outside [0,1]
        # even though every input is finite; keep the exact bound and emit no
        # non-finite JSON display number.
        display_value = None
    return {"numerator": str(n), "denominator": str(d), "value": display_value}


@dataclass(frozen=True)
class _Bound:
    n: int
    d: int
    closed: bool
    negative_index: int | None


@dataclass(frozen=True)
class _Interval:
    lo: _Bound
    hi: _Bound


def _cmp(a: _Bound, b: _Bound):
    left, right = a.n * b.d, b.n * a.d
    return (left > right) - (left < right)


def _valid(lo: _Bound, hi: _Bound):
    order = _cmp(lo, hi)
    return order < 0 or (order == 0 and lo.closed and hi.closed)


def _bound_json(bound: _Bound):
    result = _fraction_json(bound.n, bound.d)
    result.update(closed=bound.closed, negative_index=bound.negative_index)
    return result


def _interval_json(interval: _Interval):
    return {"lower": _bound_json(interval.lo), "upper": _bound_json(interval.hi)}


def _merge(intervals: Iterable[_Interval]):
    # Fraction is used only to order the O(number-of-positive) intervals.
    ordered = sorted(intervals, key=lambda x: (Fraction(x.lo.n, x.lo.d), not x.lo.closed))
    result: list[_Interval] = []
    for item in ordered:
        if not result:
            result.append(item)
            continue
        current = result[-1]
        order = _cmp(item.lo, current.hi)
        if order > 0 or (order == 0 and not (item.lo.closed or current.hi.closed)):
            result.append(item)
            continue
        upper_order = _cmp(item.hi, current.hi)
        if upper_order > 0:
            upper = item.hi
        elif upper_order < 0:
            upper = current.hi
        else:
            upper = _Bound(current.hi.n, current.hi.d,
                           current.hi.closed or item.hi.closed, None)
        result[-1] = _Interval(current.lo, upper)
    return result


def _shape(intervals: list[_Interval]):
    if not intervals:
        return "empty"
    if len(intervals) > 1:
        return "disconnected"
    return "singleton" if _cmp(intervals[0].lo, intervals[0].hi) == 0 else "interval"


def _winner(ri, ti, keys, numerator: int, denominator: int):
    return min(range(len(ri)), key=lambda i: (
        numerator * ri[i] + (denominator - numerator) * ti[i], int(keys[i])))


def _weight(numerator, denominator):
    if not isinstance(numerator, Integral) or not isinstance(denominator, Integral):
        raise ValueError("weight numerator and denominator must be integers")
    numerator, denominator = int(numerator), int(denominator)
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise ValueError("weight must lie in [0,1] with a positive denominator")
    return numerator, denominator


def exact_winner(dr, dt, tie_keys, numerator: int, denominator: int = 10):
    """Return the stable local winner under an exact rational alpha."""
    r = np.asarray(dr)
    r, t, _, keys = _validate(dr, dt, np.zeros(r.shape, dtype=bool), tie_keys)
    numerator, denominator = _weight(numerator, denominator)
    ri, ti = _integer_scores(r, t)
    return _winner(ri, ti, keys, numerator, denominator)


def fixed_correct(dr, dt, positive, tie_keys, numerator: int, denominator: int = 10):
    """Return whether the stable exact-rational-weight top-1 is positive."""
    r, t, p, keys = _validate(dr, dt, positive, tie_keys)
    numerator, denominator = _weight(numerator, denominator)
    ri, ti = _integer_scores(r, t)
    return bool(p[_winner(ri, ti, keys, numerator, denominator)])


def grid_oracle(dr, dt, positive, tie_keys):
    """Evaluate the fixed rational grid alpha=0/10,...,10/10 exactly."""
    r, t, p, keys = _validate(dr, dt, positive, tie_keys)
    ri, ti = _integer_scores(r, t)
    winners = [_winner(ri, ti, keys, i, 10) for i in range(11)]
    correct = [bool(p[i]) for i in winners]
    return {"attainable": any(correct), "correct_numerators": [
        i for i, value in enumerate(correct) if value], "correct": correct,
        "winners": winners, "denominator": 10}


def solve_query(dr, dt, positive, tie_keys):
    """Solve exact top-1 recoverability by one scalar alpha for a query.

    A positive p beats a negative n iff its linear score is smaller, or its
    score is equal and tie_keys[p] < tie_keys[n].  Other positives do not need
    to be beaten.  optimistic_intervals permit the positive side of any tie.
    Weak Pareto dominance means <= in both branches and < in at least one;
    equal score vectors are explicitly not Pareto dominance here.

    Certificate bounds retain their active local negative index (None means
    the domain [0,1]).  They are sufficient to verify each impossible interval
    by recomputing that negative's inequality from the frozen input distances.
    """
    r, t, p, keys = _validate(dr, dt, positive, tie_keys)
    ri, ti = _integer_scores(r, t)
    slopes = [a - b for a, b in zip(ri, ti)]
    positive_indices = np.flatnonzero(p).tolist()
    negative_indices = np.flatnonzero(~p).tolist()
    feasible, optimistic, certificates = [], [], []
    all_weak = bool(positive_indices)
    all_strict_both = bool(positive_indices)

    for pi in positive_indices:
        lo, hi = _Bound(0, 1, True, None), _Bound(1, 1, True, None)
        constant_blocker, optimistic_constant_blocker = None, None
        weak_witness, strict_both_witness = None, None
        rp, tp, slope_p, key_p = ri[pi], ti[pi], slopes[pi], int(keys[pi])
        for ni in negative_indices:
            rn, tn = ri[ni], ti[ni]
            if rn <= rp and tn <= tp and (rn < rp or tn < tp):
                if weak_witness is None:
                    weak_witness = ni
            if rn < rp and tn < tp and strict_both_witness is None:
                strict_both_witness = ni
            # alpha * a <= c; equality is permitted only if p wins the tie.
            a, c = slope_p - slopes[ni], tn - tp
            closed = key_p < int(keys[ni])
            if a == 0:
                if c < 0:
                    optimistic_constant_blocker = ni
                    constant_blocker = ni
                elif c == 0 and not closed and constant_blocker is None:
                    constant_blocker = ni
                continue
            if a > 0:
                # Ratios stay unreduced internally; positive denominators and
                # integer cross multiplication make every comparison exact.
                comparison = c * hi.d - hi.n * a
                if comparison < 0 or (comparison == 0 and hi.closed and not closed):
                    hi = _Bound(c, a, closed, ni)
            else:
                n, d = -c, -a
                comparison = n * lo.d - lo.n * d
                if comparison > 0 or (comparison == 0 and lo.closed and not closed):
                    lo = _Bound(n, d, closed, ni)

        opt_lo = _Bound(lo.n, lo.d, True, lo.negative_index)
        opt_hi = _Bound(hi.n, hi.d, True, hi.negative_index)
        actual_valid = constant_blocker is None and _valid(lo, hi)
        optimistic_valid = optimistic_constant_blocker is None and _valid(opt_lo, opt_hi)
        interval = _Interval(lo, hi)
        opt_interval = _Interval(opt_lo, opt_hi)
        if actual_valid:
            feasible.append(interval)
        if optimistic_valid:
            optimistic.append(opt_interval)
        all_weak = all_weak and weak_witness is not None
        all_strict_both = all_strict_both and strict_both_witness is not None
        certificates.append({
            "positive_index": pi, "positive_tie_key": int(keys[pi]),
            "attainable": actual_valid, "optimistic": optimistic_valid,
            "lower_bound": _bound_json(lo), "upper_bound": _bound_json(hi),
            "constant_blocker": constant_blocker,
            "optimistic_constant_blocker": optimistic_constant_blocker,
            "negative_pareto_dominator": weak_witness,
            "negative_strict_both_dominator": strict_both_witness,
            "attainable_interval": _interval_json(interval) if actual_valid else None,
            "optimistic_interval": _interval_json(opt_interval) if optimistic_valid else None,
            "infeasibility_reason": None if actual_valid else (
                "constant_constraint" if constant_blocker is not None else (
                    "lower_exceeds_upper" if _cmp(lo, hi) > 0 else "open_singleton")),
        })

    feasible, optimistic = _merge(feasible), _merge(optimistic)
    exact_width = sum((Fraction(x.hi.n, x.hi.d) - Fraction(x.lo.n, x.lo.d)
                       for x in feasible), Fraction(0))
    if not positive_indices:
        classification = "NO_POSITIVE"
    elif feasible:
        classification = "LINEAR_RECOVERABLE"
    elif optimistic:
        classification = "TIE_ONLY_BLOCKED"
    elif all_weak:
        classification = "ALL_POSITIVES_NEGATIVE_PARETO_DOMINATED"
    else:
        classification = "UNDOMINATED_POSITIVE_LINEAR_UNSUPPORTED"
    return {
        "attainable_intervals": [_interval_json(x) for x in feasible],
        "optimistic_intervals": [_interval_json(x) for x in optimistic],
        "attainable": bool(feasible), "optimistic": bool(optimistic),
        "positive_count": len(positive_indices), "negative_count": len(negative_indices),
        "width_float": float(exact_width),
        "width_exact": _fraction_json(exact_width.numerator, exact_width.denominator),
        "shape": _shape(feasible), "classification": classification,
        "every_positive_weakly_negative_dominated": all_weak,
        "every_positive_strictly_both_negative_dominated": all_strict_both,
        "undominated_positive_exists": bool(positive_indices) and not all_weak,
        "dominance_definition": "negative <= positive in both branches and < in at least one",
        "certificates": certificates,
    }
