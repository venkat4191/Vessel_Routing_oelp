"""
Stage 3 — Part 1: Route Validator
===================================
Physical and commercial feasibility checks on every candidate route.
Acts as a hard gate before any route enters the RMP column pool.

Checks (ordered cheapest-first for fast rejection):
  CHECK 0  — Vessel class exists
  CHECK 1  — Minimum 3 ports
  CHECK 2  — Maximum 15 ports
  CHECK 3  — No duplicate ports
  CHECK 4  — All ports exist in database
  CHECK 5  — Vessel draft <= port draft limit at every port
  CHECK 6  — Distance entry exists for every leg (including return)
  CHECK 7  — Canal feasibility (Panama draft + vessel type)
  CHECK 8  — At least one valid OD demand pair covered (CIRCULAR — FIX 3)
  CHECK 9  — Valid frequency (1 or 2)
  CHECK 10 — Positive potential demand
  CHECK 11 — Cabotage rules (FIX 4)
  CHECK 12 — Transit time feasibility for at least one OD pair (FIX 1)

Returns: (is_valid: bool, reason: str)
Speed: <2ms per route (pure dict lookups, no LP or API calls)
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CABOTAGE_PROTECTED_REGIONS


def _compute_transit_days(port_sequence: list, o: str, d: str,
                           distances: dict, speed_knots: float,
                           dwell_days: float) -> float:
    """
    Compute the minimum transit time (days) from port o to port d on this route.

    For a CIRCULAR service [A, B, C, D], transit from B to D is:
      sail(B→C) + dwell(C) + sail(C→D)
    Transit from D to B (return direction, wrapping around):
      sail(D→A) + dwell(A) + sail(A→B)

    We find o and d in the circular sequence and sum the intermediate legs.
    Returns the shortest transit around the cycle for this OD pair.
    """
    n = len(port_sequence)
    cycle = port_sequence + port_sequence   # duplicate for circular arithmetic

    # Find start index of o
    try:
        o_idx = port_sequence.index(o)
    except ValueError:
        return float('inf')

    # Walk forward from o until we find d (up to n-1 steps)
    transit = 0.0
    for step in range(1, n):
        frm = cycle[o_idx + step - 1]
        to  = cycle[o_idx + step]
        dist_rec = distances.get((frm, to))
        if dist_rec is None:
            return float('inf')
        transit += dist_rec.distance_nm / (speed_knots * 24.0)
        if to == d:
            break
        transit += dwell_days   # port dwell at intermediate ports
    return transit


def validate(
    port_sequence: list,
    vessel_class:  str,
    fleet:         dict,
    ports:         dict,
    distances:     dict,
    demand:        dict,
    frequency:     int = 1,
) -> tuple:
    """
    Run all 12 checks. Returns (True, "OK") or (False, "CHECK_N: reason").
    """
    from utils.config import PORT_DWELL_DAYS_BY_CLASS, PORT_DAYS_PER_CALL

    # ── CHECK 0: Vessel class exists ──────────────────────────────────────────
    vessel = fleet.get(vessel_class)
    if vessel is None:
        return False, f"CHECK0: Unknown vessel class '{vessel_class}'"

    # ── CHECK 1: Minimum 3 ports ──────────────────────────────────────────────
    if len(port_sequence) < 3:
        return False, f"CHECK1: Route has {len(port_sequence)} port(s) — minimum is 3"

    # ── CHECK 2: Maximum 15 ports ──────────────────────────────────────────────
    if len(port_sequence) > 15:
        return False, f"CHECK2: Route has {len(port_sequence)} ports — maximum is 15"

    # ── CHECK 3: No duplicate ports ───────────────────────────────────────────
    seen = set()
    for p in port_sequence:
        if p in seen:
            return False, f"CHECK3: Duplicate port '{p}' in sequence"
        seen.add(p)

    # ── CHECK 4: All ports exist in database ──────────────────────────────────
    for p in port_sequence:
        if p not in ports:
            return False, f"CHECK4: Port '{p}' not found in ports database"

    # ── CHECK 5: Draft feasibility at every port ──────────────────────────────
    for p in port_sequence:
        if vessel.draft > ports[p].draft:
            return False, (
                f"CHECK5: Draft fail at {ports[p].name} ({p}) — "
                f"vessel {vessel.draft}m > port limit {ports[p].draft}m"
            )

    # ── CHECK 6: Distance entry for every leg + return leg ───────────────────
    legs_full = ([(port_sequence[i], port_sequence[i+1])
                  for i in range(len(port_sequence) - 1)]
                 + [(port_sequence[-1], port_sequence[0])])
    for (frm, to) in legs_full:
        if (frm, to) not in distances:
            return False, f"CHECK6: No distance entry for leg {frm}→{to}"

    # ── CHECK 7: Canal feasibility ────────────────────────────────────────────
    for (frm, to) in legs_full:
        dist_rec = distances[(frm, to)]
        if dist_rec.is_panama:
            if vessel.panama_fee is None:
                return False, (
                    f"CHECK7: Leg {frm}→{to} uses Panama but "
                    f"{vessel_class} cannot transit Panama"
                )
            if dist_rec.draft is not None and vessel.draft > dist_rec.draft:
                return False, (
                    f"CHECK7: Panama draft fail {frm}→{to} — "
                    f"vessel {vessel.draft}m > canal limit {dist_rec.draft}m"
                )
        if dist_rec.is_suez and vessel.suez_fee is None:
            return False, (
                f"CHECK7: Leg {frm}→{to} uses Suez but "
                f"{vessel_class} cannot transit Suez"
            )

    # ── CHECK 8: FIX 3 — Circular coverage ───────────────────────────────────
    # A liner service is a CYCLE: [A,B,C] sails A→B→C→A→B→C...
    # Cargo is carried in BOTH directions around the loop.
    # Old implementation only checked forward (i < j) — missed all return-leg demand.
    n = len(port_sequence)
    cycle = port_sequence + port_sequence
    covers_any = False
    for start in range(n):
        for length in range(1, n):
            o = cycle[start]
            d = cycle[start + length]
            if o != d and (o, d) in demand:
                covers_any = True
                break
        if covers_any:
            break

    if not covers_any:
        return False, (
            "CHECK8: No demand OD pair covered by this circular route "
            "(checked both forward and return directions)"
        )

    # ── CHECK 9: Valid frequency ──────────────────────────────────────────────
    if frequency not in (1, 2):
        return False, f"CHECK9: Invalid frequency {frequency} — must be 1 or 2"

    # ── CHECK 10: Positive potential demand ───────────────────────────────────
    potential_ffe = 0.0
    for start in range(n):
        for length in range(1, n):
            od = (cycle[start], cycle[start + length])
            if od[0] != od[1] and od in demand:
                potential_ffe += demand[od].ffe_per_week
    if potential_ffe <= 0:
        return False, "CHECK10: Zero potential FFE demand on this route (all directions)"

    # ── CHECK 11: FIX 4 — Cabotage rules ─────────────────────────────────────
    # Cabotage law prohibits foreign-flagged vessels from carrying cargo between
    # two ports within the same protected domestic region consecutively.
    # Example: US Jones Act — cannot carry USNYK→USLAX on a foreign vessel.
    # We check: for any demand pair (o, d) this route would serve, if both
    # ports share a CABOTAGE_PROTECTED_REGIONS cabotage_region, reject.
    for start in range(n):
        for length in range(1, n):
            o = cycle[start]
            d = cycle[start + length]
            if o == d or (o, d) not in demand:
                continue
            o_region = ports[o].cabotage_region
            d_region = ports[d].cabotage_region
            if (o_region == d_region and o_region in CABOTAGE_PROTECTED_REGIONS):
                return False, (
                    f"CHECK11: Cabotage violation — {o} and {d} are both in "
                    f"protected region '{o_region}'. Foreign vessels cannot carry "
                    f"domestic cargo between these ports."
                )

    # ── CHECK 12: FIX 1 — Transit time feasibility ───────────────────────────
    # For each OD pair covered, the actual sailing time from o to d on this
    # route must be <= demand[od].max_transit_days.
    # A route that covers ZERO OD pairs within their transit time limit is useless.
    # NOTE: we only need ONE valid OD pair; others can still be excluded in the LP.
    dwell = PORT_DWELL_DAYS_BY_CLASS.get(vessel_class, PORT_DAYS_PER_CALL)
    speed = vessel.design_speed   # use design speed for conservative estimate
    transit_ok = False
    for start in range(n):
        for length in range(1, n):
            o = cycle[start]
            d = cycle[start + length]
            if o == d or (o, d) not in demand:
                continue
            max_td = demand[(o, d)].max_transit_days
            actual = _compute_transit_days(port_sequence, o, d, distances, speed, dwell)
            if actual <= max_td + 0.5:   # +0.5 day tolerance for rounding
                transit_ok = True
                break
        if transit_ok:
            break

    if not transit_ok:
        return False, (
            "CHECK12: Transit time infeasible — no OD pair on this route can be "
            "served within its max_transit_days contract limit at design speed. "
            "Route is too long or too slow for all covered demand pairs."
        )

    return True, "OK"


def validate_batch(routes: list, fleet, ports, distances, demand) -> list:
    results = []
    for (seq, vc, freq) in routes:
        ok, reason = validate(seq, vc, fleet, ports, distances, demand, freq)
        results.append(((seq, vc, freq), ok, reason))
    return results


def filter_valid(routes: list, fleet, ports, distances, demand) -> list:
    return [
        (seq, vc, freq) for (seq, vc, freq) in routes
        if validate(seq, vc, fleet, ports, distances, demand, freq)[0]
    ]
