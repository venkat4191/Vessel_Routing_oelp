"""
Stage 3 — Part 2: Reduced Cost Calculator
==========================================
The pricing decision engine of Column Generation.

THEORY
------
After the RMP LP solves, it returns two sets of dual prices:

  α_od  (demand dual)   — shadow price of the demand constraint for OD pair (o,d)
                          = revenue lost per FFE of unserved demand on that pair
                          = 0   if the pair is not covered by any existing route
                          > 0   if the pair IS covered (marginal value of 1 more FFE)

  μ_s   (capacity dual) — shadow price of the capacity constraint on route s
                          = 0   if route s has spare capacity (not congested)
                          > 0   if route s is fully loaded (congested)

For a CANDIDATE new route s', the reduced cost is:

    rc(s') = -cost(s') + Σ_{od in s'} (rev_od - α_od) × f_od*

Where f_od* solves the PRICING SUBPROBLEM (knapsack):

    max   Σ (rev_od - α_od) × f_od
    s.t.  Σ f_od  ≤  capacity_s'        (vessel capacity)
          f_od    ≤  demand_od           (can't carry more than exists)
          f_od    ≥  0

This is solved greedily: sort OD pairs by adjusted revenue (rev_od - α_od)
descending, fill greedily up to capacity. Optimal because all items have
the same "weight" (1 FFE of capacity consumed regardless of OD pair).

INTERPRETATION
--------------
  rc(s') > CG_RC_TOLERANCE  →  adding this route improves the LP objective
                                → agents should propose it to the RMP
  rc(s') ≤ 0                →  this route cannot improve the LP
                                → discard it

Key insight for agents:
  - Uncovered OD pairs:  α_od = 0  →  adj_rev = full freight rate  (rich target)
  - Covered, congested:  α_od = μ_s > 0  →  adj_rev = rev - μ   (some value)
  - Covered, slack:      α_od = rev_od   →  adj_rev = 0          (no value)

So agents should always prioritise UNCOVERED high-revenue OD pairs first.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CG_RC_TOLERANCE


# ── Coverage helper ────────────────────────────────────────────────────────────

def compute_coverage(port_sequence: list, demand: dict) -> list:
    """
    Return list of (o, d) OD pairs covered by this CIRCULAR liner service.

    IMPORTANT — Liner services are cyclic rotations:
      A vessel on route [A, B, C] sails A→B→C→A→B→C→... continuously.
      This means it can carry cargo in BOTH directions around the cycle:
        - Outbound : A→B, A→C, B→C  (forward sub-paths)
        - Return   : C→A             (direct return leg)
      The original one-directional implementation missed ALL return-leg demand,
      leaving revenue on the table and producing wrong dual prices.

    Implementation: treat the sequence as a cycle by doubling it, then extract
    all consecutive sub-paths of length 1 to n-1 (we stop at n-1 to avoid
    covering the full round-trip back to the same port).
    """
    covered = set()
    n = len(port_sequence)
    if n < 2:
        return []

    # Circular sequence: double it so wrap-around sub-paths appear naturally
    cycle = port_sequence + port_sequence

    for start in range(n):                  # each port as a potential origin
        for length in range(1, n):          # paths of 1 up to n-1 legs (not a full loop)
            o = cycle[start]
            d = cycle[start + length]
            if o != d and (o, d) in demand:
                covered.add((o, d))

    return list(covered)


# ── Pricing subproblem (greedy knapsack) ──────────────────────────────────────

def solve_pricing_subproblem(
    covered_ods:  list,
    capacity_ffe: float,
    alpha:        dict,
    demand:       dict,
) -> tuple:
    """
    Solve the knapsack pricing subproblem greedily.

    Returns:
      (total_adj_revenue, cargo_plan)
      cargo_plan = list of (od, ffe_loaded, adj_rev_per_ffe)
    """
    if not covered_ods or capacity_ffe <= 0:
        return 0.0, []

    # Build items: (adj_rev_per_ffe, od, max_ffe_loadable)
    items = []
    for od in covered_ods:
        rec      = demand[od]
        rev_od   = rec.revenue_per_ffe
        alpha_od = alpha.get(od, 0.0)
        adj_rev  = rev_od - alpha_od
        if adj_rev > 0:
            items.append((adj_rev, od, rec.ffe_per_week))

    # Sort by adjusted revenue per FFE descending (greedy optimal for uniform weight)
    items.sort(key=lambda x: x[0], reverse=True)

    remaining  = capacity_ffe
    total_adj  = 0.0
    cargo_plan = []

    for (adj_rev, od, max_ffe) in items:
        if remaining <= 0:
            break
        load = min(max_ffe, remaining)
        total_adj  += adj_rev * load
        cargo_plan.append((od, round(load, 2), round(adj_rev, 4)))
        remaining  -= load

    return round(total_adj, 4), cargo_plan


# ── Main reduced cost function ─────────────────────────────────────────────────

def compute_reduced_cost(
    port_sequence: list,
    vessel_class:  str,
    alpha:         dict,
    fleet:         dict,
    ports:         dict,
    distances:     dict,
    demand:        dict,
    frequency:     int = 1,
) -> dict:
    """
    Compute the reduced cost of adding a candidate route to the RMP.

    Parameters
    ----------
    port_sequence : ordered list of UNLOCODEs
    vessel_class  : one of the 6 vessel classes
    alpha         : demand duals from RMP  { (o,d): α_od }
                    Pass {} for initial pricing (before first LP solve)
    fleet/ports/distances/demand : from stage0 loader
    frequency     : 1 = weekly service, 2 = bi-weekly

    Returns
    -------
    dict with keys:
      rc              : reduced cost value (positive = improving)
      is_improving    : True if rc > CG_RC_TOLERANCE
      adj_revenue     : Σ (rev_od - α_od) × f_od*  from knapsack
      operating_cost  : weekly operating cost of the route
      cargo_plan      : [(od, ffe_loaded, adj_rev_per_ffe), ...]
      covered_ods     : all OD pairs this route covers
      utilisation_pct : % of vessel capacity used in knapsack solution
      n_ods_covered   : number of OD pairs reachable
      n_ods_loaded    : number of OD pairs loaded in knapsack
      vessel_capacity : capacity in FFE
      error           : set if route is infeasible (rc = -inf)
    """
    from stage1.fleet_profiler import route_economics

    vessel = fleet.get(vessel_class)
    if vessel is None:
        return _error_result(f"Unknown vessel class '{vessel_class}'")

    # Operating cost via fleet_profiler
    econ = route_economics(vessel, port_sequence, ports, distances, frequency)
    if not econ.feasible:
        return _error_result(f"Route infeasible: {econ.infeasible_reason}")

    op_cost  = econ.weekly_total_cost
    capacity = vessel.capacity_ffe

    # Coverage
    covered_ods = compute_coverage(port_sequence, demand)
    if not covered_ods:
        return _error_result("No demand OD pairs covered by this route")

    # Pricing subproblem
    adj_revenue, cargo_plan = solve_pricing_subproblem(
        covered_ods, capacity, alpha, demand
    )

    # Reduced cost
    rc = adj_revenue - op_cost

    # Utilisation
    total_loaded = sum(ffe for (_, ffe, _) in cargo_plan)
    utilisation  = round(total_loaded / capacity * 100, 1) if capacity > 0 else 0.0

    return {
        "rc":              round(rc, 2),
        "is_improving":    rc > CG_RC_TOLERANCE,
        "adj_revenue":     round(adj_revenue, 2),
        "operating_cost":  round(op_cost, 2),
        "cargo_plan":      cargo_plan,
        "covered_ods":     covered_ods,
        "utilisation_pct": utilisation,
        "n_ods_covered":   len(covered_ods),
        "n_ods_loaded":    len(cargo_plan),
        "vessel_capacity": capacity,
        "cycle_days":      round(econ.cycle_days, 1),
        "vessels_needed":  econ.vessels_needed,
    }


def _error_result(reason: str) -> dict:
    return {
        "rc": -float('inf'),
        "is_improving": False,
        "adj_revenue": 0.0,
        "operating_cost": 0.0,
        "cargo_plan": [],
        "covered_ods": [],
        "utilisation_pct": 0.0,
        "n_ods_covered": 0,
        "n_ods_loaded": 0,
        "vessel_capacity": 0,
        "error": reason,
    }


# ── Batch helper: rank candidates by reduced cost ─────────────────────────────

def rank_candidates(
    candidates: list,
    alpha:      dict,
    fleet:      dict,
    ports:      dict,
    distances:  dict,
    demand:     dict,
) -> list:
    """
    Given list of (port_sequence, vessel_class, frequency) tuples,
    compute RC for each and return sorted best-first.

    Returns list of (rc_dict, port_sequence, vessel_class, frequency).
    Only returns routes where rc > -inf (skips totally infeasible ones).
    """
    results = []
    for (seq, vc, freq) in candidates:
        rc = compute_reduced_cost(seq, vc, alpha, fleet, ports, distances, demand, freq)
        if rc["rc"] > -float('inf'):
            results.append((rc, seq, vc, freq))

    results.sort(key=lambda x: x[0]["rc"], reverse=True)
    return results


# ── Quick heuristic: best vessel for a route ──────────────────────────────────

def best_vessel_for_route(
    port_sequence: list,
    alpha:         dict,
    fleet:         dict,
    ports:         dict,
    distances:     dict,
    demand:        dict,
    prefer_large:  bool = True,
) -> tuple:
    """
    Try all 6 vessel classes and return the (vessel_class, rc_dict) with
    the highest reduced cost.

    prefer_large=True  tries largest vessel first (good for high-demand routes)
    prefer_large=False tries smallest first (good for thin/niche routes)
    """
    order = ["Super_panamax", "Post_panamax", "Panamax_2400",
             "Panamax_1200", "Feeder_800", "Feeder_450"]
    if not prefer_large:
        order = list(reversed(order))

    best_vc = None
    best_rc = None

    for vc in order:
        rc = compute_reduced_cost(
            port_sequence, vc, alpha, fleet, ports, distances, demand
        )
        if rc["rc"] == -float('inf'):
            continue
        if best_rc is None or rc["rc"] > best_rc["rc"]:
            best_rc = rc
            best_vc = vc

    return best_vc, best_rc


# ── Run as script — full test suite ───────────────────────────────────────────

if __name__ == "__main__":
    from stage0.loader         import load_all, validate as validate_data
    from stage1.demand_intel   import run as run_intel
    from stage1.fleet_profiler import run as run_fleet
    from stage2.seed_generator import run as run_seeds

    data  = load_all(verbose=False)
    validate_data(data)
    intel = run_intel(data, verbose=False)
    fp    = run_fleet(data, verbose=False)

    fleet     = data['fleet']
    ports     = data['ports']
    distances = data['distances']
    demand    = data['demand']

    print("=" * 65)
    print("  Stage 3 — Part 2: Reduced Cost Calculator — Test Suite")
    print("=" * 65)

    # ── TEST BLOCK 1: Zero-alpha pricing (before first LP solve) ──────────────
    # With alpha=0 for all OD pairs, adj_rev = full freight rate
    # This represents the initial pricing pass on seed routes
    print("\n── Block 1: Zero-alpha pricing (initial, no LP yet) ──────────")
    alpha_zero = {}

    routes_b1 = [
        ("Asia-Europe Main",      ["CNSHA","CNYTN","SGSIN","EGPSD","DEBRV","NLRTM"], "Super_panamax"),
        ("Korea-Japan-HK-Europe", ["KRPUS","JPYOK","HKHKG","EGPSD","DEBRV","NLRTM"], "Super_panamax"),
        ("Transpacific",          ["CNSHA","CNYTN","KRPUS","USLAX"],                 "Super_panamax"),
        ("Asia-WestAfrica",       ["CNSHA","SGSIN","ESALG","NGAPP"],                 "Panamax_2400"),
        ("Asia-SouthAmerica",     ["CNSHA","SGSIN","ESALG","BRSSZ"],                 "Panamax_2400"),
        ("SAmerica-Europe",       ["BRSSZ","ESALG","DEBRV"],                         "Panamax_2400"),
    ]

    for name, seq, vc in routes_b1:
        rc = compute_reduced_cost(seq, vc, alpha_zero, fleet, ports, distances, demand)
        sign = "✓" if rc["is_improving"] else "✗"
        print(f"\n  {sign} {name}")
        print(f"    Vessel  : {vc}  ({rc['vessel_capacity']} FFE)")
        print(f"    Cycle   : {rc.get('cycle_days','?')}d  |  Vessels needed: {rc.get('vessels_needed','?')}")
        print(f"    RC      : ${rc['rc']:>14,.0f}")
        print(f"    AdjRev  : ${rc['adj_revenue']:>14,.0f}  |  OpCost: ${rc['operating_cost']:>12,.0f}")
        print(f"    Util    : {rc['utilisation_pct']}%  |  ODs loaded: {rc['n_ods_loaded']}/{rc['n_ods_covered']}")
        if rc['cargo_plan']:
            top3 = rc['cargo_plan'][:3]
            plan_str = "  ".join(f"{od[0]}->{od[1]}: {ffe:.0f}FFE @ ${adj:.0f}" for od,ffe,adj in top3)
            print(f"    Top cargo: {plan_str}")

    # ── TEST BLOCK 2: Post-LP alpha (simulate realistic duals) ───────────────
    print("\n\n── Block 2: Simulated post-LP alpha (partial coverage) ────────")
    # Simulate: seed routes cover some OD pairs — those pairs get alpha = revenue
    # Uncovered pairs keep alpha = 0
    seeds = run_seeds(data, intel, fp, verbose=False)
    covered_by_seeds = set()
    for s in seeds:
        covered_by_seeds |= set(compute_coverage(s.port_sequence, demand))

    # Covered pairs: alpha = ~80% of revenue (route not fully congested)
    # Uncovered pairs: alpha = 0 (nobody serves them yet)
    alpha_sim = {od: demand[od].revenue_per_ffe * 0.8
                 for od in covered_by_seeds}

    routes_b2 = [
        # These cover UNCOVERED pairs → should still show positive RC
        ("Korea→Bremerhaven (uncovered)",  ["KRPUS","SGSIN","DEBRV"],      "Panamax_2400"),
        ("Japan→Rotterdam (uncovered)",    ["JPYOK","SGSIN","NLRTM"],      "Panamax_2400"),
        ("Malaysia→Europe (uncovered)",    ["MYTPP","EGPSD","DEBRV"],      "Super_panamax"),
        # These cover ONLY already-covered pairs → RC should be low/negative
        ("Repeat Asia-Europe (covered)",   ["CNSHA","SGSIN","DEBRV"],      "Super_panamax"),
    ]

    for name, seq, vc in routes_b2:
        rc = compute_reduced_cost(seq, vc, alpha_sim, fleet, ports, distances, demand)
        sign = "✓" if rc["is_improving"] else "✗"
        covered_new = [od for od in rc['covered_ods'] if od not in covered_by_seeds]
        print(f"\n  {sign} {name}")
        print(f"    RC        : ${rc['rc']:>14,.0f}  |  Improving: {rc['is_improving']}")
        print(f"    AdjRev    : ${rc['adj_revenue']:>14,.0f}  |  OpCost: ${rc['operating_cost']:>12,.0f}")
        print(f"    New ODs   : {len(covered_new)} uncovered pairs in this route")

    # ── TEST BLOCK 3: best_vessel_for_route ───────────────────────────────────
    print("\n\n── Block 3: best_vessel_for_route() ───────────────────────────")
    test_seqs = [
        ["KRPUS","JPYOK","HKHKG","EGPSD","DEBRV","NLRTM"],   # long haul, needs big ship
        ["AOLAD","UYMVD"],                                     # shallow ports, needs feeder
        ["CNSHA","SGSIN","NGAPP"],                             # Africa run
    ]
    for seq in test_seqs:
        vc, rc = best_vessel_for_route(seq, alpha_zero, fleet, ports, distances, demand)
        if vc:
            print(f"\n  Route : {' → '.join(ports[p].name for p in seq)}")
            print(f"  Best vessel: {vc}  RC=${rc['rc']:,.0f}  util={rc['utilisation_pct']}%")
        else:
            print(f"\n  Route : {seq}  → no feasible vessel")

    # ── TEST BLOCK 4: Edge cases ───────────────────────────────────────────────
    print("\n\n── Block 4: Edge cases ────────────────────────────────────────")
    edge_cases = [
        ("Bad vessel class",      ["CNSHA","NLRTM"], "Giant_Ship",    1, True),
        ("Draft fail",            ["AOLAD","NLRTM"], "Super_panamax", 1, True),
        ("Valid 2-port",          ["CNSHA","NLRTM"], "Super_panamax", 1, False),
        ("All alpha = revenue",   None,              None,            None, None),  # special
    ]

    # Case: all alpha = revenue (every OD fully priced out)
    alpha_full = {od: rec.revenue_per_ffe for od, rec in demand.items()}
    rc_full = compute_reduced_cost(
        ["CNSHA","DEBRV"], "Super_panamax", alpha_full, fleet, ports, distances, demand
    )
    print(f"\n  All-alpha-saturated route: RC=${rc_full['rc']:,.0f}  "
          f"(expected ≤0 since adj_rev=0 but cost>0)")

    rc_bad = compute_reduced_cost(
        ["CNSHA","NLRTM"], "Giant_Ship", alpha_zero, fleet, ports, distances, demand
    )
    print(f"  Bad vessel class: error='{rc_bad.get('error')}'  rc={rc_bad['rc']}")

    rc_draft = compute_reduced_cost(
        ["AOLAD","NLRTM"], "Super_panamax", alpha_zero, fleet, ports, distances, demand
    )
    print(f"  Draft fail: error='{rc_draft.get('error')}'  rc={rc_draft['rc']}")

    # ── TEST BLOCK 5: rank_candidates() ───────────────────────────────────────
    print("\n\n── Block 5: rank_candidates() — top 5 from mixed pool ─────────")
    pool = [
        (["KRPUS","JPYOK","EGPSD","DEBRV"],        "Super_panamax", 1),
        (["CNSHA","EGPSD","DEBRV"],                 "Super_panamax", 1),
        (["MYTPP","EGPSD","NLRTM"],                 "Super_panamax", 1),
        (["AOLAD","UYMVD"],                         "Feeder_450",    1),
        (["CNSHA","SGSIN","ESALG","NGAPP"],          "Panamax_2400",  1),
        (["CNSHA","SGSIN","BRSSZ"],                  "Panamax_2400",  1),
        (["DEBRV","NLRTM","GBFXT","USEWR"],          "Super_panamax", 1),
    ]
    ranked = rank_candidates(pool, alpha_zero, fleet, ports, distances, demand)
    for i, (rc, seq, vc, freq) in enumerate(ranked[:5], 1):
        names = " → ".join(ports[p].name for p in seq)
        print(f"  #{i}  RC=${rc['rc']:>14,.0f}  {names}  [{vc}]")

    print("\n" + "=" * 65)
    print("  Part 2 complete — reduced_cost.py ready for Part 3 (RMP)")
    print("=" * 65)
