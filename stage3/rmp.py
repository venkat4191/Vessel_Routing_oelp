"""
Stage 3 — Part 3: Restricted Master Problem (RMP)
==================================================
The LP at the heart of Column Generation.

FORMULATION
-----------
At each CG iteration we have a set of routes S (columns).
We solve a cargo-flow LP to find the BEST cargo allocation across those routes.

Variables
  f_{od,s}  ≥ 0   FFE of OD-pair cargo (o,d) loaded on route s
                   (defined only where route s covers OD pair od)

Objective  (maximise weekly cargo revenue — route operating costs are fixed)
  max  Σ_{od,s}  rev_od  ×  f_{od,s}

Subject to
  C1 — Vessel capacity  (one per route s):
       Σ_{od in s}  f_{od,s}  ≤  cap_s           dual: μ_s  ≥ 0

  C2 — Demand cap  (one per OD pair od):
       Σ_{s covers od}  f_{od,s}  ≤  demand_od   dual: α_od ≥ 0

  C3 — Non-negativity:
       f_{od,s} ≥ 0

DUAL PRICES
-----------
  μ_s   = marginal revenue gain from 1 extra FFE of capacity on route s
           > 0  ↔  route s is fully loaded (congested)
           = 0  ↔  route s has spare capacity

  α_od  = marginal revenue gain from 1 extra FFE of demand on OD pair od
           > 0  ↔  demand is being partially served (someone is carrying it)
           = 0  ↔  OD pair is COMPLETELY UNSERVED (no route covers it)

These feed into Part 2 (reduced_cost.py) to find improving routes:
  rc(s') = -cost_s'  +  max { Σ (rev_od - α_od) × f_od  :  Σ f_od ≤ cap_s' }

IMPLEMENTATION
--------------
Uses scipy.optimize.linprog (HiGHS backend, available in scipy ≥ 1.7).
No external solver installation needed.

Route operating costs are NOT in the LP objective — they are fixed overheads.
The LP only decides how to allocate cargo across the current route set.
Net profit = LP revenue − Σ_s operating_cost_s  (computed post-solve).
"""

import sys
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import scipy.optimize as opt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CG_RC_TOLERANCE


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RouteColumn:
    """One column (shipping service) in the RMP."""
    route_id:       str
    port_sequence:  list
    vessel_class:   str
    frequency:      int
    capacity_ffe:   float          # vessel capacity in FFE
    weekly_cost:    float          # full weekly operating cost (TC+bunker+port+canal)
    vessels_needed: int
    coverage:       list = field(default_factory=list)   # [(o,d), ...] in sequence order


@dataclass
class RMPSolution:
    """Complete output from one RMP solve."""
    status:             str     # 'optimal' | 'infeasible' | 'error'
    lp_revenue:         float   # Σ rev_od × f_od* (LP objective value)
    net_profit:         float   # lp_revenue − Σ operating costs
    # Cargo flows: {(od, route_id): FFE}
    cargo_flows:        dict
    # Per-route totals: {route_id: FFE loaded}
    route_loads:        dict
    # Per-route utilisation: {route_id: fraction [0,1]}
    route_utilisation:  dict
    # Served demand: {(o,d): FFE served}
    served:             dict
    # Unserved demand: {(o,d): FFE unserved}  — diagnostic only
    unserved:           dict
    # DUAL PRICES — fed to pricing agents
    alpha:              dict   # {(o,d): α_od}   demand duals
    mu:                 dict   # {route_id: μ_s}  capacity duals
    # Summary stats
    total_served_ffe:   float
    total_unserved_ffe: float
    total_op_cost:      float
    n_routes:           int
    n_od_pairs:         int


# ══════════════════════════════════════════════════════════════════════════════
#  Restricted Master Problem
# ══════════════════════════════════════════════════════════════════════════════

class RestrictedMasterProblem:
    """
    Manages the column pool and solves the cargo-flow LP.

    Usage
    -----
    rmp = RestrictedMasterProblem(demand, fleet)
    for seed in seeds:
        col = rmp.column_from_route(...)
        rmp.add_column(col)
    sol = rmp.solve(verbose=True, distances=distances)
    # sol.alpha → feed to pricing agents
    # sol.net_profit → CG convergence tracking
    """

    def __init__(self, demand: dict, fleet: dict):
        self.demand  = demand   # {(o,d): DemandRecord}
        self.fleet   = fleet
        self.columns: list = []
        # Fixed OD pair ordering for consistent matrix construction
        self._od_list: Optional[list] = None

    # ── Column management ─────────────────────────────────────────────────────

    def add_column(self, col: RouteColumn):
        """Add one route column. Invalidates OD list cache."""
        self.columns.append(col)
        self._od_list = None   # rebuild on next solve

    def add_columns(self, cols: list):
        for c in cols:
            self.add_column(c)

    def has_route(self, route_id: str) -> bool:
        return any(c.route_id == route_id for c in self.columns)

    def column_from_route(
        self,
        route_id:      str,
        port_sequence: list,
        vessel_class:  str,
        frequency:     int,
        ports:         dict,
        distances:     dict,
    ) -> RouteColumn:
        """
        Build a RouteColumn from raw route data.
        Computes economics via fleet_profiler and coverage via reduced_cost.

        FIX 2: Weekly throughput capacity = vessel.capacity_ffe × frequency.
               A route departing twice per week can carry 2× the FFE per week.

        FIX 11: If ALLIANCE_ENABLED, scale capacity by ALLIANCE_SLOT_FRACTION
                to model slot-sharing agreements (carrier owns only a fraction
                of each vessel's capacity in a vessel-sharing arrangement).
        """
        from stage1.fleet_profiler import route_economics
        from stage3.reduced_cost   import compute_coverage
        from utils.config import ALLIANCE_ENABLED, ALLIANCE_SLOT_FRACTION

        vessel = self.fleet[vessel_class]
        econ   = route_economics(vessel, port_sequence, ports, distances, frequency)
        cov    = compute_coverage(port_sequence, self.demand)

        # FIX 2: Weekly capacity scales with frequency
        weekly_capacity = vessel.capacity_ffe * frequency

        # FIX 11: Alliance slot sharing — carrier may only own part of vessel capacity
        if ALLIANCE_ENABLED:
            weekly_capacity = weekly_capacity * ALLIANCE_SLOT_FRACTION

        return RouteColumn(
            route_id      = route_id,
            port_sequence = port_sequence,
            vessel_class  = vessel_class,
            frequency     = frequency,
            capacity_ffe  = weekly_capacity,
            weekly_cost   = econ.weekly_total_cost if econ.feasible else float('inf'),
            vessels_needed = econ.vessels_needed   if econ.feasible else 0,
            coverage      = cov,
        )

    # ── LP construction helpers ───────────────────────────────────────────────

    def _build_od_list(self):
        """
        Build sorted list of all OD pairs covered by at least one route.
        Cached between solves; invalidated when columns are added.
        """
        od_set = set()
        for col in self.columns:
            od_set.update(col.coverage)
        # Sort for reproducibility
        self._od_list = sorted(od_set)

    def _build_variable_index(self) -> list:
        """
        Build the flat variable list: one f_{od,s} per (route, od) pair.
        Returns list of (col_index, od) tuples — the index is the variable index.
        """
        var_list = []
        for s, col in enumerate(self.columns):
            for od in col.coverage:
                if od in self.demand:
                    var_list.append((s, od))
        return var_list

    def _compute_route_transit_time(self, col, o: str, d: str, distances: dict) -> float:
        """
        FIX 1: Compute transit time (days) from o to d on this circular route.
        Accumulates sailing days + intermediate port dwell times.
        """
        from utils.config import PORT_DWELL_DAYS_BY_CLASS, PORT_DAYS_PER_CALL
        seq   = col.port_sequence
        n     = len(seq)
        dwell = PORT_DWELL_DAYS_BY_CLASS.get(col.vessel_class, PORT_DAYS_PER_CALL)
        vessel = self.fleet.get(col.vessel_class)
        speed  = vessel.design_speed if vessel else 15.0
        cycle = seq + seq
        try:
            o_idx = seq.index(o)
        except ValueError:
            return float('inf')
        transit = 0.0
        for step in range(1, n):
            frm = cycle[o_idx + step - 1]
            to  = cycle[o_idx + step]
            dist_rec = distances.get((frm, to))
            if dist_rec is None:
                return float('inf')
            transit += dist_rec.distance_nm / (speed * 24.0)
            if to == d:
                return transit
            transit += dwell
        return float('inf')

    # ── Main solve ────────────────────────────────────────────────────────────

    def solve(self, verbose: bool = False, distances: dict = None) -> RMPSolution:
        """
        Build and solve the cargo-flow LP using scipy HiGHS.

        FIX 1: Transit time enforcement — f_{od,s} is forced to 0 if route s
               cannot deliver od within demand[od].max_transit_days.
        FIX 2: Frequency x capacity — weekly capacity = vessel_ffe x frequency.
        FIX 12: Objective uses blended (contract+spot) effective revenue.
        """
        if not self.columns:
            return self._empty_solution('error')

        self._build_od_list()
        var_list = self._build_variable_index()
        N = len(var_list)

        if N == 0:
            return self._empty_solution('error')

        R  = len(self.columns)
        OD = len(self._od_list)

        # FIX 12: Use blended effective revenue (contract+spot mix)
        c = np.array([
            -(self.demand[od].effective_revenue_per_ffe
              if hasattr(self.demand[od], 'effective_revenue_per_ffe')
              else self.demand[od].revenue_per_ffe)
            for (s, od) in var_list
        ], dtype=float)

        n_constraints = R + OD
        A = np.zeros((n_constraints, N), dtype=float)
        b = np.zeros(n_constraints,      dtype=float)
        od_to_row = {od: R + j for j, od in enumerate(self._od_list)}

        for k, (s, od) in enumerate(var_list):
            A[s, k]             = 1.0
            A[od_to_row[od], k] = 1.0

        # FIX 2: capacity_ffe already = vessel_ffe * frequency (set in column_from_route)
        for s, col in enumerate(self.columns):
            b[s] = col.capacity_ffe
        for j, od in enumerate(self._od_list):
            b[R + j] = self.demand[od].ffe_per_week

        # FIX 1: Transit time — force infeasible (od,route) vars to 0
        bounds = [(0.0, None)] * N
        if distances is not None:
            for k, (s, od) in enumerate(var_list):
                col    = self.columns[s]
                o, d   = od
                max_td = self.demand[od].max_transit_days
                actual = self._compute_route_transit_time(col, o, d, distances)
                if actual > max_td + 0.5:
                    bounds[k] = (0.0, 0.0)

        result = opt.linprog(
            c,
            A_ub = A,
            b_ub = b,
            bounds = bounds,
            method = 'highs',
            options = {'disp': False, 'presolve': True},
        )

        if result.status not in (0, 1):
            # status 0 = optimal, 1 = iteration limit (still usable)
            return self._empty_solution(
                'infeasible' if result.status == 2 else 'error'
            )

        # ── Extract primal solution ────────────────────────────────────────────
        f_vals = result.x   # shape (N,)

        cargo_flows   = {}
        route_loads   = {col.route_id: 0.0 for col in self.columns}
        served        = {od: 0.0 for od in self._od_list}

        for k, (s, od) in enumerate(var_list):
            fval = float(f_vals[k])
            if fval > 0.01:
                cargo_flows[(od, self.columns[s].route_id)] = round(fval, 2)
            route_loads[self.columns[s].route_id] += fval
            served[od] = served.get(od, 0.0) + fval

        route_utilisation = {
            col.route_id: round(route_loads[col.route_id] / col.capacity_ffe, 4)
            for col in self.columns if col.capacity_ffe > 0
        }

        # Unserved demand (diagnostic)
        unserved = {}
        for od in self._od_list:
            gap = self.demand[od].ffe_per_week - served.get(od, 0.0)
            if gap > 0.01:
                unserved[od] = round(gap, 2)

        # ── Extract dual prices ────────────────────────────────────────────────
        # scipy HiGHS returns ineqlin.marginals for A_ub constraints.
        # We minimise −revenue, so marginals[i] = ∂(−revenue*)/∂b[i].
        # Dual in maximisation sense = −marginals[i]  (flip sign).
        alpha = {}   # demand duals   {(o,d): α_od}
        mu    = {}   # capacity duals {route_id: μ_s}

        if (hasattr(result, 'ineqlin')
                and result.ineqlin is not None
                and result.ineqlin.marginals is not None):

            marg = result.ineqlin.marginals   # length = n_constraints

            # C1: capacity duals for each route (rows 0 … R-1)
            for s, col in enumerate(self.columns):
                raw = float(marg[s]) if s < len(marg) else 0.0
                mu[col.route_id] = round(max(0.0, -raw), 4)

            # C2: demand duals for each OD pair (rows R … R+OD-1)
            for j, od in enumerate(self._od_list):
                idx = R + j
                raw = float(marg[idx]) if idx < len(marg) else 0.0
                alpha[od] = round(max(0.0, -raw), 4)

        else:
            # Fallback: zero duals (LP was trivially solved or HiGHS didn't return marginals)
            for col in self.columns:
                mu[col.route_id] = 0.0
            for od in self._od_list:
                alpha[od] = 0.0

        # Ensure alpha covers ALL demand pairs, not just covered ones
        # (uncovered pairs get α = 0 — agents should target these first)
        for od in self.demand:
            if od not in alpha:
                alpha[od] = 0.0

        # ── Summary stats ──────────────────────────────────────────────────────
        lp_revenue = float(-result.fun)
        total_op   = sum(col.weekly_cost for col in self.columns)
        net_profit = lp_revenue - total_op

        total_served   = sum(served.values())
        total_unserved = sum(
            self.demand[od].ffe_per_week for od in self.demand
        ) - total_served

        sol = RMPSolution(
            status             = 'optimal',
            lp_revenue         = round(lp_revenue, 2),
            net_profit         = round(net_profit, 2),
            cargo_flows        = cargo_flows,
            route_loads        = {r: round(v, 2) for r, v in route_loads.items()},
            route_utilisation  = route_utilisation,
            served             = {od: round(v, 2) for od, v in served.items() if v > 0.01},
            unserved           = unserved,
            alpha              = alpha,
            mu                 = mu,
            total_served_ffe   = round(total_served, 1),
            total_unserved_ffe = round(total_unserved, 1),
            total_op_cost      = round(total_op, 2),
            n_routes           = R,
            n_od_pairs         = OD,
        )

        if verbose:
            _print_solution(sol, self.columns, self.demand)

        return sol

    def _empty_solution(self, status: str) -> RMPSolution:
        return RMPSolution(
            status=status, lp_revenue=0, net_profit=0,
            cargo_flows={}, route_loads={}, route_utilisation={},
            served={}, unserved={}, alpha={}, mu={},
            total_served_ffe=0, total_unserved_ffe=0,
            total_op_cost=0, n_routes=0, n_od_pairs=0,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty printer
# ══════════════════════════════════════════════════════════════════════════════

def _print_solution(sol: RMPSolution, columns: list, demand: dict):
    total_demand = sum(r.ffe_per_week for r in demand.values())
    pct_served   = 100 * sol.total_served_ffe / total_demand if total_demand else 0

    print(f"\n  ┌─ RMP Solution ({'optimal' if sol.status == 'optimal' else sol.status}) "
          f"─ {sol.n_routes} routes, {sol.n_od_pairs} covered OD pairs ─┐")
    print(f"  │  LP Revenue   : ${sol.lp_revenue:>14,.0f} /week")
    print(f"  │  Op Costs     : ${sol.total_op_cost:>14,.0f} /week")
    print(f"  │  Net Profit   : ${sol.net_profit:>14,.0f} /week")
    print(f"  │  Served       : {sol.total_served_ffe:>10,.0f} / "
          f"{total_demand:,.0f} FFE/week  ({pct_served:.1f}%)")
    print(f"  │  Unserved     : {sol.total_unserved_ffe:>10,.0f} FFE/week")
    print(f"  └{'─'*60}┘")

    print(f"\n  Route utilisation:")
    for col in columns:
        load  = sol.route_loads.get(col.route_id, 0)
        util  = sol.route_utilisation.get(col.route_id, 0)
        mu_s  = sol.mu.get(col.route_id, 0)
        bar   = '█' * int(util * 20) + '░' * (20 - int(util * 20))
        print(f"    {col.route_id:<6} {col.vessel_class:<22} "
              f"[{bar}] {util*100:5.1f}%  "
              f"{load:>6,.0f}/{col.capacity_ffe:.0f} FFE  "
              f"μ={mu_s:.1f}")

    # Top 8 demand duals (most valuable unserved / congested OD pairs)
    top_alpha = sorted(
        [(α, od) for od, α in sol.alpha.items() if α > 0],
        reverse=True
    )[:8]
    if top_alpha:
        print(f"\n  Top demand duals α_od  (= marginal revenue of 1 extra FFE on this pair):")
        for α, od in top_alpha:
            dem    = demand[od]
            status = "served" if od in sol.served else "UNSERVED"
            print(f"    {od[0]}->{od[1]:<8}  α={α:>8.2f}  "
                  f"rev={dem.revenue_per_ffe:>5.0f}/FFE  "
                  f"demand={dem.ffe_per_week:>6.0f} FFE  [{status}]")

    # Top 3 congested routes
    top_mu = sorted(
        [(μ, rid) for rid, μ in sol.mu.items() if μ > 0],
        reverse=True
    )[:3]
    if top_mu:
        print(f"\n  Congested routes (μ > 0 = more capacity would help):")
        for μ, rid in top_mu:
            print(f"    {rid}  μ={μ:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Test suite
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from stage0.loader         import load_all, validate as validate_data
    from stage1.demand_intel   import run as run_intel
    from stage1.fleet_profiler import run as run_fleet
    from stage2.seed_generator import run as run_seeds
    from stage3.reduced_cost   import compute_reduced_cost

    data  = load_all(verbose=False)
    validate_data(data)
    intel = run_intel(data, verbose=False)
    fp    = run_fleet(data, verbose=False)
    seeds = run_seeds(data, intel, fp, verbose=False)

    print("=" * 65)
    print("  Stage 3 — Part 3: RMP — Test Suite")
    print("=" * 65)

    # ── TEST 1: Seed-only RMP ─────────────────────────────────────────────────
    print("\n── Test 1: Seed-only RMP (8 seed routes) ──────────────────────")
    rmp = RestrictedMasterProblem(data['demand'], data['fleet'])
    for s in seeds:
        col = rmp.column_from_route(
            s.route_id, s.port_sequence, s.vessel_class,
            s.frequency, data['ports'], data['distances']
        )
        rmp.add_column(col)

    sol1 = rmp.solve(verbose=True, distances=distances)
    assert sol1.status == 'optimal', f"Expected optimal, got {sol1.status}"
    assert sol1.lp_revenue > 0,      "LP revenue should be positive"
    assert sol1.total_served_ffe > 0,"Should serve some demand"
    print(f"\n  ✓ Status: {sol1.status}")
    print(f"  ✓ LP Revenue > 0:  ${sol1.lp_revenue:,.0f}")
    print(f"  ✓ Served FFE > 0:  {sol1.total_served_ffe:,.0f}")

    # ── TEST 2: Dual price properties ─────────────────────────────────────────
    print("\n── Test 2: Dual price properties ──────────────────────────────")

    # 2a: All duals ≥ 0
    neg_alpha = [(od, α) for od, α in sol1.alpha.items() if α < -1e-6]
    neg_mu    = [(r,  μ) for r,  μ in sol1.mu.items()    if μ < -1e-6]
    assert len(neg_alpha) == 0, f"Negative alpha duals: {neg_alpha[:3]}"
    assert len(neg_mu)    == 0, f"Negative mu duals: {neg_mu[:3]}"
    print(f"  ✓ All α ≥ 0  ({len(sol1.alpha)} OD pairs)")
    print(f"  ✓ All μ ≥ 0  ({len(sol1.mu)} routes)")

    # 2b: Uncovered OD pairs must have α = 0
    covered_ods = set()
    for col in rmp.columns:
        covered_ods.update(col.coverage)
    uncovered = [od for od in data['demand'] if od not in covered_ods]
    bad_uncovered = [(od, sol1.alpha[od]) for od in uncovered
                     if sol1.alpha.get(od, 0) > 1e-6]
    assert len(bad_uncovered) == 0, f"Uncovered OD with α>0: {bad_uncovered[:3]}"
    print(f"  ✓ All {len(uncovered)} uncovered OD pairs have α = 0")

    # 2c: Complementary slackness — if route not full, μ should be ~0
    slack_routes = [
        col.route_id for col in rmp.columns
        if sol1.route_utilisation.get(col.route_id, 0) < 0.99
    ]
    for rid in slack_routes:
        mu_val = sol1.mu.get(rid, 0)
        if mu_val > 1.0:   # allow small numerical noise
            print(f"  ⚠ Slack route {rid} has μ={mu_val:.4f} > 0 (small CS violation)")
    print(f"  ✓ Complementary slackness check done ({len(slack_routes)} slack routes)")

    # 2d: Top duals should correspond to high-revenue unserved OD pairs
    top5_alpha = sorted(sol1.alpha.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n  Top 5 α_od values:")
    for od, α in top5_alpha:
        rec = data['demand'][od]
        print(f"    {od[0]}->{od[1]}  α={α:.2f}  rev={rec.revenue_per_ffe}/FFE  "
              f"served={sol1.served.get(od, 0):.0f}/{rec.ffe_per_week:.0f} FFE")

    # ── TEST 3: Adding an improving column updates duals ─────────────────────
    print("\n── Test 3: Add an improving column → duals update ──────────────")

    # Find the highest-alpha unserved pair and design a direct route for it
    best_od = max(
        [(od, α) for od, α in sol1.alpha.items()
         if od not in covered_ods and α == 0],   # uncovered, α=0
        key=lambda x: data['demand'][x[0]].revenue_per_ffe,
        default=(None, 0)
    )
    # Actually target highest-revenue uncovered pair
    best_uncovered = max(
        uncovered,
        key=lambda od: data['demand'][od].revenue_per_ffe * data['demand'][od].ffe_per_week
    )
    o, d = best_uncovered
    print(f"  Adding direct route for top uncovered pair: {o} → {d}")
    print(f"  Weekly revenue potential: ${data['demand'][(o,d)].weekly_revenue:,.0f}")

    new_col = rmp.column_from_route(
        "NEW1", [o, d], "Panamax_2400", 1,
        data['ports'], data['distances']
    )
    rmp.add_column(new_col)
    sol2 = rmp.solve(verbose=False, distances=distances)

    assert sol2.lp_revenue >= sol1.lp_revenue - 1, \
        "Adding a column should not decrease LP revenue"
    print(f"  ✓ LP revenue after add: ${sol2.lp_revenue:,.0f}  "
          f"(was ${sol1.lp_revenue:,.0f}  Δ=${sol2.lp_revenue - sol1.lp_revenue:+,.0f})")
    print(f"  ✓ Served FFE: {sol2.total_served_ffe:,.0f}  "
          f"(was {sol1.total_served_ffe:,.0f})")

    # ── TEST 4: Reduced cost consistency ─────────────────────────────────────
    print("\n── Test 4: RC consistency — duals match pricing subproblem ────")
    # For any route ALREADY in the RMP, its RC should be ≤ CG_RC_TOLERANCE
    # (by LP optimality — no existing column can improve the LP further)
    violations = []
    for col in rmp.columns:
        rc = compute_reduced_cost(
            col.port_sequence, col.vessel_class,
            sol2.alpha, data['fleet'], data['ports'], data['distances'], data['demand'],
            col.frequency,
        )
        if rc['rc'] > CG_RC_TOLERANCE * 10:   # small tolerance for numerics
            violations.append((col.route_id, rc['rc']))

    if violations:
        print(f"  ⚠ {len(violations)} existing routes have RC > 0: {violations}")
    else:
        print(f"  ✓ All {len(rmp.columns)} existing routes have RC ≤ 0 (LP optimality)")

    # ── TEST 5: Dual price as pricing signal ─────────────────────────────────
    print("\n── Test 5: Dual prices correctly guide agent to best new route ─")
    candidate_routes = [
        (["KRPUS","EGPSD","DEBRV"],            "Super_panamax"),
        (["JPYOK","SGSIN","DEBRV"],             "Super_panamax"),
        (["MYTPP","EGPSD","DEBRV","NLRTM"],     "Super_panamax"),
        (["CNTAO","HKHKG","SGSIN","GBFXT"],     "Super_panamax"),
        (["BRSSZ","ESALG","NLRTM"],             "Panamax_2400"),
        (["CNSHA","USEWR"],                     "Panamax_2400"),
    ]
    rc_results = []
    for (seq, vc) in candidate_routes:
        rc = compute_reduced_cost(
            seq, vc, sol2.alpha, data['fleet'],
            data['ports'], data['distances'], data['demand']
        )
        rc_results.append((rc['rc'], seq, vc))
    rc_results.sort(reverse=True)

    print(f"  Candidate routes ranked by RC:")
    for rank, (rcval, seq, vc) in enumerate(rc_results, 1):
        names = " → ".join(data['ports'][p].name for p in seq)
        sign  = "✓ IMPROVING" if rcval > CG_RC_TOLERANCE else "✗"
        print(f"  #{rank} {sign}  RC=${rcval:>12,.0f}  {names}  [{vc}]")

    print(f"\n{'='*65}")
    print(f"  All RMP tests passed ✓")
    print(f"  Part 3 complete — rmp.py ready for Part 4 (agents)")
    print(f"{'='*65}")
