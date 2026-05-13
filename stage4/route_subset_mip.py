"""
Stage 4 — Part 2: Route-subset MILP (branch-and-bound via HiGHS)
===============================================================
Selects a profit-maximising subset of CG routes and a joint cargo allocation.

Variables
---------
  f_{od,r} ≥ 0   FFE of OD pair od carried on route r (only for od ∈ coverage(r))
  x_r ∈ {0,1}    operate route r or not

Objective (maximise, implemented as minimise of negative)
  Σ_{od,r}  rev_od · f_{od,r}  −  Σ_r  weekly_cost_r · x_r

Constraints
-----------
  Σ_{od} f_{od,r} ≤ cap_r · x_r          vessel capacity if the service runs
  Σ_{r covers od} f_{od,r} ≤ demand_od   OD demand caps
  Optional: Σ_{r uses class v} vessels_needed_r · x_r ≤ cap_v  (see config)

Solver: `scipy.optimize.milp` (HiGHS MIP). MIP relative gap from `BB_MIP_GAP`.

Reads : outputs/cg_routes.json (via `cg_columns.load_cg_columns`)
Writes: outputs/bb_result.json
"""

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import OUTPUTS_DIR, BB_MIP_GAP, STAGE4_MAX_VESSELS_PER_CLASS
from utils.bb_accounting import recompute_bb_financials
from stage0.loader import load_all, validate as validate_data
from stage3.rmp import RouteColumn
from stage4.cg_columns import load_cg_columns, default_cg_routes_path


@dataclass
class BBResult:
    status: str
    mip_gap: float
    lp_revenue: float
    net_profit: float
    n_routes_selected: int
    n_routes_pool: int
    total_served_ffe: float
    selected_route_ids: List[str]
    cargo_flows: Dict[str, float]  # "(o,d)|route_id" -> FFE
    route_loads: Dict[str, float]
    weekly_op_cost_selected: float
    total_vessels_used: float
    vessels_used_by_class: Dict[str, float]
    vessels_cap_by_class: Dict[str, float]
    solve_time_sec: float
    message: str


def _var_index_map(
    columns: List[RouteColumn], demand: dict
) -> Tuple[List[Tuple[int, Tuple[str, str]]], List[Tuple[str, str]]]:
    """
    Flatten (route_idx, od) for every od in column.coverage that exists in demand.
    Returns var_list and sorted unique od_list appearing in those vars.
    """
    var_list: List[Tuple[int, Tuple[str, str]]] = []
    for s, col in enumerate(columns):
        for od in col.coverage:
            if od in demand:
                var_list.append((s, od))
    od_set = {od for _, od in var_list}
    od_list = sorted(od_set)
    return var_list, od_list


def build_and_solve(
    columns: List[RouteColumn],
    demand: dict,
    max_vessels_per_class: Optional[Dict[str, float]] = None,
    mip_rel_gap: float = BB_MIP_GAP,
    time_limit: Optional[float] = None,
    min_served_ffe: Optional[float] = None,
    min_routes_selected: Optional[int] = None,
    coverage_bonus_per_ffe: float = 0.0,
    min_net_profit: Optional[float] = None,
    vessel_mix_targets: Optional[Dict[str, float]] = None,
    vessel_mix_target_fraction_of_cap: Optional[float] = None,
    vessel_mix_penalty_per_vessel: float = 0.0,
    min_selected_route_utilization: Optional[float] = None,
    large_vessel_bonus_per_ffe_capacity: float = 0.0,
    route_activation_penalty: float = 0.0,
    ports: Optional[dict] = None,
) -> BBResult:
    """
    Construct the MILP and solve. `columns` should match CG output (finite costs).
    """
    R = len(columns)
    if R == 0:
        return BBResult(
            status="empty", mip_gap=0.0, lp_revenue=0.0, net_profit=0.0,
            n_routes_selected=0, n_routes_pool=0, total_served_ffe=0.0,
            selected_route_ids=[], cargo_flows={}, route_loads={},
            weekly_op_cost_selected=0.0,
            total_vessels_used=0.0,
            vessels_used_by_class={},
            vessels_cap_by_class={},
            solve_time_sec=0.0,
            message="no columns",
        )

    var_list, od_list = _var_index_map(columns, demand)
    K = len(var_list)
    n_vars = K + R

    if K == 0:
        return BBResult(
            status="no_coverage", mip_gap=0.0, lp_revenue=0.0,
            net_profit=-sum(c.weekly_cost for c in columns if c.weekly_cost < 1e300),
            n_routes_selected=0, n_routes_pool=R, total_served_ffe=0.0,
            selected_route_ids=[], cargo_flows={}, route_loads={},
            weekly_op_cost_selected=0.0,
            total_vessels_used=0.0,
            vessels_used_by_class={},
            vessels_cap_by_class={},
            solve_time_sec=0.0,
            message="no OD variables",
        )

    caps = max_vessels_per_class or STAGE4_MAX_VESSELS_PER_CLASS
    mix_pen = float(vessel_mix_penalty_per_vessel or 0.0)
    mix_targets: Dict[str, float] = {}
    if vessel_mix_targets:
        mix_targets = {k: float(v) for k, v in vessel_mix_targets.items() if v is not None}
    elif vessel_mix_target_fraction_of_cap is not None and caps:
        frac = float(vessel_mix_target_fraction_of_cap)
        for vc, cap in caps.items():
            if cap is None:
                continue
            mix_targets[vc] = float(cap) * frac

    mix_classes = sorted(
        vc for vc in mix_targets.keys()
        if mix_pen > 0 and vc in {c.vessel_class for c in columns}
    )
    M = len(mix_classes)

    # Objective: min  -Σ (rev + bonus)·f + Σ cost·x
    # coverage_bonus_per_ffe allows soft push toward higher served volume.
    n_vars = K + R + 2 * M
    c = np.zeros(n_vars, dtype=float)
    for k, (s, od) in enumerate(var_list):
        c[k] = -(float(demand[od].revenue_per_ffe) + float(coverage_bonus_per_ffe))
    for s in range(R):
        # Optional big-ship preference (soft): subtract bonus*capacity from fixed cost.
        c[K + s] = float(columns[s].weekly_cost) - (
            float(large_vessel_bonus_per_ffe_capacity) * float(columns[s].capacity_ffe)
        ) + float(route_activation_penalty)
    # Vessel-mix deviation penalty (L1): penalize |used_vc - target_vc|
    # Implemented via pos/neg slack vars per vessel class.
    for i in range(M):
        c[K + R + 2 * i] = mix_pen      # pos
        c[K + R + 2 * i + 1] = mix_pen  # neg

    # Map (s, od) -> k
    sod_to_k = {(s, od): k for k, (s, od) in enumerate(var_list)}
    od_to_ks: Dict[Tuple[str, str], List[int]] = {}
    for k, (s, od) in enumerate(var_list):
        od_to_ks.setdefault(od, []).append(k)
    route_to_ks: Dict[int, List[int]] = {}
    for k, (s, od) in enumerate(var_list):
        route_to_ks.setdefault(s, []).append(k)

    rows: List[np.ndarray] = []
    lb: List[float] = []
    ub: List[float] = []

    # C1: per-leg capacity linking (enables unload/reload slot reuse)
    # For each route leg l in s:
    #   Σ_{od traversing leg l on route s} f_{od,s} <= cap_s * x_s
    # This is more realistic than a single route-wide capacity constraint.
    route_leg_to_ks: Dict[Tuple[int, int], List[int]] = {}
    for k, (s, od) in enumerate(var_list):
        seq = columns[s].port_sequence
        try:
            i = seq.index(od[0])
            j = seq.index(od[1])
        except ValueError:
            continue
        if i >= j:
            continue
        for leg in range(i, j):
            route_leg_to_ks.setdefault((s, leg), []).append(k)

    for s, col in enumerate(columns):
        n_legs = max(len(col.port_sequence) - 1, 0)
        for leg in range(n_legs):
            row = np.zeros(n_vars)
            for k in route_leg_to_ks.get((s, leg), []):
                row[k] = 1.0
            row[K + s] = -float(col.capacity_ffe)
            rows.append(row)
            lb.append(-np.inf)
            ub.append(0.0)

    # C2: demand caps
    for od in od_list:
        row = np.zeros(n_vars)
        for k in od_to_ks[od]:
            row[k] = 1.0
        rows.append(row)
        lb.append(-np.inf)
        ub.append(float(demand[od].ffe_per_week))

    # C3: optional fleet caps per vessel class
    if caps:
        by_class: Dict[str, List[int]] = {}
        for s, col in enumerate(columns):
            by_class.setdefault(col.vessel_class, []).append(s)
        for vc, cap in caps.items():
            if vc not in by_class or cap is None:
                continue
            row = np.zeros(n_vars)
            for s in by_class[vc]:
                row[K + s] = float(columns[s].vessels_needed)
            rows.append(row)
            lb.append(-np.inf)
            ub.append(float(cap))

    # C3b: optional vessel-mix targets (soft, via deviation variables)
    # Σ vessels_needed x  - pos + neg == target
    if M > 0:
        by_class: Dict[str, List[int]] = {}
        for s, col in enumerate(columns):
            by_class.setdefault(col.vessel_class, []).append(s)
        for i, vc in enumerate(mix_classes):
            if vc not in by_class:
                continue
            row = np.zeros(n_vars)
            for s in by_class[vc]:
                row[K + s] = float(columns[s].vessels_needed)
            pos_idx = K + R + 2 * i
            neg_idx = K + R + 2 * i + 1
            row[pos_idx] = -1.0
            row[neg_idx] = 1.0
            tgt = float(mix_targets.get(vc, 0.0))
            rows.append(row)
            lb.append(tgt)
            ub.append(tgt)

    # C4: optional minimum served-FFE floor
    if min_served_ffe is not None and float(min_served_ffe) > 0:
        row = np.zeros(n_vars)
        row[:K] = 1.0
        rows.append(row)
        lb.append(float(min_served_ffe))
        ub.append(np.inf)

    # C4b: optional minimum number of selected routes
    # Σ_s x_s >= min_routes_selected
    if min_routes_selected is not None and int(min_routes_selected) > 0:
        row = np.zeros(n_vars)
        row[K:K + R] = 1.0
        rows.append(row)
        lb.append(float(int(min_routes_selected)))
        ub.append(np.inf)

    # C5: optional per-selected-route utilization floor
    # For each route s:
    #    Σ_od f_{od,s} - min_util*cap_s*x_s >= 0
    util = (
        float(min_selected_route_utilization)
        if min_selected_route_utilization is not None else None
    )
    if util is not None and util > 0:
        util = max(0.0, min(1.0, util))
        for s, col in enumerate(columns):
            row = np.zeros(n_vars)
            for k in route_to_ks.get(s, []):
                row[k] = 1.0
            row[K + s] = -float(util) * float(col.capacity_ffe)
            rows.append(row)
            lb.append(0.0)
            ub.append(np.inf)

    # C6: optional minimum net-profit floor
    # Σ (rev+bonus)f - Σ cost x >= min_net_profit
    if min_net_profit is not None:
        row = np.zeros(n_vars)
        for k, (s, od) in enumerate(var_list):
            row[k] = float(demand[od].revenue_per_ffe) + float(coverage_bonus_per_ffe)
        for s in range(R):
            row[K + s] = -float(columns[s].weekly_cost)
        rows.append(row)
        lb.append(float(min_net_profit))
        ub.append(np.inf)

    A = np.vstack(rows)
    constraint = LinearConstraint(A, lb=np.array(lb), ub=np.array(ub))

    lb_x = np.zeros(n_vars)
    ub_x = np.full(n_vars, np.inf)
    ub_x[K:K + R] = 1.0
    # pos/neg deviation vars are unbounded above; keep as inf
    bounds = Bounds(lb_x, ub_x)

    integrality = np.zeros(n_vars, dtype=int)
    integrality[K:K + R] = 1
    # deviation vars are continuous

    opts = {
        "mip_rel_gap": float(mip_rel_gap),
        "presolve": True,
    }
    if time_limit is not None:
        opts["time_limit"] = float(time_limit)

    t0 = time.time()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraint,
        options=opts,
    )
    elapsed = time.time() - t0

    if res.x is None or not res.success:
        return BBResult(
            status="failed",
            mip_gap=float(getattr(res, "mip_gap", 0.0) or 0.0),
            lp_revenue=0.0,
            net_profit=0.0,
            n_routes_selected=0,
            n_routes_pool=R,
            total_served_ffe=0.0,
            selected_route_ids=[],
            cargo_flows={},
            route_loads={},
            weekly_op_cost_selected=0.0,
            total_vessels_used=0.0,
            vessels_used_by_class={},
            vessels_cap_by_class={k: float(v) for k, v in (caps or {}).items() if v is not None} if caps else {},
            solve_time_sec=round(elapsed, 3),
            message=str(getattr(res, "message", res)),
        )

    z = res.x
    fvals = z[:K]
    xvals = z[K:K + R]

    selected = [
        columns[s].route_id
        for s in range(R)
        if xvals[s] > 0.5
    ]

    cargo_flows: Dict[str, float] = {}
    route_loads: Dict[str, float] = {columns[s].route_id: 0.0 for s in range(R)}
    lp_revenue = 0.0

    for k, (s, od) in enumerate(var_list):
        fv = float(fvals[k])
        if fv <= 1e-6:
            continue
        rid = columns[s].route_id
        key = f"{od[0]}->{od[1]}|{rid}"
        cargo_flows[key] = round(fv, 4)
        route_loads[rid] = route_loads.get(rid, 0.0) + fv
        lp_revenue += fv * float(demand[od].revenue_per_ffe)

    op_cost = sum(
        float(columns[s].weekly_cost) * float(xvals[s])
        for s in range(R)
    )
    for rid in route_loads:
        route_loads[rid] = round(route_loads[rid], 4)

    total_served = sum(
        fv for k, fv in enumerate(fvals) if fv > 1e-6
    )

    net_profit = lp_revenue - op_cost
    lp_revenue_out = round(lp_revenue, 2)
    net_profit_out = round(net_profit, 2)
    if ports is not None and cargo_flows:
        rid_to_seq = {columns[s].route_id: list(columns[s].port_sequence) for s in range(R)}
        lp_revenue_out, _handling, net_profit_out = recompute_bb_financials(
            demand,
            ports,
            cargo_flows,
            op_cost,
            rid_to_seq,
        )

    vessels_used_by_class: Dict[str, float] = {}
    for s in range(R):
        if float(xvals[s]) <= 0.5:
            continue
        vc = columns[s].vessel_class
        vessels_used_by_class[vc] = vessels_used_by_class.get(vc, 0.0) + float(columns[s].vessels_needed)
    vessels_used_by_class = {k: round(v, 4) for k, v in vessels_used_by_class.items()}
    total_vessels_used = round(sum(vessels_used_by_class.values()), 4)
    vessels_cap_by_class = {k: float(v) for k, v in (caps or {}).items() if v is not None} if caps else {}

    return BBResult(
        status="optimal" if res.status == 0 else f"status_{res.status}",
        mip_gap=float(getattr(res, "mip_gap", 0.0) or 0.0),
        lp_revenue=lp_revenue_out,
        net_profit=net_profit_out,
        n_routes_selected=len(selected),
        n_routes_pool=R,
        total_served_ffe=round(total_served, 2),
        selected_route_ids=selected,
        cargo_flows=cargo_flows,
        route_loads={k: v for k, v in route_loads.items() if v > 1e-6},
        weekly_op_cost_selected=round(op_cost, 2),
        total_vessels_used=total_vessels_used,
        vessels_used_by_class=vessels_used_by_class,
        vessels_cap_by_class=vessels_cap_by_class,
        solve_time_sec=round(elapsed, 3),
        message=str(getattr(res, "message", "OK")),
    )


def save_bb_result(result: BBResult, path: Optional[str] = None):
    path = path or os.path.join(OUTPUTS_DIR, "bb_result.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = asdict(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_stage4(
    data: Optional[dict] = None,
    routes_path: Optional[str] = None,
    mip_rel_gap: float = BB_MIP_GAP,
    verbose: bool = True,
) -> BBResult:
    if data is None:
        data = load_all(verbose=False)
        validate_data(data)
    cols = load_cg_columns(data, routes_path=routes_path or default_cg_routes_path())
    result = build_and_solve(
        cols, data["demand"], mip_rel_gap=mip_rel_gap, ports=data["ports"]
    )
    save_bb_result(result)
    if verbose:
        _print_summary(result, data)
    return result


def _print_summary(result: BBResult, data: dict):
    total_d = sum(r.ffe_per_week for r in data["demand"].values())
    pct = 100 * result.total_served_ffe / total_d if total_d else 0
    print()
    print("=" * 60)
    print("  Stage 4 — Route-subset MILP (HiGHS)")
    print("=" * 60)
    print(f"  Status:            {result.status}")
    print(f"  MIP gap:           {result.mip_gap * 100:.2f}%")
    print(f"  Pool / selected:   {result.n_routes_pool} → {result.n_routes_selected} routes")
    print(f"  LP revenue:        ${result.lp_revenue:,.0f} / week")
    print(f"  Op cost (sel.):    ${result.weekly_op_cost_selected:,.0f} / week")
    print(f"  Net profit:        ${result.net_profit:,.0f} / week")
    print(f"  Served:            {result.total_served_ffe:,.0f} / {total_d:,.0f} FFE  ({pct:.1f}%)")
    print(f"  Solve time:        {result.solve_time_sec}s")
    print(f"  → outputs/bb_result.json")
    print("=" * 60)


if __name__ == "__main__":
    # argv[1] = LLM key (unused); argv[2] = optional MIP gap override
    gap = float(sys.argv[2]) if len(sys.argv) > 2 else BB_MIP_GAP
    data = load_all(verbose=False)
    validate_data(data)
    run_stage4(data, mip_rel_gap=gap, verbose=True)
    sys.exit(0)
