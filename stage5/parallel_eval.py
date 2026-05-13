"""
Stage 5 — Parallel evaluation of network designs under demand stress
====================================================================
Runs several **independent** RMP cargo-flow LPs (same column sets as Stage 3,
no column generation) under different demand multipliers, in parallel.

Scenarios (default)
-------------------
  * **mip_baseline** — Stage 4 MIP-selected routes, base demand
  * **full_cg_baseline** — entire `cg_routes.json` pool, base demand
  * **mip_demand_90 / 110** — MIP subset with ±10% FFE volume
  * **mip_peak** — MIP subset with all OD volumes × `PEAK_DEMAND_FACTOR`

Each worker process loads fresh data (pickle-safe), builds an RMP with the
chosen route subset, solves once, and returns KPIs.

Output: `outputs/parallel_eval.json`
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (  # noqa: E402
    OUTPUTS_DIR,
    PARALLEL_EVAL_MAX_WORKERS,
    PEAK_DEMAND_FACTOR,
)


@dataclass
class ScenarioSpec:
    name: str
    demand_mult: float
    route_set: str  # "mip_subset" | "full_cg"


def _scale_demand(demand: dict, mult: float) -> dict:
    from stage0.loader import DemandRecord

    out = {}
    for k, v in demand.items():
        out[k] = DemandRecord(
            origin=v.origin,
            destination=v.destination,
            ffe_per_week=v.ffe_per_week * mult,
            revenue_per_ffe=v.revenue_per_ffe,
            max_transit_days=v.max_transit_days,
        )
    return out


def _route_records_for_set(
    route_set: str,
    cg_path: str,
    bb_path: str,
) -> List[dict]:
    with open(cg_path, encoding="utf-8") as f:
        pool = json.load(f)
    if route_set == "full_cg":
        return pool
    with open(bb_path, encoding="utf-8") as f:
        bb = json.load(f)
    sel = set(bb.get("selected_route_ids", []))
    return [r for r in pool if r["route_id"] in sel]


def evaluate_one_local(
    project_root: str,
    scenario: ScenarioSpec,
    cg_path: str,
    bb_path: str,
) -> Dict[str, Any]:
    """Single scenario (for sequential use or testing)."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    from stage0.loader import load_all, validate as validate_data
    from stage3.rmp import RestrictedMasterProblem

    data = load_all(verbose=False)
    validate_data(data)
    if scenario.demand_mult != 1.0:
        data = {**data, "demand": _scale_demand(data["demand"], scenario.demand_mult)}

    records = _route_records_for_set(scenario.route_set, cg_path, bb_path)
    demand = data["demand"]
    fleet = data["fleet"]
    ports = data["ports"]
    distances = data["distances"]
    total_d = sum(r.ffe_per_week for r in demand.values())

    rmp = RestrictedMasterProblem(demand, fleet)
    for rec in records:
        if float(rec.get("weekly_cost", 0)) >= 1e300:
            continue
        col = rmp.column_from_route(
            rec["route_id"],
            rec["port_sequence"],
            rec["vessel_class"],
            int(rec.get("frequency", 1)),
            ports,
            distances,
        )
        rmp.add_column(col)

    sol = rmp.solve(verbose=False, distances=distances)
    pct = 100.0 * sol.total_served_ffe / total_d if total_d else 0.0
    return {
        "scenario": scenario.name,
        "demand_mult": scenario.demand_mult,
        "route_set": scenario.route_set,
        "n_routes": len(rmp.columns),
        "lp_revenue": sol.lp_revenue,
        "net_profit": sol.net_profit,
        "total_op_cost": sol.total_op_cost,
        "served_ffe": sol.total_served_ffe,
        "pct_served": round(pct, 2),
        "rmp_status": sol.status,
    }


def _worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    spec = ScenarioSpec(**payload["scenario"])
    return evaluate_one_local(
        payload["project_root"],
        spec,
        payload["cg_path"],
        payload["bb_path"],
    )


def default_scenarios() -> List[ScenarioSpec]:
    return [
        ScenarioSpec("mip_baseline", 1.0, "mip_subset"),
        ScenarioSpec("full_cg_baseline", 1.0, "full_cg"),
        ScenarioSpec("mip_demand_90", 0.9, "mip_subset"),
        ScenarioSpec("mip_demand_110", 1.1, "mip_subset"),
        ScenarioSpec("mip_peak", PEAK_DEMAND_FACTOR, "mip_subset"),
    ]


def run_parallel(
    project_root: Optional[str] = None,
    scenarios: Optional[List[ScenarioSpec]] = None,
    cg_path: Optional[str] = None,
    bb_path: Optional[str] = None,
    max_workers: int = PARALLEL_EVAL_MAX_WORKERS,
) -> List[Dict[str, Any]]:
    project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cg_path = cg_path or os.path.join(project_root, "outputs", "cg_routes.json")
    bb_path = bb_path or os.path.join(project_root, "outputs", "bb_result.json")
    scenarios = scenarios or default_scenarios()

    payloads = [
        {
            "project_root": project_root,
            "scenario": asdict(s),
            "cg_path": cg_path,
            "bb_path": bb_path,
        }
        for s in scenarios
    ]

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(max_workers, len(payloads))) as ex:
        futures = {ex.submit(_worker, p): p for p in payloads}
        for fut in as_completed(futures):
            results.append(fut.result())
    results.sort(key=lambda r: r["scenario"])
    return results


def save_results(results: List[Dict[str, Any]], path: Optional[str] = None):
    path = path or os.path.join(OUTPUTS_DIR, "parallel_eval.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scenarios": results}, f, indent=2)


def run(
    project_root: Optional[str] = None,
    sequential: bool = False,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cg_path = os.path.join(root, "outputs", "cg_routes.json")
    bb_path = os.path.join(root, "outputs", "bb_result.json")
    scenarios = default_scenarios()

    if sequential or len(scenarios) == 1:
        results = [
            evaluate_one_local(root, s, cg_path, bb_path) for s in scenarios
        ]
    else:
        results = run_parallel(root, scenarios, cg_path, bb_path)

    save_results(results)
    if verbose:
        print()
        print("=" * 62)
        print("  Stage 5 — Parallel evaluation (RMP stress scenarios)")
        print("=" * 62)
        for r in sorted(results, key=lambda x: x["scenario"]):
            print(f"  {r['scenario']:<22}  profit=${r['net_profit']/1e6:>7.2f}M  "
                  f"served={r['pct_served']:>5.1f}%  routes={r['n_routes']}")
        print(f"  → {os.path.join(OUTPUTS_DIR, 'parallel_eval.json')}")
        print("=" * 62)
    return results


if __name__ == "__main__":
    seq = "--sequential" in sys.argv
    run(sequential=seq, verbose=True)
