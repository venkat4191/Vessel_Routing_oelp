"""
Stage 4 — Part 1: Rehydrate column pool from CG output
======================================================
Reads `outputs/cg_routes.json` (written by Stage 3) and rebuilds
`RouteColumn` objects identical to those used in the RMP: same coverage
rules as `reduced_cost.compute_coverage`, same economics from
`fleet_profiler.route_economics`.

Downstream (Part 2) uses these columns for the route-subset MILP.
"""

import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import OUTPUTS_DIR
from stage3.rmp import RestrictedMasterProblem, RouteColumn


def distinct_port_count(rec: dict) -> int:
    """Number of distinct ports in `port_sequence` (loop-friendly)."""
    seq = rec.get("port_sequence") or []
    if not seq:
        return 0
    return len(set(seq))


def default_cg_routes_path() -> str:
    return os.path.join(OUTPUTS_DIR, "cg_routes.json")


def load_cg_route_records(path: Optional[str] = None) -> list:
    """Load raw list of route dicts from JSON."""
    path = path or default_cg_routes_path()
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_columns_from_records(
    records: list,
    demand: dict,
    fleet: dict,
    ports: dict,
    distances: dict,
    skip_infeasible: bool = True,
    min_ports: Optional[int] = None,
    max_ports: Optional[int] = None,
) -> List[RouteColumn]:
    """
    Turn JSON records into `RouteColumn` instances using the same
    `RestrictedMasterProblem.column_from_route` pipeline as Stage 3.

    min_ports / max_ports
        If set, keep only routes whose distinct port count lies in
        [min_ports, max_ports] (inclusive). Example: 6–14 favors longer
        multi-port services over 4–5 port shuttles (when CG supplies them).
    """
    rmp = RestrictedMasterProblem(demand, fleet)
    columns: List[RouteColumn] = []
    for rec in records:
        wc = float(rec.get("weekly_cost", 0))
        if skip_infeasible and (wc >= 1e300 or wc != wc):  # inf or nan
            continue
        npt = distinct_port_count(rec)
        if min_ports is not None and npt < int(min_ports):
            continue
        if max_ports is not None and npt > int(max_ports):
            continue
        col = rmp.column_from_route(
            rec["route_id"],
            rec["port_sequence"],
            rec["vessel_class"],
            int(rec.get("frequency", 1)),
            ports,
            distances,
        )
        columns.append(col)
    return columns


def load_cg_columns(data: dict, routes_path: Optional[str] = None) -> List[RouteColumn]:
    """
    Convenience: load JSON from disk and build columns.

    Parameters
    ----------
    data : dict from stage0.load_all()
    routes_path : optional override for cg_routes.json location
    """
    records = load_cg_route_records(routes_path)
    return build_columns_from_records(
        records,
        data["demand"],
        data["fleet"],
        data["ports"],
        data["distances"],
    )


def summarize_columns(columns: List[RouteColumn], demand: dict) -> dict:
    """Lightweight stats for sanity checks and CLI output."""
    od_set = set()
    for c in columns:
        od_set.update(c.coverage)
    return {
        "n_routes": len(columns),
        "n_distinct_od_covered": len(od_set),
        "n_demand_pairs": len(demand),
        "total_weekly_op_cost": sum(c.weekly_cost for c in columns),
        "total_vessels_if_all_operated": sum(c.vessels_needed for c in columns),
    }


if __name__ == "__main__":
    from stage0.loader import load_all, validate as validate_data

    data = load_all(verbose=False)
    validate_data(data)

    # argv[1] = optional LLM key (ignored); argv[2] = optional cg_routes.json path
    path = sys.argv[2] if len(sys.argv) > 2 else default_cg_routes_path()
    if not os.path.isfile(path):
        print(f"ERROR: missing {path}")
        sys.exit(1)

    cols = load_cg_columns(data, routes_path=path)
    stats = summarize_columns(cols, data["demand"])

    print("=" * 60)
    print("  Stage 4 — Part 1: CG columns rehydration")
    print("=" * 60)
    print(f"  Source: {path}")
    print(f"  Routes loaded:              {stats['n_routes']}")
    print(f"  Distinct OD pairs covered:  {stats['n_distinct_od_covered']}")
    print(f"  Total demand OD pairs:      {stats['n_demand_pairs']}")
    print(f"  Sum weekly op cost (all):   ${stats['total_weekly_op_cost']:,.0f}")
    print(f"  Vessels if all operated:    {stats['total_vessels_if_all_operated']}")
    print("=" * 60)

    assert stats["n_routes"] > 0, "expected at least one route"
    assert stats["n_distinct_od_covered"] > 0, "expected some OD coverage"
    print("  ✓ Part 1 checks passed")
    sys.exit(0)
