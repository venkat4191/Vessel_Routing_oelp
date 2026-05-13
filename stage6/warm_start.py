"""
Stage 6 — Warm-start refinement (CG ↔ MIP alternation)
======================================================
Uses **Stage 4 MIP-selected routes** as the next **Stage 3 seeds**, reruns
column generation to grow the pool around a profitable core, then re-solves
the route-subset MILP. Repeats for several rounds to push **demand coverage**
and **net profit** upward.

Flow per round
--------------
1. `cg_loop.run` starting from current seeds (`snapshot_tag=warm_r{k}` saves a checkpoint)
2. `route_subset_mip.build_and_solve` on the new `cg_routes.json` pool
3. Next seeds = MIP-selected routes (reloaded from pool JSON)

Stops early when MIP-served FFE share ≥ `WARM_START_TARGET_COVERAGE_PCT`.

Output
------
* Updates `outputs/cg_result.json`, `outputs/cg_routes.json`, `outputs/bb_result.json`
* `outputs/warm_start_history.json` — per-round metrics
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (  # noqa: E402
    OUTPUTS_DIR,
    WARM_START_MAX_ROUNDS,
    WARM_START_CG_MAX_ITER,
    WARM_START_TARGET_COVERAGE_PCT,
    WARM_START_TOP_K_EXACT,
    BB_MIP_GAP,
)


@dataclass
class WarmRoundLog:
    round: int
    cg_status: str
    cg_iterations: int
    cg_routes: int
    cg_pct_served: float
    cg_net_profit: float
    mip_status: str
    mip_selected_routes: int
    mip_net_profit: float
    mip_pct_served: float


def _records_to_seeds(records: List[dict]) -> List[Any]:
    from stage2.seed_generator import SeedRoute

    seeds = []
    for r in records:
        seeds.append(
            SeedRoute(
                route_id=r["route_id"],
                port_sequence=r["port_sequence"],
                vessel_class=r["vessel_class"],
                frequency=int(r.get("frequency", 1)),
                weekly_cost=float(r.get("weekly_cost", 0)),
                vessels_needed=int(r.get("vessels_needed", 0)),
                source="warm_start",
            )
        )
    return seeds


def _seeds_from_bb(pool: List[dict], selected_ids: List[str]) -> List[Any]:
    sel = set(selected_ids)
    recs = [r for r in pool if r["route_id"] in sel]
    return _records_to_seeds(recs)


def run(
    data: Optional[dict] = None,
    api_key: Optional[str] = None,
    key_ring: list = None,
    max_rounds: int = WARM_START_MAX_ROUNDS,
    cg_max_iter: int = WARM_START_CG_MAX_ITER,
    target_coverage_pct: float = WARM_START_TARGET_COVERAGE_PCT,
    start_coverage_floor_pct: float = 42.0,
    coverage_bonus_per_ffe: float = 15.0,
    profit_floor_ratio: float = 0.80,
    verbose: bool = True,
    verbose_cg: bool = True,
) -> Dict[str, Any]:
    from stage0.loader import load_all, validate as validate_data
    from stage1.demand_intel import run as run_intel
    from stage1.fleet_profiler import run as run_fleet
    from stage3.cg_loop import run as cg_run
    from stage4.cg_columns import default_cg_routes_path, load_cg_route_records
    from stage4.route_subset_mip import build_and_solve, load_cg_columns, save_bb_result

    if data is None:
        data = load_all(verbose=False)
        validate_data(data)

    intel = run_intel(data, verbose=False)
    fp = run_fleet(data, verbose=False)

    bb_path = os.path.join(OUTPUTS_DIR, "bb_result.json")
    cg_path = default_cg_routes_path()
    if not os.path.isfile(bb_path):
        raise FileNotFoundError(
            f"Missing {bb_path} — run Stage 4 first (`python stage4/route_subset_mip.py`)."
        )
    with open(bb_path, encoding="utf-8") as f:
        bb = json.load(f)
    pool = load_cg_route_records(cg_path)
    seeds = _seeds_from_bb(pool, bb["selected_route_ids"])
    if not seeds:
        raise RuntimeError("MIP selected no routes — cannot warm-start.")

    total_demand = sum(r.ffe_per_week for r in data["demand"].values())
    history: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None

    for rnd in range(1, max_rounds + 1):
        tag = f"warm_r{rnd}"
        if verbose:
            print(f"\n── Warm-start round {rnd}/{max_rounds}  (seeds={len(seeds)}) ──")

        cg_res = cg_run(
            data,
            seeds,
            fleet_profile=fp,
            api_key=api_key,
            key_ring=key_ring,
            max_iter=cg_max_iter,
            top_k_exact=WARM_START_TOP_K_EXACT,
            verbose=verbose and verbose_cg,
            snapshot_tag=tag,
        )

        cols = load_cg_columns(data, routes_path=cg_path)

        # Adaptive coverage forcing:
        # ramp minimum served share from 42% toward target across rounds.
        start_pct = float(start_coverage_floor_pct)
        ramp_den = max(1, max_rounds - 1)
        planned_floor = min(
            target_coverage_pct,
            start_pct + (target_coverage_pct - start_pct) * (rnd - 1) / ramp_den,
        )
        min_served_ffe = total_demand * (planned_floor / 100.0)

        mip = build_and_solve(
            cols,
            data["demand"],
            mip_rel_gap=BB_MIP_GAP,
            min_served_ffe=min_served_ffe,
            coverage_bonus_per_ffe=float(coverage_bonus_per_ffe),
            min_net_profit=(
                best["mip_net_profit"] * float(profit_floor_ratio)
                if best and best.get("mip_net_profit") is not None else None
            ),
            ports=data["ports"],
        )
        if mip.status == "failed":
            # Fallback to pure-profit solve if floor is infeasible.
            mip = build_and_solve(cols, data["demand"], mip_rel_gap=BB_MIP_GAP, ports=data["ports"])
        save_bb_result(mip)

        mip_pct = 100.0 * mip.total_served_ffe / total_demand if total_demand else 0.0
        entry = WarmRoundLog(
            round=rnd,
            cg_status=cg_res.status,
            cg_iterations=cg_res.total_iterations,
            cg_routes=cg_res.total_routes,
            cg_pct_served=cg_res.pct_served,
            cg_net_profit=cg_res.final_net_profit,
            mip_status=mip.status,
            mip_selected_routes=mip.n_routes_selected,
            mip_net_profit=mip.net_profit,
            mip_pct_served=round(mip_pct, 2),
        )
        history.append(asdict(entry))
        history[-1]["planned_floor_pct"] = round(planned_floor, 2)

        score = (mip.net_profit, mip_pct)
        if best is None or score > (best["_score"][0], best["_score"][1]):
            best = {
                "round": rnd,
                "mip_net_profit": mip.net_profit,
                "mip_pct_served": mip_pct,
                "n_selected": mip.n_routes_selected,
                "_score": score,
            }

        if verbose:
            print(f"     CG: {cg_res.total_routes} routes  served {cg_res.pct_served:.1f}%  "
                  f"profit ${cg_res.final_net_profit/1e6:.2f}M")
            print(f"     MIP: {mip.n_routes_selected} routes  served {mip_pct:.1f}%  "
                  f"profit ${mip.net_profit/1e6:.2f}M")

        if mip_pct >= target_coverage_pct:
            if verbose:
                print(f"\n  ✓ Target coverage {target_coverage_pct}% reached — stopping.")
            break

        pool = load_cg_route_records(cg_path)
        seeds = _seeds_from_bb(pool, mip.selected_route_ids)
        if len(seeds) < 1:
            break

    if best and "_score" in best:
        del best["_score"]

    out = {
        "rounds_run": len(history),
        "target_coverage_pct": target_coverage_pct,
        "best": best,
        "history": history,
    }
    hist_path = os.path.join(OUTPUTS_DIR, "warm_start_history.json")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print()
        print("=" * 60)
        print("  Stage 6 — Warm-start complete")
        print("=" * 60)
        if best:
            print(f"  Best round:        {best.get('round')}")
            print(f"  Best MIP profit:   ${best.get('mip_net_profit', 0):,.0f} / week")
            print(f"  Best MIP served:   {best.get('mip_pct_served', 0):.1f}%")
        print(f"  → {hist_path}")
        print("=" * 60)

    return out


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--quiet-cg"]
    quiet_cg = "--quiet-cg" in sys.argv
    api_key = argv[0] if argv else None
    run(api_key=api_key, verbose=True, verbose_cg=not quiet_cg)
