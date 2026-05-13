"""
Multi-run + merge + refinement pipeline runner for LinerNet.

Goal: maximize demand coverage and profit using feedback loops:
  1) Run Stage 0→4 twice from scratch (independent runs).
  2) Merge the two Stage3 route pools, then re-solve Stage4 on the merged pool
     with a coverage bonus (soft multi-objective).
  3) Take top 75% of merged-selected routes as seeds; re-run Stage3→4 for
     two refinement iterations, each time reseeding from Stage4.

Outputs are written under:
  outputs/merge_pipeline/
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import traceback
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stage0.loader import load_all, validate as validate_data
from stage1.demand_intel import run as run_intel
from stage1.fleet_profiler import run as run_fleet
from stage2.seed_generator import SeedRoute, run as run_seeds
from stage3.cg_loop import run as run_cg
from stage4.cg_columns import (
    load_cg_route_records,
    build_columns_from_records,
    distinct_port_count,
)
from stage4.route_subset_mip import build_and_solve, save_bb_result, BBResult
from utils.config import STAGE4_ROUTE_MAX_PORTS, STAGE4_ROUTE_MIN_PORTS
from utils.env import get_api_keys, get_gemini_key, load_dotenv


def _save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _total_demand_ffe(data: dict) -> float:
    return float(sum(r.ffe_per_week for r in data["demand"].values()))


def _served_pct(served_ffe: float, data: dict) -> float:
    td = _total_demand_ffe(data)
    return 100.0 * float(served_ffe) / td if td else 0.0


def _balanced_coverage_bonus_per_ffe(data: dict) -> float:
    """
    Equal-priority profit + coverage scalarization.

    Stage 4 solves: max (profit + bonus * served_ffe).
    To give profit and coverage similar priority, we scale served_ffe into $ using
    the demand-weighted average revenue per FFE:

        bonus = (Σ demand_od * rev_od) / (Σ demand_od)
    """
    tot = 0.0
    rev = 0.0
    for rec in data["demand"].values():
        q = float(rec.ffe_per_week)
        tot += q
        rev += q * float(rec.revenue_per_ffe)
    return float(rev / tot) if tot else 0.0


def _route_econ_from_bb(
    data: dict,
    bb: BBResult,
    route_records: List[dict],
) -> Dict[str, dict]:
    """
    Compute per-route served_ffe and profit_contrib from BB cargo flows.
    Profit_contrib is approximate: revenue from allocated flows - weekly_cost.
    """
    demand = data["demand"]
    cost_by_id = {r.get("route_id"): float(r.get("weekly_cost", 0.0)) for r in route_records}

    served_by_route: Dict[str, float] = {}
    rev_by_route: Dict[str, float] = {}

    for flow_key, flow_val in (bb.cargo_flows or {}).items():
        try:
            od_str, rid = flow_key.split("|", 1)
            o, d = od_str.split("->", 1)
        except ValueError:
            continue
        qty = float(flow_val)
        if qty <= 0:
            continue
        served_by_route[rid] = served_by_route.get(rid, 0.0) + qty
        if (o, d) in demand:
            rev_by_route[rid] = rev_by_route.get(rid, 0.0) + qty * float(demand[(o, d)].revenue_per_ffe)

    out: Dict[str, dict] = {}
    for rid in bb.selected_route_ids:
        cost = float(cost_by_id.get(rid, 0.0))
        rev = float(rev_by_route.get(rid, 0.0))
        served = float(served_by_route.get(rid, 0.0))
        out[rid] = {
            "served_ffe": round(served, 4),
            "revenue": round(rev, 2),
            "weekly_cost": round(cost, 2),
            "profit_contrib": round(rev - cost, 2),
        }
    return out


def _top_seed_routes_from_bb(
    *,
    data: dict,
    bb: BBResult,
    route_records: List[dict],
    seed_fraction: float,
    seed_count: Optional[int] = None,
    coverage_bonus_per_ffe: float,
    seed_tag: str,
    route_min_ports: Optional[int] = None,
) -> List[SeedRoute]:
    """
    Pick top fraction of selected routes as next seeds using a merged score:
      score = profit_contrib + coverage_bonus_per_ffe * served_ffe
    """
    seed_fraction = float(seed_fraction)
    seed_fraction = max(0.0, min(1.0, seed_fraction))
    if not bb.selected_route_ids:
        return []

    per_route = _route_econ_from_bb(data, bb, route_records)
    rec_by_id = {r.get("route_id"): r for r in route_records}

    scored: List[Tuple[float, str]] = []
    for rid in bb.selected_route_ids:
        pr = per_route.get(rid, {})
        served = float(pr.get("served_ffe", 0.0))
        prof = float(pr.get("profit_contrib", 0.0))
        score = prof + float(coverage_bonus_per_ffe) * served
        scored.append((score, rid))
    scored.sort(reverse=True)

    if seed_count is not None:
        k = max(1, int(seed_count))
    else:
        k = max(1, int(math.ceil(len(scored) * seed_fraction)))
    k = min(k, len(scored)) if scored else 0
    chosen = [rid for _, rid in scored[:k]]

    min_distinct = 3
    if route_min_ports is not None and int(route_min_ports) > 0:
        min_distinct = int(route_min_ports)

    seeds: List[SeedRoute] = []
    for idx, rid in enumerate(chosen, start=1):
        rec = rec_by_id.get(rid)
        if not rec:
            continue
        seq = rec.get("port_sequence") or []
        if len(seq) < 3:
            continue
        if distinct_port_count(rec) < min_distinct:
            continue
        seeds.append(
            SeedRoute(
                route_id=f"{seed_tag}_{idx:03d}",
                port_sequence=seq,
                vessel_class=rec.get("vessel_class", ""),
                frequency=int(rec.get("frequency", 1)),
                cycle_days=float(rec.get("cycle_days", 0.0) or 0.0),
                vessels_needed=int(rec.get("vessels_needed", 0) or 0),
                weekly_cost=float(rec.get("weekly_cost", 0.0) or 0.0),
                source="merge_seed",
                rationale=(
                    f"Top {k} seeds from {seed_tag}"
                    if seed_count is not None
                    else f"Top {int(seed_fraction*100)}% from {seed_tag}"
                ),
            )
        )
    return seeds


def _canonical_route_key(rec: dict) -> Tuple:
    """Key for deduping route records across runs."""
    return (
        tuple(rec.get("port_sequence") or []),
        str(rec.get("vessel_class") or ""),
        int(rec.get("frequency", 1) or 1),
    )


def _merge_route_pools(pools: List[List[dict]]) -> List[dict]:
    """
    Merge CG route records from multiple runs.
    If duplicates exist, keep the version with lower weekly_cost.
    """
    best: Dict[Tuple, dict] = {}
    for pool in pools:
        for rec in pool:
            key = _canonical_route_key(rec)
            wc = float(rec.get("weekly_cost", 0.0) or 0.0)
            if key not in best:
                best[key] = rec
                continue
            prev = best[key]
            prev_wc = float(prev.get("weekly_cost", 0.0) or 0.0)
            if wc < prev_wc:
                best[key] = rec

    merged: List[dict] = []
    for i, rec in enumerate(best.values(), start=1):
        out = dict(rec)
        out["route_id"] = f"M{i:04d}"
        merged.append(out)
    return merged


def _build_stage4_columns(
    *,
    data: dict,
    route_records: List[dict],
    route_min_ports: Optional[int],
    route_max_ports: Optional[int],
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List, str]:
    """
    Build MILP columns with optional port-count filter. If the filter removes
    every route, fall back to unfiltered (and log).
    """
    _log = log or (lambda _m: None)

    def _cols(min_p: Optional[int], max_p: Optional[int]):
        return build_columns_from_records(
            route_records,
            data["demand"],
            data["fleet"],
            data["ports"],
            data["distances"],
            skip_infeasible=True,
            min_ports=min_p,
            max_ports=max_p,
        )

    cols = _cols(route_min_ports, route_max_ports)
    note = "filtered"
    if len(cols) == 0 and (route_min_ports is not None or route_max_ports is not None):
        _log(
            "Stage 4: port-count filter removed all routes; "
            f"min_ports={route_min_ports}, max_ports={route_max_ports} — using unfiltered pool."
        )
        cols = _cols(None, None)
        note = "unfiltered_fallback"
    return cols, note


def _solve_stage4_from_records(
    *,
    data: dict,
    route_records: List[dict],
    out_bb_path: str,
    coverage_bonus_per_ffe: float,
    min_net_profit: Optional[float] = None,
    min_served_ffe: Optional[float] = None,
    min_routes_selected: Optional[int] = None,
    vessel_mix_target_fraction_of_cap: Optional[float] = None,
    vessel_mix_penalty_per_vessel: float = 0.0,
    min_selected_route_utilization: Optional[float] = None,
    large_vessel_bonus_per_ffe_capacity: float = 0.0,
    route_activation_penalty: float = 0.0,
    route_min_ports: Optional[int] = None,
    route_max_ports: Optional[int] = None,
    two_phase: bool = False,
    phase1_coverage_bonus_multiplier: float = 10.0,
    coverage_floor_ratio: float = 0.98,
    log: Optional[Callable[[str], None]] = None,
) -> BBResult:
    _log = log or (lambda _m: None)
    cols, pool_note = _build_stage4_columns(
        data=data,
        route_records=route_records,
        route_min_ports=route_min_ports,
        route_max_ports=route_max_ports,
        log=_log,
    )

    def _solve(
        *,
        cov_bonus: float,
        floor: Optional[float],
    ) -> BBResult:
        return build_and_solve(
            cols,
            data["demand"],
            coverage_bonus_per_ffe=float(cov_bonus),
            min_net_profit=min_net_profit,
            min_served_ffe=floor,
            min_routes_selected=min_routes_selected,
            vessel_mix_target_fraction_of_cap=vessel_mix_target_fraction_of_cap,
            vessel_mix_penalty_per_vessel=float(vessel_mix_penalty_per_vessel),
            min_selected_route_utilization=min_selected_route_utilization,
            large_vessel_bonus_per_ffe_capacity=float(large_vessel_bonus_per_ffe_capacity),
            route_activation_penalty=float(route_activation_penalty),
            ports=data["ports"],
        )

    if not two_phase:
        bb = _solve(cov_bonus=float(coverage_bonus_per_ffe), floor=min_served_ffe)
        bb.message = f"{bb.message} | pool={pool_note}"
        save_bb_result(bb, path=out_bb_path)
        return bb

    # Phase A: coverage-first (strong bonus on served FFE).
    mult = max(1.0, float(phase1_coverage_bonus_multiplier))
    bb1 = _solve(cov_bonus=float(coverage_bonus_per_ffe) * mult, floor=None)
    ratio = max(0.0, min(1.0, float(coverage_floor_ratio)))
    floor_candidate = float(bb1.total_served_ffe) * ratio
    if min_served_ffe is not None:
        floor_candidate = max(floor_candidate, float(min_served_ffe))
    floor = float(floor_candidate)
    if bb1.status == "failed" or bb1.total_served_ffe <= 0:
        bb1.message = f"{bb1.message} | pool={pool_note} | two_phase=phase1_failed"
        save_bb_result(bb1, path=out_bb_path)
        return bb1

    # Phase B: maximize profit subject to near-peak coverage.
    bb2 = _solve(cov_bonus=0.0, floor=floor)
    if bb2.status == "failed":
        _log(
            f"Stage 4 two-phase: phase-2 infeasible at floor={floor:.2f} FFE; "
            "keeping phase-1 solution."
        )
        bb1.message = (
            f"{bb1.message} | pool={pool_note} | two_phase=phase1_only_infeasible_phase2"
        )
        save_bb_result(bb1, path=out_bb_path)
        return bb1

    bb2.message = (
        f"{bb2.message} | pool={pool_note} | two_phase=coverage_floor "
        f"{ratio:.3f} (phase1_served={bb1.total_served_ffe:.2f})"
    )
    save_bb_result(bb2, path=out_bb_path)
    return bb2


def run_merge_architecture(
    gemini_key: str | None = None,
    log: Callable[[str], None] = print,
    objective_mode: str = "balanced",
    coverage_bonus_per_ffe: Optional[float] = None,
    seed_fraction: float = 0.75,
    target_coverage_pct: float = 80.0,
    seed_count: int = 15,
    refinement_iters: int = 2,
    cg_verbose: bool = True,
    use_llm_intel: bool = True,
    use_llm_seeds: bool = True,
    use_llm_cg: bool = True,
    cg_max_iter: Optional[int] = None,
    max_transship_hops: Optional[int] = None,
    api_keys: Optional[List[str]] = None,
    vessel_mix_target_fraction_of_cap: Optional[float] = None,
    vessel_mix_penalty_per_vessel: float = 0.0,
    min_selected_route_utilization: Optional[float] = None,
    large_vessel_bonus_per_ffe_capacity: float = 35.0,
    route_activation_penalty: float = 0.0,
    auto_tune_stage4_policy: bool = True,
    use_route_port_filter: bool = True,
    route_min_ports: Optional[int] = None,
    route_max_ports: Optional[int] = None,
    two_phase_stage4: bool = True,
    two_phase_initial_runs: bool = False,
    phase1_coverage_bonus_multiplier: float = 10.0,
    coverage_floor_ratio: float = 0.98,
    default_vessel_mix_fraction_of_cap: float = 0.85,
) -> dict:
    """
    Execute the requested architecture and write a full summary JSON.

    Route pool shape: optional distinct-port window (default 6–14) keeps longer
    services and drops short 4–5 port shuttles so Stage 4 can favor high-capacity
    ships on multi-port strings (override with ``route_min_ports`` /
    ``route_max_ports`` or ``use_route_port_filter=False``).

    Two-phase Stage 4 (merged pool + refinements by default): phase A maximizes
    coverage with a strong bonus multiplier; phase B fixes served FFE to a high
    floor and maximizes profit (bonus 0). Initial CG runs stay single-phase
    unless ``two_phase_initial_runs`` is True.
    """
    load_dotenv(PROJECT_ROOT)
    # Key rotation: if multiple keys are present, cycle them across LLM-using calls.
    keyring = get_api_keys(PROJECT_ROOT, explicit_keys=api_keys) or []
    if gemini_key:
        # Backward compatible: single explicit key overrides ring.
        keyring = [gemini_key.strip()]
    key_idx = 0

    def next_key() -> Optional[str]:
        nonlocal key_idx
        if not keyring:
            return None
        k = keyring[key_idx % len(keyring)]
        key_idx += 1
        return k

    out_dir = os.path.join(PROJECT_ROOT, "outputs", "merge_pipeline")
    os.makedirs(out_dir, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        os.remove(os.path.join(out_dir, "summary.json"))

    try:
        log("Loading and validating data (Stage 0)...")
        data = load_all(verbose=False)
        validate_data(data)

        mode = (objective_mode or "balanced").strip().lower()
        if mode not in ("profit", "coverage", "balanced"):
            mode = "balanced"
        if coverage_bonus_per_ffe is None:
            if mode == "profit":
                bonus = 0.0
            elif mode == "coverage":
                # Strongly coverage-first; still linear and feasible.
                bonus = 10.0 * _balanced_coverage_bonus_per_ffe(data)
            else:
                bonus = _balanced_coverage_bonus_per_ffe(data)
        else:
            bonus = float(coverage_bonus_per_ffe)

        mix_frac_effective = vessel_mix_target_fraction_of_cap
        if mix_frac_effective is None:
            mix_frac_effective = float(default_vessel_mix_fraction_of_cap)

        eff_min_ports: Optional[int] = None
        eff_max_ports: Optional[int] = None
        if use_route_port_filter:
            eff_min_ports = (
                route_min_ports
                if route_min_ports is not None
                else (
                    STAGE4_ROUTE_MIN_PORTS
                    if STAGE4_ROUTE_MIN_PORTS is not None
                    else 6
                )
            )
            eff_max_ports = (
                route_max_ports
                if route_max_ports is not None
                else (
                    STAGE4_ROUTE_MAX_PORTS
                    if STAGE4_ROUTE_MAX_PORTS is not None
                    else 14
                )
            )

        log("Stage 1A: demand intelligence...")
        intel = run_intel(
            data,
            api_key=(next_key() or None) if use_llm_intel else None,
            key_ring=keyring if use_llm_intel else None,
            verbose=False,
        )

        target_coverage_pct = float(target_coverage_pct)
        if target_coverage_pct <= 0:
            target_coverage_pct = 0.0
        if target_coverage_pct > 100:
            target_coverage_pct = 100.0
        target_floor_ffe = _total_demand_ffe(data) * (target_coverage_pct / 100.0)

        log("Stage 1B: fleet profiler...")
        fleet_profile = run_fleet(data, verbose=False)

        _ = max_transship_hops

        initial_runs = []
        cg_pools: List[List[dict]] = []

        for run_idx in (1, 2):
            tag = f"run{run_idx}"
            log(f"Stage 2 (run {run_idx}/2): seed generation...")
            seeds = run_seeds(
                data,
                intel,
                fleet_profile,
                api_key=(next_key() or None) if use_llm_seeds else None,
                key_ring=keyring if use_llm_seeds else None,
                verbose=False,
            )

            log(f"Stage 3 (run {run_idx}/2): column generation...")
            cg = run_cg(
                data,
                seeds,
                fleet_profile=fleet_profile,
                api_key=(next_key() or None) if use_llm_cg else None,
                key_ring=keyring if use_llm_cg else None,
                verbose=bool(cg_verbose),
                snapshot_tag=tag,
                max_iter=int(cg_max_iter) if cg_max_iter is not None else 100,
                target_coverage_pct=target_coverage_pct,
                min_iter_before_profit_stop=2,
            )

            cg_routes_path = os.path.join(PROJECT_ROOT, "outputs", f"cg_routes_{tag}.json")
            pool = load_cg_route_records(cg_routes_path)
            cg_pools.append(pool)

            bb_path = os.path.join(out_dir, f"bb_{tag}.json")
            log(f"Stage 4 (run {run_idx}/2): route-subset MIP...")
            bb = _solve_stage4_from_records(
                data=data,
                route_records=pool,
                out_bb_path=bb_path,
                coverage_bonus_per_ffe=float(bonus),
                vessel_mix_target_fraction_of_cap=mix_frac_effective,
                vessel_mix_penalty_per_vessel=float(vessel_mix_penalty_per_vessel),
                min_selected_route_utilization=min_selected_route_utilization,
                large_vessel_bonus_per_ffe_capacity=float(large_vessel_bonus_per_ffe_capacity),
                route_activation_penalty=float(route_activation_penalty),
                route_min_ports=eff_min_ports,
                route_max_ports=eff_max_ports,
                two_phase=bool(two_phase_initial_runs),
                phase1_coverage_bonus_multiplier=float(phase1_coverage_bonus_multiplier),
                coverage_floor_ratio=float(coverage_floor_ratio),
                log=log,
            )

            initial_runs.append(
                {
                    "tag": tag,
                    "cg": {
                        "status": cg.status,
                        "routes": cg.total_routes,
                        "served_pct": cg.pct_served,
                        "net_profit": cg.final_net_profit,
                    },
                    "bb": {
                        "status": bb.status,
                        "selected_routes": bb.n_routes_selected,
                        "served_ffe": bb.total_served_ffe,
                        "served_pct": round(_served_pct(bb.total_served_ffe, data), 4),
                        "net_profit": bb.net_profit,
                        "bb_json": os.path.relpath(bb_path, PROJECT_ROOT),
                    },
                    "cg_routes_json": os.path.relpath(cg_routes_path, PROJECT_ROOT),
                }
            )

        # Enforce: merged selection uses MORE routes than either run selected.
        min_routes_after_merge = max(
            int(initial_runs[0]["bb"]["selected_routes"]),
            int(initial_runs[1]["bb"]["selected_routes"]),
        ) + 1

        log("Merging the two Stage 3 route pools (merge agent)...")
        merged_pool = _merge_route_pools(cg_pools)
        merged_routes_path = os.path.join(out_dir, "cg_routes_merged.json")
        _save_json(merged_routes_path, merged_pool)

        tuned = {
            "coverage_bonus_multiplier": 1.0,
            "vessel_mix_target_fraction_of_cap": mix_frac_effective,
            "vessel_mix_penalty_per_vessel": float(vessel_mix_penalty_per_vessel),
            "min_selected_route_utilization": min_selected_route_utilization,
            "large_vessel_bonus_per_ffe_capacity": float(large_vessel_bonus_per_ffe_capacity),
            "route_activation_penalty": float(route_activation_penalty),
        }
        tuning_trials = []
        if auto_tune_stage4_policy:
            log("Auto-tuning Stage 4 policy on merged pool...")
            candidates = [
                # (cov_mult, mix_frac, mix_pen, util_floor, large_bonus, route_pen)
                (1.00, mix_frac_effective, float(vessel_mix_penalty_per_vessel), min_selected_route_utilization, float(large_vessel_bonus_per_ffe_capacity), float(route_activation_penalty)),
                (1.25, 0.80, 120000.0, 0.80, 10.0, 0.0),
                (1.50, 0.85, 180000.0, 0.85, 20.0, 50000.0),
                (1.75, 0.90, 250000.0, 0.85, 30.0, 100000.0),
                (2.00, 0.92, 300000.0, 0.90, 35.0, 120000.0),
                (2.25, 0.95, 350000.0, 0.90, 40.0, 150000.0),
            ]
            best_key = None
            best_cfg = None
            for i, (cov_mult, mix_frac, mix_pen, util_floor, lbonus, rpen) in enumerate(candidates, start=1):
                bb_try = _solve_stage4_from_records(
                    data=data,
                    route_records=merged_pool,
                    out_bb_path=os.path.join(out_dir, f"bb_tune_{i}.json"),
                    coverage_bonus_per_ffe=float(bonus) * float(cov_mult),
                    vessel_mix_target_fraction_of_cap=mix_frac,
                    vessel_mix_penalty_per_vessel=float(mix_pen),
                    min_selected_route_utilization=util_floor,
                    large_vessel_bonus_per_ffe_capacity=float(lbonus),
                    route_activation_penalty=float(rpen),
                    route_min_ports=eff_min_ports,
                    route_max_ports=eff_max_ports,
                    two_phase=False,
                    log=log,
                )
                cov = round(_served_pct(bb_try.total_served_ffe, data), 6)
                prof = float(bb_try.net_profit)
                nsel = int(bb_try.n_routes_selected)
                # Lexicographic: maximize coverage, then profit, then fewer routes.
                key = (cov, prof, -nsel)
                trial = {
                    "trial": i,
                    "config": {
                        "coverage_bonus_multiplier": float(cov_mult),
                        "vessel_mix_target_fraction_of_cap": mix_frac,
                        "vessel_mix_penalty_per_vessel": float(mix_pen),
                        "min_selected_route_utilization": util_floor,
                        "large_vessel_bonus_per_ffe_capacity": float(lbonus),
                        "route_activation_penalty": float(rpen),
                    },
                    "bb": {
                        "status": bb_try.status,
                        "served_pct": cov,
                        "net_profit": prof,
                        "selected_routes": nsel,
                    },
                }
                tuning_trials.append(trial)
                if best_key is None or key > best_key:
                    best_key = key
                    best_cfg = trial["config"]
            if best_cfg:
                tuned.update(best_cfg)
                log(
                    "Auto-tune picked policy: "
                    f"cov_mult={tuned['coverage_bonus_multiplier']}, "
                    f"mix_frac={tuned['vessel_mix_target_fraction_of_cap']}, "
                    f"mix_pen={tuned['vessel_mix_penalty_per_vessel']}, "
                    f"util_floor={tuned['min_selected_route_utilization']}, "
                    f"large_bonus={tuned['large_vessel_bonus_per_ffe_capacity']}, "
                    f"route_pen={tuned['route_activation_penalty']}"
                )

        log("Stage 4 on merged pool (profit + coverage + tuned policy)...")
        merged_bb_path = os.path.join(out_dir, "bb_merged.json")
        merged_bb = _solve_stage4_from_records(
            data=data,
            route_records=merged_pool,
            out_bb_path=merged_bb_path,
            coverage_bonus_per_ffe=float(bonus) * float(tuned["coverage_bonus_multiplier"]),
            vessel_mix_target_fraction_of_cap=tuned["vessel_mix_target_fraction_of_cap"],
            vessel_mix_penalty_per_vessel=float(tuned["vessel_mix_penalty_per_vessel"]),
            min_selected_route_utilization=tuned["min_selected_route_utilization"],
            large_vessel_bonus_per_ffe_capacity=float(tuned["large_vessel_bonus_per_ffe_capacity"]),
            route_activation_penalty=float(tuned["route_activation_penalty"]),
            route_min_ports=eff_min_ports,
            route_max_ports=eff_max_ports,
            two_phase=bool(two_phase_stage4),
            phase1_coverage_bonus_multiplier=float(phase1_coverage_bonus_multiplier),
            coverage_floor_ratio=float(coverage_floor_ratio),
            min_served_ffe=float(target_floor_ffe) if target_floor_ffe > 0 else None,
            min_routes_selected=int(min_routes_after_merge),
            log=log,
        )

        log(f"Creating seed set from merged selection (seed_count={int(seed_count)})...")
        seeds_from_merged = _top_seed_routes_from_bb(
            data=data,
            bb=merged_bb,
            route_records=merged_pool,
            seed_fraction=float(seed_fraction),
            seed_count=int(seed_count) if seed_count is not None else None,
            coverage_bonus_per_ffe=float(bonus),
            seed_tag="SEED_MERGED",
            route_min_ports=eff_min_ports,
        )
        seeds_path = os.path.join(out_dir, "seeds_from_merged.json")
        _save_json(
            seeds_path,
            [
                {
                    "route_id": s.route_id,
                    "port_sequence": s.port_sequence,
                    "vessel_class": s.vessel_class,
                    "frequency": s.frequency,
                    "cycle_days": s.cycle_days,
                    "vessels_needed": s.vessels_needed,
                    "weekly_cost": s.weekly_cost,
                    "source": s.source,
                    "rationale": s.rationale,
                }
                for s in seeds_from_merged
            ],
        )

        seeds = list(seeds_from_merged)
        refinements = []
        for it in range(1, int(refinement_iters) + 1):
            tag = f"refine{it}"
            log(f"Refinement {it}/{refinement_iters}: Stage 3 (seeded CG)...")
            cg = run_cg(
                data,
                seeds,
                fleet_profile=fleet_profile,
                api_key=(next_key() or None) if use_llm_cg else None,
                key_ring=keyring if use_llm_cg else None,
                verbose=bool(cg_verbose),
                snapshot_tag=tag,
                max_iter=int(cg_max_iter) if cg_max_iter is not None else 100,
                target_coverage_pct=target_coverage_pct,
                min_iter_before_profit_stop=2,
            )

            cg_routes_path = os.path.join(PROJECT_ROOT, "outputs", f"cg_routes_{tag}.json")
            pool = load_cg_route_records(cg_routes_path)

            bb_path = os.path.join(out_dir, f"bb_{tag}.json")
            log(f"Refinement {it}/{refinement_iters}: Stage 4 (seeded pool)...")
            bb = _solve_stage4_from_records(
                data=data,
                route_records=pool,
                out_bb_path=bb_path,
                coverage_bonus_per_ffe=float(bonus) * float(tuned["coverage_bonus_multiplier"]),
                vessel_mix_target_fraction_of_cap=tuned["vessel_mix_target_fraction_of_cap"],
                vessel_mix_penalty_per_vessel=float(tuned["vessel_mix_penalty_per_vessel"]),
                min_selected_route_utilization=tuned["min_selected_route_utilization"],
                large_vessel_bonus_per_ffe_capacity=float(tuned["large_vessel_bonus_per_ffe_capacity"]),
                route_activation_penalty=float(tuned["route_activation_penalty"]),
                route_min_ports=eff_min_ports,
                route_max_ports=eff_max_ports,
                two_phase=bool(two_phase_stage4),
                phase1_coverage_bonus_multiplier=float(phase1_coverage_bonus_multiplier),
                coverage_floor_ratio=float(coverage_floor_ratio),
                min_served_ffe=float(target_floor_ffe) if target_floor_ffe > 0 else None,
                log=log,
            )

            # feedback loop: reseed from this iteration's BB selection
            seeds = _top_seed_routes_from_bb(
                data=data,
                bb=bb,
                route_records=pool,
                seed_fraction=float(seed_fraction),
                seed_count=int(seed_count) if seed_count is not None else None,
                coverage_bonus_per_ffe=float(bonus),
                seed_tag=f"SEED_{tag.upper()}",
                route_min_ports=eff_min_ports,
            )

            refinements.append(
                {
                    "tag": tag,
                    "cg": {
                        "status": cg.status,
                        "routes": cg.total_routes,
                        "served_pct": cg.pct_served,
                        "net_profit": cg.final_net_profit,
                    },
                    "bb": {
                        "status": bb.status,
                        "selected_routes": bb.n_routes_selected,
                        "served_ffe": bb.total_served_ffe,
                        "served_pct": round(_served_pct(bb.total_served_ffe, data), 4),
                        "net_profit": bb.net_profit,
                        "bb_json": os.path.relpath(bb_path, PROJECT_ROOT),
                    },
                    "cg_routes_json": os.path.relpath(cg_routes_path, PROJECT_ROOT),
                    "n_seeds_next": len(seeds),
                }
            )

        summary = {
            "objective": {
                "priority": "equal (profit + demand coverage)" if mode == "balanced" else mode,
                "objective_mode": mode,
                "coverage_bonus_per_ffe": float(bonus),
                "seed_fraction": float(seed_fraction),
                "target_coverage_pct": float(target_coverage_pct),
                "seed_count": int(seed_count),
                "target_floor_ffe": float(target_floor_ffe),
                "refinement_iters": int(refinement_iters),
                "cg_verbose": bool(cg_verbose),
                "use_llm_intel": bool(use_llm_intel),
                "use_llm_seeds": bool(use_llm_seeds),
                "use_llm_cg": bool(use_llm_cg),
                "cg_max_iter": int(cg_max_iter) if cg_max_iter is not None else None,
                "max_transship_hops": int(max_transship_hops) if max_transship_hops is not None else None,
                "vessel_mix_target_fraction_of_cap": (
                    float(vessel_mix_target_fraction_of_cap)
                    if vessel_mix_target_fraction_of_cap is not None else None
                ),
                "vessel_mix_penalty_per_vessel": float(vessel_mix_penalty_per_vessel),
                "min_selected_route_utilization": (
                    float(min_selected_route_utilization)
                    if min_selected_route_utilization is not None else None
                ),
                "large_vessel_bonus_per_ffe_capacity": float(large_vessel_bonus_per_ffe_capacity),
                "route_activation_penalty": float(route_activation_penalty),
                "api_keys_in_ring": len(keyring),
                "auto_tune_stage4_policy": bool(auto_tune_stage4_policy),
                "tuned_policy": tuned,
                "use_route_port_filter": bool(use_route_port_filter),
                "route_min_ports": eff_min_ports,
                "route_max_ports": eff_max_ports,
                "two_phase_stage4": bool(two_phase_stage4),
                "two_phase_initial_runs": bool(two_phase_initial_runs),
                "phase1_coverage_bonus_multiplier": float(phase1_coverage_bonus_multiplier),
                "coverage_floor_ratio": float(coverage_floor_ratio),
                "default_vessel_mix_fraction_of_cap": float(default_vessel_mix_fraction_of_cap),
                "vessel_mix_target_fraction_of_cap_effective": float(mix_frac_effective),
            },
            "policy_tuning_trials": tuning_trials,
            "initial_runs": initial_runs,
            "merge": {
                "merged_pool_routes": len(merged_pool),
                "merged_routes_json": os.path.relpath(merged_routes_path, PROJECT_ROOT),
                "min_routes_selected_after_merge": int(min_routes_after_merge),
                "bb": {
                    "status": merged_bb.status,
                    "selected_routes": merged_bb.n_routes_selected,
                    "served_ffe": merged_bb.total_served_ffe,
                    "served_pct": round(_served_pct(merged_bb.total_served_ffe, data), 4),
                    "net_profit": merged_bb.net_profit,
                    "bb_json": os.path.relpath(merged_bb_path, PROJECT_ROOT),
                },
                "seed_json": os.path.relpath(seeds_path, PROJECT_ROOT),
                "n_seed_routes": len(seeds_from_merged),
            },
            "refinements": refinements,
            "final": {
                "bb_json": refinements[-1]["bb"]["bb_json"] if refinements else os.path.relpath(merged_bb_path, PROJECT_ROOT),
                "headline": (refinements[-1]["bb"] if refinements else {
                    "status": merged_bb.status,
                    "selected_routes": merged_bb.n_routes_selected,
                    "served_ffe": merged_bb.total_served_ffe,
                    "served_pct": round(_served_pct(merged_bb.total_served_ffe, data), 4),
                    "net_profit": merged_bb.net_profit,
                }),
            },
        }

        summary_path = os.path.join(out_dir, "summary.json")
        _save_json(summary_path, summary)
        log(f"Done. See outputs/merge_pipeline/summary.json")
        return summary

    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else None
    run_merge_architecture(gemini_key=key)