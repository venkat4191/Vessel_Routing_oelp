"""
One-command pipeline runner for LinerNet.

Runs:
  Stage 0 -> Stage 1A -> Stage 1B -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5 -> Stage 6

Gemini key:
  - optional argv[1], else read from .env as GEMINI_API_KEY
"""

import json
import os
import sys
import traceback
from typing import Callable

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from stage0.loader import load_all, validate as validate_data
from stage1.demand_intel import run as run_intel
from stage1.fleet_profiler import run as run_fleet
from stage2.seed_generator import run as run_seeds, seeds_to_dict
from stage3.cg_loop import run as run_cg
from stage4.route_subset_mip import run_stage4
from stage5.parallel_eval import run as run_parallel_eval
from stage6.warm_start import run as run_warm_start
from utils.env import get_gemini_key, get_api_keys, load_dotenv
from utils.config import WARM_START_TARGET_COVERAGE_PCT
from utils.paper_profit import (
    breakdown_to_dict,
    evaluate_paper_profit,
    load_services_from_bb,
)


def _save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _intel_to_dict(intel) -> dict:
    return {
        "top_hubs": intel.top_hubs,
        "top_od_pairs": [
            {
                "origin": od.origin,
                "destination": od.destination,
                "ffe_per_week": od.ffe_per_week,
                "revenue_per_ffe": od.revenue_per_ffe,
                "weekly_revenue": od.weekly_revenue,
                "priority_rank": od.priority_rank,
            }
            for od in intel.od_priorities[:50]
        ],
        "hub_rankings": [
            {
                "unlocode": hs.unlocode,
                "name": hs.name,
                "d_region": hs.d_region,
                "hub_score": hs.hub_score,
                "classification": hs.llm_classification,
                "reasoning": hs.llm_reasoning,
            }
            for hs in intel.hub_rankings
        ],
        "llm_enriched": intel.llm_enriched,
        "llm_summary": intel.llm_summary,
        "total_weekly_demand_ffe": intel.total_weekly_demand_ffe,
        "total_max_revenue": intel.total_max_revenue,
    }


def run_all(
    gemini_key: str = None,
    log: Callable[[str], None] = print,
    target_coverage_pct: float = None,
    coverage_bonus_per_ffe: float = 15.0,
    profit_floor_ratio: float = 0.80,
) -> dict:
    load_dotenv(PROJECT_ROOT)
    key = get_gemini_key(PROJECT_ROOT, gemini_key)

    # Load all keys from api_keys_store.py / .env for rotation.
    # Keys are used one at a time — next key is only tried on 429 (rate-limit).
    keyring = get_api_keys(PROJECT_ROOT) or ([key] if key else [])
    if gemini_key:
        keyring = [gemini_key.strip()]

    # primary_key is always keyring[0]; the full ring is passed to every stage
    # so llm_client rotates keys internally on 429 — exhaustion is global, not per-stage.
    primary_key = keyring[0] if keyring else None

    import contextlib
    with contextlib.suppress(FileNotFoundError):
        os.remove(os.path.join(PROJECT_ROOT, "outputs", "pipeline_summary.json"))

    try:
        log("Stage 0: loading and validating data...")
        data = load_all(verbose=False)
        validate_data(data)

        log("Stage 1A: demand intelligence...")
        intel = run_intel(data, api_key=primary_key, key_ring=keyring, verbose=False)
        _save_json(os.path.join(PROJECT_ROOT, "outputs", "demand_intel.json"), _intel_to_dict(intel))

        log("Stage 1B: fleet profiler...")
        fleet_profile = run_fleet(data, verbose=False)

        log("Stage 2: seed generation...")
        seeds = run_seeds(data, intel, fleet_profile, api_key=primary_key, key_ring=keyring, verbose=False)
        _save_json(os.path.join(PROJECT_ROOT, "outputs", "seeds.json"), seeds_to_dict(seeds))

        for loop_iter in range(5):
            log(f"--- Iteration {loop_iter+1}/5 ---")
            log("Stage 3: column generation loop...")
            cg_result = run_cg(
                data,
                seeds,
                fleet_profile=fleet_profile,
                api_key=primary_key,
                key_ring=keyring,
                verbose=False,
            )

            log("Stage 4: branch and bound (MIP)...")
            bb_result = run_stage4(data, verbose=False)
            
            if loop_iter < 4:
                paper_sv = load_services_from_bb(PROJECT_ROOT)
                
                if loop_iter == 0:
                    frac_str, fraction = "1/4th", 1.0 / 4.0
                elif loop_iter == 1:
                    frac_str, fraction = "1/3rd", 1.0 / 3.0
                elif loop_iter == 2:
                    frac_str, fraction = "1/2", 1.0 / 2.0
                else:
                    frac_str, fraction = "Full", 1.0
                    
                top_n = max(1, int(len(paper_sv) * fraction))
                
                route_profit = {}
                for r in paper_sv:
                    rid = r.get("route_id")
                    cost = r.get("weekly_cost", 0.0)
                    rev = 0.0
                    for flow_key, flow_val in bb_result.cargo_flows.items():
                        if flow_key.endswith(f"|{rid}"):
                            od_str = flow_key.split("|")[0]
                            o, d = od_str.split("->")
                            if (o, d) in data["demand"]:
                                rev += flow_val * data["demand"][(o, d)].revenue_per_ffe
                    route_profit[rid] = rev - cost
                
                paper_sv.sort(key=lambda r: route_profit.get(r.get("route_id"), -float('inf')), reverse=True)
                
                from stage2.seed_generator import SeedRoute
                seeds = []
                for idx, r in enumerate(paper_sv[:top_n]):
                    seq = r.get("port_sequence", [])
                    if len(seq) >= 3:
                        seeds.append(SeedRoute(
                            route_id=f"S_I{loop_iter+1}_{idx+1:02d}",
                            port_sequence=seq,
                            vessel_class=r.get("vessel_class", ""),
                            frequency=r.get("frequency", 1),
                            cycle_days=r.get("cycle_days", 0.0),
                            vessels_needed=r.get("vessels_needed", 0),
                            weekly_cost=r.get("weekly_cost", 0.0),
                            source="bb_top25",
                            rationale=f"Top 25% from iteration {loop_iter+1}"
                        ))
                log(f"Selected {len(seeds)} top routes ({frac_str}) as seeds for iteration {loop_iter+2}.")

        log("Stage 5: parallel scenario evaluation...")
        stage5 = run_parallel_eval(verbose=False)

        log("Stage 6: warm-start refinement...")
        stage6 = run_warm_start(
            data=data,
            api_key=primary_key,
            key_ring=keyring,
            verbose=False,
            verbose_cg=False,
            target_coverage_pct=(
                float(target_coverage_pct)
                if target_coverage_pct is not None else WARM_START_TARGET_COVERAGE_PCT
            ),
            coverage_bonus_per_ffe=float(coverage_bonus_per_ffe),
            profit_floor_ratio=float(profit_floor_ratio),
        )
    except Exception as exc:
        log(f"ERROR: {type(exc).__name__}: {exc}")
        log(traceback.format_exc())
        raise

    paper_sv = load_services_from_bb(PROJECT_ROOT)
    paper_br = evaluate_paper_profit(data, paper_sv)
    paper_dict = breakdown_to_dict(paper_br)
    _save_json(os.path.join(PROJECT_ROOT, "outputs", "paper_profit.json"), paper_dict)

    summary = {
        "cg": {
            "status": cg_result.status,
            "routes": cg_result.total_routes,
            "served_pct": cg_result.pct_served,
            "net_profit": cg_result.final_net_profit,
        },
        "bb": {
            "status": bb_result.status,
            "selected_routes": bb_result.n_routes_selected,
            "served_ffe": bb_result.total_served_ffe,
            "net_profit": bb_result.net_profit,
        },
        "paper_profit": paper_dict,
        "stage5_scenarios": len(stage5),
        "stage6_rounds": stage6.get("rounds_run", 0),
        "stage6_best": stage6.get("best", {}),
        "final_headline": {
            "net_profit": stage6.get("best", {}).get("mip_net_profit", bb_result.net_profit),
            "served_pct": stage6.get("best", {}).get("mip_pct_served"),
            "selected_routes": stage6.get("best", {}).get("n_selected"),
            "paper_eta_musd_per_week": paper_dict.get("eta_million_usd_per_week"),
            "paper_pct_served": paper_br.details.get("pct_served"),
        },
    }
    _save_json(os.path.join(PROJECT_ROOT, "outputs", "pipeline_summary.json"), summary)
    log("Done. See outputs/pipeline_summary.json")
    return summary


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else None
    run_all(gemini_key=key)

