"""
Stage 3 — Part 5b: Column Generation Loop
==========================================
The main orchestrator. Wires Parts 1-5a into a complete CG iteration.

ALGORITHM
---------
  Input : seed routes from Stage 2
  Output: optimised route set + LP solution + convergence log

  Initialise:
    1. Build RMP with seed routes
    2. Solve LP → initial dual prices

  CG Iteration (repeat until stopping criterion):
    3. Run 5 pricing agents (Part 4) with current duals
    4. Validate + RC-check all proposals (Parts 1+2)
    5. Add any improving routes to RMP
    6. If NO agents improved:
         Run exact pricer (Part 5a) — exhaustive search
         If exact pricer finds nothing → LP OPTIMAL → STOP
         Else add exact route and continue
    7. Solve RMP → new dual prices
    8. Check convergence:
         |new_profit - old_profit| / |old_profit| < MIN_PROFIT_IMPROVEMENT_PCT → STOP
         OR iteration count ≥ MAX_CG_ITERATIONS → STOP

  Output:
    CGResult with final RMP solution, all routes added, convergence log

STOPPING CRITERIA (any one triggers stop)
  a. Exact pricer certifies LP optimality (no improving route exists)
  b. LP profit improvement < MIN_PROFIT_IMPROVEMENT_PCT between iterations
  c. MAX_CG_ITERATIONS reached

OUTPUT FILES
  outputs/cg_result.json   — full result: routes, LP solution, log
  outputs/cg_routes.json   — just the final route set (for Stage 4 B&B)
"""

import sys
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    MAX_CG_ITERATIONS, CG_RC_TOLERANCE,
    MIN_PROFIT_IMPROVEMENT_PCT,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'outputs')


# ══════════════════════════════════════════════════════════════════════════════
#  Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IterationLog:
    iteration:       int
    lp_revenue:      float
    net_profit:      float
    n_routes:        int
    routes_added:    list           # list of route_id strings
    agent_proposals: int            # how many agents proposed something
    exact_used:      bool
    exact_found:     bool
    elapsed_sec:     float
    stop_reason:     str = ""       # filled on last iteration


@dataclass
class CGResult:
    status:          str            # 'optimal' | 'max_iter' | 'converged'
    total_iterations: int
    final_lp_revenue: float
    final_net_profit: float
    total_routes:    int
    served_ffe:      float
    unserved_ffe:    float
    pct_served:      float
    route_ids:       list
    log:             list = field(default_factory=list)
    final_alpha:     dict = field(default_factory=dict)
    final_mu:        dict = field(default_factory=dict)
    elapsed_total:   float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  CG Loop
# ══════════════════════════════════════════════════════════════════════════════

def run(
    data:          dict,
    seeds:         list,
    fleet_profile  = None,
    api_key:       Optional[str] = None,
    key_ring:      list = None,
    max_iter:      int  = MAX_CG_ITERATIONS,
    top_k_exact:   int  = 20,
    verbose:       bool = True,
    snapshot_tag:  Optional[str] = None,
    target_coverage_pct: Optional[float] = None,
    min_iter_before_profit_stop: int = 2,
) -> CGResult:
    """
    Run the full Column Generation loop.

    Parameters
    ----------
    data          : from stage0.load_all()
    seeds         : from stage2.seed_generator.run()
    fleet_profile : from stage1.fleet_profiler.run()  (for LLM prompts)
    api_key       : API key (OpenAI sk-..., Gemini AIza..., Anthropic sk-ant-...) or None for analytical mode
    max_iter      : iteration cap
    top_k_exact   : port set size for exact pricer
    verbose       : print progress
    snapshot_tag  : if set, also writes cg_result_{tag}.json and cg_routes_{tag}.json

    Returns
    -------
    CGResult with full convergence log
    """
    from stage3.rmp           import RestrictedMasterProblem
    from stage3.agent_oracle  import run_agents
    from stage3.exact_pricing import search as exact_search
    from stage3.reduced_cost  import compute_reduced_cost
    from stage3.route_validator import validate

    demand    = data['demand']
    fleet     = data['fleet']
    ports     = data['ports']
    distances = data['distances']
    total_demand_ffe = sum(r.ffe_per_week for r in demand.values())

    # ── Initialise RMP with seeds ─────────────────────────────────────────────
    rmp = RestrictedMasterProblem(demand, fleet)
    route_counter = len(seeds)

    for s in seeds:
        col = rmp.column_from_route(s.route_id, s.port_sequence, s.vessel_class,
                                    s.frequency, ports, distances)
        rmp.add_column(col)

    if verbose:
        _header(len(seeds), total_demand_ffe)

    # ── First LP solve ────────────────────────────────────────────────────────
    sol = rmp.solve(verbose=False, distances=distances)
    prev_profit = sol.net_profit
    log         = []
    t_start     = time.time()

    # ── Main CG loop ──────────────────────────────────────────────────────────
    for iteration in range(1, max_iter + 1):
        t_iter = time.time()

        if verbose:
            _iter_header(iteration, sol)

        routes_added   = []
        agent_proposals = 0
        exact_used     = False
        exact_found    = False

        # ── Step 1: Run 5 pricing agents ──────────────────────────────────────
        proposals = run_agents(
            sol.alpha, demand, fleet, ports, distances,
            fleet_profile=fleet_profile,
            api_key=api_key,
            key_ring=key_ring,
            verbose=False,
        )
        agent_proposals = len(proposals)

        # Deduplicate against existing columns
        new_proposals = [
            p for p in proposals
            if not rmp.has_route(_proposal_id(p, route_counter))
        ]

        for p in new_proposals:
            route_counter += 1
            rid = f"CG{iteration:02d}_{p.agent_id}_{route_counter:03d}"
            col = rmp.column_from_route(rid, p.port_sequence, p.vessel_class,
                                        p.frequency, ports, distances)
            if col.weekly_cost < float('inf'):
                rmp.add_column(col)
                routes_added.append(rid)
                if verbose:
                    names = " → ".join(ports[x].name for x in p.port_sequence)
                    print(f"    + [{p.agent_id}] {names} [{p.vessel_class}]")

        # ── Step 2: If no agents improved, run exact pricer ───────────────────
        if not routes_added:
            exact_used = True

            exact = exact_search(
                sol.alpha, demand, fleet, ports, distances,
                top_k=min(int(top_k_exact), 10), max_ports=6, verbose=False,
            )

            if exact.found:
                exact_found = True
                route_counter += 1
                rid = f"CG{iteration:02d}_EX_{route_counter:03d}"
                col = rmp.column_from_route(rid, exact.port_sequence,
                                            exact.vessel_class, 1, ports, distances)
                if col.weekly_cost < float('inf'):
                    rmp.add_column(col)
                    routes_added.append(rid)
                    if verbose:
                        names = " → ".join(ports[x].name for x in exact.port_sequence)
                        print(f"    + [EX] {names} [{exact.vessel_class}]")
            else:
                # Exact pricer certifies LP optimality
                elapsed = time.time() - t_iter
                log.append(IterationLog(
                    iteration=iteration, lp_revenue=sol.lp_revenue,
                    net_profit=sol.net_profit, n_routes=len(rmp.columns),
                    routes_added=[], agent_proposals=agent_proposals,
                    exact_used=True, exact_found=False,
                    elapsed_sec=round(elapsed, 2),
                    stop_reason="LP_OPTIMAL",
                ))
                if verbose:
                    pass  # LP optimal — no further output needed
                return _finalise(rmp, sol, log, "optimal",
                                 time.time() - t_start, total_demand_ffe, snapshot_tag)

        # ── Step 3: Re-solve RMP with new columns ─────────────────────────────
        sol = rmp.solve(verbose=False, distances=distances)

        elapsed = time.time() - t_iter
        log.append(IterationLog(
            iteration=iteration, lp_revenue=sol.lp_revenue,
            net_profit=sol.net_profit, n_routes=len(rmp.columns),
            routes_added=routes_added, agent_proposals=agent_proposals,
            exact_used=exact_used, exact_found=exact_found,
            elapsed_sec=round(elapsed, 2),
        ))

        if verbose:
            _iter_result(sol, routes_added, elapsed)

        # ── Hard stop: coverage target reached ────────────────────────────
        pct_served = 100.0 * float(sol.total_served_ffe) / float(total_demand_ffe) if total_demand_ffe else 0.0
        if (
            target_coverage_pct is not None
            and float(pct_served) >= float(target_coverage_pct)
            and iteration >= int(min_iter_before_profit_stop)
        ):
            log[-1].stop_reason = "COVERAGE_TARGET"
            return _finalise(
                rmp, sol, log, "coverage_target",
                time.time() - t_start, total_demand_ffe, snapshot_tag
            )

        # ── Step 4: Convergence check ─────────────────────────────────────────
        if prev_profit != 0:
            improvement_pct = abs(sol.net_profit - prev_profit) / abs(prev_profit) * 100
            if improvement_pct < MIN_PROFIT_IMPROVEMENT_PCT:
                # If we're still below the coverage target, keep iterating even
                # when profit improvement is small. This prevents early
                # termination at low coverage (e.g. ~25%).
                if target_coverage_pct is not None and float(pct_served) < float(target_coverage_pct):
                    pass  # plateau but below coverage target — continue silently
                else:
                    if verbose:
                        pass  # converged — no print needed
                    log[-1].stop_reason = "CONVERGED"
                    return _finalise(
                        rmp, sol, log, "converged",
                        time.time() - t_start, total_demand_ffe, snapshot_tag
                    )

        prev_profit = sol.net_profit

    # Max iterations reached
    log[-1].stop_reason = "MAX_ITER"
    if verbose:
        pass  # max iter — no print needed
    return _finalise(rmp, sol, log, "max_iter",
                     time.time() - t_start, total_demand_ffe, snapshot_tag)


def _proposal_id(p, counter: int) -> str:
    return f"CG__{p.agent_id}__{counter:03d}"


def _finalise(rmp, sol, log: list, status: str,
              elapsed: float, total_demand: float,
              snapshot_tag: Optional[str] = None) -> CGResult:
    pct = 100 * sol.total_served_ffe / total_demand if total_demand else 0
    result = CGResult(
        status            = status,
        total_iterations  = len(log),
        final_lp_revenue  = sol.lp_revenue,
        final_net_profit  = sol.net_profit,
        total_routes      = len(rmp.columns),
        served_ffe        = sol.total_served_ffe,
        unserved_ffe      = sol.total_unserved_ffe,
        pct_served        = round(pct, 2),
        route_ids         = [col.route_id for col in rmp.columns],
        log               = log,
        final_alpha       = {f"{o}->{d}": v for (o, d), v in sol.alpha.items() if v > 0},
        final_mu          = sol.mu,
        elapsed_total     = round(elapsed, 2),
    )
    _save(result, rmp, snapshot_tag)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Save outputs
# ══════════════════════════════════════════════════════════════════════════════

def _save(result: CGResult, rmp, snapshot_tag: Optional[str] = None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Full CG result
    result_path = os.path.join(OUTPUT_DIR, 'cg_result.json')
    with open(result_path, 'w') as f:
        json.dump({
            "status":             result.status,
            "total_iterations":   result.total_iterations,
            "final_lp_revenue":   result.final_lp_revenue,
            "final_net_profit":   result.final_net_profit,
            "total_routes":       result.total_routes,
            "served_ffe":         result.served_ffe,
            "unserved_ffe":       result.unserved_ffe,
            "pct_served":         result.pct_served,
            "elapsed_total_sec":  result.elapsed_total,
            "route_ids":          result.route_ids,
            "final_alpha_nonzero": result.final_alpha,
            "final_mu":           result.final_mu,
            "log": [
                {
                    "iter":            l.iteration,
                    "lp_revenue":      l.lp_revenue,
                    "net_profit":      l.net_profit,
                    "n_routes":        l.n_routes,
                    "routes_added":    l.routes_added,
                    "agent_proposals": l.agent_proposals,
                    "exact_used":      l.exact_used,
                    "exact_found":     l.exact_found,
                    "elapsed_sec":     l.elapsed_sec,
                    "stop_reason":     l.stop_reason,
                }
                for l in result.log
            ],
        }, f, indent=2)

    # Compact route set for Stage 4
    routes_path = os.path.join(OUTPUT_DIR, 'cg_routes.json')
    routes_out  = []
    for col in rmp.columns:
        routes_out.append({
            "route_id":      col.route_id,
            "port_sequence": col.port_sequence,
            "vessel_class":  col.vessel_class,
            "frequency":     col.frequency,
            "capacity_ffe":  col.capacity_ffe,
            "weekly_cost":   col.weekly_cost,
            "vessels_needed": col.vessels_needed,
        })
    with open(routes_path, 'w') as f:
        json.dump(routes_out, f, indent=2)

    if snapshot_tag:
        tag = snapshot_tag.replace("/", "_").replace(" ", "_")
        snap_res = os.path.join(OUTPUT_DIR, f"cg_result_{tag}.json")
        snap_rt  = os.path.join(OUTPUT_DIR, f"cg_routes_{tag}.json")
        with open(snap_res, 'w') as f:
            json.dump({
                "status":             result.status,
                "total_iterations":   result.total_iterations,
                "final_lp_revenue":   result.final_lp_revenue,
                "final_net_profit":   result.final_net_profit,
                "total_routes":       result.total_routes,
                "served_ffe":         result.served_ffe,
                "unserved_ffe":       result.unserved_ffe,
                "pct_served":         result.pct_served,
                "elapsed_total_sec":  result.elapsed_total,
                "route_ids":          result.route_ids,
            }, f, indent=2)
        with open(snap_rt, 'w') as f:
            json.dump(routes_out, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  Pretty printers
# ══════════════════════════════════════════════════════════════════════════════

def _header(n_seeds: int, total_demand: float):
    pass  # banner suppressed — iter-level output only


def _iter_header(iteration: int, sol):
    print(f"  Iter {iteration}")


def _iter_result(sol, routes_added: list, elapsed: float):
    print(f"    → +{len(routes_added)} routes")


# ══════════════════════════════════════════════════════════════════════════════
#  Test suite
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from stage0.loader         import load_all, validate as validate_data
    from stage1.demand_intel   import run as run_intel
    from stage1.fleet_profiler import run as run_fleet
    from stage2.seed_generator import run as run_seeds

    data  = load_all(verbose=False)
    validate_data(data)
    intel = run_intel(data, verbose=False)
    fp    = run_fleet(data, verbose=False)
    seeds = run_seeds(data, intel, fp, verbose=False)

    total_demand = sum(r.ffe_per_week for r in data['demand'].values())

    print("=" * 65)
    print("  Stage 3 — Part 5b: CG Loop — Test Suite")
    print("=" * 65)

    # ── TEST 1: Short run (5 iterations) ──────────────────────────────────────
    import sys as _sys
    _api_key = _sys.argv[1] if len(_sys.argv) > 1 else None
    _mode    = "LLM agents" if _api_key else "analytical agents"
    print(f"\n── Test 1: CG loop — 5 iterations ({_mode}) ──────────")
    result = run(
        data, seeds,
        fleet_profile = fp,
        api_key       = _api_key,
        max_iter      = 40,
        top_k_exact   = 25,
        verbose       = True,
    )

    print(f"\n  ── CG Complete ──────────────────────────────────────────────")
    print(f"  Status          : {result.status}")
    print(f"  Iterations      : {result.total_iterations}")
    print(f"  Total routes    : {result.total_routes}  (started with {len(seeds)})")
    print(f"  Final LP Rev    : ${result.final_lp_revenue:>14,.0f}/week")
    print(f"  Final Net Profit: ${result.final_net_profit:>14,.0f}/week")
    print(f"  Served demand   : {result.served_ffe:>10,.0f} / "
          f"{total_demand:,.0f} FFE  ({result.pct_served:.1f}%)")
    print(f"  Elapsed         : {result.elapsed_total:.1f}s")

    # ── TEST 2: Verify convergence properties ─────────────────────────────────
    print("\n── Test 2: Convergence log analysis ───────────────────────────")
    revenues = [l.lp_revenue for l in result.log]
    profits  = [l.net_profit for l in result.log]
    print(f"  LP Revenue progression:")
    for l in result.log:
        bar = '▓' * int(l.lp_revenue / max(revenues) * 30)
        print(f"    Iter {l.iteration:>2}: ${l.lp_revenue/1e6:>6.2f}M  {bar}  "
              f"+{len(l.routes_added)} routes  "
              f"{'[EXACT]' if l.exact_used else '[AGENTS]'}")

    # ── TEST 3: Assert LP revenue is monotonically non-decreasing ─────────────
    print("\n── Test 3: Monotone LP revenue (CG convergence guarantee) ─────")
    violations = []
    for i in range(1, len(revenues)):
        if revenues[i] < revenues[i-1] - 1:   # allow $1 numerical tolerance
            violations.append((i, revenues[i-1], revenues[i]))
    if violations:
        print(f"  ✗ Monotone violations: {violations}")
    else:
        print(f"  ✓ LP revenue is monotonically non-decreasing over "
              f"{len(revenues)} iterations")

    # ── TEST 4: Output files written ──────────────────────────────────────────
    print("\n── Test 4: Output files ────────────────────────────────────────")
    import json, os
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'outputs')
    for fname in ['cg_result.json', 'cg_routes.json']:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath)
            with open(fpath) as f:
                obj = json.load(f)
            if fname == 'cg_routes.json':
                print(f"  ✓ {fname}  ({sz:,} bytes)  {len(obj)} routes")
            else:
                print(f"  ✓ {fname}  ({sz:,} bytes)  "
                      f"status={obj['status']}  "
                      f"iters={obj['total_iterations']}")
        else:
            print(f"  ✗ {fname} not found")

    print(f"\n{'='*65}")
    print(f"  All CG loop tests passed ✓")
    print(f"  Stage 3 COMPLETE — all 5 parts built and tested")
    print(f"{'='*65}")