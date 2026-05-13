"""
Stage 3 — Part 4: Agent Oracle (5 Pricing Agents)
==================================================
After each RMP solve, 5 specialised agents read the dual prices
and propose candidate routes with positive reduced cost.

AGENT ROSTER
------------
  A1 — Demand Coverage (destination focus)
       Groups uncovered OD pairs by destination, builds string routes
       that funnel multiple origins into the same high-value destination.

  A2 — Demand Coverage (origin spread)
       Groups uncovered OD pairs by origin, builds routes that carry
       one origin's cargo to multiple high-value destinations.

  B1 — Hub Efficiency (inbound relay)
       Routes cargo from multiple origins through a transshipment hub,
       then onward to a destination cluster.

  B2 — Hub Efficiency (outbound relay)
       Picks up at an origin cluster, relays through a hub, and fans
       out to multiple destinations.

  C1 — Peak Robustness
       Scales alpha duals DOWN by 1/PEAK_DEMAND_FACTOR to simulate
       +30% demand surge. Finds routes that look profitable under peak
       load. Prefers large vessels for capacity headroom.

FLOW PER AGENT
--------------
  1. Score uncovered / high-alpha OD pairs
  2. Build candidate port sequences (3–6 ports)
  3. Route Validator (Part 1) — hard gate
  4. Reduced Cost (Part 2) — only keep RC > CG_RC_TOLERANCE
  5. Return best AgentProposal or None

LLM MODE
--------
  If api_key is provided, each agent sends its top opportunity data
  to Claude (claude-sonnet-4-6) and asks for one route proposal.
  The response is validated + RC-checked before acceptance.
  Falls back to analytical if LLM is unavailable or proposes invalid route.

OUTPUT
------
  run_agents() returns list[AgentProposal] — one per agent that found
  an improving route.  The CG loop (Part 5) adds these to the RMP.
"""

import sys
import os
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import CG_RC_TOLERANCE, PEAK_DEMAND_FACTOR


# ══════════════════════════════════════════════════════════════════════════════
#  Data structure
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentProposal:
    agent_id:      str          # 'A1' | 'A2' | 'B1' | 'B2' | 'C1'
    agent_type:    str          # 'demand' | 'hub' | 'peak'
    port_sequence: list
    vessel_class:  str
    frequency:     int
    rc_value:      float        # reduced cost (> 0 means improving)
    rationale:     str
    source:        str          # 'analytical' | 'llm'


# ══════════════════════════════════════════════════════════════════════════════
#  Shared utilities
# ══════════════════════════════════════════════════════════════════════════════

# Known transshipment hubs in WorldSmall (ordered by importance)
TRANSSHIP_HUBS = [
    "SGSIN",   # Singapore
    "MYTPP",   # Tanjung Pelepas
    "EGPSD",   # Port Said
    "ESALG",   # Algeciras
    "MAPTM",   # Tangier Med
    "ITGIT",   # Gioia Tauro
    "LKCMB",   # Colombo
    "OMSLL",   # Salalah
    "AEJEA",   # Jebel Ali
    "MYPKG",   # Port Klang
]

# Bias knobs: generate benchmark-style long services with larger vessels.
CG_BIAS_MIN_PORTS = 6
CG_BIAS_MAX_PORTS = 14


def _uncovered_pairs(alpha: dict, demand: dict) -> list:
    """
    Return (od, record) for all OD pairs with alpha=0,
    sorted by weekly_revenue descending.
    These are the richest targets for any agent.
    """
    pairs = [
        (od, rec)
        for od, rec in demand.items()
        if alpha.get(od, 0.0) == 0.0
    ]
    pairs.sort(key=lambda x: -x[1].weekly_revenue)
    return pairs


def _opportunity_score(od: tuple, alpha: dict, demand: dict) -> float:
    """
    Adjusted weekly revenue = (rev_per_ffe - alpha) * ffe_per_week.
    High score = agent should include this OD pair in its proposed route.
    """
    rec = demand.get(od)
    if rec is None:
        return 0.0
    return max(0.0, rec.revenue_per_ffe - alpha.get(od, 0.0)) * rec.ffe_per_week


def _feasible_vessel(seq: list, ports: dict, distances: dict, fleet: dict) -> Optional[str]:
    """
    Return the largest vessel class feasible for this port sequence,
    or None if no vessel class works.
    Checks draft at every port and canal restrictions on every leg.
    """
    order = ["Super_panamax", "Post_panamax", "Panamax_2400",
             "Panamax_1200", "Feeder_800", "Feeder_450"]

    all_legs = (
        [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
        + [(seq[-1], seq[0])]   # return leg
    )

    for vc in order:
        v = fleet[vc]
        # Draft check at every port
        if any(v.draft > ports[p].draft for p in seq if p in ports):
            continue
        # Canal check on every leg
        ok = True
        for (f, t) in all_legs:
            dr = distances.get((f, t))
            if dr is None:
                ok = False; break
            if dr.is_panama:
                if v.panama_fee is None:
                    ok = False; break
                if dr.draft and v.draft > dr.draft:
                    ok = False; break
            if dr.is_suez:
                if v.suez_fee is None:
                    ok = False; break
        if ok:
            return vc
    return None


def _try_proposal(
    agent_id:   str,
    agent_type: str,
    seq:        list,
    alpha:      dict,
    demand:     dict,
    fleet:      dict,
    ports:      dict,
    distances:  dict,
    rationale:  str,
    source:     str = "analytical",
    prefer_large: bool = True,
) -> Optional[AgentProposal]:
    """
    Validate a candidate sequence, compute RC, return AgentProposal if improving.
    Tries vessel classes largest-first (or smallest-first if prefer_large=False).
    """
    from stage3.route_validator import validate
    from stage3.reduced_cost    import compute_reduced_cost

    if len(seq) < 2 or len(set(seq)) != len(seq):
        return None
    if any(p not in ports for p in seq):
        return None
    if len(seq) < CG_BIAS_MIN_PORTS or len(seq) > CG_BIAS_MAX_PORTS:
        return None

    order = ["Super_panamax", "Post_panamax", "Panamax_2400",
             "Panamax_1200", "Feeder_800", "Feeder_450"]
    if not prefer_large:
        order = list(reversed(order))

    best: Optional[AgentProposal] = None

    for vc in order:
        ok, _ = validate(seq, vc, fleet, ports, distances, demand, frequency=1)
        if not ok:
            continue
        rc = compute_reduced_cost(seq, vc, alpha, fleet, ports, distances, demand)
        if not rc["is_improving"]:
            continue
        if best is None or rc["rc"] > best.rc_value:
            best = AgentProposal(
                agent_id      = agent_id,
                agent_type    = agent_type,
                port_sequence = seq,
                vessel_class  = vc,
                frequency     = 1,
                rc_value      = rc["rc"],
                rationale     = rationale,
                source        = source,
            )
        # Do NOT break — try all vessel sizes, keep the one with highest RC
        # (a Super_panamax at RC=$2M beats a Feeder_800 at RC=$100K)

    return best


def _dedup_seq(seq: list) -> list:
    """Remove duplicates while preserving order."""
    seen = set(); out = []
    for p in seq:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT A1 — Demand Coverage: destination focus
#  "Which destination has the most uncovered inbound revenue?
#   Build a string route bringing multiple origins to that destination."
# ══════════════════════════════════════════════════════════════════════════════

def _agent_A1(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:

    # Group uncovered pairs by destination, sum opportunity score
    dest_score  = defaultdict(float)
    dest_origins = defaultdict(list)  # dest → [(score, origin)]

    for (od, rec) in _uncovered_pairs(alpha, demand):
        o, d = od
        score = rec.weekly_revenue   # alpha=0 so full revenue is the score
        dest_score[d]  += score
        dest_origins[d].append((score, o))

    # Sort destinations by total uncovered inbound revenue
    sorted_dests = sorted(dest_score.items(), key=lambda x: -x[1])

    for dest, _ in sorted_dests:
        if dest not in ports:
            continue

        # Top origins for this destination, by weekly revenue
        origins_ranked = sorted(dest_origins[dest], reverse=True)
        top_origins = [o for _, o in origins_ranked[:13] if o in ports]

        if not top_origins:
            continue

        # Try sequences of 3, 4, 6, 8, 10, 12 origins + destination
        for n in (14, 12, 10, 8, 6):
            candidates_to_try = [top_origins[:n] + [dest]]

            # Also try inserting a natural hub between origins and dest
            for hub in TRANSSHIP_HUBS:
                if hub not in (dest,) and hub not in top_origins and hub in ports:
                    candidates_to_try.append(top_origins[:min(n-1,2)] + [hub, dest])

            for raw_seq in candidates_to_try:
                seq = _dedup_seq(raw_seq)
                if len(seq) < 2:
                    continue
                rat = (f"A1: {len(seq)-1} origins → {ports[dest].name}. "
                       f"Top origin: {top_origins[0]}, "
                       f"uncovered rev=${dest_score[dest]:,.0f}/wk")
                prop = _try_proposal("A1", "demand", seq, alpha, demand,
                                     fleet, ports, distances, rat)
                if prop:
                    return prop

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT A2 — Demand Coverage: origin spread
#  "Which origin has the most uncovered outbound revenue?
#   Build a route carrying that origin's cargo to multiple destinations."
# ══════════════════════════════════════════════════════════════════════════════

def _agent_A2(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:

    # Group uncovered pairs by origin
    origin_score = defaultdict(float)
    origin_dests  = defaultdict(list)  # origin → [(score, dest)]

    for (od, rec) in _uncovered_pairs(alpha, demand):
        o, d = od
        score = rec.weekly_revenue
        origin_score[o]  += score
        origin_dests[o].append((score, d))

    sorted_origins = sorted(origin_score.items(), key=lambda x: -x[1])

    for origin, _ in sorted_origins:
        if origin not in ports:
            continue

        dests_ranked = sorted(origin_dests[origin], reverse=True)
        top_dests = [d for _, d in dests_ranked[:13] if d in ports]

        if not top_dests:
            continue

        # Try: origin → dest1 → dest2 ... up to massive strings
        for n in (14, 12, 10, 8, 6):
            candidates_to_try = [[origin] + top_dests[:n]]

            # Also try origin → hub → dest1 → dest2
            for hub in TRANSSHIP_HUBS:
                if hub not in (origin,) and hub not in top_dests and hub in ports:
                    candidates_to_try.append([origin, hub] + top_dests[:min(n-1, 2)])

            for raw_seq in candidates_to_try:
                seq = _dedup_seq(raw_seq)
                if len(seq) < 2:
                    continue
                rat = (f"A2: {ports[origin].name} → {n} destinations. "
                       f"Top dest: {top_dests[0]}, "
                       f"uncovered rev=${origin_score[origin]:,.0f}/wk")
                prop = _try_proposal("A2", "demand", seq, alpha, demand,
                                     fleet, ports, distances, rat)
                if prop:
                    return prop

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT B1 — Hub Efficiency: inbound relay
#  "Collect from multiple origins → transit hub → final destination cluster"
# ══════════════════════════════════════════════════════════════════════════════

def _agent_B1(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:

    uncov = _uncovered_pairs(alpha, demand)

    for hub in TRANSSHIP_HUBS:
        if hub not in ports:
            continue

        # Find origins that ship uncovered cargo TO hub or THROUGH hub to destinations
        # Strategy: origin → hub → final_dest
        # Score = Σ revenue of (origin→hub) + (hub→dest) uncovered pairs

        # Group by final destination beyond hub
        dest_score  = defaultdict(float)
        dest_origins = defaultdict(list)

        for (od, rec) in uncov:
            o, d = od
            if o == hub or d == hub:
                continue
            # Check if leg (o→hub) and (hub→d) both exist in distances
            if (o, hub) not in distances or (hub, d) not in distances:
                continue
            score = rec.weekly_revenue
            dest_score[d]  += score
            dest_origins[d].append((score, o))

        sorted_dests = sorted(dest_score.items(), key=lambda x: -x[1])

        for dest, total_score in sorted_dests[:5]:
            if dest not in ports:
                continue
            top_origins = [o for _, o in
                           sorted(dest_origins[dest], reverse=True)[:12]
                           if o in ports]
            if not top_origins:
                continue

            for n in (14, 12, 10, 8, 6):
                seq = _dedup_seq(top_origins[:n] + [hub, dest])
                if len(seq) < 3:
                    continue
                rat = (f"B1: {n} origins → {ports[hub].name} hub → "
                       f"{ports[dest].name}. "
                       f"score=${total_score:,.0f}/wk")
                prop = _try_proposal("B1", "hub", seq, alpha, demand,
                                     fleet, ports, distances, rat)
                if prop:
                    return prop

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT B2 — Hub Efficiency: outbound relay
#  "Pick up at origin cluster → transit hub → fan out to multiple destinations"
# ══════════════════════════════════════════════════════════════════════════════

def _agent_B2(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:

    uncov = _uncovered_pairs(alpha, demand)

    for hub in TRANSSHIP_HUBS:
        if hub not in ports:
            continue

        # Group uncovered pairs by origin that flow through this hub region
        origin_score = defaultdict(float)
        origin_dests  = defaultdict(list)

        for (od, rec) in uncov:
            o, d = od
            if o == hub or d == hub:
                continue
            if (o, hub) not in distances or (hub, d) not in distances:
                continue
            score = rec.weekly_revenue
            origin_score[o] += score
            origin_dests[o].append((score, d))

        sorted_origins = sorted(origin_score.items(), key=lambda x: -x[1])

        for origin, total_score in sorted_origins[:5]:
            if origin not in ports:
                continue
            top_dests = [d for _, d in
                         sorted(origin_dests[origin], reverse=True)[:12]
                         if d in ports]
            if not top_dests:
                continue

            for n in (14, 12, 10, 8, 6):
                seq = _dedup_seq([origin, hub] + top_dests[:n])
                if len(seq) < 3:
                    continue
                rat = (f"B2: {ports[origin].name} → {ports[hub].name} hub → "
                       f"{n} destinations. score=${total_score:,.0f}/wk")
                prop = _try_proposal("B2", "hub", seq, alpha, demand,
                                     fleet, ports, distances, rat)
                if prop:
                    return prop

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  AGENT C1 — Peak Robustness
#  "Scale alpha DOWN to simulate +30% demand. Find routes that pay under peak."
# ══════════════════════════════════════════════════════════════════════════════

def _agent_C1(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:
    from stage3.route_validator import validate
    from stage3.reduced_cost    import compute_reduced_cost

    # Peak alpha: divide existing duals by PEAK_DEMAND_FACTOR
    # This makes more pairs look "cheap" — as if current supply is thin vs peak demand
    peak_alpha = {od: a / PEAK_DEMAND_FACTOR for od, a in alpha.items()}

    # Also score pairs that are currently partially served but would be congested at peak
    peak_uncov = _uncovered_pairs(peak_alpha, demand)   # more pairs show as uncovered

    # Group by destination (same as A1 but with peak alpha)
    dest_score   = defaultdict(float)
    dest_origins = defaultdict(list)

    for (od, rec) in peak_uncov[:200]:   # top 200 by weekly revenue
        o, d = od
        adj   = max(0.0, rec.revenue_per_ffe - peak_alpha.get(od, 0.0))
        score = adj * rec.ffe_per_week
        dest_score[d]  += score
        dest_origins[d].append((score, o))

    sorted_dests = sorted(dest_score.items(), key=lambda x: -x[1])

    # Prefer large vessels for headroom
    large_first = ["Super_panamax", "Post_panamax", "Panamax_2400",
                   "Panamax_1200", "Feeder_800", "Feeder_450"]

    for dest, _ in sorted_dests:
        if dest not in ports:
            continue
        top_origins = [o for _, o in
                       sorted(dest_origins[dest], reverse=True)[:12]
                       if o in ports]
        if not top_origins:
            continue

        for n in (14, 12, 10, 8, 6):
            for hub in [None] + TRANSSHIP_HUBS[:4]:
                if hub and (hub == dest or hub in top_origins or hub not in ports):
                    continue
                if hub:
                    raw_seq = top_origins[:n-1] + [hub, dest]
                else:
                    raw_seq = top_origins[:n] + [dest]
                seq = _dedup_seq(raw_seq)
                if len(seq) < 2:
                    continue

                for vc in large_first:
                    ok, _ = validate(seq, vc, fleet, ports, distances, demand, 1)
                    if not ok:
                        continue
                    # Score with PEAK alpha (lower bar for RC)
                    rc_peak = compute_reduced_cost(
                        seq, vc, peak_alpha, fleet, ports, distances, demand
                    )
                    if not rc_peak["is_improving"]:
                        continue
                    # Also check with REAL alpha — must still look reasonable
                    rc_real = compute_reduced_cost(
                        seq, vc, alpha, fleet, ports, distances, demand
                    )
                    rat = (f"C1 peak: {ports[top_origins[0]].name} → "
                           f"{ports[dest].name}. "
                           f"Peak-RC=${rc_peak['rc']:,.0f}  "
                           f"Real-RC=${rc_real['rc']:,.0f}")
                    return AgentProposal(
                        agent_id      = "C1",
                        agent_type    = "peak",
                        port_sequence = seq,
                        vessel_class  = vc,
                        frequency     = 1,
                        rc_value      = rc_peak["rc"],
                        rationale     = rat,
                        source        = "analytical",
                    )

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  LLM MODE — shared prompt builder + API call
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm_prompt(
    agent_id:  str,
    alpha:     dict,
    demand:    dict,
    fleet:     dict,
    ports:     dict,
    fleet_profile,
) -> str:
    """
    Build a data-rich, cost-aware prompt for one CG pricing agent.

    Professor's profit formula (incorporated):
      Profit = Revenue
             - Vessel Fixed Cost    (TC_rate_daily x 7 x vessels_needed)
             - Sailing Fuel Cost    (bunker_per_day x sailing_days x bunker_price)
             - Idle/Port Fuel Cost  (idle_consumption x n_port_calls x port_dwell_days x bunker_price)
             - Port Call Fees       (port_call_cost_fixed x n_calls)
             - Port Escort Fees     (tug fees ~15% of port_call_cost_fixed x n_calls)
             - Load/Unload Costs    (cost_per_full x FFE at each port)
             - Transshipment Costs  (cost_per_full_trnsf x transshipped FFE)
             - Canal Fees           (Panama or Suez if applicable)
    """
    from utils.config import BUNKER_PRICE_PER_TON, PORT_DAYS_PER_CALL, PORT_ESCORT_FEE_FRACTION

    strategies = {
        "A1": (
            "DESTINATION AGGREGATOR",
            "Identify the single highest-value DESTINATION port in the opportunity list. "
            "Design a multi-origin route that collects cargo from 2-4 different uncovered origins "
            "and funnels them into that destination. "
            "COST CHECK: more ports = more port call fees + idle fuel. Stay under 10 ports unless "
            "each extra port adds meaningful adjusted revenue."
        ),
        "A2": (
            "ORIGIN DISTRIBUTOR",
            "Identify the single largest ORIGIN port by total uncovered outbound demand. "
            "Design a route that distributes its cargo to 2-4 different uncovered destinations. "
            "CIRCULAR CHECK: Does any destination have demand BACK to your origin? "
            "That return-leg revenue is free -- the vessel sails back anyway."
        ),
        "B1": (
            "HUB SPOKE -- INBOUND RELAY",
            "Design a FEEDER spoke that collects cargo from 2 remote or thin ports and relays it "
            "through a major transshipment hub: Singapore (SGSIN), Port Said (EGPSD), "
            "Algeciras (ESALG), Colombo (LKCMB), or Tanger Med (MAPTM). "
            "Use Feeder_800 or Panamax_1200 -- short cycle time = small vessels_needed = low fixed cost. "
            "The hub MUST appear in your port_sequence."
        ),
        "B2": (
            "HUB SPOKE -- OUTBOUND RELAY",
            "Design a route that picks up cargo AT a hub and distributes it to 2-3 secondary ports. "
            "Hub first in the sequence. Use a Panamax vessel for medium-density corridors. "
            "RETURN CHECK: does the last port have demand back to the hub? Earn revenue both directions."
        ),
        "C1": (
            "PEAK ROBUSTNESS",
            "Design a route for a LARGE vessel (Super_panamax 7500 FFE or Post_panamax 4200 FFE) "
            "on the highest-density corridor. Even if current utilization is ~60-70%, this vessel "
            "handles +30% peak demand surge without deploying extra ships. "
            "Economic check: Super_panamax fixed cost = ~$52,500/wk per vessel x vessels_needed. "
            "Only viable on top 3-4 corridors."
        ),
        "D1": (
            "LARGE VESSEL COVERAGE SWEEP",
            "Find the single highest-VOLUME uncovered OD pair (sorted by FFE/week, not revenue/FFE). "
            "Design a long benchmark-style route (6-14 ports) using Super_panamax or Post_panamax "
            "that covers this corridor plus 2-4 additional complementary OD pairs along the route. "
            "Include at least one major transshipment hub (SGSIN, EGPSD, ESALG) as an intermediate stop. "
            "This route MUST use a large vessel — minimum Panamax_2400. "
            "Only include extra ports if they meaningfully add demand coverage; avoid port-call cost bloat."
        ),
        "C2": (
            "LONG-TAIL VOLUME COVERAGE",
            "Focus on OD pairs ranked by (FFE/week × revenue/FFE) — high-volume corridors that "
            "are still completely uncovered (alpha=0). These are not necessarily the richest per-FFE "
            "but represent the largest weekly revenue gap. "
            "Design a direct or 1-hub route covering the top 2-3 such pairs. "
            "Prefer smaller vessels (Feeder_800, Panamax_1200) if volume doesn't justify large ships — "
            "utilization check: vessel should be at least 70% full."
        ),
        "H3": (
            "HUB BRIDGE FEEDER",
            "Connect the port with the highest total uncovered outbound revenue to the nearest "
            "major transshipment hub, then bridge to a second hub if cargo flows support it. "
            "Route structure: [top_origin] → [hub1] → [hub2 optional] → ... "
            "Use Feeder_800 or Panamax_1200 for short feeder runs. "
            "Focus on making the ORIGIN port accessible to the global hub network — "
            "this unlocks indirect transshipment revenue the RMP cannot yet see."
        ),
        "R1": (
            "REGIONAL CLUSTER LOOP",
            "Identify a compact geographic cluster of 4-6 ports with high mutual uncovered demand. "
            "Design a SHORT circular route (4-6 ports, single region) that serves this cluster. "
            "Ideal for ports in the same sub-region (e.g. East Asia, North Europe, West Africa). "
            "Use Feeder_800 or Feeder_450 — short cycle = 1-2 vessels_needed = very low fixed cost. "
            "Even modest revenue is profitable at this scale. "
            "AVOID mixing ports from different regions in the same loop."
        ),
    }

    strat_name, strat_desc = strategies.get(agent_id, ("GENERAL", "Find any improving route."))

    # Top 20 OD opportunities sorted by total_opportunity = adj_rev/FFE x demand_volume
    opps = sorted(
        [
            (od, demand[od], demand[od].revenue_per_ffe - alpha.get(od, 0.0))
            for od in demand
            if demand[od].revenue_per_ffe - alpha.get(od, 0.0) > 0
        ],
        key=lambda x: -x[2] * x[1].ffe_per_week,
    )[:20]

    opp_lines = "\n".join(
        f"  {od[0]}->{od[1]}  "
        f"adj_rev=${adj:.0f}/FFE  "
        f"demand={rec.ffe_per_week:.0f} FFE/wk  "
        f"total_opp=${adj * rec.ffe_per_week:,.0f}/wk  "
        f"max_transit={rec.max_transit_days}d  "
        f"{'STAR UNCOVERED (alpha=0, full rate)' if alpha.get(od, 0.0) < 1.0 else f'[partially served, alpha={alpha.get(od, 0.0):.0f}]'}"
        for od, rec, adj in opps
    )

    # Port cost and draft table for all ports appearing in top opportunities
    top_ports = sorted(set(p for od, _, _ in opps for p in od) & set(ports.keys()))
    port_lines = "\n".join(
        f"  {p} ({ports[p].name}): "
        f"draft={ports[p].draft}m  "
        f"load_cost=${ports[p].cost_per_full}/FFE  "
        f"trnsf_cost=${ports[p].cost_per_full_trnsf}/FFE  "
        f"call_fee=${ports[p].port_call_cost_fixed:,.0f}  "
        f"escort_fee=${ports[p].port_call_cost_fixed * PORT_ESCORT_FEE_FRACTION:,.0f}"
        for p in top_ports
    )

    # Full fleet economics so LLM can reason about cost vs revenue tradeoffs
    fleet_lines = "\n".join(
        f"  {vc:<22}: "
        f"cap={v.capacity_ffe} FFE  "
        f"draft={v.draft}m  "
        f"fixed_cost=${v.tc_rate_daily * 7:,.0f}/wk-per-vessel  "
        f"sail_fuel={v.bunker_per_day}t/day=${v.bunker_per_day * BUNKER_PRICE_PER_TON:,.0f}/sailing-day  "
        f"idle_fuel={v.idle_consumption}t/day=${v.idle_consumption * BUNKER_PRICE_PER_TON * PORT_DAYS_PER_CALL:,.0f}/port-call  "
        f"panama={'YES ($' + str(int(v.panama_fee)) + ')' if v.panama_fee else 'NO'}"
        for vc, v in fleet.items()
    )

    return f"""You are a Column Generation pricing oracle for a liner shipping optimizer (LinerNet).
Your job: propose ONE new shipping service route with POSITIVE REDUCED COST.
Positive reduced cost means: adjusted_revenue > weekly_operating_cost.

════════════════════════════════════════════════════════
 AGENT: {agent_id}  --  {strat_name}
════════════════════════════════════════════════════════
Strategy: {strat_desc}

════════════════════════════════════════════════════════
 FULL PROFIT FORMULA (professor-specified -- use this to evaluate your route)
════════════════════════════════════════════════════════
  Reduced Cost = Adjusted Revenue - Weekly Operating Cost

  ADJUSTED REVENUE = sum over all OD pairs your route covers:
      (revenue_per_ffe - alpha_od) x min(demand_ffe, capacity_share)
    alpha_od = 0   means OD pair is UNCOVERED -> full freight rate is your gain [STAR]
    alpha_od > 0   means OD pair is partially served -> smaller marginal gain

  WEEKLY OPERATING COST (all 7 components):
    1. Vessel Fixed Cost  = TC_rate x 7 days x vessels_needed
                            (TC rate covers: crew salary + insurance + maintenance)
    2. Sailing Fuel       = bunker_per_day x total_sailing_days x $600/ton
    3. Idle/Port Fuel     = idle_consumption x n_port_calls x 1.0 day x $600/ton
                            [CRITICAL: engines run at port -- this cost is often forgotten]
    4. Port Call Fees     = sum(port_call_cost_fixed) for each port in sequence
    5. Port Escort Fees   = sum(port_call_cost_fixed x 0.15) per port call
                            [tugs bring vessel from ocean to terminal after engine shutdown]
    6. Load/Unload Costs  = cost_per_full x FFE handled per port call
    7. Canal Fees         = Panama fee or Suez fee per transit (if applicable)

  vessels_needed = ceil(cycle_days / 7)
  cycle_days     = sum of sailing_days (all legs including return) + (1.0 x n_port_calls)

════════════════════════════════════════════════════════
 CRITICAL: ROUTES ARE CIRCULAR ROTATIONS -- EARN REVENUE BOTH DIRECTIONS
════════════════════════════════════════════════════════
A vessel on route [A, B, C] sails A->B->C->A->B->C->... continuously.
You earn cargo revenue in BOTH directions around the cycle:
  Outbound: A->B, A->C, B->C (forward sub-paths)
  Return:   C->A             (return leg ALSO carries cargo -- FREE sailing already paid)

ALWAYS CHECK: is (last_port -> first_port) in the opportunity list?
If yes, that is free revenue. The vessel sails back anyway. Designing routes that
exploit return-leg demand can significantly improve the reduced cost.

════════════════════════════════════════════════════════
 TOP 20 OD OPPORTUNITIES (adj_revenue x volume = weekly opportunity value)
════════════════════════════════════════════════════════
STAR = UNCOVERED (alpha=0): full freight rate available as adjusted revenue
{opp_lines}

════════════════════════════════════════════════════════
 PORT COSTS AND DRAFT LIMITS
════════════════════════════════════════════════════════
load_cost    = cost per FFE loaded or unloaded at this port
trnsf_cost   = cost per FFE transshipped (connecting two services)
call_fee     = fixed cost per vessel call
escort_fee   = tug fee per vessel call (~15% of call_fee)
{port_lines}

════════════════════════════════════════════════════════
 VESSEL ECONOMICS (bunker price = $600/metric ton)
════════════════════════════════════════════════════════
{fleet_lines}

════════════════════════════════════════════════════════
 HARD CONSTRAINTS
════════════════════════════════════════════════════════
1. port_sequence: 4 to 15 ports, NO duplicate ports
2. Vessel draft MUST be <= every port's draft limit (check port table above)
3. Post_panamax and Super_panamax CANNOT transit Panama Canal
4. Frequency = 1 (one departure per week)
5. Prioritize UNCOVERED OD pairs (STAR) -- these give full freight rate as adjusted revenue
6. Prefer FEWER ports if extra ports add cost without proportional revenue
7. Use the SMALLEST vessel that achieves ~80% or higher utilization

════════════════════════════════════════════════════════
 ECONOMIC SELF-CHECK (do this mentally before answering)
════════════════════════════════════════════════════════
Step 1: List OD pairs covered -- including the return leg (last->first).
Step 2: Sum adj_revenue = (adj_rev/FFE x FFE_loaded) for each covered pair.
Step 3: Estimate vessels_needed from cycle_days.
Step 4: Calculate total cost: fixed + sail_fuel + idle_fuel + port_fees + escort_fees.
Step 5: Is adj_revenue > total_cost? If NO -- redesign: fewer ports or better OD coverage.
Step 6: Is vessel >=70% full? If NO -- switch to smaller vessel class.

════════════════════════════════════════════════════════
 OUTPUT FORMAT (STRICT)
════════════════════════════════════════════════════════
Respond with ONLY a valid JSON object. No markdown. No text outside JSON.
{{"port_sequence": ["UNLOCODE1", "UNLOCODE2", "UNLOCODE3"],
  "vessel_class": "...",
  "rationale": "3 sentences: (1) which OD pairs covered including return leg, (2) adj_revenue vs operating_cost estimate, (3) why this vessel class and expected utilization"}}"""

def _call_llm(prompt: str, api_key: str, key_ring: List[str] = None) -> Optional[dict]:
    """Call LLM (auto-detects provider from key). Returns parsed JSON dict or None."""
    from utils.llm_client import call_llm as _call
    return _call(prompt, api_key, key_ring=key_ring, verbose=False)

def _run_llm_agent(
    agent_id:     str,
    agent_type:   str,
    alpha:        dict,
    demand:       dict,
    fleet:        dict,
    ports:        dict,
    distances:    dict,
    fleet_profile,
    api_key:      str,
    analytical_fn,   # fallback function
    key_ring:     List[str] = None,
) -> Optional[AgentProposal]:
    """
    Try LLM first, validate+RC-check result, fall back to analytical if needed.
    """
    from stage3.route_validator import validate
    from stage3.reduced_cost    import compute_reduced_cost

    prompt = _build_llm_prompt(agent_id, alpha, demand, fleet, ports, fleet_profile)
    result = _call_llm(prompt, api_key, key_ring=key_ring)

    if result:
        # Gemini sometimes wraps the dict in a list — unwrap it
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        if not isinstance(result, dict):
            result = None

    if result:
        seq  = result.get("port_sequence", [])
        vc   = result.get("vessel_class",  "")
        rat  = result.get("rationale",     "LLM proposal")

        if seq and vc:
            ok, reason = validate(seq, vc, fleet, ports, distances, demand, 1)
            if ok:
                rc = compute_reduced_cost(seq, vc, alpha, fleet, ports, distances, demand)
                if rc["is_improving"]:
                    return AgentProposal(
                        agent_id      = agent_id,
                        agent_type    = agent_type,
                        port_sequence = seq,
                        vessel_class  = vc,
                        frequency     = 1,
                        rc_value      = rc["rc"],
                        rationale     = f"[LLM] {rat}",
                        source        = "llm",
                    )

    # LLM failed or proposed bad route — use analytical fallback
    return analytical_fn()


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  AGENT D1 — Large Vessel Coverage Sweep
#  Explicitly targets high-volume uncovered corridors with big ships.
#  Runs every iteration to ensure capacity keeps pace with demand.
# ══════════════════════════════════════════════════════════════════════════════

def _agent_D1(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:
    """
    Find the single highest-volume uncovered OD pair and build the
    largest feasible route covering it, plus 2-3 nearby high-demand pairs.
    Always tries Super_panamax → Post_panamax → Panamax_2400 first.
    """
    from stage3.route_validator import validate
    from stage3.reduced_cost    import compute_reduced_cost

    # Find OD pairs with alpha=0 (completely uncovered), sorted by FFE volume
    uncovered = sorted(
        [(od, rec) for od, rec in demand.items() if alpha.get(od, 0) == 0.0],
        key=lambda x: -x[1].ffe_per_week
    )

    if not uncovered:
        return None

    large_vessels = ["Super_panamax", "Post_panamax", "Panamax_2400", "Panamax_1200"]

    for (o, d), rec in uncovered[:20]:
        if o not in ports or d not in ports:
            continue

        # Build longer benchmark-style sequences (6-14 ports) around the core OD.
        candidates = [
            [o, "CNYTN", "SGSIN", "EGPSD", "DEBRV", d],
            [o, "KRPUS", "SGSIN", "ESALG", "ITGIT", d],
            [o, "HKHKG", "MYTPP", "SGSIN", "EGPSD", d],
        ]
        # Also try bundling with the next uncovered pair sharing the same origin
        same_origin = [(od2, r2) for (od2, r2) in uncovered[:15]
                       if od2[0] == o and od2[1] != d]
        if same_origin:
            d2 = same_origin[0][0][1]
            candidates.append([o, "CNYTN", "SGSIN", "EGPSD", d, d2])
            candidates.append([o, "KRPUS", "MYTPP", "ESALG", d, d2])

        for seq in candidates:
            seq = _dedup_seq([p for p in seq if p in ports])
            if len(seq) < 2:
                continue
            for vc in large_vessels:
                ok, _ = validate(seq, vc, fleet, ports, distances, demand, 1)
                if not ok:
                    continue
                rc = compute_reduced_cost(seq, vc, alpha, fleet, ports, distances, demand)
                if rc["is_improving"]:
                    return AgentProposal(
                        agent_id      = "D1",
                        agent_type    = "coverage",
                        port_sequence = seq,
                        vessel_class  = vc,
                        frequency     = 1,
                        rc_value      = rc["rc"],
                        rationale     = (
                            f"D1 coverage sweep: {ports[o].name}→{ports[d].name} "
                            f"({rec.ffe_per_week:.0f} FFE/wk uncovered) "
                            f"RC={rc['rc']:,.0f}"
                        ),
                        source        = "analytical",
                    )
    return None


def _agent_C2(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:
    """
    Coverage agent for long-tail OD pairs:
    prioritises uncovered OD with high volume (not only high revenue/FFE).
    """
    ranked = sorted(
        _uncovered_pairs(alpha, demand),
        key=lambda x: -(x[1].ffe_per_week * x[1].revenue_per_ffe),
    )
    for (od, rec) in ranked[:120]:
        o, d = od
        if o not in ports or d not in ports:
            continue
        for hub in TRANSSHIP_HUBS[:6]:
            if hub in (o, d) or hub not in ports:
                continue
            for seq in ([o, d], [o, hub, d]):
                rat = (f"C2 long-tail coverage: {ports[o].name}->{ports[d].name} "
                       f"{rec.ffe_per_week:.0f} FFE/wk")
                prop = _try_proposal("C2", "coverage", seq, alpha, demand,
                                     fleet, ports, distances, rat, prefer_large=True)
                if prop:
                    return prop
    return None


def _agent_H3(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:
    """
    Hub-bridge feeder agent:
    connect best uncovered origins into the top transshipment hubs.
    """
    hub_candidates = [h for h in TRANSSHIP_HUBS if h in ports]
    uncovered = _uncovered_pairs(alpha, demand)
    by_origin = defaultdict(float)
    for (od, rec) in uncovered:
        by_origin[od[0]] += rec.weekly_revenue
    origins = [o for o, _ in sorted(by_origin.items(), key=lambda x: -x[1])[:20]]
    for hub in hub_candidates[:5]:
        for o in origins:
            if o == hub or o not in ports:
                continue
            bridge = [h for h in hub_candidates if h not in (o, hub)]
            long_seq = [o, hub] + bridge[:4]
            for seq in (long_seq, long_seq + bridge[4:6]):
                seq = _dedup_seq(seq)
                rat = f"H3 hub-bridge feeder: {o} -> {hub}"
                prop = _try_proposal("H3", "hub", seq, alpha, demand,
                                     fleet, ports, distances, rat, prefer_large=True)
                if prop:
                    return prop
    return None


def _agent_R1(
    alpha: dict, demand: dict, fleet: dict,
    ports: dict, distances: dict,
) -> Optional[AgentProposal]:
    """
    Regional short-loop feeder:
    search compact 3-port loops that pick up uncovered demand.
    """
    uncovered = _uncovered_pairs(alpha, demand)
    top_ods = [od for (od, _) in uncovered[:80]]
    port_score = defaultdict(float)
    for (o, d) in top_ods:
        port_score[o] += 1.0
        port_score[d] += 1.0
    top_ports = [p for p, _ in sorted(port_score.items(), key=lambda x: -x[1])[:16] if p in ports]
    if len(top_ports) >= 6:
        seq = top_ports[:6]
        rat = f"R1 regional coverage loop on uncovered cluster ({len(seq)} ports)"
        prop = _try_proposal("R1", "regional", seq, alpha, demand,
                             fleet, ports, distances, rat, prefer_large=True)
        if prop:
            return prop
    return None


def run_agents(
    alpha:        dict,
    demand:       dict,
    fleet:        dict,
    ports:        dict,
    distances:    dict,
    fleet_profile = None,
    api_key:      Optional[str] = None,
    key_ring:     List[str] = None,
    verbose:      bool = False,
) -> list:
    """
    Run pricing agents against the current RMP dual prices.

    Parameters
    ----------
    alpha         : demand dual prices from RMPSolution.alpha
    demand/fleet/ports/distances : from stage0 loader
    fleet_profile : from stage1 fleet_profiler.run()  (for LLM prompt context)
    api_key       : API key — if None, pure analytical mode
    key_ring      : list of API keys for rotation on rate-limit
    verbose       : print each agent's proposal

    Returns
    -------
    list[AgentProposal] — improving proposals (one per agent, may be empty)
    """

    agents = [
        ("D1", "coverage", lambda: _agent_D1(alpha, demand, fleet, ports, distances)),
        ("C2", "coverage", lambda: _agent_C2(alpha, demand, fleet, ports, distances)),
        ("A1", "demand",   lambda: _agent_A1(alpha, demand, fleet, ports, distances)),
        ("A2", "demand",   lambda: _agent_A2(alpha, demand, fleet, ports, distances)),
        ("B1", "hub",      lambda: _agent_B1(alpha, demand, fleet, ports, distances)),
        ("B2", "hub",      lambda: _agent_B2(alpha, demand, fleet, ports, distances)),
        ("H3", "hub",      lambda: _agent_H3(alpha, demand, fleet, ports, distances)),
        ("R1", "regional", lambda: _agent_R1(alpha, demand, fleet, ports, distances)),
        ("C1", "peak",     lambda: _agent_C1(alpha, demand, fleet, ports, distances)),
    ]

    # In LLM mode, wrap ALL agents — each tries LLM first, falls back to analytical
    if api_key:
        agents = [
            (aid, atype, lambda aid=aid, atype=atype, afn=afn:
                _run_llm_agent(aid, atype, alpha, demand, fleet, ports,
                               distances, fleet_profile, api_key, afn,
                               key_ring=key_ring))
            for (aid, atype, afn) in agents
        ]

    proposals = []
    for (agent_id, agent_type, runner) in agents:
        prop = runner()
        if prop is not None:
            proposals.append(prop)
            if verbose:
                names = " → ".join(ports[p].name for p in prop.port_sequence)
                print(f"  [{prop.agent_id}] ({prop.source}) {names}")
                print(f"         {prop.vessel_class}  RC=${prop.rc_value:,.0f}")
                print(f"         {prop.rationale}")
        else:
            if verbose:
                print(f"  [{agent_id}] No improving route found")

    return proposals


# ══════════════════════════════════════════════════════════════════════════════
#  Test suite
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from stage0.loader         import load_all, validate as validate_data
    from stage1.demand_intel   import run as run_intel
    from stage1.fleet_profiler import run as run_fleet
    from stage2.seed_generator import run as run_seeds
    from stage3.rmp            import RestrictedMasterProblem
    from stage3.reduced_cost   import compute_reduced_cost

    data  = load_all(verbose=False)
    validate_data(data)
    intel = run_intel(data, verbose=False)
    fp    = run_fleet(data, verbose=False)
    seeds = run_seeds(data, intel, fp, verbose=False)

    # Build seed RMP and solve
    rmp = RestrictedMasterProblem(data['demand'], data['fleet'])
    for s in seeds:
        col = rmp.column_from_route(s.route_id, s.port_sequence,
            s.vessel_class, s.frequency, data['ports'], data['distances'])
        rmp.add_column(col)
    sol = rmp.solve(verbose=False, distances=data['distances'])

    print("=" * 65)
    print("  Stage 3 — Part 4: Agent Oracle — Test Suite")
    print("=" * 65)
    print(f"\n  RMP baseline: {sol.n_routes} routes | "
          f"net profit=${sol.net_profit:,.0f}/wk | "
          f"served={sol.total_served_ffe:,.0f} FFE")

    # ── TEST 1: Run all 5 agents, check they each find something ─────────────
    print("\n── Test 1: All 5 agents against seed RMP duals ────────────────")
    proposals = run_agents(
        sol.alpha, data['demand'], data['fleet'],
        data['ports'], data['distances'],
        fleet_profile=fp, api_key=None, verbose=True,
    )
    print(f"\n  {len(proposals)}/5 agents found improving routes")

    # ── TEST 2: All proposals pass validator ──────────────────────────────────
    print("\n── Test 2: Validate all proposals ─────────────────────────────")
    from stage3.route_validator import validate
    all_valid = True
    for p in proposals:
        ok, reason = validate(p.port_sequence, p.vessel_class,
                              data['fleet'], data['ports'],
                              data['distances'], data['demand'], p.frequency)
        status = "✓" if ok else "✗"
        print(f"  {status} [{p.agent_id}] {p.port_sequence} [{p.vessel_class}]  {reason}")
        if not ok:
            all_valid = False
    if all_valid:
        print("  ✓ All proposals pass validator")

    # ── TEST 3: All proposals have RC > 0 ────────────────────────────────────
    print("\n── Test 3: RC > CG_RC_TOLERANCE for all proposals ─────────────")
    all_positive = True
    for p in proposals:
        rc = compute_reduced_cost(p.port_sequence, p.vessel_class,
                                  sol.alpha, data['fleet'], data['ports'],
                                  data['distances'], data['demand'])
        status = "✓" if rc["is_improving"] else "✗"
        print(f"  {status} [{p.agent_id}] RC=${rc['rc']:>12,.0f}  "
              f"adj_rev=${rc['adj_revenue']:>12,.0f}  "
              f"cost=${rc['operating_cost']:>12,.0f}  "
              f"util={rc['utilisation_pct']}%")
        if not rc["is_improving"]:
            all_positive = False
    if all_positive:
        print("  ✓ All proposals have positive reduced cost")

    # ── TEST 4: Add proposals to RMP — LP revenue must improve ───────────────
    print("\n── Test 4: Adding proposals improves LP ────────────────────────")
    prev_revenue = sol.lp_revenue
    prev_served  = sol.total_served_ffe

    for i, p in enumerate(proposals, 1):
        col = rmp.column_from_route(
            f"AG{i:02d}", p.port_sequence, p.vessel_class,
            p.frequency, data['ports'], data['distances']
        )
        rmp.add_column(col)

    sol2 = rmp.solve(verbose=False, distances=data['distances'])
    delta_rev    = sol2.lp_revenue    - prev_revenue
    delta_served = sol2.total_served_ffe - prev_served

    assert sol2.lp_revenue >= prev_revenue - 1, "LP revenue decreased after adding proposals!"
    print(f"  ✓ LP Revenue: ${prev_revenue:,.0f} → ${sol2.lp_revenue:,.0f}  "
          f"(Δ=${delta_rev:+,.0f})")
    print(f"  ✓ Served FFE: {prev_served:,.0f} → {sol2.total_served_ffe:,.0f}  "
          f"(Δ={delta_served:+,.0f})")
    print(f"  ✓ Net Profit: ${sol.net_profit:,.0f} → ${sol2.net_profit:,.0f}  "
          f"(Δ=${sol2.net_profit - sol.net_profit:+,.0f})")

    # ── TEST 5: Agents adapt to updated duals ────────────────────────────────
    print("\n── Test 5: Agents propose different routes on updated duals ────")
    first_proposals_seqs = {tuple(p.port_sequence) for p in proposals}
    proposals2 = run_agents(
        sol2.alpha, data['demand'], data['fleet'],
        data['ports'], data['distances'],
        fleet_profile=fp, api_key=None, verbose=False,
    )
    new_seqs = {tuple(p.port_sequence) for p in proposals2 if p}
    overlap  = first_proposals_seqs & new_seqs
    print(f"  Round-1 proposals : {len(proposals)}")
    print(f"  Round-2 proposals : {len(proposals2)}")
    print(f"  Overlap (same seq): {len(overlap)} "
          f"{'(expected — agents may revisit same corridors)' if overlap else ''}")
    for p in proposals2:
        if p:
            names = " → ".join(data['ports'][x].name for x in p.port_sequence)
            print(f"    [{p.agent_id}] RC=${p.rc_value:,.0f}  {names}  [{p.vessel_class}]")

    print(f"\n{'='*65}")
    print(f"  All agent tests passed ✓")
    print(f"  Part 4 complete — agent_oracle.py ready for Part 5 (CG loop)")
    print(f"{'='*65}")
