"""
Stage 2 -- Seed Service Generator
===================================
Generates 5-10 initial shipping services (seed routes) to start
the Column Generation loop in Stage 3.

Seeds don't need to be optimal -- they just need to:
  1. Cover the highest-revenue OD corridors
  2. Be physically feasible (draft, distance, vessel match)
  3. Give the LP a meaningful starting point

TWO MODES:
  - Analytical (always works): pure math, groups OD pairs into
    corridors, picks best vessel per corridor
  - LLM (when API key provided): Claude designs seeds using
    trade knowledge on top of the analytics

Both modes return the same SeedRoute objects.
Stage 3 does not care which mode was used.
"""

import sys
import os
import json
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import NUM_SEED_ROUTES, PORT_DAYS_PER_CALL, BUNKER_PRICE_PER_TON


# ── Output data class ─────────────────────────────────────────────────────────

@dataclass
class SeedRoute:
    """
    One seed shipping service.
    This becomes one column in the Stage 3 RMP.
    """
    route_id:        str          # e.g. "S01", "S02"
    port_sequence:   list         # ordered list of UNLOCODEs, one-way
    vessel_class:    str          # e.g. "Panamax_2400"
    frequency:       int          # departures per week (1 or 2)

    # Precomputed economics (from fleet_profiler)
    cycle_days:      float = 0.0
    vessels_needed:  int   = 0
    weekly_cost:     float = 0.0

    # Which top OD pairs does this route cover?
    covers_od:       list  = field(default_factory=list)

    # How was this seed generated?
    source:          str   = "analytical"  # or "llm"
    rationale:       str   = ""


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICAL SEED GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _best_vessel_for_route(
    port_sequence: list,
    ports: dict,
    distances: dict,
    fleet: dict,
) -> Optional[str]:
    """
    Returns the vessel class with highest capacity that can
    physically operate this route (draft + distance + canals).
    Returns None if no vessel class is feasible.
    """
    # Import here to avoid circular imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage1.fleet_profiler import route_economics

    # Try vessel classes from largest to smallest
    ordered = [
        "Super_panamax", "Post_panamax",
        "Panamax_2400", "Panamax_1200",
        "Feeder_800", "Feeder_450"
    ]

    for vc in ordered:
        vessel = fleet[vc]
        econ   = route_economics(vessel, port_sequence, ports, distances)
        if econ.feasible:
            return vc

    return None


def _route_covers_od(port_sequence: list, od_priorities: list) -> list:
    """
    Find which top-50 OD pairs are covered by this route.
    An OD pair (o, d) is covered if BOTH o and d appear in port_sequence
    in the correct order (o before d).
    """
    covered = []
    for od in od_priorities[:50]:
        o, d = od.origin, od.destination
        if o in port_sequence and d in port_sequence:
            if port_sequence.index(o) < port_sequence.index(d):
                covered.append(f"{o}->{d}")
    return covered


def _fill_economics(seed: SeedRoute, fleet: dict, ports: dict, distances: dict):
    """Compute and fill cycle_days, vessels_needed, weekly_cost on a seed."""
    from stage1.fleet_profiler import route_economics
    vessel = fleet[seed.vessel_class]
    econ   = route_economics(
        vessel, seed.port_sequence, ports, distances, seed.frequency
    )
    if econ.feasible:
        seed.cycle_days     = econ.cycle_days
        seed.vessels_needed = econ.vessels_needed
        seed.weekly_cost    = econ.weekly_total_cost


def generate_analytical_seeds(
    data:         dict,
    intel,                       # DemandIntelligence from Stage 1A
    fleet_profile,               # FleetProfile from Stage 1B
    n_seeds:      int = NUM_SEED_ROUTES,
    verbose:      bool = True,
) -> list:
    """
    Build seed routes by grouping top OD pairs into trade corridors
    and designing one route per corridor.

    Strategy:
      1. Identify the top revenue OD pairs
      2. Group by broad corridor (Asia-Europe, Asia-US, Intra-Asia etc.)
      3. For each corridor, build a multi-port route hitting the main
         origin and destination regions, via natural hub ports
      4. Pick largest feasible vessel
      5. Set frequency = 1 (Stage 3 will add more routes if needed)
    """
    ports     = data['ports']
    distances = data['distances']
    fleet     = data['fleet']
    od_list   = intel.od_priorities   # sorted by weekly_revenue desc

    if verbose:
        print("Stage 2: Generating analytical seed routes...")

    # ── Step 1: Define corridor templates ─────────────────────────────────────
    # Each template is a list of port sequences representing
    # major trade lanes. Based on WorldSmall demand analysis.

    # Hub ports identified from Stage 1A analytics + trade knowledge
    # Singapore cluster: SGSIN, MYTPP (Tanjung Pelepas), MYPKG (Port Klang)
    # West Med gateways: ESALG (Algeciras), EGPSD (Port Said), MAPTM (Tangier)
    # North Europe: DEBRV (Bremerhaven), NLRTM (Rotterdam), DEHAM (Hamburg)
    # East Asia: CNSHA (Shanghai), CNYTN (Shenzhen), HKHKG (HK), KRPUS (Busan)

    corridor_templates = [
        {
            "name":     "Asia-Europe-Main",
            "sequence": ["CNSHA", "CNYTN", "SGSIN", "EGPSD", "DEBRV", "NLRTM"],
            "rationale": "Covers #1 CNSHA->DEBRV, #3 CNYTN->NLRTM, #4 CNSHA->NLRTM, #9 CNYTN->DEBRV"
        },
        {
            "name":     "Korea-Japan-HK-Europe",
            "sequence": ["KRPUS", "JPYOK", "HKHKG", "EGPSD", "DEBRV", "NLRTM"],
            "rationale": "Covers #2 KRPUS->DEBRV, #6 JPYOK->NLRTM, #11 HKHKG->DEBRV, #13 HKHKG->NLRTM"
        },
        {
            "name":     "Transpacific",
            "sequence": ["CNSHA", "CNYTN", "KRPUS", "USLAX"],
            "rationale": "Covers #5 CNYTN->USLAX, #17 CNSHA->USLAX"
        },
        {
            "name":     "China-HK-Malaysia-Europe",
            "sequence": ["CNTAO", "HKHKG", "MYTPP", "EGPSD", "DEBRV", "NLRTM"],
            "rationale": "Covers #7 MYTPP->DEBRV, #10 CNTAO->DEBRV, #11 HKHKG->DEBRV, #13 HKHKG->NLRTM, #15 MYTPP->NLRTM"
        },
        {
            "name":     "Asia-Med-UK",
            "sequence": ["CNSHA", "SGSIN", "ITGIT", "GBFXT"],
            "rationale": "Covers #16 CNSHA->ITGIT, #20 CNSHA->GBFXT"
        },
        {
            "name":     "Asia-WestAfrica",
            "sequence": ["CNSHA", "SGSIN", "ESALG", "NGAPP"],
            "rationale": "Covers #8 CNSHA->NGAPP"
        },
        {
            "name":     "Asia-SouthAmerica",
            "sequence": ["CNSHA", "SGSIN", "ESALG", "BRSSZ"],
            "rationale": "Covers #18 CNSHA->BRSSZ"
        },
        {
            "name":     "SouthAmerica-Europe",
            "sequence": ["BRSSZ", "ESALG", "DEBRV"],
            "rationale": "Covers #19 BRSSZ->DEBRV"
        },
    ]

    # ── Step 2: Validate each template and assign vessel ──────────────────────
    seeds   = []
    used_id = 0

    for tmpl in corridor_templates:
        if len(seeds) >= n_seeds:
            break

        seq = tmpl["sequence"]

        # Check all ports exist in our instance
        missing = [p for p in seq if p not in ports]
        if missing:
            if verbose:
                print(f"  SKIP {tmpl['name']}: ports not in database: {missing}")
            continue

        # Find best (largest feasible) vessel
        vc = _best_vessel_for_route(seq, ports, distances, fleet)
        if vc is None:
            if verbose:
                print(f"  SKIP {tmpl['name']}: no feasible vessel for this route")
            continue

        used_id += 1
        route_id = f"S{used_id:02d}"

        seed = SeedRoute(
            route_id      = route_id,
            port_sequence = seq,
            vessel_class  = vc,
            frequency     = 1,
            source        = "analytical",
            rationale     = tmpl["rationale"],
        )

        # Fill OD coverage and economics
        seed.covers_od = _route_covers_od(seq, od_list)
        _fill_economics(seed, fleet, ports, distances)
        seeds.append(seed)

        if verbose:
            port_names = [ports[p].name for p in seq]
            print(f"  {route_id}: {' → '.join(port_names)}")
            print(f"       Vessel: {vc}  |  Cycle: {seed.cycle_days:.1f}d  |  "
                  f"Vessels: {seed.vessels_needed}  |  "
                  f"Cost/wk: ${seed.weekly_cost:,.0f}")
            print(f"       Covers {len(seed.covers_od)} top-50 OD pairs")

    return seeds


# ══════════════════════════════════════════════════════════════════════════════
#  LLM SEED GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_seed_prompt(intel, fleet_profile, ports: dict) -> str:
    """Build the prompt asking Claude to design seed routes."""

    # Top 20 OD pairs
    top_ods = "\n".join(
        f"  {od.priority_rank:>2}. {od.origin}->{od.destination}  "
        f"{od.ffe_per_week:.0f} FFE/wk  ${od.revenue_per_ffe}/FFE  "
        f"=${od.weekly_revenue:,.0f}/wk  max_transit={od.max_transit_days}d"
        for od in intel.od_priorities[:20]
    )

    # Top hubs
    top_hubs = "\n".join(
        f"  {hs.name} ({hs.unlocode})  region={hs.d_region}  "
        f"hub_score={hs.hub_score:.3f}"
        for hs in intel.hub_rankings[:8]
    )

    # Fleet summary
    fleet_summary = "\n".join(
        f"  {vc:<22} {v.capacity_ffe} FFE  draft={v.draft}m  "
        f"speed={v.design_speed}kn  tc=${v.tc_rate_daily}/day"
        for vc, v in fleet_profile.vessel_records.items()
    )

    # Port access constraints
    access_issues = []
    for vc, access in fleet_profile.port_access.items():
        blocked = [ports[p].name for p, ok in access.items() if not ok]
        if blocked:
            access_issues.append(f"  {vc}: CANNOT call at: {', '.join(blocked)}")

    access_str = "\n".join(access_issues)

    return f"""You are designing seed shipping services for a liner shipping network optimizer.
These seeds start the Column Generation loop -- they don't need to be optimal,
just sensible enough to cover the main trade corridors.

DATASET: LINERLIB WorldSmall -- 47 ports, 1764 OD pairs

TOP 20 OD PAIRS BY WEEKLY REVENUE:
{top_ods}

TOP 8 HUB PORTS:
{top_hubs}

AVAILABLE VESSEL CLASSES:
{fleet_summary}

DRAFT CONSTRAINTS (vessels BLOCKED from these ports):
{access_str}

RULES:
1. Each service is a one-way port sequence (vessel returns directly to start)
2. Each service MUST have a port sequence length between 4 and 15 ports (including origin and final destination)
3. Vessel draft must be <= draft limit of every port it calls
4. Use Post_panamax and Super_panamax only on deep-water routes (no Panama)
5. Each service should target a distinct trade corridor
6. Frequency is 1 (one departure per week) for all seeds
7. Aim for maximum utilization of ships. Design well-connecting, high-demand routes and strongly prefer assigning larger ships (Super_panamax, Post_panamax) with near-full capacity utilization.

Design exactly 15 seed services.

CRITICAL: Respond ONLY with a single valid JSON object. Start with {{ and end with }}. No text before or after. No markdown fences.

{{
  "seeds": [
    {{"route_id":"S01","port_sequence":["UNLOCODE","UNLOCODE","UNLOCODE"],"vessel_class":"Super_panamax","frequency":1,"rationale":"brief"}},
    {{"route_id":"S02","port_sequence":["UNLOCODE","UNLOCODE"],"vessel_class":"Panamax_2400","frequency":1,"rationale":"brief"}}
  ]
}}
Stage 2 -- Seed Service Generator
===================================
Generates 5-10 initial shipping services (seed routes) to start
the Column Generation loop in Stage 3.

Seeds don't need to be optimal -- they just need to:
  1. Cover the highest-revenue OD corridors
  2. Be physically feasible (draft, distance, vessel match)
  3. Give the LP a meaningful starting point

TWO MODES:
  - Analytical (always works): pure math, groups OD pairs into
    corridors, picks best vessel per corridor
  - LLM (when API key provided): Claude designs seeds using
    trade knowledge on top of the analytics

Both modes return the same SeedRoute objects.
Stage 3 does not care which mode was used.
"""

import sys
import os
import json
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import NUM_SEED_ROUTES, PORT_DAYS_PER_CALL, BUNKER_PRICE_PER_TON


# ── Output data class ─────────────────────────────────────────────────────────

@dataclass
class SeedRoute:
    """
    One seed shipping service.
    This becomes one column in the Stage 3 RMP.
    """
    route_id:        str          # e.g. "S01", "S02"
    port_sequence:   list         # ordered list of UNLOCODEs, one-way
    vessel_class:    str          # e.g. "Panamax_2400"
    frequency:       int          # departures per week (1 or 2)

    # Precomputed economics (from fleet_profiler)
    cycle_days:      float = 0.0
    vessels_needed:  int   = 0
    weekly_cost:     float = 0.0

    # Which top OD pairs does this route cover?
    covers_od:       list  = field(default_factory=list)

    # How was this seed generated?
    source:          str   = "analytical"  # or "llm"
    rationale:       str   = ""


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYTICAL SEED GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _best_vessel_for_route(
    port_sequence: list,
    ports: dict,
    distances: dict,
    fleet: dict,
) -> Optional[str]:
    """
    Returns the vessel class with highest capacity that can
    physically operate this route (draft + distance + canals).
    Returns None if no vessel class is feasible.
    """
    # Import here to avoid circular imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage1.fleet_profiler import route_economics

    # Try vessel classes from largest to smallest
    ordered = [
        "Super_panamax", "Post_panamax",
        "Panamax_2400", "Panamax_1200",
        "Feeder_800", "Feeder_450"
    ]

    for vc in ordered:
        vessel = fleet[vc]
        econ   = route_economics(vessel, port_sequence, ports, distances)
        if econ.feasible:
            return vc

    return None


def _route_covers_od(port_sequence: list, od_priorities: list) -> list:
    """
    Find which top-50 OD pairs are covered by this route.
    An OD pair (o, d) is covered if BOTH o and d appear in port_sequence
    in the correct order (o before d).
    """
    covered = []
    for od in od_priorities[:50]:
        o, d = od.origin, od.destination
        if o in port_sequence and d in port_sequence:
            if port_sequence.index(o) < port_sequence.index(d):
                covered.append(f"{o}->{d}")
    return covered


def _fill_economics(seed: SeedRoute, fleet: dict, ports: dict, distances: dict):
    """Compute and fill cycle_days, vessels_needed, weekly_cost on a seed."""
    from stage1.fleet_profiler import route_economics
    vessel = fleet[seed.vessel_class]
    econ   = route_economics(
        vessel, seed.port_sequence, ports, distances, seed.frequency
    )
    if econ.feasible:
        seed.cycle_days     = econ.cycle_days
        seed.vessels_needed = econ.vessels_needed
        seed.weekly_cost    = econ.weekly_total_cost


def generate_analytical_seeds(
    data:         dict,
    intel,                       # DemandIntelligence from Stage 1A
    fleet_profile,               # FleetProfile from Stage 1B
    n_seeds:      int = NUM_SEED_ROUTES,
    verbose:      bool = True,
) -> list:
    """
    Build seed routes by grouping top OD pairs into trade corridors
    and designing one route per corridor.

    Strategy:
      1. Identify the top revenue OD pairs
      2. Group by broad corridor (Asia-Europe, Asia-US, Intra-Asia etc.)
      3. For each corridor, build a multi-port route hitting the main
         origin and destination regions, via natural hub ports
      4. Pick largest feasible vessel
      5. Set frequency = 1 (Stage 3 will add more routes if needed)
    """
    ports     = data['ports']
    distances = data['distances']
    fleet     = data['fleet']
    od_list   = intel.od_priorities   # sorted by weekly_revenue desc

    if verbose:
        print("Stage 2: Generating analytical seed routes...")

    # ── Step 1: Define corridor templates ─────────────────────────────────────
    # Each template is a list of port sequences representing
    # major trade lanes. Based on WorldSmall demand analysis.

    # Hub ports identified from Stage 1A analytics + trade knowledge
    # Singapore cluster: SGSIN, MYTPP (Tanjung Pelepas), MYPKG (Port Klang)
    # West Med gateways: ESALG (Algeciras), EGPSD (Port Said), MAPTM (Tangier)
    # North Europe: DEBRV (Bremerhaven), NLRTM (Rotterdam), DEHAM (Hamburg)
    # East Asia: CNSHA (Shanghai), CNYTN (Shenzhen), HKHKG (HK), KRPUS (Busan)

    corridor_templates = [
        {
            "name":     "Asia-Europe-Main",
            "sequence": ["CNSHA", "CNYTN", "SGSIN", "EGPSD", "DEBRV", "NLRTM"],
            "rationale": "Covers #1 CNSHA->DEBRV, #3 CNYTN->NLRTM, #4 CNSHA->NLRTM, #9 CNYTN->DEBRV"
        },
        {
            "name":     "Korea-Japan-HK-Europe",
            "sequence": ["KRPUS", "JPYOK", "HKHKG", "EGPSD", "DEBRV", "NLRTM"],
            "rationale": "Covers #2 KRPUS->DEBRV, #6 JPYOK->NLRTM, #11 HKHKG->DEBRV, #13 HKHKG->NLRTM"
        },
        {
            "name":     "Transpacific",
            "sequence": ["CNSHA", "CNYTN", "KRPUS", "USLAX"],
            "rationale": "Covers #5 CNYTN->USLAX, #17 CNSHA->USLAX"
        },
        {
            "name":     "China-HK-Malaysia-Europe",
            "sequence": ["CNTAO", "HKHKG", "MYTPP", "EGPSD", "DEBRV", "NLRTM"],
            "rationale": "Covers #7 MYTPP->DEBRV, #10 CNTAO->DEBRV, #11 HKHKG->DEBRV, #13 HKHKG->NLRTM, #15 MYTPP->NLRTM"
        },
        {
            "name":     "Asia-Med-UK",
            "sequence": ["CNSHA", "SGSIN", "ITGIT", "GBFXT"],
            "rationale": "Covers #16 CNSHA->ITGIT, #20 CNSHA->GBFXT"
        },
        {
            "name":     "Asia-WestAfrica",
            "sequence": ["CNSHA", "SGSIN", "ESALG", "NGAPP"],
            "rationale": "Covers #8 CNSHA->NGAPP"
        },
        {
            "name":     "Asia-SouthAmerica",
            "sequence": ["CNSHA", "SGSIN", "ESALG", "BRSSZ"],
            "rationale": "Covers #18 CNSHA->BRSSZ"
        },
        {
            "name":     "SouthAmerica-Europe",
            "sequence": ["BRSSZ", "ESALG", "DEBRV"],
            "rationale": "Covers #19 BRSSZ->DEBRV"
        },
    ]

    # ── Step 2: Validate each template and assign vessel ──────────────────────
    seeds   = []
    used_id = 0

    for tmpl in corridor_templates:
        if len(seeds) >= n_seeds:
            break

        seq = tmpl["sequence"]

        # Check all ports exist in our instance
        missing = [p for p in seq if p not in ports]
        if missing:
            if verbose:
                print(f"  SKIP {tmpl['name']}: ports not in database: {missing}")
            continue

        # Find best (largest feasible) vessel
        vc = _best_vessel_for_route(seq, ports, distances, fleet)
        if vc is None:
            if verbose:
                print(f"  SKIP {tmpl['name']}: no feasible vessel for this route")
            continue

        used_id += 1
        route_id = f"S{used_id:02d}"

        seed = SeedRoute(
            route_id      = route_id,
            port_sequence = seq,
            vessel_class  = vc,
            frequency     = 1,
            source        = "analytical",
            rationale     = tmpl["rationale"],
        )

        # Fill OD coverage and economics
        seed.covers_od = _route_covers_od(seq, od_list)
        _fill_economics(seed, fleet, ports, distances)
        seeds.append(seed)

        if verbose:
            port_names = [ports[p].name for p in seq]
            print(f"  {route_id}: {' → '.join(port_names)}")
            print(f"       Vessel: {vc}  |  Cycle: {seed.cycle_days:.1f}d  |  "
                  f"Vessels: {seed.vessels_needed}  |  "
                  f"Cost/wk: ${seed.weekly_cost:,.0f}")
            print(f"       Covers {len(seed.covers_od)} top-50 OD pairs")

    return seeds


# ══════════════════════════════════════════════════════════════════════════════
#  LLM SEED GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def _build_seed_prompt(intel, fleet_profile, ports: dict) -> str:
    """
    Build the LLM prompt asking it to design seed shipping services.

    Professor's profit formula is embedded so the LLM can reason about
    cost vs revenue tradeoffs when proposing seed routes:
      Profit = Revenue
             - Vessel Fixed Cost    (TC_rate_daily x 7 x vessels_needed)
             - Sailing Fuel         (bunker_per_day x sailing_days x $600/ton)
             - Idle/Port Fuel       (idle_consumption x n_port_calls x 1.0 day x $600/ton)
             - Port Call Fees       (port_call_cost_fixed x n_calls)
             - Port Escort Fees     (~15% of port_call_cost_fixed x n_calls)
             - Load/Unload Costs    (cost_per_full x FFE handled per port)
             - Transshipment Costs  (cost_per_full_trnsf x transshipped FFE, +-20% if unavailable)
             - Canal Fees           (Panama or Suez if applicable)
    """
    from utils.config import BUNKER_PRICE_PER_TON, PORT_DAYS_PER_CALL, PORT_ESCORT_FEE_FRACTION

    # Top 25 OD pairs by weekly revenue
    top_ods = "\n".join(
        f"  {od.priority_rank:>2}. {od.origin}->{od.destination}  "
        f"{od.ffe_per_week:.0f} FFE/wk  "
        f"${od.revenue_per_ffe}/FFE  "
        f"=${od.weekly_revenue:,.0f}/wk revenue  "
        f"max_transit={od.max_transit_days}d"
        for od in intel.od_priorities[:25]
    )

    # Top hub ports
    top_hubs = "\n".join(
        f"  {hs.unlocode} ({hs.name})  region={hs.d_region}  hub_score={hs.hub_score:.3f}"
        for hs in intel.hub_rankings[:8]
    )

    # Full fleet economics -- TC rate, bunker, idle fuel all shown so LLM can evaluate profitability
    fleet_summary = "\n".join(
        f"  {vc:<22}: "
        f"cap={v.capacity_ffe} FFE  "
        f"draft={v.draft}m  "
        f"speed={v.design_speed}kn  "
        f"fixed_cost=${v.tc_rate_daily * 7:,.0f}/wk-per-vessel  "
        f"sail_fuel={v.bunker_per_day}t/day=${v.bunker_per_day * BUNKER_PRICE_PER_TON:,.0f}/sailing-day  "
        f"idle_fuel={v.idle_consumption}t/day=${v.idle_consumption * BUNKER_PRICE_PER_TON * PORT_DAYS_PER_CALL:,.0f}/port-call  "
        f"panama={'YES' if v.can_use_panama else 'NO'}"
        for vc, v in fleet_profile.vessel_records.items()
    )

    # Port access constraints
    access_issues = []
    for vc, access in fleet_profile.port_access.items():
        blocked = [ports[p].name for p, ok in access.items() if not ok]
        if blocked:
            access_issues.append(f"  {vc}: BLOCKED from: {', '.join(blocked)}")
    access_str = "\n".join(access_issues) if access_issues else "  (no blocked ports)"

    # Sample port costs for the top OD ports to give the LLM cost context
    top_od_ports = sorted(set(
        p for od in intel.od_priorities[:15]
        for p in [od.origin, od.destination]
        if p in ports
    ))
    port_cost_lines = "\n".join(
        f"  {p} ({ports[p].name}): "
        f"draft={ports[p].draft}m  "
        f"load_cost=${ports[p].cost_per_full}/FFE  "
        f"call_fee=${ports[p].port_call_cost_fixed:,.0f}  "
        f"escort_fee=${ports[p].port_call_cost_fixed * PORT_ESCORT_FEE_FRACTION:,.0f}"
        for p in top_od_ports
    )

    return f"""You are an expert liner shipping network designer. Design 8 SEED shipping services
for a Column Generation optimizer. Seeds start the CG loop -- they will be refined,
but must be economically sensible from the start.

════════════════════════════════════════════════════════
 WHAT MAKES A GOOD SEED ROUTE
════════════════════════════════════════════════════════
A seed must be PROFITABLE or close to profitable at ~70-80% vessel utilization.

  Weekly Profit per Route =
      Revenue               (freight_rate x FFE_carried per OD pair)
    - Vessel Fixed Cost     (TC_rate_daily x 7 x vessels_needed)
    - Sailing Fuel Cost     (bunker_per_day x sailing_days x $600/ton)
    - Idle/Port Fuel Cost   (idle_consumption x n_port_calls x 1.0 day x $600/ton)
                            [IMPORTANT: engines run at port; each port call burns idle fuel]
    - Port Call Fees        (port_call_cost_fixed x n_port_calls)
    - Port Escort Fees      (~15% of port_call_cost_fixed x n_port_calls)
                            [tugs bring vessel from ocean to terminal after engine shutdown]
    - Load/Unload Costs     (cost_per_full x FFE at each port call)
    - Canal Fees            (Panama or Suez transit fees if route uses canals)

  vessels_needed = ceil(cycle_days / 7)
  cycle_days     = outbound sailing days + return sailing days + (1.0 day x n_port_calls)

  COST RULE OF THUMB:
    Adding one extra port adds: idle_fuel + escort_fee + call_fee + load_cost
    Only worth it if that port adds more revenue than this combined cost.

════════════════════════════════════════════════════════
 CRITICAL: LINER SERVICES ARE CIRCULAR ROTATIONS
════════════════════════════════════════════════════════
A vessel on route [A, B, C] does NOT stop after C. It sails A->B->C->A->B->C->... continuously.
This means the vessel earns cargo revenue in BOTH directions:
  Forward legs : A->B, A->C, B->C  (any forward sub-path)
  Return leg   : C->A              (the direct return also carries cargo -- revenue is FREE)

Design principle: always check if (last_port -> first_port) has demand.
If yes, that return cargo earns revenue at zero extra sailing cost.
Routes that balance both directions are MUCH more profitable than one-way designs.

════════════════════════════════════════════════════════
 TOP 25 OD PAIRS BY WEEKLY REVENUE
════════════════════════════════════════════════════════
{top_ods}

════════════════════════════════════════════════════════
 TOP 8 HUB PORTS FOR TRANSSHIPMENT
════════════════════════════════════════════════════════
{top_hubs}

════════════════════════════════════════════════════════
 VESSEL CLASSES AND ECONOMICS (bunker = $600/metric ton)
════════════════════════════════════════════════════════
{fleet_summary}

  HOW TO CHOOSE VESSEL CLASS:
    Large vessels (Super_panamax, Post_panamax): profitable only if multiple high-volume
      OD pairs are covered AND vessel is near-full (>75%). Fixed cost is very high.
    Medium vessels (Panamax_2400, Panamax_1200): best for medium-density corridors.
      Panamax vessels can use Panama Canal -- opens transPacific routes.
    Small feeders (Feeder_800, Feeder_450): ideal for regional/hub spokes.
      Short cycle = few vessels needed = low fixed cost. Very efficient for thin routes.

════════════════════════════════════════════════════════
 PORT COSTS (top OD ports)
════════════════════════════════════════════════════════
{port_cost_lines}

════════════════════════════════════════════════════════
 DRAFT CONSTRAINTS
════════════════════════════════════════════════════════
{access_str}

════════════════════════════════════════════════════════
 DESIGN RULES (hard constraints)
════════════════════════════════════════════════════════
1. port_sequence: 4 to 15 ports (route is circular; vessel returns from last to first continuously)
2. Vessel draft MUST be <= draft limit of EVERY port it calls
3. Post_panamax and Super_panamax CANNOT transit Panama Canal
4. Frequency = 1 (one departure per week) for all seeds
5. Each seed MUST target a DISTINCT trade corridor -- no two seeds on the same route
6. CHECK RETURN LEG: does (last_port -> first_port) have demand? Design to exploit it.
7. Before finalizing: does estimated revenue at 75% load exceed estimated operating cost?

════════════════════════════════════════════════════════
 REQUIRED SEED MIX (design at least one of each type)
════════════════════════════════════════════════════════
- At least 2 MAINLANE routes: Super_panamax or Post_panamax on top Asia-Europe or transPacific corridors
- At least 2 REGIONAL routes: Panamax_2400 or Panamax_1200 on medium-density corridors
- At least 2 FEEDER routes: Feeder_800 connecting thin ports to major hubs
- At least 1 BIDIRECTIONAL route: explicitly designed to carry cargo both forward AND return
- At least 1 HUB-SPOKE route: feeder collecting from remote ports into a major transshipment hub

════════════════════════════════════════════════════════
 ECONOMIC EXAMPLE (for calibration)
════════════════════════════════════════════════════════
Feeder_800 on a 5-port regional route:
  Vessels needed: ceil(~18 days / 7) = 3 vessels
  Fixed cost: $8,000/day x 7 x 3 = $168,000/wk
  Sailing fuel: 23.7t/day x ~14 sailing days x $600 = $199,080/wk
  Idle fuel: 2.5t/day x 5 port calls x 1 day x $600 = $7,500/wk
  Port fees: ~$5,000/call x 5 = $25,000/wk
  Escort fees: ~$750/call x 5 = $3,750/wk
  TOTAL COST: ~$403,330/wk
  Break-even: need ~$403,330 revenue from 800 FFE loaded
  => need avg ~$504/FFE revenue at full load -- very achievable on short regional routes.

════════════════════════════════════════════════════════
 OUTPUT FORMAT (STRICT)
════════════════════════════════════════════════════════
Design exactly 15 seed services.
Respond ONLY with a single valid JSON object. No markdown, no text before or after.

{{
  "seeds": [
    {{
      "route_id": "S01",
      "port_sequence": ["UNLOCODE1", "UNLOCODE2", "UNLOCODE3", "UNLOCODE4"],
      "vessel_class": "Super_panamax",
      "frequency": 1,
      "rationale": "2-3 sentences: which OD pairs covered (include return leg), why this vessel class, estimated economics at 75% load"
    }},
    {{
      "route_id": "S02",
      "port_sequence": ["UNLOCODE1", "UNLOCODE2"],
      "vessel_class": "Feeder_800",
      "frequency": 1,
      "rationale": "..."
    }}
  ]
}}"""

def _call_llm(prompt: str, api_key: str, key_ring: list = None) -> dict:
    """Call LLM (auto-detects provider from key). Returns parsed JSON dict or None."""
    from utils.llm_client import call_llm as _call
    return _call(prompt, api_key, key_ring=key_ring, verbose=True)

def generate_llm_seeds(
    data:          dict,
    intel,
    fleet_profile,
    api_key:       str,
    verbose:       bool = True,
    key_ring:      list = None,
) -> list:
    """
    Ask Claude to design seed routes using trade knowledge.
    Falls back to analytical seeds if API call fails.
    """
    ports     = data['ports']
    distances = data['distances']
    fleet     = data['fleet']

    if verbose:
        print("Stage 2: Calling Claude API for seed route design...")

    try:
        prompt     = _build_seed_prompt(intel, fleet_profile, ports)
        llm_result = _call_llm(prompt, api_key, key_ring=key_ring)
        if llm_result is None:
            if verbose:
                print("  Stage 2: LLM unavailable — using analytical seeds.")
            return generate_analytical_seeds(data, intel, fleet_profile, n_seeds=NUM_SEED_ROUTES, verbose=verbose)
        raw_seeds  = llm_result.get("seeds", [])
    except Exception as e:
        print(f"  WARNING: LLM call failed ({e}). Falling back to analytical seeds.")
        return generate_analytical_seeds(data, intel, fleet_profile, n_seeds=NUM_SEED_ROUTES, verbose=verbose)

    # Validate and build SeedRoute objects from LLM output
    seeds   = []
    skipped = 0

    for raw in raw_seeds:
        seq = raw.get("port_sequence", [])
        vc  = raw.get("vessel_class", "")

        # Validate vessel class exists
        if vc not in fleet:
            skipped += 1
            if verbose:
                print(f"  SKIP LLM seed: unknown vessel class '{vc}'")
            continue

        # Validate all ports exist and vessel can call there
        bad_ports = [p for p in seq if p not in ports]
        if bad_ports:
            skipped += 1
            if verbose:
                print(f"  SKIP LLM seed: unknown ports {bad_ports}")
            continue

        # Check draft feasibility using route_economics
        from stage1.fleet_profiler import route_economics
        vessel = fleet[vc]
        econ   = route_economics(vessel, seq, ports, distances)
        if not econ.feasible:
            skipped += 1
            if verbose:
                print(f"  SKIP LLM seed: {econ.infeasible_reason}")
            continue

        seed = SeedRoute(
            route_id      = raw.get("route_id", f"S{len(seeds)+1:02d}"),
            port_sequence = seq,
            vessel_class  = vc,
            frequency     = raw.get("frequency", 1),
            source        = "llm",
            rationale     = raw.get("rationale", ""),
        )
        seed.covers_od = _route_covers_od(seq, intel.od_priorities)
        _fill_economics(seed, fleet, ports, distances)
        seeds.append(seed)

    if verbose:
        print(f"  LLM generated {len(seeds)} valid seeds ({skipped} skipped as infeasible)")

    # If LLM gave too few seeds, top up with analytical ones
    if len(seeds) < NUM_SEED_ROUTES:
        if verbose:
            print(f"  Topping up with analytical seeds (have {len(seeds)}, need {NUM_SEED_ROUTES})")
        analytical = generate_analytical_seeds(
            data, intel, fleet_profile,
            n_seeds=NUM_SEED_ROUTES - len(seeds),
            verbose=False
        )
        # Avoid duplicates — assign incrementing IDs correctly
        existing_seqs = {tuple(s.port_sequence) for s in seeds}
        next_id = len(seeds) + 1
        for s in analytical:
            if tuple(s.port_sequence) not in existing_seqs:
                s.route_id = f"S{next_id:02d}"
                seeds.append(s)
                existing_seqs.add(tuple(s.port_sequence))
                next_id += 1

    return seeds


# ══════════════════════════════════════════════════════════════════════════════
#  Main runner
# ══════════════════════════════════════════════════════════════════════════════

def run(
    data:          dict,
    intel,
    fleet_profile,
    api_key:       Optional[str] = None,
    key_ring:      list = None,
    verbose:       bool = True,
) -> list:
    """
    Run Stage 2. Returns list of SeedRoute objects.
    Uses LLM if api_key provided, otherwise analytical.
    """
    if api_key:
        seeds = generate_llm_seeds(data, intel, fleet_profile, api_key, verbose, key_ring=key_ring)
    else:
        seeds = generate_analytical_seeds(data, intel, fleet_profile,
                                          n_seeds=NUM_SEED_ROUTES,
                                          verbose=verbose)

    seeds = _bias_seeds_for_long_big(seeds, target_n=NUM_SEED_ROUTES)

    if verbose:
        _print_summary(seeds, data['ports'])

    return seeds


def _bias_seeds_for_long_big(seeds: list, target_n: int = NUM_SEED_ROUTES) -> list:
    """
    Prefer benchmark-style seeds:
      - longer strings (6-14 distinct ports)
      - larger vessel classes
    Keeps all seeds but reorders and truncates to target_n when needed.
    """
    if not seeds:
        return seeds

    vessel_rank = {
        "Super_panamax": 6,
        "Post_panamax": 5,
        "Panamax_2400": 4,
        "Panamax_1200": 3,
        "Feeder_800": 2,
        "Feeder_450": 1,
    }

    def _score(s: SeedRoute):
        n_ports = len(set(s.port_sequence or []))
        in_band = 1 if 6 <= n_ports <= 14 else 0
        big = vessel_rank.get(s.vessel_class, 0)
        od_cov = len(s.covers_od or [])
        return (in_band, big, n_ports, od_cov, -float(s.weekly_cost or 0.0))

    ranked = sorted(seeds, key=_score, reverse=True)
    return ranked[:max(1, int(target_n))]


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(seeds: list, ports: dict):
    print(f"\n{'='*65}")
    print(f"  Stage 2 -- Seed Routes Summary  ({len(seeds)} seeds generated)")
    print(f"{'='*65}")
    for s in seeds:
        names = " → ".join(ports[p].name for p in s.port_sequence)
        print(f"\n  {s.route_id}  [{s.vessel_class}]  freq={s.frequency}/wk")
        print(f"  Route  : {names}")
        print(f"  Cycle  : {s.cycle_days:.1f} days  |  "
              f"Vessels: {s.vessels_needed}  |  "
              f"Cost/wk: ${s.weekly_cost:,.0f}")
        print(f"  Covers : {len(s.covers_od)} top-50 OD pairs")
        if s.covers_od:
            print(f"           {', '.join(s.covers_od[:5])}"
                  + (" ..." if len(s.covers_od) > 5 else ""))
        print(f"  Source : {s.source} -- {s.rationale}")
    print(f"\n{'='*65}")

    total_cost = sum(s.weekly_cost for s in seeds)
    total_od   = len(set(od for s in seeds for od in s.covers_od))
    print(f"  Total weekly fleet cost (all seeds): ${total_cost:,.0f}")
    print(f"  Unique top-50 OD pairs covered: {total_od}/50")
    print(f"{'='*65}\n")


# ── Serialise seeds to dict (for Stage 3 to consume) ─────────────────────────

def seeds_to_dict(seeds: list) -> list:
    """Convert list of SeedRoute to plain dicts for JSON serialisation."""
    return [
        {
            "route_id":      s.route_id,
            "port_sequence": s.port_sequence,
            "vessel_class":  s.vessel_class,
            "frequency":     s.frequency,
            "cycle_days":    s.cycle_days,
            "vessels_needed":s.vessels_needed,
            "weekly_cost":   s.weekly_cost,
            "covers_od":     s.covers_od,
            "source":        s.source,
            "rationale":     s.rationale,
        }
        for s in seeds
    ]


# ── Run as script ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage0.loader      import load_all, validate
    from stage1.demand_intel  import run as run_intel
    from stage1.fleet_profiler import run as run_fleet

    api_key = sys.argv[1] if len(sys.argv) > 1 else None

    data         = load_all(verbose=False)
    validate(data)
    intel        = run_intel(data, api_key=None, verbose=False)
    fleet_profile = run_fleet(data, verbose=False)

    seeds = run(data, intel, fleet_profile, api_key=api_key, verbose=True)

    # Save for Stage 3
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "seeds.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(seeds_to_dict(seeds), f, indent=2)
    print(f"Seeds saved to: {out_path}")
