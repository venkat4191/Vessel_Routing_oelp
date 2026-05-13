"""
Stage 1A -- Demand Intelligence
================================
Runs two things:

  PART 1 (pure math, no LLM):
    - Builds a weighted port graph from demand data
    - Computes betweenness centrality for every port
    - Ranks all 1764 OD pairs by weekly revenue
    - Identifies top corridors and trade regions
    - Produces a complete DemandIntelligence object

  PART 2 (LLM enrichment):
    - Sends the analytical results to Claude API
    - Claude adds trade domain knowledge on top
      (e.g. "SIN is primary hub due to geographic crossroads, not just centrality")
    - Returns enriched hub scores and corridor commentary
    - API key passed as argument -- works on local machine

Output: DemandIntelligence object used by Stage 2 (seed generator)
"""

import sys
import os
import json
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import PEAK_DEMAND_FACTOR


# ── Output data classes ───────────────────────────────────────────────────────

@dataclass
class HubScore:
    unlocode:          str
    name:              str
    d_region:          str
    betweenness:       float        # raw graph centrality score
    total_demand_in:   float        # total FFE/week arriving here
    total_demand_out:  float        # total FFE/week departing here
    total_revenue_in:  float        # $ revenue from inbound demand
    total_revenue_out: float        # $ revenue from outbound demand
    hub_score:         float        # combined score (0-1, higher = better hub)
    llm_classification: str = ""   # e.g. "Primary Hub", "Secondary Hub", etc.
    llm_reasoning:     str = ""    # LLM explanation


@dataclass
class ODPriority:
    origin:          str
    destination:     str
    ffe_per_week:    float
    revenue_per_ffe: float
    weekly_revenue:  float
    max_transit_days: int
    priority_rank:   int           # 1 = highest revenue


@dataclass
class DemandIntelligence:
    """Complete output of Stage 1A. Passed to Stage 2."""

    # Hub rankings (all 47 ports, sorted by hub_score descending)
    hub_rankings:       list        # list of HubScore

    # OD pair priorities (all 1764, sorted by weekly_revenue descending)
    od_priorities:      list        # list of ODPriority

    # Quick lookups
    top_hubs:           list        # top 5 hub UNLOCODEs
    top_od_pairs:       list        # top 20 OD pairs by revenue

    # Corridor map: trade region -> [port UNLOCODEs]
    region_map:         dict

    # Summary stats
    total_weekly_demand_ffe:    float
    total_max_revenue:          float
    avg_revenue_per_ffe:        float

    # LLM enrichment (None if API not called)
    llm_enriched:       bool = False
    llm_summary:        str  = ""


# ══════════════════════════════════════════════════════════════════════════════
#  PART 1 — Pure analytical computation
# ══════════════════════════════════════════════════════════════════════════════

def build_port_graph(demand: dict, ports: dict) -> nx.DiGraph:
    """
    Build a directed weighted graph where:
      - Nodes = port UNLOCODEs
      - Edges = OD pairs, weighted by weekly_revenue
    """
    G = nx.DiGraph()

    # Add all instance ports as nodes with attributes
    all_ports = set()
    for (o, d) in demand:
        all_ports.add(o)
        all_ports.add(d)

    for p in all_ports:
        port = ports[p]
        G.add_node(p, name=port.name, region=port.d_region)

    # Add edges weighted by weekly revenue
    for (o, d), rec in demand.items():
        G.add_edge(o, d,
                   weight=rec.weekly_revenue,
                   ffe=rec.ffe_per_week,
                   revenue_per_ffe=rec.revenue_per_ffe)

    return G


def compute_hub_scores(
    G: nx.DiGraph,
    demand: dict,
    ports: dict,
) -> list:
    """
    Compute betweenness centrality and demand-based scores for every port.
    Returns list of HubScore sorted by hub_score descending.
    """
    # Betweenness centrality on undirected version (port connectivity)
    G_undirected = G.to_undirected()
    betweenness = nx.betweenness_centrality(G_undirected, weight='weight', normalized=True)

    # Demand aggregates per port
    demand_in  = defaultdict(float)
    demand_out = defaultdict(float)
    rev_in     = defaultdict(float)
    rev_out    = defaultdict(float)

    for (o, d), rec in demand.items():
        demand_out[o] += rec.ffe_per_week
        demand_in[d]  += rec.ffe_per_week
        rev_out[o]    += rec.weekly_revenue
        rev_in[d]     += rec.weekly_revenue

    # Total demand and revenue for normalization
    max_demand = max(demand_in[p] + demand_out[p]
                     for p in G.nodes()) or 1.0
    max_rev    = max(rev_in[p] + rev_out[p]
                     for p in G.nodes()) or 1.0
    max_betw   = max(betweenness.values()) or 1.0

    hub_scores = []
    for p in G.nodes():
        port = ports[p]

        # Normalise each component to 0-1
        b_norm    = betweenness[p] / max_betw
        d_norm    = (demand_in[p] + demand_out[p]) / max_demand
        r_norm    = (rev_in[p] + rev_out[p]) / max_rev

        # Combined score: betweenness 50%, demand volume 25%, revenue 25%
        combined  = 0.50 * b_norm + 0.25 * d_norm + 0.25 * r_norm

        hub_scores.append(HubScore(
            unlocode          = p,
            name              = port.name,
            d_region          = port.d_region,
            betweenness       = round(betweenness[p], 4),
            total_demand_in   = round(demand_in[p],  1),
            total_demand_out  = round(demand_out[p], 1),
            total_revenue_in  = round(rev_in[p],  0),
            total_revenue_out = round(rev_out[p], 0),
            hub_score         = round(combined, 4),
        ))

    hub_scores.sort(key=lambda x: x.hub_score, reverse=True)

    # Label top hubs with simple classification (LLM will refine this)
    for i, hs in enumerate(hub_scores):
        if i == 0:
            hs.llm_classification = "Primary Hub"
        elif i < 3:
            hs.llm_classification = "Secondary Hub"
        elif i < 8:
            hs.llm_classification = "Regional Hub"
        else:
            hs.llm_classification = "Origin/Destination Port"

    return hub_scores


def rank_od_pairs(demand: dict) -> list:
    """
    Rank all OD pairs by weekly revenue (descending).
    Returns list of ODPriority.
    """
    pairs = []
    for (o, d), rec in demand.items():
        pairs.append(ODPriority(
            origin           = o,
            destination      = d,
            ffe_per_week     = rec.ffe_per_week,
            revenue_per_ffe  = rec.revenue_per_ffe,
            weekly_revenue   = rec.weekly_revenue,
            max_transit_days = rec.max_transit_days,
            priority_rank    = 0,   # set below
        ))

    pairs.sort(key=lambda x: x.weekly_revenue, reverse=True)
    for i, p in enumerate(pairs):
        p.priority_rank = i + 1

    return pairs


def build_region_map(ports: dict, instance_ports: set) -> dict:
    """Group instance ports by trade region."""
    region_map = defaultdict(list)
    for p in instance_ports:
        region_map[ports[p].d_region].append(p)
    return dict(region_map)


# ══════════════════════════════════════════════════════════════════════════════
#  PART 2 — LLM enrichment via Anthropic API
# ══════════════════════════════════════════════════════════════════════════════

def _build_llm_prompt(hub_scores: list, od_priorities: list, ports: dict) -> str:
    """Build the prompt we send to Claude for trade domain enrichment."""

    top_hubs = hub_scores[:10]
    top_ods  = od_priorities[:20]

    hub_lines = "\n".join(
        f"  {i+1}. {hs.name} ({hs.unlocode}) region={hs.d_region} "
        f"betweenness={hs.betweenness:.3f} hub_score={hs.hub_score:.3f} "
        f"demand_in={hs.total_demand_in:.0f} FFE/wk"
        for i, hs in enumerate(top_hubs)
    )

    od_lines = "\n".join(
        f"  {od.priority_rank}. {od.origin}->{od.destination} "
        f"{od.ffe_per_week:.0f} FFE/wk @ ${od.revenue_per_ffe}/FFE "
        f"= ${od.weekly_revenue:,.0f}/wk  transit_max={od.max_transit_days}d"
        for od in top_ods
    )

    return f"""You are an expert in global liner shipping network design.

I have run betweenness centrality and demand analysis on the LINERLIB WorldSmall 
instance (47 ports, 1764 OD pairs). Here are the mathematical results.

TOP 10 PORTS BY HUB SCORE (betweenness + demand volume + revenue):
{hub_lines}

TOP 20 OD PAIRS BY WEEKLY REVENUE:
{od_lines}

Your task — respond with ONLY a valid JSON object.
IMPORTANT: Your ENTIRE response must be ONLY a valid JSON object.
Do NOT use markdown code fences. Do NOT write any text before or after the JSON.
Start with {{ and end with }}.

{{
  "hub_classifications": {{
    "<UNLOCODE>": {{
      "classification": "Primary Hub | Secondary Hub | Regional Hub | Origin/Destination Port",
      "reasoning": "one sentence",
      "hub_score_adjusted": 0.0
    }}
  }},
  "summary": "3-4 sentence overall assessment"
}}

Only include the top 15 most important ports in hub_classifications.
Keep reasoning to ONE sentence per port to stay concise."""


def call_llm(prompt: str, api_key: str, key_ring: list = None) -> dict:
    """Call LLM (auto-detects provider from key). Returns parsed JSON dict or None."""
    from utils.llm_client import call_llm as _call
    return _call(prompt, api_key, key_ring=key_ring, verbose=True)

def enrich_with_llm(
    intel: DemandIntelligence,
    api_key: str,
    ports: dict,
    verbose: bool = True,
    key_ring: list = None,
) -> DemandIntelligence:
    """
    Call Claude API and update hub classifications with trade domain knowledge.
    Modifies intel in-place and returns it.
    """
    if verbose:
        print("\n  Stage 1A: Calling Claude API for trade domain enrichment...")

    prompt = _build_llm_prompt(intel.hub_rankings, intel.od_priorities, ports)

    llm_result = call_llm(prompt, api_key, key_ring=key_ring)

    # If LLM call failed, return intel unchanged (analytical mode)
    if llm_result is None:
        if verbose:
            print("  Stage 1A: LLM unavailable — using analytical results only.")
        return intel

    # Update hub classifications with LLM knowledge
    hub_by_code = {hs.unlocode: hs for hs in intel.hub_rankings}
    classifications = llm_result.get("hub_classifications", {})

    for code, info in classifications.items():
        if code in hub_by_code:
            hub_by_code[code].llm_classification = info.get("classification", "")
            hub_by_code[code].llm_reasoning       = info.get("reasoning", "")
            # Optionally update hub score with LLM adjustment
            adj = info.get("hub_score_adjusted")
            if adj is not None:
                hub_by_code[code].hub_score = round(float(adj), 4)

    # Re-sort after LLM adjustments
    intel.hub_rankings.sort(key=lambda x: x.hub_score, reverse=True)
    intel.top_hubs = [hs.unlocode for hs in intel.hub_rankings[:5]]

    intel.llm_enriched = True
    intel.llm_summary  = llm_result.get("summary", "")

    # Store corridor and seed suggestions in summary
    corridors   = llm_result.get("key_corridors", [])
    seed_sugg   = llm_result.get("seed_route_suggestions", [])
    intel._llm_corridors    = corridors
    intel._llm_seed_suggestions = seed_sugg

    if verbose:
        print(f"  LLM enrichment complete.")
        print(f"  Summary: {intel.llm_summary}")
        print(f"  Corridors identified: {len(corridors)}")
        print(f"  Seed route suggestions: {len(seed_sugg)}")

    return intel


# ══════════════════════════════════════════════════════════════════════════════
#  Main runner
# ══════════════════════════════════════════════════════════════════════════════

def run(
    data: dict,
    api_key: Optional[str] = None,
    key_ring: list = None,
    verbose: bool = True,
) -> DemandIntelligence:
    """
    Run Stage 1A.
    If api_key is provided, runs LLM enrichment.
    Otherwise returns analytical results only.
    """
    demand         = data['demand']
    ports          = data['ports']
    instance_ports = data['instance_ports']

    if verbose:
        print("Stage 1A: Running Demand Intelligence...")

    # ── Part 1: Pure analytics ─────────────────────────────────────────────
    G           = build_port_graph(demand, ports)
    hub_scores  = compute_hub_scores(G, demand, ports)
    od_pairs    = rank_od_pairs(demand)
    region_map  = build_region_map(ports, instance_ports)

    total_demand  = sum(r.ffe_per_week   for r in demand.values())
    total_revenue = sum(r.weekly_revenue for r in demand.values())
    avg_rate      = total_revenue / total_demand if total_demand else 0

    intel = DemandIntelligence(
        hub_rankings             = hub_scores,
        od_priorities            = od_pairs,
        top_hubs                 = [hs.unlocode for hs in hub_scores[:5]],
        top_od_pairs             = od_pairs[:20],
        region_map               = region_map,
        total_weekly_demand_ffe  = total_demand,
        total_max_revenue        = total_revenue,
        avg_revenue_per_ffe      = round(avg_rate, 2),
    )

    if verbose:
        _print_summary(intel, ports)

    # ── Part 2: LLM enrichment (optional) ─────────────────────────────────
    if api_key:
        intel = enrich_with_llm(intel, api_key, ports, verbose, key_ring=key_ring)

    return intel


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(intel: DemandIntelligence, ports: dict):
    print(f"\n{'='*60}")
    print(f"  Stage 1A -- Demand Intelligence")
    print(f"{'='*60}")
    print(f"  Total demand   : {intel.total_weekly_demand_ffe:>10,.0f} FFE/week")
    print(f"  Max revenue    : ${intel.total_max_revenue:>12,.0f}/week")
    print(f"  Avg rate       : ${intel.avg_revenue_per_ffe:>8,.2f}/FFE")
    print(f"\n  Top 10 Hubs by score (betweenness + demand + revenue):")
    print(f"  {'Rank':<5} {'Port':<25} {'Region':<25} {'Score':>6} {'Classification'}")
    print(f"  {'-'*85}")
    for i, hs in enumerate(intel.hub_rankings[:10]):
        print(f"  {i+1:<5} {hs.name:<25} {hs.d_region:<25} {hs.hub_score:>6.3f}  {hs.llm_classification}")

    print(f"\n  Top 15 OD Pairs by Weekly Revenue:")
    print(f"  {'Rank':<5} {'Origin':>7} {'Dest':>7} {'FFE/wk':>8} {'$/FFE':>7} {'Rev/wk':>12}")
    print(f"  {'-'*55}")
    for od in intel.od_priorities[:15]:
        print(f"  {od.priority_rank:<5} {od.origin:>7} {od.destination:>7} "
              f"{od.ffe_per_week:>8.0f} {od.revenue_per_ffe:>7.0f} "
              f"${od.weekly_revenue:>11,.0f}")

    print(f"\n  Trade regions in instance:")
    for region, port_list in sorted(intel.region_map.items()):
        names = [ports[p].name for p in port_list]
        print(f"    {region:<30} {', '.join(names)}")
    print(f"{'='*60}\n")


# ── Run as script ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage0.loader import load_all, validate

    # Check for API key as command-line argument
    api_key = sys.argv[1] if len(sys.argv) > 1 else None
    if not api_key:
        print("Note: No API key provided. Running analytics only.")
        print("To run with LLM enrichment: python3 demand_intel.py <your-api-key>")

    data = load_all(verbose=False)
    validate(data)
    intel = run(data, api_key=api_key, verbose=True)

    # Save output as JSON for other stages to use
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs", "demand_intel.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Serialise to JSON
    output = {
        "top_hubs": intel.top_hubs,
        "top_od_pairs": [
            {
                "origin": od.origin, "destination": od.destination,
                "ffe_per_week": od.ffe_per_week,
                "revenue_per_ffe": od.revenue_per_ffe,
                "weekly_revenue": od.weekly_revenue,
                "priority_rank": od.priority_rank,
            }
            for od in intel.od_priorities[:50]
        ],
        "hub_rankings": [
            {
                "unlocode": hs.unlocode, "name": hs.name,
                "d_region": hs.d_region, "hub_score": hs.hub_score,
                "classification": hs.llm_classification,
                "reasoning": hs.llm_reasoning,
            }
            for hs in intel.hub_rankings
        ],
        "llm_enriched": intel.llm_enriched,
        "llm_summary":  intel.llm_summary,
        "total_weekly_demand_ffe": intel.total_weekly_demand_ffe,
        "total_max_revenue": intel.total_max_revenue,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Output saved to: {out_path}")
