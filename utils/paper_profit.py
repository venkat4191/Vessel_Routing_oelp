"""
Paper-aligned LSNDP profit (Dutta et al. 2024, Appendix A–B; LINERLIB / Brouer et al.)
======================================================================================
η  =  R_total  −  C_reject  −  C_handle  −  C_NDP

Use `evaluate_paper_profit(data, service_records)` for benchmark-comparable scoring.
Column-generation stages still use the internal LP surrogate; this module is the
canonical η for comparison to Table 2 in the paper.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from stage1.fleet_profiler import route_economics
from utils.config import (
    PAPER_FLEET_AVAILABLE,
    PAPER_REJECTION_PENALTY_PER_FFE,
    MAX_TRANSSHIP_HOPS,
)

EPS = 1e-6
INF_CAP = 1e18


@dataclass
class PaperProfitBreakdown:
    """
    Full profit decomposition per professor's formula:

      Profit = Revenue – (Vessel Fixed Cost + Vessel Dynamic Cost
                        + Container Load/Unload Costs + Transshipment Costs
                        + Port Call Costs + Port Escort Fees)

    Mapping to fields:
      Revenue              -> r_total
      Vessel Fixed Cost    -> c_service   (TC charter: crew + insurance + maintenance)
      Vessel Dynamic Cost  -> c_voyage_port_canal (sailing fuel) + c_idle (port-dwell fuel)
      Load/Unload Costs    -> c_handle    (cost_per_full x FFE, via MCF cargo assignment)
      Transshipment Costs  -> c_handle    (cost_per_full_trnsf x transshipped FFE, in MCF)
      Port Call Costs      -> c_voyage_port_canal (port_call_cost_fixed x n_calls)
      Port Escort Fees     -> c_escort    (tug fees, ~15% of port call fixed cost)
      Rejection Penalty    -> c_reject    (academic penalty for unserved demand)
      Unused fleet cost    -> c_unused    (opportunity cost of idle vessels)
    """
    eta: float
    r_total: float
    c_reject: float
    c_handle: float
    c_ndp: float
    c_service: float
    c_unused: float
    c_voyage_port_canal: float
    c_idle: float          # NEW: idle/port-dwell fuel cost (professor's formula)
    c_escort: float        # NEW: port escort / tug fees (professor's formula)
    served_ffe: float
    total_demand_ffe: float
    unserved_ffe: float
    details: Dict[str, Any] = field(default_factory=dict)


def _fractional_vessels(cycle_days: float, frequency: int) -> float:
    return max(cycle_days * float(frequency) / 7.0, 0.0)


def _compute_c_ndp_and_enrich(
    service_records: List[dict],
    data: dict,
) -> Tuple[float, float, float, float, float, float, List[dict]]:
    """
    Compute C_NDP (network design cost) per the professor's formula:

      C_NDP = Vessel Fixed Cost + Vessel Dynamic Cost + Port Call Costs + Port Escort Fees + Canal Fees

    Where:
      Vessel Fixed Cost  = TC_rate_daily x 7 x vessels_needed  (crew, insurance, maintenance)
      Vessel Dynamic Cost = Sailing Fuel + Idle/Port Fuel
        Sailing Fuel = bunker_per_day x sailing_days x bunker_price
        Idle Fuel    = idle_consumption x n_port_calls x port_dwell_days x bunker_price
      Port Call Costs = port_call_cost_fixed x n_port_calls  (per vessel call)
      Port Escort Fees = port_call_cost_fixed x PORT_ESCORT_FEE_FRACTION x n_port_calls
                         (tugs bring vessel from ocean to terminal after engine shutdown)
      Canal Fees = Panama + Suez transit fees if applicable

    Note: Load/Unload Costs and Transshipment Costs are computed separately in the MCF
    (they depend on actual cargo flows) and appear in C_handle.
    """
    from utils.config import PORT_ESCORT_FEE_FRACTION
    fleet = data["fleet"]
    ports = data["ports"]
    distances = data["distances"]

    c_service = 0.0       # vessel fixed cost (TC charter)
    c_voyage_pc = 0.0     # sailing fuel + port call fees + canal fees
    c_idle = 0.0          # idle/port-dwell fuel (NEW -- professor's formula)
    c_escort = 0.0        # port escort / tug fees (NEW -- professor's formula)
    enriched: List[dict] = []
    used_nv: DefaultDict[str, float] = defaultdict(float)

    for rec in service_records:
        vc = rec["vessel_class"]
        seq = rec["port_sequence"]
        freq = int(rec.get("frequency", 1))
        vessel = fleet[vc]
        econ = route_economics(vessel, seq, ports, distances, freq)
        if not econ.feasible:
            raise ValueError(
                f"Paper C_NDP: infeasible route {rec.get('route_id')}: {econ.infeasible_reason}"
            )

        nvf = _fractional_vessels(econ.cycle_days, freq)
        int_nv = max(econ.vessels_needed, 1)
        scale = nvf / float(int_nv)

        # Vessel Fixed Cost: TC charter hire (covers crew, insurance, maintenance)
        c_service += econ.weekly_tc_cost * scale

        # Vessel Dynamic Cost (sailing fuel) + Port Call Fees + Canal Fees
        c_voyage_pc += (
            econ.weekly_bunker * scale
            + econ.weekly_port_cost * scale
            + econ.weekly_canal_fee * scale
        )

        # Idle / Port-Dwell Fuel (professor's formula: idling fuel consumption at port)
        # RouteEconomics now carries weekly_idle_bunker after fleet_profiler fix
        c_idle += econ.weekly_idle_bunker * scale

        # Port Escort / Tug Fees (professor's formula)
        # RouteEconomics now carries weekly_escort_cost after fleet_profiler fix
        c_escort += econ.weekly_escort_cost * scale

        used_nv[vc] += max(nvf, 0.0)
        enriched.append({
            **rec,
            "nv_frac": nvf,
            "cap_ffe": float(vessel.capacity_ffe),
            "cycle_days": econ.cycle_days,
        })

    c_unused = 0.0
    if PAPER_FLEET_AVAILABLE:
        for vc, vlim in PAPER_FLEET_AVAILABLE.items():
            vessel = fleet.get(vc)
            if vessel is None:
                continue
            spare = float(vlim) - used_nv.get(vc, 0.0)
            c_unused += -spare * vessel.tc_rate_daily * 7.0

    c_ndp = c_service + c_unused + c_voyage_pc + c_idle + c_escort
    return c_ndp, c_service, c_unused, c_voyage_pc, c_idle, c_escort, enriched


def _build_static_master_edges(
    enriched_services: List[dict],
    data: dict,
) -> Tuple[
    List[Tuple[int, int, float, float, int]],
    Dict[Tuple[int, str], int],
    int,
]:
    """Returns master_edge list (u,v,cost,cap), key_to_node, n_nodes."""
    ports_db = data["ports"]
    key_to_node: Dict[Tuple[int, str], int] = {}
    n = 0
    for sid, svc in enumerate(enriched_services):
        for p in svc["port_sequence"]:
            if (sid, p) not in key_to_node:
                key_to_node[(sid, p)] = n
                n += 1

    # Each edge: (u, v, cost, cap, hop_inc)
    # hop_inc == 1 means this edge represents a transshipment transfer.
    master: List[Tuple[int, int, float, float, int]] = []

    for sid, svc in enumerate(enriched_services):
        seq = svc["port_sequence"]
        nvf = max(svc["nv_frac"], 0.0)
        cap_leg = nvf * svc["cap_ffe"]
        if cap_leg <= EPS:
            continue
        m = len(seq)
        for i in range(m):
            p = seq[i]
            q = seq[(i + 1) % m]
            u = key_to_node[(sid, p)]
            v = key_to_node[(sid, q)]
            master.append((u, v, 0.0, cap_leg, 0))

    port_to_sids: DefaultDict[str, List[int]] = defaultdict(list)
    for sid, svc in enumerate(enriched_services):
        for p in set(svc["port_sequence"]):
            port_to_sids[p].append(sid)

    for p, sids in port_to_sids.items():
        if p not in ports_db:
            continue
        pt = ports_db[p].cost_per_full_trnsf

        # FIX 10: Transshipment Storage / Dwell Cost
        # Containers waiting at a hub for the next connecting service incur
        # storage fees ($5-15/FFE/day at major hubs like Singapore, Rotterdam).
        # Average wait = half of weekly service interval = 3.5 days.
        # Storage add-on = TRANSSHIP_STORAGE_COST_PER_FFE_DAY x TRANSSHIP_AVG_WAIT_DAYS
        try:
            from utils.config import TRANSSHIP_STORAGE_COST_PER_FFE_DAY, TRANSSHIP_AVG_WAIT_DAYS
            storage_addon = TRANSSHIP_STORAGE_COST_PER_FFE_DAY * TRANSSHIP_AVG_WAIT_DAYS
        except ImportError:
            storage_addon = 28.0  # fallback: $8/day x 3.5 days = $28/FFE
        pt_total = pt + storage_addon  # handling fee + hub storage cost

        for i, a in enumerate(sids):
            for b in sids[i + 1:]:
                ua = key_to_node[(a, p)]
                ub = key_to_node[(b, p)]
                master.append((ua, ub, pt_total, INF_CAP, 1))
                master.append((ub, ua, pt_total, INF_CAP, 1))


    return master, key_to_node, n


def _dijkstra_path(
    graph: Dict[int, List[int]],
    edges: List[List[float]],
    S: int,
    T: int,
    max_transship_hops: int,
) -> Optional[Tuple[List[int], float]]:
    # Constrained shortest path in the expanded state space:
    # state = (node, used_transship_hops).
    dist: Dict[Tuple[int, int], float] = {(S, 0): 0.0}
    prev: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    pq: List[Tuple[float, int, int]] = [(0.0, S, 0)]
    while pq:
        d, u, used = heapq.heappop(pq)
        if d > dist.get((u, used), float("inf")) + 1e-9:
            continue
        if u == T:
            # Reconstruct from the reached (T, used) state.
            path_e: List[int] = []
            cur_u = T
            cur_used = used
            while not (cur_u == S and cur_used == 0):
                pu, pused, ei = prev[(cur_u, cur_used)]
                path_e.append(ei)
                cur_u, cur_used = pu, pused
            path_e.reverse()
            return path_e, d
        for ei in graph[u]:
            e = edges[ei]
            u0 = int(e[0])
            v = int(e[1])
            c = e[2]
            cap = e[3]
            fl = e[4]
            hop_inc = int(e[6])
            res = cap - fl
            if res <= EPS:
                continue
            nu = used + hop_inc
            if nu > max_transship_hops:
                continue
            nd = d + c
            if nd < dist.get((v, nu), float("inf")) - 1e-9:
                dist[(v, nu)] = nd
                prev[(v, nu)] = (u, used, ei)
                heapq.heappush(pq, (nd, v, nu))
    return None


def _run_greedy_mcf(
    master_tpl: List[Tuple[int, int, float, float, int]],
    n_service_nodes: int,
    key_to_node: Dict[Tuple[int, str], int],
    enriched_services: List[dict],
    data: dict,
    max_transship_hops: int,
) -> Tuple[float, float, float, Dict[str, float]]:
    ports_db = data["ports"]
    demand = data["demand"]

    master_flow = [0.0] * len(master_tpl)

    demand_items = sorted(
        demand.items(),
        key=lambda it: -it[1].revenue_per_ffe,
    )

    r_total = 0.0
    c_handle = 0.0
    served_map: Dict[str, float] = {}
    served_ffe = 0.0

    for (o, dport), rec in demand_items:
        qty = float(rec.ffe_per_week)
        rev = float(rec.revenue_per_ffe)
        dr = qty

        while dr > EPS:
            S = n_service_nodes
            T = n_service_nodes + 1
            edges: List[List[float]] = []
            graph: Dict[int, List[int]] = defaultdict(list)

            def add_edge(
                u: int,
                v: int,
                cost: float,
                cap: float,
                midx: int,
                hop_inc: int,
            ):
                idx = len(edges)
                # Edge fields: u, v, cost, cap, flow, midx (master idx), hop_inc
                edges.append([float(u), float(v), cost, cap, 0.0, float(midx), float(hop_inc)])
                graph[u].append(idx)

            for i, (u, v, c, cap, hop_inc) in enumerate(master_tpl):
                res = cap - master_flow[i]
                if res > EPS:
                    add_edge(u, v, c, res, i, hop_inc)

            for sid, svc in enumerate(enriched_services):
                if o in svc["port_sequence"]:
                    u = key_to_node[(sid, o)]
                    add_edge(S, u, ports_db[o].cost_per_full, INF_CAP, -1, 0)
                if dport in svc["port_sequence"]:
                    u = key_to_node[(sid, dport)]
                    add_edge(u, T, ports_db[dport].cost_per_full, INF_CAP, -1, 0)

            path_res = _dijkstra_path(
                graph, edges, S, T, max_transship_hops=int(max_transship_hops)
            )
            if path_res is None:
                break
            path_e, path_cost = path_res
            path_cap = min(
                edges[ei][3] - edges[ei][4]
                for ei in path_e
            )
            push = min(dr, path_cap)
            if push <= EPS:
                break

            r_total += push * rev
            c_handle += path_cost * push

            for ei in path_e:
                midx = int(edges[ei][5])
                if midx >= 0:
                    master_flow[midx] += push
            dr -= push
            served_ffe += push
            key_s = f"{o}->{dport}"
            served_map[key_s] = served_map.get(key_s, 0.0) + push

    total_dem = sum(r.ffe_per_week for r in demand.values())
    unserved = max(total_dem - served_ffe, 0.0)
    return r_total, c_handle, served_ffe, served_map


def evaluate_paper_profit(
    data: dict,
    service_records: List[dict],
    rejection_penalty: Optional[float] = None,
    max_transship_hops: Optional[int] = None,
) -> PaperProfitBreakdown:
    """
    Compute η and full decomposition for a set of weekly liner services.

    Parameters
    ----------
    data : output of `stage0.load_all()`
    service_records : each dict needs `port_sequence`, `vessel_class`, optional `frequency`, `route_id`
    rejection_penalty : $/FFE for unserved demand (default: PAPER_REJECTION_PENALTY_PER_FFE)
    """
    pen = (
        float(rejection_penalty)
        if rejection_penalty is not None
        else float(PAPER_REJECTION_PENALTY_PER_FFE)
    )
    mth = (
        int(max_transship_hops)
        if max_transship_hops is not None
        else int(MAX_TRANSSHIP_HOPS)
    )

    if not service_records:
        total_dem = sum(r.ffe_per_week for r in data["demand"].values())
        c_rej = pen * total_dem
        return PaperProfitBreakdown(
            eta=-c_rej,
            r_total=0.0,
            c_reject=c_rej,
            c_handle=0.0,
            c_ndp=0.0,
            c_service=0.0,
            c_unused=0.0,
            c_voyage_port_canal=0.0,
            c_idle=0.0,
            c_escort=0.0,
            served_ffe=0.0,
            total_demand_ffe=total_dem,
            unserved_ffe=total_dem,
            details={"n_services": 0},
        )

    c_ndp, c_svc, c_unused, c_vpc, c_idle, c_escort, enriched = _compute_c_ndp_and_enrich(
        service_records, data
    )
    master_tpl, key_to_node, n_nodes = _build_static_master_edges(enriched, data)
    r_tot, c_handle, served, _smap = _run_greedy_mcf(
        master_tpl, n_nodes, key_to_node, enriched, data, max_transship_hops=mth
    )
    total_dem = sum(r.ffe_per_week for r in data["demand"].values())
    unserved = max(total_dem - served, 0.0)
    c_reject = pen * unserved
    eta = r_tot - c_reject - c_handle - c_ndp

    return PaperProfitBreakdown(
        eta=round(eta, 2),
        r_total=round(r_tot, 2),
        c_reject=round(c_reject, 2),
        c_handle=round(c_handle, 2),
        c_ndp=round(c_ndp, 2),
        c_service=round(c_svc, 2),
        c_unused=round(c_unused, 2),
        c_voyage_port_canal=round(c_vpc, 2),
        c_idle=round(c_idle, 2),
        c_escort=round(c_escort, 2),
        served_ffe=round(served, 2),
        total_demand_ffe=round(total_dem, 2),
        unserved_ffe=round(unserved, 2),
        details={
            "formula": (
                "eta = R_total - C_reject - C_handle - C_NDP  |  "
                "C_NDP = C_service + C_unused + C_voyage_port_canal + C_idle + C_escort  |  "
                "Professor formula: Profit = Revenue - VesselFixed - VesselDynamic(sail+idle) "
                "- LoadUnload - Transship - PortCallFees - PortEscortFees - CanalFees"
            ),
            "n_services": len(service_records),
            "rejection_penalty_per_ffe": pen,
            "max_transship_hops": int(mth),
            "pct_served": round(100.0 * served / total_dem, 4) if total_dem else 0.0,
        },
    )


def breakdown_to_dict(b: PaperProfitBreakdown) -> dict:
    d = {k: getattr(b, k) for k in (
        "eta", "r_total", "c_reject", "c_handle", "c_ndp",
        "c_service", "c_unused", "c_voyage_port_canal",
        "c_idle", "c_escort",                              # NEW: professor's formula components
        "served_ffe", "total_demand_ffe", "unserved_ffe",
    )}
    d["details"] = b.details
    d["eta_million_usd_per_week"] = round(b.eta / 1e6, 4)
    # Professor's formula breakdown for reporting
    d["professor_formula_breakdown"] = {
        "revenue":              round(b.r_total, 2),
        "vessel_fixed_cost":    round(b.c_service, 2),
        "vessel_dynamic_cost":  round(b.c_voyage_port_canal + b.c_idle, 2),
        "  sailing_fuel":       round(b.c_voyage_port_canal, 2),
        "  idle_port_fuel":     round(b.c_idle, 2),
        "load_unload_costs":    round(b.c_handle, 2),
        "transship_costs":      "(included in load_unload_costs above via MCF)",
        "port_call_costs":      "(included in vessel_dynamic_cost above)",
        "port_escort_fees":     round(b.c_escort, 2),
        "rejection_penalty":    round(b.c_reject, 2),
        "profit_eta":           round(b.eta, 2),
    }
    return d


def load_services_from_bb(
    project_root: str,
) -> List[dict]:
    """Load selected routes from outputs/cg_routes.json ∩ bb_result.json."""
    import json
    import os

    cg_path = os.path.join(project_root, "outputs", "cg_routes.json")
    bb_path = os.path.join(project_root, "outputs", "bb_result.json")
    with open(cg_path, encoding="utf-8") as f:
        pool = json.load(f)
    with open(bb_path, encoding="utf-8") as f:
        bb = json.load(f)
    sel = set(bb.get("selected_route_ids", []))
    return [r for r in pool if r.get("route_id") in sel]
