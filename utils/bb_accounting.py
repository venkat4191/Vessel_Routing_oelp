from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

from utils.config import (
    TRANSSHIP_COST_FALLBACK_RATIO,
    TRANSSHIP_AVG_WAIT_DAYS,
    TRANSSHIP_STORAGE_COST_PER_FFE_DAY,
)


def _effective_revenue_per_ffe(demand_rec) -> float:
    if hasattr(demand_rec, "effective_revenue_per_ffe"):
        return float(demand_rec.effective_revenue_per_ffe)
    return float(demand_rec.revenue_per_ffe)


def _transship_ffe_cost(port, ffe: float) -> float:
    tr = float(getattr(port, "cost_per_full_trnsf", 0.0) or 0.0)
    cf = float(getattr(port, "cost_per_full", 0.0) or 0.0)
    if tr > 0.0:
        rate = tr
    else:
        rate = float(TRANSSHIP_COST_FALLBACK_RATIO) * cf
    rate += float(TRANSSHIP_STORAGE_COST_PER_FFE_DAY) * float(TRANSSHIP_AVG_WAIT_DAYS)
    return ffe * rate


def handling_cost_for_od_flow(
    ports: Mapping,
    port_sequence: List[str],
    o: str,
    d: str,
    ffe: float,
) -> float:
    seq = port_sequence
    if not seq or ffe <= 0:
        return 0.0
    try:
        i = seq.index(o)
        j = seq.index(d)
    except ValueError:
        return 0.0
    if i >= j:
        return 0.0
    po = ports.get(o)
    pd = ports.get(d)
    if po is None or pd is None:
        return 0.0
    cost = ffe * float(po.cost_per_full) + ffe * float(pd.cost_per_full)
    for k in range(i + 1, j):
        p = seq[k]
        pt = ports.get(p)
        if pt is None:
            continue
        cost += _transship_ffe_cost(pt, ffe)
    return cost


def accounting_revenue_and_handling(
    demand: dict,
    ports: Mapping,
    cargo_flows: Dict[str, float],
    route_id_to_sequence: Dict[str, List[str]],
) -> Tuple[float, float]:
    revenue = 0.0
    handling = 0.0
    for key, qty in (cargo_flows or {}).items():
        try:
            od_str, rid = key.split("|", 1)
            o, d = od_str.split("->", 1)
        except ValueError:
            continue
        ffe = float(qty)
        if ffe <= 1e-9:
            continue
        od = (o, d)
        if od not in demand:
            continue
        revenue += ffe * _effective_revenue_per_ffe(demand[od])
        seq = route_id_to_sequence.get(rid) or []
        handling += handling_cost_for_od_flow(ports, seq, o, d, ffe)
    return revenue, handling


def recompute_bb_financials(
    demand: dict,
    ports: Mapping,
    cargo_flows: Dict[str, float],
    weekly_op_cost_selected: float,
    route_id_to_sequence: Dict[str, List[str]],
) -> Tuple[float, float, float]:
    rev, hand = accounting_revenue_and_handling(
        demand, ports, cargo_flows, route_id_to_sequence
    )
    net = float(rev) - float(weekly_op_cost_selected) - float(hand)
    return round(rev, 2), round(hand, 2), round(net, 2)
