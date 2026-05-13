# tests/test_stage0.py
# Quick sanity checks for Stage 0 — run with: python tests/test_stage0.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage0.loader import run_stage0

def test_stage0():
    data = run_stage0(verbose=False)

    # ── Port data ──────────────────────────────────────────────────────────
    assert len(data["port_data"]) == 435,           "Expected 435 ports"
    sgsin = data["port_data"]["SGSIN"]
    assert sgsin.name == "Singapore",               "Singapore name wrong"
    assert sgsin.draft == 13.5,                     "Singapore draft wrong"
    assert sgsin.d_region == "Singapore",           "Singapore region wrong"

    # ── Demand ────────────────────────────────────────────────────────────
    assert len(data["od_matrix"]) == 1764,          "Expected 1764 OD pairs"
    assert len(data["demand_ports"]) == 47,         "Expected 47 demand ports"
    od = data["od_matrix"][("CNYTN", "USLAX")]
    assert od.ffe_per_week == 1865,                 "Largest OD pair FFE wrong"
    assert od.revenue_per_ffe == 1620,              "Largest OD pair revenue wrong"

    # ── Fleet ─────────────────────────────────────────────────────────────
    assert len(data["fleet"]) == 6,                 "Expected 6 vessel classes"
    pp = data["fleet"]["Post_panamax"]
    assert not pp.can_use_panama,                   "Post_panamax should not use Panama"
    assert pp.can_use_suez,                         "Post_panamax should use Suez"
    sp = data["fleet"]["Super_panamax"]
    assert sp.capacity_ffe == 7500,                 "Super_panamax capacity wrong"

    # ── Graph ─────────────────────────────────────────────────────────────
    g = data["graph"]
    assert g.number_of_nodes() == 47,               "Graph should have 47 nodes"
    assert g.number_of_edges() > 0,                 "Graph has no edges"
    assert g.has_edge("SGSIN", "HKHKG"),            "Missing SIN→HKG edge"

    # ── Dist lookup ───────────────────────────────────────────────────────
    entry = data["dist_lookup"][("SGSIN", "HKHKG")]
    assert entry.distance_nm > 0,                   "Distance SIN→HKG should be > 0"

    print("All Stage 0 tests passed ✓")
    stats = data["stats"]
    print(f"  Demand ports : {stats['n_demand_ports']}")
    print(f"  OD pairs     : {stats['n_od_pairs']}")
    print(f"  Total demand : {stats['total_ffe_per_week']:,.0f} FFE/week")
    print(f"  Fleet classes: {stats['n_vessel_classes']}")
    print(f"  Graph edges  : {stats['n_graph_edges']}")

if __name__ == "__main__":
    test_stage0()
