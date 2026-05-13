"""
Stage 0 - Data Foundation
=========================
Reads the 4 raw CSV files and builds 4 clean data structures.
No intelligence, no optimization -- pure organisation.

Output structures
-----------------
ports      : dict { unlocode -> PortRecord }
demand     : dict { (origin, dest) -> DemandRecord }
fleet      : dict { vessel_class -> VesselRecord }
distances  : dict { (from, to) -> DistRecord }
instance_ports : set of the 47 UNLOCODEs active in WorldSmall
"""

import csv
import sys
import os
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    PORTS_FILE, DIST_FILE, FLEET_FILE, DEMAND_FILE, BUNKER_PRICE_PER_TON,
    TRANSSHIP_COST_FALLBACK_RATIO, TRANSSHIP_COST_MAX_RATIO,
    BLENDED_REVENUE_FACTOR, FUEL_TYPE_VLSFO, FUEL_TYPE_LNG, FUEL_TYPE_HFO,
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PortRecord:
    unlocode:               str
    name:                   str
    country:                str
    cabotage_region:        str
    d_region:               str
    longitude:              float
    latitude:               float
    draft:                  float    # max vessel draft allowed (m)
    cost_per_full:          float    # $/FFE for a loaded box at this port
    cost_per_full_trnsf:    float    # $/FFE for a transshipped box
    port_call_cost_fixed:   float    # fixed $ per vessel call
    port_call_cost_per_ffe: float    # variable $ per FFE handled

    def total_call_cost(self, ffe: float = 0) -> float:
        """Total cost of one vessel call handling `ffe` loaded FFE."""
        return self.port_call_cost_fixed + self.port_call_cost_per_ffe * ffe


@dataclass
class DemandRecord:
    origin:           str
    destination:      str
    ffe_per_week:     float   # weekly volume
    revenue_per_ffe:  float   # freight rate $/FFE  (spot rate — headline rate from CSV)
    max_transit_days: int     # max allowed delivery time

    # FIX 12: Revenue Tiers — Contract vs Spot
    # ~65% of cargo moves on long-term contracts at 85% of spot rate.
    # effective_revenue_per_ffe accounts for the real blended rate carriers actually receive.
    # BLENDED_REVENUE_FACTOR = 0.65*0.85 + 0.35*1.0 = 0.9025
    @property
    def effective_revenue_per_ffe(self) -> float:
        """Blended revenue: weighted average of contract rate and spot rate."""
        return self.revenue_per_ffe * BLENDED_REVENUE_FACTOR

    @property
    def weekly_revenue(self) -> float:
        """Gross potential weekly revenue at full spot rate (used for ranking only)."""
        return self.ffe_per_week * self.revenue_per_ffe

    @property
    def effective_weekly_revenue(self) -> float:
        """Realistic weekly revenue accounting for contract vs spot mix."""
        return self.ffe_per_week * self.effective_revenue_per_ffe


@dataclass
class VesselRecord:
    vessel_class:     str
    capacity_ffe:     int
    tc_rate_daily:    float   # USD/day charter hire (covers crew, insurance, maintenance)
    draft:            float   # vessel draft (m)
    min_speed:        float   # knots
    max_speed:        float   # knots
    design_speed:     float   # knots — reference speed (used for fuel curve base)
    bunker_per_day:   float   # metric tons/day at design_speed
    idle_consumption: float   # metric tons/day at anchor/port
    panama_fee:       Optional[float]   # None = vessel cannot use Panama
    suez_fee:         Optional[float]   # None = vessel cannot use Suez

    # FIX 9: Dual-Fuel / LNG — fuel type field
    # "VLSFO" = default IMO 2020 low-sulfur fuel (replaces HFO post-2020)
    # "LNG"   = dual-fuel LNG (lower emissions, higher newbuild cost)
    # "HFO"   = heavy fuel oil (only legal with scrubbers after IMO 2020)
    # "MDO"   = marine diesel oil (small feeders in emission control areas)
    fuel_type: str = FUEL_TYPE_VLSFO

    def can_use_panama(self) -> bool:
        return self.panama_fee is not None and self.draft <= 12.0

    def can_use_suez(self) -> bool:
        return self.suez_fee is not None

    def bunker_rate_at_speed(self, speed_knots: float) -> float:
        """
        Fuel consumption in metric tons/day at arbitrary speed.
        Uses the admiralty cubic law: consumption ∝ speed^3.
        """
        from utils.config import BUNKER_SPEED_EXPONENT
        ratio = speed_knots / self.design_speed
        return self.bunker_per_day * (ratio ** BUNKER_SPEED_EXPONENT)

    def fuel_price(self) -> float:
        """Return the correct bunker price per ton based on this vessel's fuel type."""
        from utils.config import (
            BUNKER_PRICE_PER_TON, HFO_PRICE_PER_TON,
            LNG_PRICE_PER_TON, MGO_PRICE_PER_TON,
            FUEL_TYPE_VLSFO, FUEL_TYPE_HFO, FUEL_TYPE_LNG, FUEL_TYPE_MDO,
        )
        prices = {
            FUEL_TYPE_VLSFO: BUNKER_PRICE_PER_TON,
            FUEL_TYPE_HFO:   HFO_PRICE_PER_TON,
            FUEL_TYPE_LNG:   LNG_PRICE_PER_TON,
            FUEL_TYPE_MDO:   MGO_PRICE_PER_TON,
        }
        return prices.get(self.fuel_type, BUNKER_PRICE_PER_TON)


@dataclass
class DistRecord:
    from_port:    str
    to_port:      str
    distance_nm:  float
    draft:        Optional[float]   # route draft limit (set for Panama routes)
    is_panama:    bool
    is_suez:      bool

    def sailing_days(self, speed_knots: float) -> float:
        """One-way sailing time in days."""
        return self.distance_nm / (speed_knots * 24.0)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ports(filepath: str = PORTS_FILE) -> dict:
    ports = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            code = row['UNLocode'].strip()
            def _f(val):
                v = val.strip()
                return float(v) if v not in ('NULL', '') else 0.0
            ports[code] = PortRecord(
                unlocode               = code,
                name                   = row['name'].strip(),
                country                = row['Country'].strip(),
                cabotage_region        = row['Cabotage_Region'].strip(),
                d_region               = row['D_Region'].strip(),
                longitude              = _f(row['Longitude']),
                latitude               = _f(row['Latitude']),
                draft                  = _f(row['Draft']),
                cost_per_full          = _f(row['CostPerFULL']),
                cost_per_full_trnsf    = _f(row['CostPerFULLTrnsf']),
                port_call_cost_fixed   = _f(row['PortCallCostFixed']),
                port_call_cost_per_ffe = _f(row['PortCallCostPerFFE']),
            )
    # ── Transshipment cost validation (Professor's formula) ────────────────────
    # "If transshipment costs aren't available in the dataset, assume ±20% of load/unload costs."
    # We enforce: 0 < trnsf_cost ≤ TRANSSHIP_COST_MAX_RATIO × cost_per_full.
    # If a port has zero or implausibly high trnsf cost, apply the fallback ratio.
    for code, port in ports.items():
        cpf = port.cost_per_full
        trnsf = port.cost_per_full_trnsf
        if cpf > 0:
            if trnsf <= 0 or trnsf > cpf * TRANSSHIP_COST_MAX_RATIO:
                # Apply fallback: use TRANSSHIP_COST_FALLBACK_RATIO × cost_per_full
                object.__setattr__(port, 'cost_per_full_trnsf', cpf * TRANSSHIP_COST_FALLBACK_RATIO) \
                    if hasattr(port, '__dataclass_fields__') else None
                # PortRecord is not frozen, so direct assignment works:
                port.cost_per_full_trnsf = cpf * TRANSSHIP_COST_FALLBACK_RATIO
    return ports


def load_demand(filepath: str = DEMAND_FILE) -> dict:
    demand = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            key = (row['Origin'].strip(), row['Destination'].strip())
            demand[key] = DemandRecord(
                origin           = row['Origin'].strip(),
                destination      = row['Destination'].strip(),
                ffe_per_week     = float(row['FFEPerWeek']),
                revenue_per_ffe  = float(row['Revenue_1']),
                max_transit_days = int(row['TransitTime']),
            )
    return demand


def load_fleet(filepath: str = FLEET_FILE) -> dict:
    fleet = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            vc = row['Vessel class'].strip()
            panama_raw = row['panamaFee'].strip()
            suez_raw   = row['suezFee'].strip()
            fleet[vc] = VesselRecord(
                vessel_class     = vc,
                capacity_ffe     = int(row['Capacity FFE']),
                tc_rate_daily    = float(row['TC rate daily (fixed Cost)']),
                draft            = float(row['draft']),
                min_speed        = float(row['minSpeed']),
                max_speed        = float(row['maxSpeed']),
                design_speed     = float(row['designSpeed']),
                bunker_per_day   = float(row['Bunker ton per day at designSpeed']),
                idle_consumption = float(row['Idle Consumption ton/day']),
                panama_fee       = float(panama_raw) if panama_raw else None,
                suez_fee         = float(suez_raw)   if suez_raw   else None,
                # FIX 9: Assign fuel type per vessel class.
                # LINERLIB fleet_data.csv has no fuel_type column, so we infer:
                # Large vessels (Super/Post-panamax) on long-haul routes → VLSFO (IMO 2020)
                # Smaller feeders in coastal/regional service → MDO in ECAs, VLSFO elsewhere
                # Real newbuilds post-2022 are increasingly LNG dual-fuel.
                fuel_type = FUEL_TYPE_VLSFO,   # all vessels default to IMO 2020 VLSFO
            )
    return fleet


def load_distances(filepath: str = DIST_FILE) -> dict:
    """Load distances. Keeps canal routes (Panama/Suez) over direct routes."""
    distances = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            key       = (row['fromUNLOCODe'].strip(), row['ToUNLOCODE'].strip())
            draft_raw = row['Draft'].strip()
            dist_nm   = float(row['Distance'])
            is_panama = row['IsPanama'].strip() == '1'
            is_suez   = row['IsSuez'].strip()   == '1'
            rec = DistRecord(
                from_port   = row['fromUNLOCODe'].strip(),
                to_port     = row['ToUNLOCODE'].strip(),
                distance_nm = dist_nm,
                draft       = float(draft_raw) if draft_raw else None,
                is_panama   = is_panama,
                is_suez     = is_suez,
            )
            if key not in distances:
                distances[key] = rec
            else:
                existing  = distances[key]
                is_canal  = is_panama or is_suez
                was_canal = existing.is_panama or existing.is_suez
                if is_canal and not was_canal:
                    distances[key] = rec
                elif is_canal == was_canal and dist_nm < existing.distance_nm:
                    distances[key] = rec
    return distances


# ── Master loader ─────────────────────────────────────────────────────────────

def load_all(verbose: bool = True) -> dict:
    """
    Load all 4 files. Returns a single dict:
        {
            'ports'          : { unlocode -> PortRecord },
            'demand'         : { (origin, dest) -> DemandRecord },
            'fleet'          : { vessel_class -> VesselRecord },
            'distances'      : { (from, to) -> DistRecord },
            'instance_ports' : set of UNLOCODEs active in WorldSmall,
        }
    """
    if verbose:
        print("Stage 0: Loading data files...")

    ports     = load_ports()
    demand    = load_demand()
    fleet     = load_fleet()
    distances = load_distances()

    # Active ports = all ports that appear in the demand file
    instance_ports = set()
    for (o, d) in demand:
        instance_ports.add(o)
        instance_ports.add(d)

    data = {
        'ports'          : ports,
        'demand'         : demand,
        'fleet'          : fleet,
        'distances'      : distances,
        'instance_ports' : instance_ports,
    }

    if verbose:
        _print_summary(data)

    return data


# ── Validation ────────────────────────────────────────────────────────────────

def validate(data: dict) -> bool:
    """
    Basic sanity checks. Returns True if all pass.
    """
    ports     = data['ports']
    demand    = data['demand']
    fleet     = data['fleet']
    distances = data['distances']
    inst      = data['instance_ports']

    errors = []

    # Every instance port must exist in ports.csv
    for p in inst:
        if p not in ports:
            errors.append(f"Instance port {p} missing from ports.csv")

    # Every OD pair must have a distance entry
    missing_dist = [(o,d) for (o,d) in demand if (o,d) not in distances]
    if missing_dist:
        errors.append(f"{len(missing_dist)} OD pairs missing from dist_dense.csv")

    # Fleet sanity
    for vc, v in fleet.items():
        if v.capacity_ffe <= 0:
            errors.append(f"Vessel {vc}: capacity <= 0")
        if v.design_speed <= 0:
            errors.append(f"Vessel {vc}: design speed <= 0")
        if v.tc_rate_daily <= 0:
            errors.append(f"Vessel {vc}: TC rate <= 0")

    # Demand sanity
    bad = [(o,d) for (o,d), r in demand.items()
           if r.ffe_per_week <= 0 or r.revenue_per_ffe <= 0]
    if bad:
        errors.append(f"{len(bad)} demand records have zero/negative FFE or revenue")

    # Distance sanity
    bad_d = [k for k, r in distances.items() if r.distance_nm <= 0]
    if bad_d:
        errors.append(f"{len(bad_d)} distances are non-positive")

    if errors:
        print("\nValidation FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        return False
    else:
        print("Validation PASSED -- all checks clean.")
        return True


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(data: dict):
    ports     = data['ports']
    demand    = data['demand']
    fleet     = data['fleet']
    distances = data['distances']
    inst      = data['instance_ports']

    total_ffe = sum(r.ffe_per_week     for r in demand.values())
    total_rev = sum(r.weekly_revenue   for r in demand.values())

    print(f"\n{'='*56}")
    print(f"  Stage 0 -- Data Load Summary ({len(inst)} active ports)")
    print(f"{'='*56}")
    print(f"  ports.csv         {len(ports):>6,} total ports in database")
    print(f"  dist_dense.csv    {len(distances):>6,} port-pair distances")
    print(f"  fleet_data.csv    {len(fleet):>6} vessel classes")
    print(f"  demand file       {len(demand):>6,} OD pairs (WorldSmall)")
    print(f"{'─'*56}")
    print(f"  Active ports      {len(inst):>6}")
    print(f"  Total demand      {total_ffe:>10,.0f} FFE / week")
    print(f"  Max possible rev  ${total_rev:>11,.0f} / week")
    print(f"{'─'*56}")
    print(f"  Vessel classes:")
    for vc, v in fleet.items():
        flags = []
        if v.can_use_panama(): flags.append("Panama")
        if v.can_use_suez():   flags.append("Suez")
        flag_str = ", ".join(flags) if flags else "No canals"
        print(f"    {vc:<22} {v.capacity_ffe:>5} FFE  "
              f"draft={v.draft:>4}m  "
              f"spd={v.design_speed}kn  "
              f"[{flag_str}]")
    print(f"{'='*56}\n")


# ── Run as script ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_all(verbose=True)
    ok   = validate(data)
    sys.exit(0 if ok else 1)
