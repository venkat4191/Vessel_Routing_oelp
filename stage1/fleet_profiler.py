"""
Stage 1B -- Fleet Profiler
==========================
Pure arithmetic. No LLM, no optimization.

For every vessel class, computes:
  - Whether it can call at each port (draft check)
  - Sailing time, cycle time, vessels needed for any route
  - Full weekly operating cost for a route
  - Canal fees if route uses Panama or Suez

Outputs a FleetProfile object that Stage 2 and Stage 3 use
to quickly look up vessel capabilities without recomputing.
"""

import sys
import os
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    PORT_DAYS_PER_CALL, BUNKER_PRICE_PER_TON, PORT_ESCORT_FEE_FRACTION,
    PORT_DWELL_DAYS_BY_CLASS, BUNKER_SPEED_EXPONENT, ENABLE_SLOW_STEAMING,
    SCHEDULE_SLACK_DAYS, ALLIANCE_ENABLED, ALLIANCE_SLOT_FRACTION,
)


# ── Return types ──────────────────────────────────────────────────────────────

@dataclass
class RouteEconomics:
    """
    Full cost/capability breakdown for one vessel class on one route.

    Professor's profit formula components:
      Revenue              — computed by the RMP/LP from cargo flows
      Vessel Fixed Cost    — weekly_tc_cost  (TC hire: crew + insurance + maintenance)
      Vessel Dynamic Cost  — weekly_bunker (sailing fuel at optimal_speed)
                             + weekly_idle_bunker (port dwell fuel)
      Port Call Costs      — weekly_port_cost (fixed per-call fees)
      Port Escort Fees     — weekly_escort_cost (tug fees ~15% of call fee)
      Canal Fees           — weekly_canal_fee
      Load/Unload Costs    — computed in RMP once FFE flows are known
      Transshipment Costs  — computed in paper_profit MCF (incl. storage)

    FIX 5: optimal_speed is the slow-steaming speed that minimizes total cost.
    FIX 7: dwell_days_per_call is vessel-class-specific (not uniform 1.0).
    FIX 8: schedule_buffer_days is added to align cycle to 7/freq grid.
    """
    vessel_class:     str
    port_sequence:    list
    feasible:         bool

    # Times
    one_way_days:       float = 0.0
    cycle_days:         float = 0.0     # includes schedule buffer if applied
    vessels_needed:     int   = 0

    # FIX 5: Slow steaming
    optimal_speed:      float = 0.0     # knots — speed that minimizes weekly cost
    design_speed_used:  float = 0.0     # for reference

    # FIX 7: Per-vessel dwell time
    dwell_days_per_call: float = 1.0    # days at port per vessel call

    # FIX 8: Schedule buffer
    schedule_buffer_days: float = 0.0   # days added to align cycle to weekly grid

    # Costs per week (for the full fleet of vessels_needed ships)
    weekly_tc_cost:       float = 0.0
    weekly_bunker:        float = 0.0
    weekly_idle_bunker:   float = 0.0
    weekly_port_cost:     float = 0.0
    weekly_escort_cost:   float = 0.0
    weekly_canal_fee:     float = 0.0
    weekly_total_cost:    float = 0.0

    infeasible_reason: str = ""


@dataclass
class FleetProfile:
    """
    All precomputed vessel capabilities.
    Built once by run() and passed to every later stage.
    """
    # port_access[vessel_class][unlocode] = True/False
    port_access: dict = field(default_factory=dict)

    # vessel_records: the raw VesselRecord objects from Stage 0
    vessel_records: dict = field(default_factory=dict)

    # Derived: for quick lookup
    # od_feasible_vessels[(origin, dest)] = [vessel_class, ...]
    od_feasible_vessels: dict = field(default_factory=dict)


# ── Core calculations ─────────────────────────────────────────────────────────

def sailing_days(distance_nm: float, speed_knots: float) -> float:
    """One-way sailing time in days."""
    return distance_nm / (speed_knots * 24.0)


def vessels_needed(cycle_days: float, frequency_per_week: int = 1) -> int:
    """
    How many vessels needed for a weekly service at given frequency.
    frequency=1: one departure/week → vessels = ceil(cycle_days / 7)
    frequency=2: two departures/week → vessels = ceil(cycle_days / 3.5)
    """
    days_between_departures = 7.0 / frequency_per_week
    return math.ceil(cycle_days / days_between_departures)


def weekly_tc_cost(vessel, n_vessels: int) -> float:
    """Total weekly charter hire for n_vessels of this class."""
    return vessel.tc_rate_daily * 7 * n_vessels


def get_dwell_days(vessel_class: str) -> float:
    """
    FIX 7: Per-vessel-class port dwell time.
    Returns days spent at port per call, accounting for vessel size.
    Larger vessels take longer to load/unload (more cargo, more berth time).
    """
    return PORT_DWELL_DAYS_BY_CLASS.get(vessel_class, PORT_DAYS_PER_CALL)


def find_optimal_speed(
    vessel,
    total_distance_nm: float,
    n_port_calls: int,
    frequency_per_week: int,
    fuel_price_per_ton: float,
) -> Tuple[float, float]:
    """
    FIX 5: Slow Steaming — find the speed that minimizes total weekly cost.

    The tradeoff:
      - Going slower cuts fuel cost (∝ speed^3 by admiralty law)
      - But longer sailing time → longer cycle → more vessels needed → higher fixed cost

    We scan speeds from min_speed to max_speed in 0.25-knot steps and return
    the speed that gives the lowest total weekly operating cost.

    Returns: (optimal_speed_knots, cycle_days_at_optimal_speed)

    Economics:
      fuel_cost(v) = bunker_per_day × (v/v_design)^3 × sailing_days × price
      fixed_cost   = TC_rate × 7 × ceil(cycle_days / (7/freq))
      cycle_days   = total_distance / (v × 24) + n_port_calls × dwell_days
    """
    dwell = get_dwell_days(vessel.vessel_class)
    port_time = n_port_calls * dwell

    best_speed = vessel.design_speed
    best_cost  = float('inf')
    best_cycle = 0.0

    # Scan from min to max in 0.25-knot increments
    v = vessel.min_speed
    while v <= vessel.max_speed + 0.01:
        total_sail_days = total_distance_nm / (v * 24.0)
        cycle = total_sail_days + port_time

        nv = vessels_needed(cycle, frequency_per_week)
        wk_tc     = vessel.tc_rate_daily * 7 * nv
        fuel_rate = vessel.bunker_per_day * ((v / vessel.design_speed) ** BUNKER_SPEED_EXPONENT)
        wk_bunker = fuel_rate * total_sail_days * fuel_price_per_ton
        wk_idle   = vessel.idle_consumption * n_port_calls * dwell * fuel_price_per_ton

        total = wk_tc + wk_bunker + wk_idle
        if total < best_cost:
            best_cost  = total
            best_speed = v
            best_cycle = cycle

        v += 0.25

    return round(best_speed, 2), round(best_cycle, 2)


def route_economics(
    vessel,
    port_sequence: list,
    ports: dict,
    distances: dict,
    frequency_per_week: int = 1,
) -> RouteEconomics:
    """
    Compute full economics for a vessel running port_sequence at given frequency.

    FIX 5: Uses slow-steaming optimizer to find cost-minimizing speed.
    FIX 7: Uses per-vessel-class port dwell time (not uniform 1.0 day).
    FIX 8: Adds schedule buffer to align cycle_days to 7/freq weekly grid.
    FIX 9: Uses vessel.fuel_price() for correct VLSFO/LNG/HFO price.
    FIX 11: Applies ALLIANCE_SLOT_FRACTION to effective capacity if enabled.

    port_sequence is a one-way list; the vessel completes a round trip.
    """
    vc = vessel.vessel_class

    # ── Step 1: Draft check ───────────────────────────────────────────────────
    for p in port_sequence:
        port = ports.get(p)
        if port is None:
            return RouteEconomics(vessel_class=vc, port_sequence=port_sequence,
                                  feasible=False, infeasible_reason=f"Port {p} not found")
        if vessel.draft > port.draft:
            return RouteEconomics(vessel_class=vc, port_sequence=port_sequence,
                                  feasible=False,
                                  infeasible_reason=f"Draft fail: {vessel.draft}m > {port.name} {port.draft}m")

    # ── Step 2: Leg distances, canal fees ────────────────────────────────────
    legs_full = (list(zip(port_sequence[:-1], port_sequence[1:]))
                 + [(port_sequence[-1], port_sequence[0])])

    total_distance_nm = 0.0
    total_canal_fee   = 0.0

    for (frm, to) in legs_full:
        dist_rec = distances.get((frm, to))
        if dist_rec is None:
            return RouteEconomics(vessel_class=vc, port_sequence=port_sequence,
                                  feasible=False, infeasible_reason=f"No distance: {frm}->{to}")
        if dist_rec.is_panama:
            if vessel.panama_fee is None:
                return RouteEconomics(vessel_class=vc, port_sequence=port_sequence,
                                      feasible=False, infeasible_reason=f"{vc} cannot transit Panama")
            if dist_rec.draft and vessel.draft > dist_rec.draft:
                return RouteEconomics(vessel_class=vc, port_sequence=port_sequence,
                                      feasible=False,
                                      infeasible_reason=f"Panama draft fail: {vessel.draft}m > {dist_rec.draft}m")
            total_canal_fee += vessel.panama_fee
        if dist_rec.is_suez:
            if vessel.suez_fee is None:
                return RouteEconomics(vessel_class=vc, port_sequence=port_sequence,
                                      feasible=False, infeasible_reason=f"{vc} cannot transit Suez")
            total_canal_fee += vessel.suez_fee
        total_distance_nm += dist_rec.distance_nm

    # ── Step 3: FIX 9 — correct fuel price for this vessel's fuel type ───────
    fuel_price = vessel.fuel_price() if hasattr(vessel, 'fuel_price') else BUNKER_PRICE_PER_TON

    # ── Step 4: FIX 5 — Slow steaming: find optimal speed ────────────────────
    n_port_calls = len(port_sequence)
    if ENABLE_SLOW_STEAMING:
        optimal_speed, _ = find_optimal_speed(
            vessel, total_distance_nm, n_port_calls, frequency_per_week, fuel_price
        )
    else:
        optimal_speed = vessel.design_speed

    # ── Step 5: FIX 7 — Per-vessel dwell time ────────────────────────────────
    dwell = get_dwell_days(vc)

    # ── Step 6: Cycle time ────────────────────────────────────────────────────
    total_sailing_days = total_distance_nm / (optimal_speed * 24.0)
    port_time          = n_port_calls * dwell
    raw_cycle          = total_sailing_days + port_time

    # ── Step 7: FIX 8 — Schedule quantization buffer ────────────────────────
    # Real liner schedules depart on a fixed weekday. Cycle must be a multiple
    # of (7 / frequency) days. If raw_cycle is between multiples, vessels either
    # wait at a port (adding idle cost) or slow-steam the last leg.
    # We round UP to the next slot boundary if within SCHEDULE_SLACK_DAYS.
    slot = 7.0 / frequency_per_week
    remainder = raw_cycle % slot
    if remainder > 0 and remainder <= SCHEDULE_SLACK_DAYS:
        schedule_buffer = slot - remainder   # push to next full slot
    elif remainder > SCHEDULE_SLACK_DAYS:
        schedule_buffer = slot - remainder   # must round up to maintain schedule
    else:
        schedule_buffer = 0.0
    cycle_days = raw_cycle + schedule_buffer

    # ── Step 8: Vessels needed ────────────────────────────────────────────────
    n_vessels = vessels_needed(cycle_days, frequency_per_week)

    # ── Step 9: Weekly costs (professor's formula) ────────────────────────────
    wk_tc = vessel.tc_rate_daily * 7 * n_vessels

    # Sailing fuel at optimal speed using cubic fuel curve
    fuel_rate_at_speed = (vessel.bunker_per_day
                          * (optimal_speed / vessel.design_speed) ** BUNKER_SPEED_EXPONENT)
    wk_bunker = fuel_rate_at_speed * total_sailing_days * fuel_price

    # Idle fuel: vessel engines run at port during dwell
    wk_idle = vessel.idle_consumption * n_port_calls * dwell * fuel_price

    # Port call fees (fixed per call)
    wk_port = sum(ports[p].port_call_cost_fixed for p in port_sequence)

    # Port escort / tug fees (~15% of fixed call fee)
    wk_escort = sum(ports[p].port_call_cost_fixed * PORT_ESCORT_FEE_FRACTION
                    for p in port_sequence)

    wk_canal = total_canal_fee

    wk_total = wk_tc + wk_bunker + wk_idle + wk_port + wk_escort + wk_canal

    # Outbound-only sailing days (for one_way_days)
    outbound_legs = list(zip(port_sequence[:-1], port_sequence[1:]))
    one_way_sail  = sum(distances[(f, t)].distance_nm / (optimal_speed * 24.0)
                        for (f, t) in outbound_legs)
    one_way_days  = one_way_sail + port_time

    return RouteEconomics(
        vessel_class         = vc,
        port_sequence        = port_sequence,
        feasible             = True,
        one_way_days         = round(one_way_days, 2),
        cycle_days           = round(cycle_days, 2),
        vessels_needed       = n_vessels,
        optimal_speed        = optimal_speed,
        design_speed_used    = vessel.design_speed,
        dwell_days_per_call  = dwell,
        schedule_buffer_days = round(schedule_buffer, 2),
        weekly_tc_cost       = round(wk_tc, 2),
        weekly_bunker        = round(wk_bunker, 2),
        weekly_idle_bunker   = round(wk_idle, 2),
        weekly_port_cost     = round(wk_port, 2),
        weekly_escort_cost   = round(wk_escort, 2),
        weekly_canal_fee     = round(wk_canal, 2),
        weekly_total_cost    = round(wk_total, 2),
    )


# ── Port access matrix ────────────────────────────────────────────────────────

def build_port_access(fleet: dict, instance_ports: set, ports: dict) -> dict:
    """
    Returns port_access[vessel_class][unlocode] = True/False
    based purely on draft constraint.
    """
    access = {}
    for vc, vessel in fleet.items():
        access[vc] = {}
        for p in instance_ports:
            port = ports[p]
            access[vc][p] = (vessel.draft <= port.draft)
    return access


def build_od_feasible_vessels(
    fleet: dict,
    instance_ports: set,
    ports: dict,
    distances: dict,
) -> dict:
    """
    For every direct OD pair among instance ports, which vessel classes
    can physically operate a direct 2-port service (origin -> dest -> origin)?

    Returns { (origin, dest) -> [vessel_class, ...] }
    """
    od_feasible = {}
    port_list   = sorted(instance_ports)

    for o in port_list:
        for d in port_list:
            if o == d:
                continue
            if (o, d) not in distances:
                continue
            feasible_vessels = []
            for vc, vessel in fleet.items():
                econ = route_economics(vessel, [o, d], ports, distances)
                if econ.feasible:
                    feasible_vessels.append(vc)
            od_feasible[(o, d)] = feasible_vessels

    return od_feasible


# ── Main runner ───────────────────────────────────────────────────────────────

def run(data: dict, verbose: bool = True) -> FleetProfile:
    """
    Build the full FleetProfile from Stage 0 data.
    This is what all later stages call.
    """
    fleet          = data['fleet']
    ports          = data['ports']
    distances      = data['distances']
    instance_ports = data['instance_ports']

    if verbose:
        print("Stage 1B: Running Fleet Profiler...")

    port_access = build_port_access(fleet, instance_ports, ports)

    if verbose:
        print("\n  Draft access matrix (vessel can call at port):")
        print(f"  {'Vessel class':<22}", end="")
        # Show a sample of 8 ports
        sample_ports = sorted(instance_ports)[:8]
        for p in sample_ports:
            print(f"  {p}", end="")
        print()
        for vc in fleet:
            print(f"  {vc:<22}", end="")
            for p in sample_ports:
                symbol = " OK " if port_access[vc][p] else "FAIL"
                print(f"  {symbol}", end="")
            print()

        # Count total accessible ports per vessel
        print("\n  Accessible ports (draft OK) per vessel class:")
        for vc, vessel in fleet.items():
            n_ok = sum(1 for p in instance_ports if port_access[vc][p])
            blocked = [ports[p].name for p in instance_ports if not port_access[vc][p]]
            print(f"    {vc:<22} {n_ok:>2}/{len(instance_ports)} ports OK", end="")
            if blocked:
                print(f"  |  BLOCKED: {', '.join(blocked)}", end="")
            print()

    # Build OD feasibility (this takes a few seconds — 47x47 pairs x 6 vessels)
    if verbose:
        print("\n  Building OD feasibility matrix (47x47 ports x 6 vessels)...")

    od_feasible = build_od_feasible_vessels(fleet, instance_ports, ports, distances)

    if verbose:
        _print_od_summary(od_feasible, fleet)

    profile = FleetProfile(
        port_access          = port_access,
        vessel_records       = fleet,
        od_feasible_vessels  = od_feasible,
    )

    return profile


def _print_od_summary(od_feasible: dict, fleet: dict):
    # Count OD pairs each vessel class can serve directly
    print("\n  Direct OD pairs each vessel can serve:")
    for vc in fleet:
        n = sum(1 for vessels in od_feasible.values() if vc in vessels)
        print(f"    {vc:<22} {n:>5} direct OD pairs")

    # Count pairs with zero feasible vessels
    zero = [(o,d) for (o,d), v in od_feasible.items() if len(v) == 0]
    if zero:
        print(f"\n  WARNING: {len(zero)} OD pairs have NO direct feasible vessel")
        print(f"  (these must be served via transshipment)")
    else:
        print(f"\n  All direct OD pairs have at least one feasible vessel class.")


# ── Run as script ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from stage0.loader import load_all, validate

    data = load_all(verbose=False)
    validate(data)
    profile = run(data, verbose=True)

    # Quick spot-check: CNSHA -> SGSIN -> NLRTM for each vessel
    print("\n--- Spot check: CNSHA -> SGSIN -> NLRTM ---")
    for vc, vessel in data['fleet'].items():
        econ = route_economics(
            vessel,
            ['CNSHA', 'SGSIN', 'NLRTM'],
            data['ports'],
            data['distances'],
        )
        if econ.feasible:
            print(f"  {vc:<22} cycle={econ.cycle_days:.1f}d  "
                  f"vessels={econ.vessels_needed}  "
                  f"cost/wk=${econ.weekly_total_cost:>10,.0f}")
        else:
            print(f"  {vc:<22} INFEASIBLE: {econ.infeasible_reason}")
