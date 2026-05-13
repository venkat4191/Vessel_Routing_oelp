# utils/types.py
# ─────────────────────────────────────────────────────────────────────────────
# Shared dataclasses used across all stages.
# Every stage imports these — never define the same structure twice.
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional


# ── Port ──────────────────────────────────────────────────────────────────────
@dataclass
class Port:
    """One port from ports.csv."""
    unlocode: str             # e.g. "SGSIN"
    name: str                 # e.g. "Singapore"
    country: str
    cabotage_region: str
    d_region: str             # Trade region used for grouping
    longitude: float
    latitude: float
    draft: float              # Max vessel draft allowed (metres)
    cost_per_full: float      # $/FFE loaded at this port
    cost_per_full_transship: float  # $/FFE transshipped
    port_call_fixed: float    # Fixed cost per vessel call ($)
    port_call_per_ffe: float  # Variable cost per FFE handled ($)


# ── Vessel ────────────────────────────────────────────────────────────────────
@dataclass
class Vessel:
    """One vessel class from fleet_data.csv."""
    vessel_class: str         # e.g. "Panamax_2400"
    capacity_ffe: int         # TEU capacity in FFE
    tc_rate_daily: float      # Time-charter rate in $/day (fixed operating cost)
    draft: float              # Vessel draft in metres
    min_speed: float          # Knots
    max_speed: float          # Knots
    design_speed: float       # Knots (used for cycle time calculations)
    bunker_per_day: float     # Tonnes of HFO/day at design speed
    idle_consumption: float   # Tonnes of HFO/day at anchor/port
    panama_fee: Optional[float] = None  # $ per canal transit (None = cannot use)
    suez_fee: Optional[float]   = None  # $ per canal transit

    @property
    def can_use_panama(self) -> bool:
        return self.panama_fee is not None and self.panama_fee > 0

    @property
    def can_use_suez(self) -> bool:
        return self.suez_fee is not None and self.suez_fee > 0


# ── OD Pair (Demand) ──────────────────────────────────────────────────────────
@dataclass
class ODPair:
    """One origin-destination demand pair from Demand_WorldSmall.csv."""
    origin: str               # UNLOCODE
    destination: str          # UNLOCODE
    ffe_per_week: float       # Weekly demand in FFE
    revenue_per_ffe: float    # Freight rate in $/FFE
    max_transit_days: int     # Maximum allowed transit time in days

    @property
    def weekly_revenue(self) -> float:
        """Total potential weekly revenue if fully served."""
        return self.ffe_per_week * self.revenue_per_ffe


# ── Distance Entry ────────────────────────────────────────────────────────────
@dataclass
class DistEntry:
    """One row from dist_dense.csv."""
    from_port: str
    to_port: str
    distance_nm: float
    draft: Optional[float]    # Draft constraint for this leg (mainly Panama)
    is_panama: bool
    is_suez: bool


# ── Route ─────────────────────────────────────────────────────────────────────
@dataclass
class Route:
    """
    A proposed or active shipping service.
    Created by the LLM Oracle or seed generator; validated by RouteValidator;
    added to the RMP as a column.
    """
    route_id: str                         # Unique identifier, e.g. "R001"
    port_sequence: list[str]              # Ordered list of UNLOCODEs
    vessel_class: str                     # Which vessel type runs this route
    frequency: int                        # Sailings per week (1 or 2)
    cycle_days: float                     # Full round-trip cycle time in days
    vessels_needed: int                   # ceil(cycle_days / 7) * frequency
    weekly_operating_cost: float          # TC rate × vessels × 7 + bunker + canal

    # Populated after LP solve
    lp_lambda: float = 0.0               # LP usage fraction (0..1)
    is_selected: bool = False            # After B&B: True = service operates

    # Which OD pairs does this route serve (origin → destination via this route)?
    od_coverage: dict = field(default_factory=dict)  # {(orig,dest): max_ffe}

    @property
    def origin(self) -> str:
        return self.port_sequence[0]

    @property
    def destination(self) -> str:
        return self.port_sequence[-1]

    @property
    def num_ports(self) -> int:
        return len(self.port_sequence)

    def __repr__(self):
        seq = " → ".join(self.port_sequence)
        return f"Route({self.route_id}: {seq} | {self.vessel_class} | {self.frequency}×/wk)"
