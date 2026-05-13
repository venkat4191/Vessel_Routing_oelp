"""
Global constants and parameters for LinerNet.
All stages import from here — never hardcode values in stage files.
"""
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")

PORTS_FILE   = os.path.join(DATA_DIR, "ports.csv")
DIST_FILE    = os.path.join(DATA_DIR, "dist_dense.csv")
FLEET_FILE   = os.path.join(DATA_DIR, "fleet_data.csv")
DEMAND_FILE  = os.path.join(DATA_DIR, "demand_worldsmall.csv")

# Instance
INSTANCE_NAME = "WorldSmall"

# Column Generation
MAX_TRANSSHIP_HOPS   = 4
MAX_CG_ITERATIONS    = 100
NUM_SEED_ROUTES      = 8
NUM_LLM_AGENTS       = 5

# Peak season
PEAK_DEMAND_FACTOR   = 1.30

# Solver
BB_MIP_GAP           = 0.02
CG_RC_TOLERANCE      = 1e-4

# Stage 4 filter
STAGE4_ROUTE_MIN_PORTS = None
STAGE4_ROUTE_MAX_PORTS = None

STAGE4_MAX_VESSELS_PER_CLASS = {
    "Feeder_450": 24.11,
    "Feeder_800": 39.10,
    "Panamax_1200": 70.40,
    "Panamax_2400": 79.19,
    "Post_panamax": 58.43,
    "Super_panamax": 11.07
}

# ── FIX 6: Fuel Pricing — IMO 2020 VLSFO (replaces HFO $600) ─────────────────
# Since Jan 2020, global sulfur cap (IMO 2020) mandates VLSFO (<0.5% sulfur)
# unless vessels have scrubbers. VLSFO trades at ~$650–750/ton vs HFO ~$500–600.
# Updated from $600 (pre-2020 HFO) to $660 (VLSFO 2024 average).
BUNKER_PRICE_PER_TON = 660        # USD/metric ton — VLSFO (IMO 2020 compliant)
HFO_PRICE_PER_TON    = 530        # USD/metric ton — HFO (only for scrubber-equipped vessels)
LNG_PRICE_PER_MMBTU  = 12.0       # USD/MMBTU — LNG for dual-fuel vessels
LNG_PRICE_PER_TON    = 700        # USD/metric ton LNG equivalent
MGO_PRICE_PER_TON    = 850        # USD/metric ton — MGO (used in emission control areas)

# Fuel type constants used in VesselRecord.fuel_type
FUEL_TYPE_VLSFO = "VLSFO"         # Default: IMO 2020 compliant low-sulfur HFO
FUEL_TYPE_HFO   = "HFO"           # Scrubber-equipped vessels only
FUEL_TYPE_LNG   = "LNG"           # Dual-fuel LNG vessels (newer newbuilds)
FUEL_TYPE_MDO   = "MDO"           # Marine Diesel Oil (small feeders in port zones)

# ── FIX 7: Port Dwell Time — per vessel class (hours at port per call) ─────────
# Real dwell time depends on port throughput, berth availability, cargo volume.
# Approximated per vessel size class (1 day = 1.0 in PORT_DAYS_PER_CALL units).
# Formula: dwell_days = BASE + SIZE_FACTOR
# Uniform 1.0 day was the old default — now per-vessel-class.
PORT_DWELL_DAYS_BY_CLASS = {
    "Feeder_450":   0.50,   # Small feeder: half day at port (fast turnaround)
    "Feeder_800":   0.75,   # Regional feeder: 18 hours
    "Panamax_1200": 1.00,   # Panamax: standard 1 day
    "Panamax_2400": 1.25,   # Large panamax: 30 hours (more cargo to handle)
    "Post_panamax": 1.75,   # Post-panamax: almost 2 days (4200 FFE load/unload)
    "Super_panamax": 2.50,  # Super-panamax: 2.5 days (7500 FFE — very large port operation)
}
PORT_DAYS_PER_CALL = 1.0  # Legacy fallback if vessel class not in dict above

# ── Port Escort / Tug Fees ────────────────────────────────────────────────────
PORT_ESCORT_FEE_FRACTION = 0.15  # ~15% of port_call_cost_fixed per vessel call

# ── Transshipment Cost Fallback (professor's ±20% rule) ───────────────────────
TRANSSHIP_COST_FALLBACK_RATIO = 1.0
TRANSSHIP_COST_MAX_RATIO      = 2.0

# ── FIX 10: Transshipment Storage / Dwell Cost ───────────────────────────────
# Containers waiting at a hub for the next connecting service incur storage fees.
# Real-world: $5–15/FFE/day at major hubs (Singapore, Rotterdam, etc.)
# Average wait = half of the connecting service frequency = 3.5 days for weekly service.
# Total: ~3.5 days × $8/FFE/day = $28/FFE added to transshipment cost.
TRANSSHIP_STORAGE_COST_PER_FFE_DAY = 8.0   # USD/FFE/day storage at hub
TRANSSHIP_AVG_WAIT_DAYS             = 3.5   # days waiting for next connecting vessel

# ── FIX 5: Slow Steaming — Speed Optimization ────────────────────────────────
# Fuel consumption scales approximately with speed^BUNKER_SPEED_EXPONENT.
# Real-world exponent is ~2.7–3.0 (cubic relationship).
# Slow steaming can cut fuel cost by 30–50% at the cost of longer cycle times.
BUNKER_SPEED_EXPONENT   = 3.0    # fuel ∝ speed^3 (admiralty law approximation)
ENABLE_SLOW_STEAMING    = True   # If True, route_economics optimizes speed

# ── FIX 8: Schedule Quantization ─────────────────────────────────────────────
# Real liner schedules depart on fixed weekdays (e.g., Shanghai every Monday).
# Cycle must be a multiple of 7/frequency days to maintain fixed-day schedule.
# We allow ±SCHEDULE_SLACK_DAYS tolerance before the optimizer adds a buffer day.
SCHEDULE_SLACK_DAYS = 0.5        # tolerance in fractional days for schedule alignment

# ── FIX 4: Cabotage — Protected Regions ──────────────────────────────────────
# Ports sharing these cabotage_region values may not be served by the same
# foreign-flagged vessel consecutively (cargo between them = domestic trade).
# Source: LINERLIB / real-world trade law.
CABOTAGE_PROTECTED_REGIONS = {
    "United States",     # Jones Act — US domestic cargo requires US-flagged vessels
    "China",             # China coastal cabotage law
    "Brazil",            # Brazilian cabotage (flagged-in-Brazil requirement)
    "Australia",         # Australian coastal trading act
    "Japan",             # Japanese cabotage rules
    "India",             # Indian coastal shipping rules
    "South Africa",      # SA cabotage regulations (in force since 2017)
}

# ── FIX 12: Revenue Tiers — Contract vs Spot Rates ────────────────────────────
# ~65% of liner cargo moves on long-term service contracts at lower but stable rates.
# ~35% is spot market. Contract rate is typically 80–90% of spot rate.
# We model this as: effective_revenue = CONTRACT_FRACTION × (rate × CONTRACT_DISCOUNT)
#                                     + SPOT_FRACTION    × rate
# Net blended revenue = rate × (CONTRACT_FRACTION × CONTRACT_DISCOUNT + SPOT_FRACTION)
CONTRACT_DEMAND_FRACTION = 0.65   # share of demand on long-term contracts
SPOT_DEMAND_FRACTION     = 0.35   # share of demand on spot market
CONTRACT_RATE_DISCOUNT   = 0.85   # contract rate = 85% of spot rate
# Blended effective revenue multiplier: 0.65*0.85 + 0.35*1.0 = 0.9025
BLENDED_REVENUE_FACTOR = CONTRACT_DEMAND_FRACTION * CONTRACT_RATE_DISCOUNT + SPOT_DEMAND_FRACTION

# ── FIX 11: Alliance / Slot-Sharing (Acknowledged Simplification) ─────────────
# In reality, ~95% of liner capacity operates under vessel-sharing agreements
# within 3 alliances (Ocean Alliance, 2M, THE Alliance). Carriers co-deploy
# vessels and sell slots on each other's services.
# This model assumes SINGLE-CARRIER operation — a known academic simplification
# standard in LSNDP literature (Brouer et al. 2014, Dutta et al. 2024).
# To model alliances, each service would have ALLIANCE_SLOT_FRACTION of capacity
# available to the optimizing carrier; the rest is sold to partners.
ALLIANCE_ENABLED       = False   # Set True to activate partial-capacity constraint
ALLIANCE_SLOT_FRACTION = 1.0     # Fraction of vessel capacity available to this carrier
# When ALLIANCE_ENABLED=True, effective capacity = capacity_ffe × ALLIANCE_SLOT_FRACTION

# Penalties
UNMET_DEMAND_PENALTY = 100_000

# Paper / LINERLIB-style
PAPER_REJECTION_PENALTY_PER_FFE = 1000.0
PAPER_FLEET_AVAILABLE = None

# Convergence
MIN_PROFIT_IMPROVEMENT_PCT = 1.0

# Stage 5
PARALLEL_EVAL_MAX_WORKERS = 4

# Stage 6
WARM_START_MAX_ROUNDS     = 6
WARM_START_CG_MAX_ITER    = 55
WARM_START_TARGET_COVERAGE_PCT = 58.0
WARM_START_TOP_K_EXACT    = 28
