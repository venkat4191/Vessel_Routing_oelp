# LinerNet v3 — All 12 Real-World Fixes Applied

## Fix Summary Table

| # | Issue | Files Changed | What Was Done |
|---|-------|--------------|---------------|
| 1 | Transit time never enforced | `route_validator.py`, `stage3/rmp.py` | CHECK 12 in validator rejects routes where zero OD pairs can be served within `max_transit_days`. In RMP LP, variables `f_{od,s}` are forced to 0 (upper bound = 0) when the route's actual sailing time to destination exceeds the demand contract limit. |
| 2 | Frequency × capacity ignored | `stage3/rmp.py` | `capacity_ffe = vessel.capacity_ffe × frequency`. A freq=2 service departs twice per week → 2× weekly throughput. Was previously identical to freq=1. |
| 3 | Circular coverage missing return leg | `stage3/route_validator.py` | CHECK 8 rewritten using doubled-cycle approach. `[A,B,C]` now covers C→A demand too, matching the fix in `reduced_cost.py`. |
| 4 | Cabotage rules not enforced | `stage3/route_validator.py`, `utils/config.py` | CHECK 11 added: rejects any OD pair where origin and destination share a `CABOTAGE_PROTECTED_REGIONS` cabotage region (US Jones Act, Brazil, Australia, China, Japan, India, South Africa). |
| 5 | No slow steaming (fixed design speed) | `stage1/fleet_profiler.py`, `utils/config.py` | `find_optimal_speed()` scans `min_speed → max_speed` in 0.25-knot steps, picks the speed that minimizes `fixed_cost + sailing_fuel + idle_fuel`. Uses admiralty cubic law: `fuel ∝ speed³`. `ENABLE_SLOW_STEAMING = True` in config. |
| 6 | Pre-IMO 2020 HFO pricing ($600) | `utils/config.py`, `stage0/loader.py` | `BUNKER_PRICE_PER_TON = 660` (VLSFO 2024 average). Added `HFO_PRICE_PER_TON = 530` (scrubber vessels), `LNG_PRICE_PER_TON = 700`, `MGO_PRICE_PER_TON = 850`. `VesselRecord.fuel_price()` returns correct price per vessel's fuel type. |
| 7 | Uniform 1-day port dwell for all vessels | `stage1/fleet_profiler.py`, `utils/config.py` | `PORT_DWELL_DAYS_BY_CLASS` dict: Feeder_450=0.5d, Feeder_800=0.75d, Panamax_1200=1.0d, Panamax_2400=1.25d, Post_panamax=1.75d, Super_panamax=2.5d. Larger vessels load more cargo and take longer. Used in all cycle time and idle fuel calculations. |
| 8 | No schedule quantization | `stage1/fleet_profiler.py`, `utils/config.py` | Schedule buffer days added to align `cycle_days` to the `7/frequency` weekly slot grid. Prevents phantom schedules (e.g., a 26.3-day cycle cannot maintain a fixed Monday departure). Buffer ≤ `SCHEDULE_SLACK_DAYS = 0.5d`. |
| 9 | Single fuel type, no dual-fuel/LNG | `stage0/loader.py`, `utils/config.py` | `VesselRecord.fuel_type` field added (defaults to `FUEL_TYPE_VLSFO`). `VesselRecord.fuel_price()` returns correct per-ton price. `VesselRecord.bunker_rate_at_speed()` computes consumption at any speed via cubic law. Config includes all four fuel type constants. |
| 10 | No transshipment storage/dwell cost | `utils/paper_profit.py`, `utils/config.py` | MCF transshipment edge cost = `cost_per_full_trnsf + storage_addon`. Storage addon = `TRANSSHIP_STORAGE_COST_PER_FFE_DAY ($8) × TRANSSHIP_AVG_WAIT_DAYS (3.5) = $28/FFE`. Containers waiting at a hub for the next connecting vessel now incur realistic holding cost. |
| 11 | Alliance/slot-sharing not modeled | `stage3/rmp.py`, `utils/config.py` | `ALLIANCE_ENABLED = False` (off by default). When enabled, `capacity_ffe = vessel_ffe × frequency × ALLIANCE_SLOT_FRACTION`. Config documents the real-world context (Ocean Alliance, 2M, THE Alliance). Academic single-carrier simplification preserved by default. |
| 12 | All revenue treated as spot rate | `stage0/loader.py`, `stage3/rmp.py`, `utils/config.py` | `DemandRecord.effective_revenue_per_ffe` = spot_rate × `BLENDED_REVENUE_FACTOR (0.9025)`. Blended factor = 65% contracts × 85% discount + 35% spot. RMP LP objective now uses blended revenue, not spot rate, for realistic profit calculation. |

## Architecture of Key Files Changed

```
utils/
  config.py          ← 8 new constant groups (Fixes 4,5,6,7,8,9,10,11,12)
  paper_profit.py    ← Fix 10: transshipment storage in MCF edges

stage0/
  loader.py          ← Fix 6,9,12: VLSFO fuel type, blended revenue on DemandRecord

stage1/
  fleet_profiler.py  ← Fix 5,7,8,9: slow steaming optimizer, per-vessel dwell,
                        schedule buffer, fuel_price() for VLSFO/LNG

stage3/
  route_validator.py ← Fix 1,3,4: transit time CHECK 12, circular CHECK 8,
                        cabotage CHECK 11 (complete rewrite)
  rmp.py             ← Fix 1,2,11,12: transit time LP zeroing, freq×capacity,
                        alliance slot fraction, blended revenue objective
  cg_loop.py         ← Patched: rmp.solve(distances=distances)
  exact_pricing.py   ← Patched: rmp.solve(distances=distances)
  agent_oracle.py    ← Patched: rmp.solve(distances=data['distances'])

stage5/
  parallel_eval.py   ← Patched: rmp.solve(distances=distances)
```

## Real-World Accuracy: Before vs After

| Metric | Before | After |
|--------|--------|-------|
| Fuel price | $600/t HFO (pre-2020) | $660/t VLSFO (IMO 2020 compliant) |
| Port dwell | 1.0 day all vessels | 0.5–2.5 days by vessel class |
| Transit time | Not checked anywhere | Enforced in validator + LP |
| Frequency cap | Identical to freq=1 | 2× capacity for freq=2 |
| Return-leg demand | Not counted | Counted via circular coverage |
| Cabotage | Not checked | US, Brazil, AU, China, JP, IN, SA |
| Fuel optimization | Design speed always | Optimal slow-steam speed |
| Transship storage | No holding cost | $28/FFE average dwell at hub |
| Revenue model | 100% spot rate | 90.25% blended (contract+spot) |
