# LinerNet — Agentic Liner Shipping Network Optimizer

Instance: WorldSmall (47 ports, 1,764 OD pairs, 6 vessel types)
Unit: FFE throughout (1 FFE = 1 forty-foot container)
Goal: Maximize weekly net profit subject to 7 hard constraints

## Build Progress
- [x] Stage 0 — Data Foundation  <- CURRENT
- [ ] Stage 1 — Parallel Intelligence
- [ ] Stage 2 — Seed Generator
- [ ] Stage 3 — Column Generation Loop
- [ ] Stage 4 — Branch and Bound
- [ ] Stage 5 — Parallel Evaluation
- [ ] Stage 6 — Warm-Start Refinement

## Quick Run

From project root:

- Full pipeline (uses `.env` key if present):
  - `python3 run_pipeline.py`
- With explicit key:
  - `python3 run_pipeline.py AIza...`
- Desktop UI:
  - `python3 ui/linernet_ui.py`
- Web frontend:
  - `python3 webapp.py`
  - open `http://127.0.0.1:8080`

