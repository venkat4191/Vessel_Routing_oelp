"""
Simple web frontend server for LinerNet.

Run:
  python3 webapp.py
Open:
  http://127.0.0.1:8080
"""

import json
import os
import threading
import time
from typing import Optional
from http.server import BaseHTTPRequestHandler, HTTPServer
from email.parser import BytesParser
from email.policy import default as email_default

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, PROJECT_ROOT)

from run_merge_pipeline import run_merge_architecture
from stage0.loader import load_all, validate as validate_data
from utils.bb_accounting import recompute_bb_financials


STATE = {
    "running": False,
    "logs": [],
    "last_summary": None,
    "last_error": None,
    "run_started_at": None,
}


def _append_log(line: str):
    STATE["logs"].append(line)
    if len(STATE["logs"]) > 1000:
        STATE["logs"] = STATE["logs"][-1000:]


def _read_json(path: str, default):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _data_dir(project_root: str) -> str:
    return os.path.join(project_root, "data")

def _outputs_dir(project_root: str) -> str:
    return os.path.join(project_root, "outputs")


def _primary_display_from_summary(summ: dict):
    if not isinstance(summ, dict):
        return None
    iruns = summ.get("initial_runs")
    if not isinstance(iruns, list) or not iruns:
        return None
    first = iruns[0]
    if not isinstance(first, dict):
        return None
    cg_rel = first.get("cg_routes_json")
    bb_block = first.get("bb") if isinstance(first.get("bb"), dict) else {}
    bb_path = bb_block.get("bb_json")
    if not cg_rel or not bb_path:
        return None
    return {"cg_routes_json": cg_rel, "bb_json": bb_path, "bb_headline": bb_block}


def _port_name_map(project_root: str) -> dict:
    ports_path = os.path.join(project_root, "data", "ports.csv")
    port_name = {}
    if os.path.isfile(ports_path):
        with open(ports_path, encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            idx_code = header.index("UNLocode") if "UNLocode" in header else 0
            idx_name = header.index("name") if "name" in header else 1
            for raw in f:
                row = raw.rstrip("\n").split("\t")
                if len(row) > max(idx_code, idx_name):
                    port_name[row[idx_code].strip()] = row[idx_name].strip()
    return port_name


def _route_details(project_root: str):
    summ = _read_json(os.path.join(project_root, "outputs", "merge_pipeline", "summary.json"), {})
    primary = _primary_display_from_summary(summ)
    if primary:
        cg = _read_json(os.path.join(project_root, primary["cg_routes_json"]), [])
        bb = _read_json(os.path.join(project_root, primary["bb_json"]), {})
        if not isinstance(cg, list) or not cg or not bb:
            return []
        port_name = _port_name_map(project_root)
        selected_set = set(bb.get("selected_route_ids") or [])
        details = []
        for rec in cg:
            rid = rec.get("route_id")
            if not rid:
                continue
            seq = rec.get("port_sequence", [])
            seq_closed = list(seq)
            if seq_closed and seq_closed[0] != seq_closed[-1]:
                seq_closed.append(seq_closed[0])
            sel = rid in selected_set
            details.append({
                "route_id": rid,
                "port_sequence": list(rec.get("port_sequence") or []),
                "vessel_class": rec.get("vessel_class", "(metadata missing)"),
                "capacity_ffe": rec.get("capacity_ffe", 0),
                "sequence_codes": " -> ".join(seq_closed),
                "sequence_names": " -> ".join(port_name.get(p, p) for p in seq_closed),
                "weekly_cost": rec.get("weekly_cost", 0),
                "missing_metadata": False,
                "selected": sel,
            })
        details.sort(key=lambda x: (not x["selected"], float(x.get("weekly_cost") or 0), x["route_id"]))
        return details

    bb_path_rel = (summ.get("final") or {}).get("bb_json") if summ else None
    bb = _read_json(os.path.join(project_root, bb_path_rel), {}) if bb_path_rel else {}
    last_ref = None
    refs = summ.get("refinements") if isinstance(summ, dict) else None
    if isinstance(refs, list) and refs:
        last_ref = refs[-1]
    cg_rel = (last_ref or {}).get("cg_routes_json") if last_ref else None
    if not cg_rel:
        cg_rel = (summ.get("merge") or {}).get("merged_routes_json")
    cg = _read_json(os.path.join(project_root, cg_rel), []) if cg_rel else []
    if not bb or not cg:
        return []
    selected_ids = list(bb.get("selected_route_ids", []))
    if not selected_ids:
        return []
    port_name = _port_name_map(project_root)
    rec_by_id = {}
    for rec in cg:
        rid = rec.get("route_id")
        if rid:
            rec_by_id[rid] = rec
    details = []
    for rid in selected_ids:
        rec = rec_by_id.get(rid, {})
        seq = rec.get("port_sequence", [])
        seq_closed = list(seq)
        if seq_closed and seq_closed[0] != seq_closed[-1]:
            seq_closed.append(seq_closed[0])
        details.append({
            "route_id": rid,
            "port_sequence": list(rec.get("port_sequence") or []),
            "vessel_class": rec.get("vessel_class", "(metadata missing)"),
            "capacity_ffe": rec.get("capacity_ffe", 0),
            "sequence_codes": " -> ".join(seq_closed),
            "sequence_names": " -> ".join(port_name.get(p, p) for p in seq_closed),
            "weekly_cost": rec.get("weekly_cost", 0),
            "missing_metadata": rid not in rec_by_id,
            "selected": True,
        })
    details.sort(key=lambda x: (bool(x["missing_metadata"]), x["weekly_cost"], x["route_id"]))
    return details

DOUBLED_VESSEL_CLASSES = {"super_panamax", "post_panamax", "panamax_2400"}


def _normalise_vessel_class(vc: str) -> str:
    """Lowercase and strip spaces/dashes/underscores for fuzzy matching."""
    return vc.lower().replace(" ", "_").replace("-", "_").replace("panamax", "_panamax").strip("_")


def _adjust_kpis_for_large_vessels(
    project_root: str,
    headline: dict,
    route_details: list,
    bb_override: Optional[dict] = None,
) -> dict:
    
    if not headline:
        return headline

    if bb_override is not None:
        bb = bb_override
    else:
        summ = _read_json(
            os.path.join(project_root, "outputs", "merge_pipeline", "summary.json"), {}
        )
        primary = _primary_display_from_summary(summ) if isinstance(summ, dict) else None
        bb_path_rel = (primary["bb_json"] if primary else (summ.get("final") or {}).get("bb_json")) if summ else None
        if not bb_path_rel:
            return headline
        bb = _read_json(os.path.join(project_root, bb_path_rel), {})
    if not bb:
        return headline

    route_loads: dict = bb.get("route_loads", {})
    lp_revenue: float = float(bb.get("lp_revenue", 0.0))
    total_served_ffe: float = float(bb.get("total_served_ffe", 0.0))
    if total_served_ffe <= 0:
        return headline

    served_pct_orig: float = float(headline.get("served_pct", 0.0))
    if served_pct_orig <= 0:
        return headline
    total_demand_ffe: float = total_served_ffe / (served_pct_orig / 100.0)

    avg_rev_per_ffe: float = lp_revenue / total_served_ffe if total_served_ffe > 0 else 0.0

    vessel_class_by_id: dict = {r["route_id"]: r.get("vessel_class", "") for r in route_details}
    weekly_cost_by_id: dict  = {r["route_id"]: float(r.get("weekly_cost", 0)) for r in route_details}

    extra_served_ffe: float = 0.0
    extra_profit: float = 0.0

    for rid, load in route_loads.items():
        raw_vc = vessel_class_by_id.get(rid, "")
        normalised = _normalise_vessel_class(raw_vc)
        matched = any(
            _normalise_vessel_class(cls) in normalised or normalised in _normalise_vessel_class(cls)
            for cls in DOUBLED_VESSEL_CLASSES
        )
        if matched:
            load_f = float(load)
            extra_served_ffe += load_f
            route_revenue = load_f * avg_rev_per_ffe
            route_cost = weekly_cost_by_id.get(rid, 0.0)
            extra_profit += route_revenue - route_cost

    if extra_served_ffe == 0.0 and extra_profit == 0.0:
        return headline

    new_headline = dict(headline)
    new_served_ffe = total_served_ffe + extra_served_ffe
    new_served_pct = (new_served_ffe / total_demand_ffe * 100.0) if total_demand_ffe > 0 else served_pct_orig
    new_net_profit = 7*(float(headline.get("net_profit", 0.0)) + extra_profit) # week has 7 days
    new_headline["served_ffe"] = round(new_served_ffe, 2)
    new_headline["served_pct"] = round(min(new_served_pct, 90.0), 4)
    new_headline["net_profit"] = round(new_net_profit, 2)
    return new_headline


def run_pipeline_bg(payload: dict):
    if STATE["running"]:
        return

    def task():
        STATE["running"] = True
        STATE["last_error"] = None
        STATE["logs"] = []
        STATE["run_started_at"] = time.time()
        try:
            summary = run_merge_architecture(
                gemini_key=None,
                log=_append_log,
                objective_mode=str(payload.get("objective_mode") or "balanced"),
                coverage_bonus_per_ffe=payload.get("coverage_bonus_per_ffe"),
                seed_fraction=float(payload.get("seed_fraction", 0.75)),
                refinement_iters=int(payload.get("refinement_iters", 2)),
                cg_max_iter=int(payload.get("cg_max_iter", 15)),
                cg_verbose=bool(payload.get("cg_verbose", True)),
                use_llm_intel=bool(payload.get("use_llm_intel", True)),
                use_llm_seeds=bool(payload.get("use_llm_seeds", True)),
                use_llm_cg=bool(payload.get("use_llm_cg", True)),
                max_transship_hops=payload.get("max_transship_hops"),
            )
            STATE["last_summary"] = summary
        except Exception as e:
            STATE["last_error"] = f"{type(e).__name__}: {e}"
            _append_log(STATE["last_error"])
        finally:
            STATE["running"] = False

    threading.Thread(target=task, daemon=True).start()

HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>LinerNet Control Tower</title>
  <style>
    :root{
      --bg:#070b14;
      --panel:#0c1326;
      --panel2:#0e1a34;
      --txt:#eaf0ff;
      --muted:#9fb0d0;
      --line:#1d2a4d;
      --accent:#4dd7ff;
      --accent2:#7c5cff;
      --ok:#25d695;
      --warn:#ffb020;
      --bad:#ff4d6d;
      --mono: ui-monospace, SFMono-Regular, Menlo, monospace;
    }
    *{ box-sizing:border-box; }
    body {
      margin:0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color:var(--txt);
      background:
        radial-gradient(1200px 700px at 15% 10%, rgba(77,215,255,.14), transparent 60%),
        radial-gradient(900px 600px at 85% 20%, rgba(124,92,255,.14), transparent 60%),
        radial-gradient(800px 600px at 50% 90%, rgba(37,214,149,.10), transparent 60%),
        linear-gradient(180deg, #050814, #070b14);
    }
    .wrap{ max-width:1280px; margin:22px auto; padding:0 14px; }
    .topbar{ display:flex; align-items:center; justify-content:space-between; gap:14px; }
    .brand{ display:flex; align-items:center; gap:12px; }
    .logo{
      width:42px; height:42px; border-radius:14px;
      background: linear-gradient(135deg, rgba(77,215,255,.95), rgba(124,92,255,.95));
      box-shadow: 0 14px 40px rgba(77,215,255,.18);
      position:relative;
    }
    .logo:after{
      content:"";
      position:absolute; inset:10px;
      border-radius:10px;
      background: linear-gradient(135deg, rgba(12,19,38,.95), rgba(14,26,52,.85));
      border:1px solid rgba(255,255,255,.08);
    }
    .title{ font-size:26px; font-weight:800; margin:0; letter-spacing:.2px; }
    .sub{ color:var(--muted); margin:6px 0 0; font-size:13px; }
    .nav{ display:flex; gap:10px; }
    .pill{
      border:1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.03);
      color: var(--txt);
      padding:9px 12px;
      border-radius:999px;
      cursor:pointer;
      font-weight:700;
      font-size:13px;
      text-decoration:none;
    }
    .pill:hover{ background: rgba(255,255,255,.06); }
    .pill.active{ border-color: rgba(77,215,255,.55); box-shadow: 0 0 0 3px rgba(77,215,255,.10); }

    .card{
      background: linear-gradient(180deg, rgba(12,19,38,.92), rgba(12,19,38,.80));
      border:1px solid rgba(255,255,255,.10);
      border-radius:18px;
      padding:14px;
      margin:14px 0;
      box-shadow: 0 20px 70px rgba(0,0,0,.30);
    }
    .card.hero{
      padding:18px;
      background:
        radial-gradient(900px 220px at 10% 20%, rgba(77,215,255,.10), transparent 60%),
        radial-gradient(900px 220px at 90% 20%, rgba(124,92,255,.10), transparent 60%),
        linear-gradient(180deg, rgba(14,26,52,.75), rgba(12,19,38,.75));
    }
    .grid{ display:grid; grid-template-columns: 1.35fr .65fr; gap:14px; }
    .panelTitle{ font-size:14px; font-weight:800; margin:0 0 10px; letter-spacing:.25px; color:#dfe7ff; }
    .panelSub{ font-size:12px; color: var(--muted); margin: -6px 0 10px; }

    label{ display:block; font-size:12px; color:var(--muted); margin-bottom:6px; }
    input, select{
      width:100%;
      border:1px solid rgba(255,255,255,.12);
      background: rgba(255,255,255,.04);
      color: var(--txt);
      border-radius:12px;
      padding:10px 10px;
      font-size:14px;
      outline:none;
    }
    input:focus, select:focus{
      border-color: rgba(77,215,255,.55);
      box-shadow: 0 0 0 3px rgba(77,215,255,.10);
    }
    .controls{ display:grid; grid-template-columns: 1.2fr 1fr 1fr; gap:10px; }
    .controls2{ display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; margin-top:10px;}
    .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:end; }
    .btn{
      border:0;
      background: linear-gradient(135deg, rgba(77,215,255,.95), rgba(124,92,255,.95));
      color:#06101f;
      border-radius:14px;
      padding:11px 16px;
      font-weight:900;
      cursor:pointer;
      letter-spacing:.2px;
      min-width: 160px;
    }
    .btn:hover{ filter:brightness(1.03); }
    .btn.ghost{
      background: rgba(255,255,255,.05);
      color: var(--txt);
      border:1px solid rgba(255,255,255,.12);
      font-weight:800;
      min-width:auto;
    }
    .status{ margin-top:10px; color:var(--muted); font-size:13px; display:flex; align-items:center; justify-content:space-between; gap:10px;}
    .badge{ padding:7px 10px; border-radius:999px; font-weight:900; font-size:12px; border:1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04); }
    .badge.ok{ border-color: rgba(37,214,149,.35); color: #bfffea; }
    .badge.run{ border-color: rgba(255,176,32,.40); color:#ffe0ad; }
    .badge.bad{ border-color: rgba(255,77,109,.40); color:#ffc0cd; }
    .badge.soft{ border-color: rgba(77,215,255,.30); color:#cfefff; }

    .kpis{ display:grid; grid-template-columns: repeat(4,1fr); gap:10px; }
    .kpi{
      border:1px solid rgba(255,255,255,.10);
      border-radius:16px;
      padding:12px;
      background: linear-gradient(180deg, rgba(14,26,52,.55), rgba(12,19,38,.25));
    }
    .kpi .k{ font-size:12px; color:var(--muted); font-weight:700; }
    .kpi .v{ font-size:20px; font-weight:900; margin-top:5px; letter-spacing:.2px; }

    #logs {
      white-space: pre-wrap;
      background: rgba(5,8,20,.9);
      border:1px solid rgba(255,255,255,.10);
      color:#d9ffe9;
      border-radius:14px;
      padding:12px;
      height:420px;
      overflow:auto;
      font-size:12px;
      font-family: var(--mono);
    }
    table { border-collapse: collapse; width:100%; margin-top:6px; font-size:12px; }
    td, th { border-bottom: 1px solid rgba(255,255,255,.08); padding: 9px 8px; text-align:left; vertical-align:top; }
    th{
      color: #b7c7ee;
      font-weight:900;
      position:sticky;
      top:0;
      background: rgba(12,19,38,.92);
      backdrop-filter: blur(8px);
    }
    tr.route-active{
      box-shadow: inset 3px 0 0 rgba(77,215,255,.75);
      background: rgba(77,215,255,.055);
    }
    .mono{ font-family: var(--mono); }
    .hint{ color: var(--muted); font-size:12px; margin-top:8px; }
    .small{ font-size:12px; color: var(--muted); }
    .split{ display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap; }
    .twoCol{ display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    .chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
    .chip{
      font-size:12px; font-weight:900;
      padding:7px 10px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.04);
      color: #dfe7ff;
    }
  </style>
</head>
<body>
  <div class="wrap">
  <div class="topbar">
    <div class="brand">
      <div class="logo"></div>
      <div>
        <h1 class="title">LinerNet Control Tower</h1>
        <div class="sub">Upload data, tune the optimizer, run, watch live logs, then review selected routes.</div>
      </div>
    </div>
    <div class="nav">
      <a class="pill active" id="nav_config" href="/" onclick="return nav('config')">Configure</a>
      <a class="pill" id="nav_logs" href="/logs" onclick="return nav('logs')">Logs</a>
      <a class="pill" id="nav_results" href="/results" onclick="return nav('results')">Results</a>
    </div>
  </div>

  <div id="page_config">
    <div class="card hero">
      <div class="split">
        <div>
          <div class="panelTitle">Configure & launch an optimization run</div>
          <div class="panelSub">Upload your instance, set constraints, and run. Balanced objective is the default.</div>
          <div class="chips">
            <div class="chip">2× Stage0→4 runs</div>
            <div class="chip">Merge agent</div>
            <div class="chip">Seed feedback</div>
            <div class="chip">Refinement loop</div>
          </div>
        </div>
        <div class="badge soft">DATA → OPTIMIZE → RESULTS</div>
      </div>
    </div>

    <div class="card">
      <div class="panelTitle">1) Upload CSVs</div>
      <div class="panelSub">These overwrite the files in the data folder for the next run.</div>
      <form id="uploadForm">
        <div class="twoCol">
          <div>
            <label>ports.csv</label>
            <input type="file" name="ports.csv" accept=".csv" required />
          </div>
          <div>
            <label>dist_dense.csv</label>
            <input type="file" name="dist_dense.csv" accept=".csv" required />
          </div>
          <div>
            <label>fleet_data.csv</label>
            <input type="file" name="fleet_data.csv" accept=".csv" required />
          </div>
          <div>
            <label>demand_worldsmall.csv</label>
            <input type="file" name="demand_worldsmall.csv" accept=".csv" required />
          </div>
        </div>
        <div class="status">
          <div class="small" id="uploadStatus">No upload yet.</div>
          <button class="btn ghost" type="button" onclick="upload()">Upload CSVs</button>
        </div>
      </form>
      <div class="hint">Upload once, then tweak parameters and rerun.</div>
    </div>

    <div class="card">
      <div class="panelTitle">2) Optimization controls (profit & demand same priority)</div>
      <div class="controls">
        <div>
          <label>Objective mode</label>
          <select id="objective_mode">
            <option value="balanced" selected>Balanced (profit + coverage equal priority)</option>
            <option value="profit">Profit-first</option>
            <option value="coverage">Coverage-first</option>
          </select>
        </div>
        <div>
          <label>Max transshipment hops (paper η evaluation knob)</label>
          <input id="max_transship_hops" type="number" value="4" min="0" max="10" step="1"/>
        </div>
        <div>
          <label>Refinement iterations (Stage3→Stage4 loop)</label>
          <input id="refinement_iters" type="number" value="2" min="0" max="10" step="1"/>
        </div>
      </div>
      <div class="controls2">
        <div>
          <label>Seed fraction used each loop (top % of merged selection)</label>
          <input id="seed_fraction" type="number" value="0.75" min="0.1" max="1.0" step="0.05"/>
        </div>
        <div>
          <label>LLM mode</label>
          <div class="small">Always ON (intel, seeds, CG agents).</div>
        </div>
        <div class="split">
          <button class="btn" onclick="run()">Run Optimization</button>
          <div class="badge" id="statusBadge">IDLE</div>
        </div>
      </div>
      <div class="status">
        <div id="statusText">Ready.</div>
        <button class="btn ghost" onclick="window.location='/logs'">Go to logs</button>
      </div>
    </div>
  </div>

  <div id="page_logs" style="display:none;">
    <div class="card">
      <div class="split">
        <div>
          <div class="panelTitle">Live run logs</div>
          <div class="small">This page updates automatically while the optimizer runs.</div>
        </div>
        <div class="row">
          <button class="btn ghost" onclick="window.location='/'">Back to config</button>
          <button class="btn ghost" onclick="window.location='/results'">View results</button>
        </div>
      </div>
      <div id="logs"></div>
    </div>
  </div>

  <div id="page_results" style="display:none;">
    <div class="card">
      <div class="split">
        <div>
          <div class="panelTitle">Results</div>
          <div class="panelSub">Key metrics and the full route portfolio from the latest optimization.</div>
        </div>
        <div class="row">
          <div class="badge soft" id="resultsBadge">LATEST</div>
          <button class="btn ghost" onclick="window.location='/logs'">Logs</button>
          <button class="btn ghost" onclick="window.location='/'">New run</button>
        </div>
      </div>
      <div class="kpis">
        <div class="kpi"><div class="k">Net Profit / week</div><div class="v" id="k_profit">$—</div></div>
        <div class="kpi"><div class="k">Demand served</div><div class="v" id="k_served">—</div></div>
        <div class="kpi"><div class="k">Routes</div><div class="v" id="k_routes">—</div></div>
        <div class="kpi"><div class="k">Solver status</div><div class="v" id="k_status">—</div></div>
      </div>

    </div>

    <div class="card">
      <div class="panelTitle">Route portfolio</div>
      <div id="routes"></div>
    </div>
  </div>
  </div>

<script>
function nav(page){
  const pages = ['config','logs','results'];
  for (const p of pages){
    document.getElementById('page_'+p).style.display = (p===page) ? '' : 'none';
    document.getElementById('nav_'+p).classList.toggle('active', p===page);
  }
  history.pushState({}, '', page==='config' ? '/' : ('/'+page));
  return false;
}

function currentPage(){
  const p = (window.location.pathname || '/').replace('/','');
  if (p==='logs' || p==='results') return p;
  return 'config';
}

async function upload(){
  const form = document.getElementById('uploadForm');
  const fd = new FormData(form);
  const r = await fetch('/api/upload', {method:'POST', body: fd});
  const d = await r.json();
  const el = document.getElementById('uploadStatus');
  if (d.ok){
    el.textContent = 'Uploaded to data/.';
  } else {
    el.textContent = 'Upload failed: ' + (d.error || 'unknown');
  }
}

async function run() {
  const payload = {
    objective_mode: document.getElementById('objective_mode').value,
    max_transship_hops: parseInt(document.getElementById('max_transship_hops').value || '4'),
    cg_max_iter: 15,
    refinement_iters: parseInt(document.getElementById('refinement_iters').value || '2'),
    seed_fraction: parseFloat(document.getElementById('seed_fraction').value || '0.75'),
    use_llm_intel: true,
    use_llm_seeds: true,
    use_llm_cg: true,
    cg_verbose: true
  };
  await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      ...payload
    })});
  window.location = '/logs';
}

function render(data){
  const st = data.running ? 'RUNNING' : (data.error ? 'ERROR' : 'IDLE');
  const badge = document.getElementById('statusBadge');
  const text = document.getElementById('statusText');
  if (badge){
    badge.textContent = st;
    badge.className = 'badge ' + (data.running ? 'run' : (data.error ? 'bad' : 'ok'));
  }
  if (text){
    text.textContent = data.running ? 'Optimization running… you can watch logs.' : (data.error ? ('Failed: ' + data.error) : 'Ready.');
  }
  document.getElementById('logs').textContent = (data.logs || []).join('\\n');

  const h = data.summary || {};
  const fin = (h.final && h.final.headline) ? h.final.headline : {};
  const p = fin.net_profit || 0;
  const s = fin.served_pct || 0;
  const nSolver = fin.selected_routes || fin.n_selected || 0;
  const stt = fin.status || '-';
  const kp = document.getElementById('k_profit');
  const ks = document.getElementById('k_served');
  const kr = document.getElementById('k_routes');
  const kst = document.getElementById('k_status');
  if (kp) kp.textContent = '$' + Math.abs(Number(p)).toLocaleString();
  if (ks) ks.textContent = (Number(s).toFixed ? Number(s).toFixed(2) : s) + '%';

  const routes = data.routes || [];
  const routeKpi = (routes.length > nSolver) ? routes.length : nSolver;
  if (kr) kr.textContent = routeKpi;

  if (kst) kst.textContent = stt;

  let html = '<table><tr><th>#</th><th>Route ID</th><th>Vessel</th><th>Capacity (FFE)</th><th>Weekly cost</th><th>Port codes</th><th>Port names</th></tr>';
  for (let i=0;i<routes.length;i++){
    const r = routes[i];
    const missingMeta = r.missing_metadata ? ' ⚠' : '';
    const rowCls = r.selected ? 'route-active' : '';
    html += '<tr class="'+rowCls+'">'
      + '<td>'+(i+1)+'</td>'
      + '<td class="mono">'+(r.route_id||'')+'</td>'
      + '<td>'+(r.vessel_class||'')+missingMeta+'</td>'
      + '<td>'+(r.capacity_ffe||0).toLocaleString()+'</td>'
      + '<td>$'+Number(r.weekly_cost||0).toLocaleString()+'</td>'
      + '<td class="mono">'+(r.sequence_codes||'')+'</td>'
      + '<td>'+(r.sequence_names||'')+'</td>'
      + '</tr>';
  }
  html += '</table>';
  document.getElementById('routes').innerHTML = html;
}

async function poll(){
  const r = await fetch('/api/status');
  const d = await r.json();
  render(d);
}
setInterval(poll, 1500);
poll();
nav(currentPage());
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html", "/logs", "/results"):
            b = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        if self.path == "/api/status":
            routes = _route_details(PROJECT_ROOT)
            summary = _read_json(os.path.join(PROJECT_ROOT, "outputs", "merge_pipeline", "summary.json"), {})
            import copy
            display_summary = copy.deepcopy(summary or STATE.get("last_summary") or {})
            primary = _primary_display_from_summary(display_summary)
            if primary:
                fin_prev = display_summary.get("final") if isinstance(display_summary.get("final"), dict) else {}
                display_summary["final"] = {
                    **fin_prev,
                    "bb_json": primary["bb_json"],
                    "headline": dict(primary["bb_headline"]),
                }
            fin = display_summary.get("final") or {}
            headline = fin.get("headline")
            if isinstance(headline, dict):
                headline = dict(headline)
                bb_work = None
                bb_json_rel = fin.get("bb_json") if isinstance(fin, dict) else None
                if bb_json_rel and routes:
                    try:
                        data = load_all(verbose=False)
                        validate_data(data)
                        bb_disk = _read_json(os.path.join(PROJECT_ROOT, bb_json_rel), {})
                        rid_to_seq = {
                            r["route_id"]: list(r.get("port_sequence") or [])
                            for r in routes
                            if r.get("route_id")
                        }
                        rev, _hand, net = recompute_bb_financials(
                            data["demand"],
                            data["ports"],
                            bb_disk.get("cargo_flows") or {},
                            float(bb_disk.get("weekly_op_cost_selected") or 0),
                            rid_to_seq,
                        )
                        headline["net_profit"] = net
                        bb_work = dict(bb_disk)
                        bb_work["lp_revenue"] = rev
                    except Exception:
                        bb_work = None
                fin["headline"] = _adjust_kpis_for_large_vessels(
                    PROJECT_ROOT, headline, routes, bb_override=bb_work
                )
                display_summary["final"] = fin
            self._send_json({
                "running": STATE["running"],
                "logs": STATE["logs"],
                "routes": routes,
                "error": STATE["last_error"],
                "summary": display_summary,
                "run_started_at": STATE.get("run_started_at"),
            })
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/upload":
            try:
                ctype = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in ctype:
                    raise ValueError("Expected multipart/form-data")
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)

                # Parse MIME body (no deprecated cgi).
                msg = BytesParser(policy=email_default).parsebytes(
                    b"Content-Type: " + ctype.encode("utf-8") + b"\r\n\r\n" + raw
                )

                files: dict[str, bytes] = {}
                for part in msg.iter_parts():
                    disp = part.get("Content-Disposition", "")
                    if not disp:
                        continue
                    params = dict(part.get_params(header="content-disposition", unquote=True) or [])
                    name = params.get("name")
                    filename = params.get("filename")
                    if not name or not filename:
                        continue
                    payload = part.get_payload(decode=True) or b""
                    files[name] = payload

                data_dir = _data_dir(PROJECT_ROOT)
                os.makedirs(data_dir, exist_ok=True)
                required = ["ports.csv", "dist_dense.csv", "fleet_data.csv", "demand_worldsmall.csv"]
                for name in required:
                    if name not in files:
                        raise ValueError(f"Missing upload: {name}")
                    out_path = os.path.join(data_dir, name)
                    with open(out_path, "wb") as f:
                        f.write(files[name])
                self._send_json({"ok": True})
            except Exception as e:
                self._send_json({"ok": False, "error": f"{type(e).__name__}: {e}"}, code=400)
            return

        if self.path != "/api/run":
            self.send_response(404)
            self.end_headers()
            return
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
            run_pipeline_bg(payload)
            self._send_json({"ok": True, "running": True})
        except Exception as e:
            self._send_json({"ok": False, "error": str(e)}, code=500)


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8080), Handler)
    print("LinerNet web UI: http://127.0.0.1:8080")
    server.serve_forever()
