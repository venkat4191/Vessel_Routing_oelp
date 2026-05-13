"""
Simple .env helper for LinerNet.
Only reads/writes key-value pairs like:
  GEMINI_API_KEY=AIza...
"""

import os
import importlib.util


def _env_path(project_root: str) -> str:
    return os.path.join(project_root, ".env")


def load_dotenv(project_root: str) -> dict:
    """
    Load .env into a dict and also inject missing keys into os.environ.
    Existing process environment values are not overwritten.
    """
    path = _env_path(project_root)
    values = {}
    if not os.path.isfile(path):
        return values

    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            values[key] = val
            if key and key not in os.environ:
                os.environ[key] = val
    return values


def save_gemini_key(project_root: str, gemini_key: str):
    """
    Save/replace GEMINI_API_KEY in .env.
    """
    path = _env_path(project_root)
    rows = []
    found = False

    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            rows = f.readlines()

    out = []
    for row in rows:
        if row.strip().startswith("GEMINI_API_KEY="):
            out.append(f"GEMINI_API_KEY={gemini_key}\n")
            found = True
        else:
            out.append(row)

    if not found:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append(f"GEMINI_API_KEY={gemini_key}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)

    os.environ["GEMINI_API_KEY"] = gemini_key


def get_gemini_key(project_root: str, explicit_key: str = None) -> str:
    """
    Priority:
      1) explicit_key arg
      2) process env GEMINI_API_KEY
      3) .env GEMINI_API_KEY
    """
    if explicit_key:
        return explicit_key.strip()

    env_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key

    vals = load_dotenv(project_root)
    return vals.get("GEMINI_API_KEY", "").strip()


def get_api_keys(project_root: str, explicit_keys: list[str] | None = None) -> list[str]:
    """
    Return a list of API keys from:
      1) explicit_keys (if provided and non-empty)
      2) process env / .env `GEMINI_API_KEYS` (comma-separated)
      3) process env / .env `GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, ...
      4) fallback to single `GEMINI_API_KEY` (if set)

    This is used for key rotation to avoid per-key rate limits.
    """
    if explicit_keys:
        keys = [str(k).strip() for k in explicit_keys if str(k).strip()]
        return keys

    # 0) local python key list file (project_root/api_keys_store.py)
    store_path = os.path.join(project_root, "api_keys_store.py")
    if os.path.isfile(store_path):
        try:
            spec = importlib.util.spec_from_file_location("api_keys_store", store_path)
            mod = importlib.util.module_from_spec(spec) if spec else None
            if spec and spec.loader and mod is not None:
                spec.loader.exec_module(mod)
                keys = [str(k).strip() for k in getattr(mod, "API_KEYS", []) if str(k).strip()]
                if keys:
                    return keys
        except Exception:
            pass

    vals = load_dotenv(project_root)

    def _get(name: str) -> str:
        v = os.environ.get(name, "").strip()
        if v:
            return v
        return str(vals.get(name, "") or "").strip()

    ring = _get("GEMINI_API_KEYS")
    if ring:
        keys = [k.strip() for k in ring.split(",") if k.strip()]
        if keys:
            return keys

    seq: list[str] = []
    for i in range(1, 51):
        v = _get(f"GEMINI_API_KEY_{i}")
        if v:
            seq.append(v)
    if seq:
        return seq

    one = _get("GEMINI_API_KEY")
    return [one] if one else []

