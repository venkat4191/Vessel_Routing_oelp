"""
utils/llm_client.py — Unified LLM Client
=========================================
Supports Google Gemini, Anthropic Claude, and OpenAI ChatGPT.
Auto-detects provider from API key format:

  AIza...      → Google Gemini     (tries models newest→oldest)  ← ACTIVE
  sk-ant-...   → Anthropic Claude  (claude-sonnet-4-6)
  sk-...       → OpenAI ChatGPT    (gpt-4o-mini)
  sk-proj-...  → OpenAI ChatGPT    (gpt-4o-mini)

Key rotation:
  call_llm() accepts an optional `key_ring` list.
  When provided, it sticks to the current key and only advances to
  the next one when a 429 (rate-limit) response is received.
  This means each key is fully exhausted before moving to the next.
"""

import json
import threading
import urllib.request
import urllib.error
from typing import Optional, List


# ── Provider detection ────────────────────────────────────────────────────────

def detect_provider(api_key: str) -> str:
    """Returns 'gemini' | 'anthropic' | 'openai' | 'unknown'"""
    if not api_key:
        return 'unknown'
    if api_key.startswith('AIza'):
        return 'gemini'
    if api_key.startswith('sk-ant-'):
        return 'anthropic'
    if api_key.startswith('sk-'):
        return 'openai'
    return 'unknown'


def provider_label(api_key: str) -> str:
    return {
        'gemini':    'Gemini (Google)',
        'anthropic': 'Claude (Anthropic)',
        'openai':    'GPT-4o-mini (OpenAI)',
        'unknown':   'Unknown provider',
    }[detect_provider(api_key)]


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_json(text: str) -> Optional[dict]:
    """
    Robustly extract JSON from LLM response.
    Handles markdown fences, leading/trailing prose, pretty printing.
    """
    if not text:
        return None

    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            candidate = part.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except Exception:
                continue

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except Exception:
        pass

    start = stripped.find("{")
    end   = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(stripped[start:end + 1])
        except Exception:
            pass

    start = stripped.find("[")
    end   = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(stripped[start:end + 1])
            if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
                return arr[0]
            return arr
        except Exception:
            pass

    return None


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _call_anthropic_single(prompt: str, api_key: str) -> Optional[dict]:
    """Try one Anthropic key. Raises HTTPError on failure."""
    payload = json.dumps({
        "model":      "claude-sonnet-4-6",
        "max_tokens": 4096,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data    = payload,
        headers = {
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    text = data["content"][0]["text"].strip()
    return _parse_json(text)


def _call_anthropic(prompt: str, api_key: str,
                    key_ring: List[str] = None,
                    verbose: bool = False) -> Optional[dict]:
    """Try each Anthropic key in ring; rotate on 429, stop on 401/403."""
    ring = [k for k in (key_ring or []) if k and k.startswith("sk-ant-")]
    if not ring:
        ring = [api_key] if api_key else []
    if not ring:
        return None

    for idx, key in enumerate(ring):
        try:
            result = _call_anthropic_single(prompt, key)
            if verbose:
                print(f"  ✓ Anthropic key {idx+1}/{len(ring)} (...{key[-4:]})")
            return result
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                msg = json.loads(body).get("error", {}).get("message", body)
            except Exception:
                msg = body[:150]
            if e.code == 429:
                if verbose:
                    print(f"  ↻ Anthropic 429 on key {idx+1} → trying next key")
                continue
            elif e.code in (400, 401, 403):
                if verbose:
                    print(f"  ✗ Anthropic auth error (HTTP {e.code}): {msg}")
                return None
            else:
                if verbose:
                    print(f"  ✗ Anthropic HTTP {e.code}: {msg}")
                return None
    if verbose:
        print("  ✗ All Anthropic keys exhausted")
    return None


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _call_openai_single(prompt: str, api_key: str) -> Optional[dict]:
    """Try one OpenAI key. Raises HTTPError on failure."""
    payload = json.dumps({
        "model":       "gpt-4o-mini",
        "max_tokens":  4096,
        "temperature": 0.2,
        "messages":    [
            {
                "role":    "system",
                "content": "You are a JSON-only assistant. Always respond with valid JSON and nothing else. No markdown, no explanation, no prose."
            },
            {"role": "user", "content": prompt}
        ],
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data    = payload,
        headers = {
            "Authorization": f"Bearer {api_key}",
            "content-type":  "application/json",
        },
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    text = data["choices"][0]["message"]["content"].strip()
    return _parse_json(text)


def _call_openai(prompt: str, api_key: str,
                 key_ring: List[str] = None,
                 verbose: bool = False) -> Optional[dict]:
    """Try each OpenAI key in ring; rotate on 429, stop on 401/403."""
    ring = [k for k in (key_ring or []) if k and k.startswith("sk-") and not k.startswith("sk-ant-")]
    if not ring:
        ring = [api_key] if api_key else []
    if not ring:
        return None

    for idx, key in enumerate(ring):
        try:
            result = _call_openai_single(prompt, key)
            if verbose:
                print(f"  ✓ OpenAI key {idx+1}/{len(ring)} (...{key[-4:]})")
            return result
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            try:
                msg = json.loads(body).get("error", {}).get("message", body)
            except Exception:
                msg = body[:150]
            if e.code == 429:
                if verbose:
                    print(f"  ↻ OpenAI 429 on key {idx+1} → trying next key")
                continue
            elif e.code in (400, 401, 403):
                if verbose:
                    print(f"  ✗ OpenAI auth error (HTTP {e.code}): {msg}")
                return None
            else:
                if verbose:
                    print(f"  ✗ OpenAI HTTP {e.code}: {msg}")
                return None
    if verbose:
        print("  ✗ All OpenAI keys exhausted")
    return None


# ── Gemini ────────────────────────────────────────────────────────────────────

GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]


def _call_gemini_model(prompt: str, api_key: str, model: str,
                       verbose: bool = False) -> Optional[dict]:
    """Try one specific Gemini model. Raises HTTPError on failure."""
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":     0.1,
            "maxOutputTokens": 8192,
        },
    }).encode()

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    req = urllib.request.Request(
        url,
        data    = payload,
        headers = {"content-type": "application/json"},
    )
    resp  = urllib.request.urlopen(req, timeout=45)
    data  = json.loads(resp.read())

    candidates = data.get("candidates", [])
    if not candidates:
        feedback = data.get("promptFeedback", {})
        raise ValueError(f"No candidates — blocked? feedback={feedback}")

    finish = candidates[0].get("finishReason", "")
    parts  = candidates[0].get("content", {}).get("parts", [])
    if not parts:
        raise ValueError(f"Empty content parts (finishReason={finish})")

    raw_text = parts[0].get("text", "").strip()
    if not raw_text:
        raise ValueError(f"Empty text in response (finishReason={finish})")

    if verbose:
        preview = raw_text[:200].replace('\n', '\\n')
        print(f"     Raw response preview: {preview!r}")

    result = _parse_json(raw_text)
    if result is None and verbose:
        print(f"     ⚠ Could not parse above text as JSON")

    return result


def _call_gemini(prompt: str, api_key: str, key_ring: List[str] = None,
                 verbose: bool = False) -> Optional[dict]:
    """
    Try Gemini models in order until one succeeds.

    Key rotation strategy (correct behaviour):
      - Use keys ONE AT A TIME — stick to the current key across all calls.
      - Only move to the next key when a 429 (rate-limit) is received.
      - On 429, try every remaining key before giving up on this model.
      - 400/401/403 = auth/bad-request — stop immediately, no rotation needed.

    This means key1 handles ALL calls until it hits its daily/minute limit,
    then key2 takes over, then key3, etc.
    """
    # Build ordered list: current key first, then rest of ring
    ring: List[str] = []
    if key_ring:
        ring = [k for k in key_ring if k and k.strip()]
    if not ring:
        ring = [api_key] if api_key else []
    if not ring:
        return None

    last_error = ""
    for model in GEMINI_MODELS:
        # Try each key in order for this model — only advance on 429
        for key_idx, active_key in enumerate(ring):
            try:
                result = _call_gemini_model(prompt, active_key, model, verbose=verbose)
                if verbose:
                    print(f"  ✓ model={model} key={key_idx+1}/{len(ring)} (...{active_key[-4:]})")
                return result

            except urllib.error.HTTPError as e:
                body = e.read().decode()
                try:
                    msg = json.loads(body).get("error", {}).get("message", body)
                except Exception:
                    msg = body[:150]
                last_error = f"{model}: HTTP {e.code} — {msg}"

                if e.code == 429:
                    # Rate-limited — try next key for the same model
                    if verbose:
                        print(f"  ↻ 429 on key {key_idx+1} (...{active_key[-4:]}) → trying next key")
                    continue  # next key in ring

                elif e.code in (400, 401, 403):
                    if verbose:
                        print(f"  ✗ {last_error} (auth error — stopping)")
                    return None

                else:
                    if verbose:
                        print(f"  ✗ {last_error} (trying next model)")
                    break  # try next model, same key order

            except Exception as e:
                last_error = f"{model}: {e}"
                if verbose:
                    print(f"  ✗ {last_error} (trying next model)")
                break  # try next model

        else:
            # All keys were 429'd for this model — try next model
            if verbose:
                print(f"  ✗ All keys rate-limited for {model} — trying next model")
            continue

    if verbose:
        print(f"  ✗ All Gemini models/keys exhausted. Last error: {last_error}")
    return None


# ── Main public function ──────────────────────────────────────────────────────

def call_llm(prompt: str, api_key: str, key_ring: List[str] = None,
             verbose: bool = False) -> Optional[dict]:
    """
    Call the LLM for the given api_key. Returns parsed JSON dict or None.
    Never raises — falls back gracefully on any error.

    For Gemini key rotation, pass key_ring=[key1, key2, ...].
    Keys are used one at a time; next key is only tried on 429.
    """
    provider = detect_provider(api_key)

    if provider == 'unknown':
        # Try to detect from key_ring if single api_key is empty
        if key_ring:
            for k in key_ring:
                p = detect_provider(k)
                if p != 'unknown':
                    provider = p
                    api_key = k
                    break
    if provider == 'unknown':
        if verbose:
            print(f"  ⚠  Cannot detect provider from key — analytical mode")
        return None

    if verbose:
        print(f"  → Calling {provider_label(api_key)}...")

    try:
        if provider == 'gemini':
            return _call_gemini(prompt, api_key, key_ring=key_ring, verbose=verbose)
        elif provider == 'anthropic':
            return _call_anthropic(prompt, api_key, key_ring=key_ring, verbose=verbose)
        elif provider == 'openai':
            return _call_openai(prompt, api_key, key_ring=key_ring, verbose=verbose)

    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            msg = json.loads(body).get("error", {}).get("message", body)
        except Exception:
            msg = body[:200]
        if verbose:
            print(f"  ⚠  API error (HTTP {e.code}): {msg}")
            print(f"     Falling back to analytical mode.")
        return None

    except Exception as e:
        if verbose:
            print(f"  ⚠  LLM call failed ({type(e).__name__}: {e})")
            print(f"     Falling back to analytical mode.")
        return None


# ── Key tester ────────────────────────────────────────────────────────────────

def test_key(api_key: str, verbose: bool = True) -> tuple:
    """Send a minimal test prompt. Returns (success: bool, message: str)."""
    provider = detect_provider(api_key)
    if provider == 'unknown':
        return False, "Cannot detect provider — key format not recognised"

    result = call_llm(
        'Reply with exactly this JSON and nothing else: {"status": "ok"}',
        api_key,
        verbose=verbose,
    )
    if result and result.get("status") == "ok":
        return True, f"{provider_label(api_key)} — key valid ✓"
    elif result is not None:
        return True, f"{provider_label(api_key)} — key valid (response: {result})"
    else:
        return False, f"{provider_label(api_key)} — call failed (see errors above)"


# ── Run as script ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("=" * 55)
    print("  LLM Client — Key Tester")
    print("=" * 55)
    print()
    print("  Supported formats:")
    print("    AIza...       → Google Gemini  ← active")
    print("    sk-ant-...    → Anthropic Claude")
    print("    sk-...        → OpenAI ChatGPT")
    print()

    if len(sys.argv) < 2:
        print("  Usage: python3 utils/llm_client.py YOUR_API_KEY")
        sys.exit(0)

    for key in sys.argv[1:]:
        print(f"  Key      : {key[:12]}...{key[-4:]}")
        print(f"  Provider : {provider_label(key)}")
        print(f"  Testing  :")
        success, msg = test_key(key, verbose=True)
        status = "✓" if success else "✗"
        print(f"  Result   : {status} {msg}")
        print()
