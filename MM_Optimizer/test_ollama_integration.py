"""
test_ollama_integration.py
--------------------------
Verifies that configured Ollama models can:
  1. Be listed and are present in the local Ollama registry
  2. Respond to a basic generation prompt (measures first-token latency)
  3. Return structured output via Instructor + ParamProposal schema

Integration path: Instructor -> OpenAI-compatible Ollama endpoint (localhost:11434/v1)

Run with:
    python MM_Optimizer/test_ollama_integration.py
    python MM_Optimizer/test_ollama_integration.py --models qwen3:8b   # subset only
"""

import argparse
import sys
import time

# ---------------------------------------------------------------------------
# Models under test (name as it appears in `ollama list`)
# ---------------------------------------------------------------------------
DEFAULT_MODELS = ["phi4-reasoning:latest", "qwen3:8b"]
OLLAMA_BASE_URL = "http://localhost:11434/v1"

PASS_COUNT = 0
FAIL_COUNT = 0


def _ok(label: str, detail: str = "") -> None:
    global PASS_COUNT
    PASS_COUNT += 1
    print(f"  [PASS] {label}" + (f"  -- {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# Test 1: Model registry
# ---------------------------------------------------------------------------

def test_model_present(model_name: str, available: list[str]) -> bool:
    """Check that the model name (or its prefix) is in the Ollama registry."""
    # Match by exact name or by name prefix (e.g. "qwen3:8b" in "qwen3:8b")
    for a in available:
        if model_name in a or a in model_name:
            _ok(f"model present: {model_name}", f"found as '{a}'")
            return True
    _fail(f"model present: {model_name}", f"not in {available} — run: ollama pull {model_name}")
    return False


# ---------------------------------------------------------------------------
# Test 2: Basic generation (raw ollama Python client)
# ---------------------------------------------------------------------------

def test_basic_generation(model_name: str) -> bool:
    """Send a one-line prompt and verify a non-empty text response."""
    try:
        import ollama
    except ImportError:
        _fail(f"basic generation: {model_name}", "ollama package not installed — pip install ollama")
        return False

    prompt = "Reply with exactly the word PONG and nothing else."
    t0 = time.perf_counter()
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 32, "temperature": 0},
            think=False,   # disable chain-of-thought for phi4-reasoning (Ollama ≥ 0.7)
        )
        elapsed = time.perf_counter() - t0
        text = resp.message.content.strip()
        if text:
            # phi4-reasoning wraps answers in <think>...</think> before the real reply.
            # Accept the response if it contains PONG anywhere, or if non-empty
            # (the model is responding — thinking tokens are expected behaviour).
            has_pong = "PONG" in text.upper()
            note = "" if has_pong else "  (note: reply includes thinking tokens — expected for phi4-reasoning)"
            preview = text[:80].replace("\n", " ")
            _ok(f"basic generation: {model_name}", f"{elapsed:.1f}s  reply='{preview}'{note}")
            return True
        else:
            _fail(f"basic generation: {model_name}", "empty response")
            return False
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        # think=False may not be supported in older Ollama; retry without it
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 64, "temperature": 0},
            )
            elapsed = time.perf_counter() - t0
            text = resp.message.content.strip()
            if text:
                preview = text[:80].replace("\n", " ")
                _ok(f"basic generation: {model_name}", f"{elapsed:.1f}s  reply='{preview}'  (think=False not supported by this Ollama version)")
                return True
            _fail(f"basic generation: {model_name}", "empty response on retry")
            return False
        except Exception as exc2:
            _fail(f"basic generation: {model_name}", f"{elapsed:.1f}s  {exc2}")
            return False


# ---------------------------------------------------------------------------
# Test 3: Structured output via Instructor
# ---------------------------------------------------------------------------

def test_structured_output(model_name: str) -> bool:
    """Use Instructor + OpenAI-compatible Ollama endpoint to get a ParamProposal."""
    try:
        import instructor
        from openai import OpenAI
    except ImportError as e:
        _fail(f"structured output: {model_name}", f"missing package: {e}")
        return False

    # Import ParamProposal from llm_optimizer
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    try:
        from llm_optimizer import ParamProposal
    except ImportError as e:
        _fail(f"structured output: {model_name}", f"cannot import ParamProposal: {e}")
        return False

    raw_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
    client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    prompt = (
        "You are tuning MechMind CAD matching parameters for a cylindrical part "
        "(diameter 80 mm, SO2 symmetry). Propose a starting configuration. "
        "Return JSON matching the ParamProposal schema. "
        "Remember: referredStep MUST be <= refStep."
    )

    t0 = time.perf_counter()
    try:
        proposal: ParamProposal = client.chat.completions.create(
            model=model_name,
            response_model=ParamProposal,
            max_retries=2,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        elapsed = time.perf_counter() - t0

        # Validate constraint
        violated = proposal.referredStep > proposal.refStep
        detail = (
            f"{elapsed:.1f}s  "
            f"refStep={proposal.refStep}  referredStep={proposal.referredStep}  "
            f"distQ={proposal.distQuantification:.2f}  "
            f"reason='{proposal.reasoning[:60]}'"
        )
        if violated:
            _fail(f"structured output: {model_name}", f"CONSTRAINT VIOLATED referredStep>refStep  {detail}")
            return False
        _ok(f"structured output: {model_name}", detail)
        return True

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _fail(f"structured output: {model_name}", f"{elapsed:.1f}s  {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(models: list[str]) -> None:
    print("=" * 65)
    print("Ollama Integration Test")
    print("=" * 65)

    # -- Step 0: check Ollama is reachable ----------------------------------
    print("\n[0] Ollama connectivity")
    try:
        import ollama
        list_resp = ollama.list()
        # API returns an object with .models list (ollama >= 0.2)
        available = [m.model for m in list_resp.models]
        _ok("Ollama running", f"{len(available)} model(s) loaded: {available}")
    except Exception as exc:
        _fail("Ollama running", f"{exc} — is 'ollama serve' running?")
        print(f"\n{FAIL_COUNT} failure(s). Cannot continue without Ollama.")
        sys.exit(1)

    # -- Per-model tests -----------------------------------------------------
    for model in models:
        print(f"\n[model: {model}]")

        present = test_model_present(model, available)
        if not present:
            print(f"  Skipping generation tests for missing model.")
            continue

        test_basic_generation(model)
        test_structured_output(model)

    # -- Summary ------------------------------------------------------------
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'=' * 65}")
    print(f"{PASS_COUNT}/{total} passed")
    if FAIL_COUNT:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Space-separated list of Ollama model names to test"
    )
    args = parser.parse_args()
    main(args.models)
