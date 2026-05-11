"""CUTE tokenizer — Streamlit chat demo (prompt → Python via Groq + local token stats).

Run from the repository root. Install Streamlit, Groq SDK, plus project dependencies:

    pip install streamlit groq
    pip install -e .

Set your Groq API key **via environment** (recommended — never commit keys to git):

    # PowerShell
    $env:GROQ_API_KEY = "your-key"

    streamlit run demo/streamlit_tokenizer_demo.py

Optionally use Streamlit secrets (``.streamlit/secrets.toml``): ``GROQ_API_KEY = "..."``

Tokenizer files default to the repository ``model/`` directory (same artifacts shipped on PyPI
via ``load_default_tokenizer()``). Override with env ``CUTE_TOKENIZER_DIR`` or the sidebar path.

**Train your own:** ``cute build --corpus ./corpus --output ./my_run`` then point the sidebar at
that folder, or replace files under ``model/`` if you want the demo default to match.

**Baseline column:** ``pip install tiktoken`` so **tiktoken/cl100k** loads (standard LLM billing proxy).

**Token counting:** ``CUTETokenizerFast`` vs baselines run **locally** on your text. Groq billing uses its own tokenizer.

This file is standalone; other files touched only for the optional build script under ``scripts/``.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Default Groq chat model (OpenAI-compatible ID on Groq Cloud).
GROQ_MODEL_DEFAULT = "openai/gpt-oss-120b"

_SYSTEM_PYTHON_ONLY = (
    "You are an expert Python programmer. Respond with Python source code only. "
    "Put all code in a single fenced markdown block: ```python ... ```. "
    "No explanations outside the code block unless inside comments."
)


def _import_cute_tokenizer() -> ModuleType:
    """Import ``cute_tokenizer`` from site-packages or from ``<repo>/src`` (no editable install)."""
    try:
        import cute_tokenizer
    except ImportError:
        src = _REPO_ROOT / "src"
        pkg_init = src / "cute_tokenizer" / "__init__.py"
        if pkg_init.is_file():
            root_str = str(src)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            import cute_tokenizer  # type: ignore[no-redef]
        else:
            raise
    return cute_tokenizer
_ENV_TOKENIZER_DIR = os.environ.get("CUTE_TOKENIZER_DIR", "").strip()


def _default_tokenizer_dir() -> Path:
    if _ENV_TOKENIZER_DIR:
        return Path(_ENV_TOKENIZER_DIR).expanduser().resolve()
    repo_model = (_REPO_ROOT / "model").resolve()
    if (repo_model / "tokenizer.json").is_file() and (repo_model / "cute_mapping.json").is_file():
        return repo_model
    cute = _import_cute_tokenizer()
    origin = getattr(cute, "__file__", None)
    if not origin:
        return repo_model
    pkg_data = Path(origin).resolve().parent / "data"
    if (pkg_data / "tokenizer.json").is_file() and (pkg_data / "cute_mapping.json").is_file():
        return pkg_data
    return repo_model


def _synthetic_python_artifact(user_prompt: str) -> str:
    """Deterministic placeholder when Groq is disabled or unavailable."""
    lines = user_prompt.splitlines() if user_prompt.strip() else ["(empty prompt)"]
    commented = "\n".join(f"# {ln}" for ln in lines)
    return (
        '"""Synthetic Python artifact for CUTE token visualization (demo stub)."""\n\n'
        f"{commented}\n\n"
        "from __future__ import annotations\n\n\n"
        "def run() -> str:\n"
        '    """Replace this stub with your model\'s completion."""\n'
        '    return "Connect an LLM here; this file is for tokenizer metrics only."\n'
    )


def _extract_python_from_markdown(raw: str) -> str:
    """Pull the best Python fenced block from model output; otherwise return stripped text."""
    text = raw.strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    return text


def _rough_token_estimate(text: str) -> int:
    """Cheap heuristic (~4 chars/token); Groq limits are on true tokens — stay under budget."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _trim_for_groq_api(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    head = "[…truncated for API context limit…]\n"
    return head + content[-(max_chars - len(head)) :]


def _groq_chat_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> str:
    try:
        from groq import Groq
        from groq.types.chat import ChatCompletionMessageParam
    except ImportError as e:
        raise RuntimeError("Install the Groq SDK: pip install groq") from e

    client = Groq(api_key=api_key)
    # Cast to proper type for Groq SDK
    typed_messages: list[ChatCompletionMessageParam] = messages  # type: ignore[assignment]
    completion = client.chat.completions.create(
        model=model,
        messages=typed_messages,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    msg = completion.choices[0].message
    return (msg.content or "").strip()


def _build_groq_messages(
    *,
    prior_chat: list[dict],
    new_user_text: str,
    max_prior_turns: int,
    per_message_char_cap: int,
    max_estimated_prompt_tokens: int,
) -> tuple[list[dict[str, str]], str | None]:
    """Rolling window + truncation so requests stay under Groq on-demand TPM/context limits."""
    system_msg = {"role": "system", "content": _SYSTEM_PYTHON_ONLY}

    prior_only = list(prior_chat)
    max_pairs = max(0, max_prior_turns) * 2
    window = prior_only[-max_pairs:] if max_pairs else []

    rolled: list[dict[str, str]] = []
    for m in window:
        role = m["role"]
        if role not in ("user", "assistant"):
            continue
        body = _trim_for_groq_api(str(m.get("content", "")), per_message_char_cap)
        rolled.append({"role": role, "content": body})

    rolled.append({"role": "user", "content": _trim_for_groq_api(new_user_text, per_message_char_cap)})

    msgs: list[dict[str, str]] = [system_msg, *rolled]

    note: str | None = None

    def _total_estimated(ms: list[dict[str, str]]) -> int:
        return sum(_rough_token_estimate(x["content"]) for x in ms)

    est = _total_estimated(msgs)
    safety = 0
    # Drop oldest turns until under prompt-side budget (heuristic).
    while est > max_estimated_prompt_tokens and len(msgs) > 2 and safety < 500:
        safety += 1
        msgs.pop(1)
        est = _total_estimated(msgs)

    if est > max_estimated_prompt_tokens:
        note = (
            f"Context still large (~{est} est. tokens). Lower **Prior conversation turns**, "
            "**Max chars per history message**, or **Estimated input token budget**."
        )

    return msgs, note


def _encode_lengths(text: str, encoders: dict[str, Callable[[str], list[int]]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, fn in encoders.items():
        try:
            out[name] = len(fn(text))
        except Exception:
            out[name] = -1  # unavailable / error
    return out


def _pick_baseline_encoder(
    encoders: dict[str, Callable[[str], list[int]]],
) -> tuple[str, Callable[[str], list[int]]] | None:
    """Prefer OpenAI-style cl100k (API-cost baseline), then GPT-2."""
    for name in ("tiktoken/cl100k", "GPT-2 (HF)"):
        if name in encoders:
            return name, encoders[name]
    return None


def _bench_encode_ms(fn: Callable[[str], list[int]], text: str, *, warmup: int = 5, reps: int = 40) -> float:
    """Mean encode time in ms for local CPU timing (not Groq latency)."""
    if not text:
        return 0.0
    for _ in range(warmup):
        fn(text)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn(text)
    return (time.perf_counter() - t0) / reps * 1000.0


def _benchmark_rows(
    prompt_counts: dict[str, int],
    code_counts: dict[str, int],
) -> list[dict[str, str | int | float]]:
    """Build table rows aligned with ``benchmarks/compression.py`` (ratio vs CUTE total)."""
    cute_p = prompt_counts.get("CUTE", -1)
    cute_c = code_counts.get("CUTE", -1)
    if cute_p < 0 or cute_c < 0:
        return []
    cute_tot = cute_p + cute_c
    rows: list[dict[str, str | int | float]] = []
    names = list(prompt_counts.keys())
    # Stable order: CUTE first, then alphabetical baselines
    if "CUTE" in names:
        names.remove("CUTE")
        ordered = ["CUTE", *sorted(names)]
    else:
        ordered = sorted(names)

    for name in ordered:
        pt = prompt_counts.get(name, -1)
        ct = code_counts.get(name, -1)
        if pt < 0 or ct < 0:
            total = -1
            rel = float("nan")
        else:
            total = pt + ct
            rel = total / cute_tot if cute_tot else float("nan")
        rows.append(
            {
                "Tokenizer": name,
                "Prompt tokens": pt if pt >= 0 else "—",
                "Python tokens": ct if ct >= 0 else "—",
                "Total tokens": total if total >= 0 else "—",
                "vs CUTE (total)": "—" if total < 0 else f"{rel:.2f}x",
            }
        )
    return rows


@st.cache_resource(show_spinner=False)
def _benchmark_encode_fns(
    tokenizer_json: str,
    mapping_json: str,
) -> dict[str, Callable[[str], list[int]]]:
    """Same baseline set as ``benchmarks/compression._build_tokenizers`` (best-effort)."""
    from cute_tokenizer import CUTETokenizerFast

    encoders: dict[str, Callable[[str], list[int]]] = {}

    cute = CUTETokenizerFast(
        tokenizer_file=tokenizer_json,
        cute_mapping_file=mapping_json,
    )
    encoders["CUTE"] = lambda t: cute(t, add_special_tokens=False).input_ids

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        encoders["tiktoken/cl100k"] = enc.encode
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer

        gpt2 = AutoTokenizer.from_pretrained("gpt2")
        encoders["GPT-2 (HF)"] = lambda t: gpt2(t, add_special_tokens=False)["input_ids"]
    except Exception:
        pass

    return encoders


def _render_benchmark_table(prompt_counts: dict[str, int], code_counts: dict[str, int]) -> None:
    rows = _benchmark_rows(prompt_counts, code_counts)
    if not rows:
        return
    with st.expander("Full tokenizer table (all loaded backends)", expanded=False):
        st.caption(
            "**vs CUTE** = baseline_total ÷ CUTE_total. Optional: `pip install tiktoken`; GPT-2 loads once."
        )
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _compute_validation_payload(
    encoders: dict[str, Callable[[str], list[int]]],
    *,
    user_prompt: str,
    code: str,
    prompt_counts: dict[str, int],
    code_counts: dict[str, int],
) -> dict[str, float | int | str | None]:
    """Snapshot for UI + chat replay: baseline vs CUTE totals, savings %, local encode ms."""
    cute_pt = prompt_counts.get("CUTE", -1)
    cute_ct = code_counts.get("CUTE", -1)
    cute_tot = cute_pt + cute_ct if cute_pt >= 0 and cute_ct >= 0 else -1
    bundle = f"{user_prompt}\n\n{code}"

    ms_c = _bench_encode_ms(encoders["CUTE"], bundle)
    pick = _pick_baseline_encoder(encoders)
    if not pick:
        return {
            "baseline_name": "",
            "bundle_ms_baseline": 0.0,
            "bundle_ms_cute": ms_c,
            "savings_pct_total": None,
            "baseline_total": -1,
            "cute_total": float(cute_tot),
        }

    bname, fn_b = pick
    ms_b = _bench_encode_ms(fn_b, bundle)
    b_pt = prompt_counts.get(bname, -1)
    b_ct = code_counts.get(bname, -1)
    base_tot = b_pt + b_ct if b_pt >= 0 and b_ct >= 0 else -1
    savings = (
        (base_tot - cute_tot) / base_tot * 100.0
        if base_tot > 0 and cute_tot >= 0
        else None
    )
    return {
        "baseline_name": bname,
        "bundle_ms_baseline": ms_b,
        "bundle_ms_cute": ms_c,
        "savings_pct_total": savings,
        "baseline_total": float(base_tot),
        "cute_total": float(cute_tot),
    }


def _render_validation_split(
    *,
    prompt_counts: dict[str, int],
    code_counts: dict[str, int],
    validation: dict[str, float | int | str | None],
    artifact_code: str | None = None,
) -> None:
    """Left = typical LLM tokenizer baseline; right = CUTE — tokens, savings, encode speed."""
    st.markdown("##### Validation split · baseline vs CUTE (efficiency & local speed)")
    bname = str(validation.get("baseline_name") or "")
    ms_b = float(validation.get("bundle_ms_baseline") or 0.0)
    ms_c = float(validation.get("bundle_ms_cute") or 0.0)
    savings = validation.get("savings_pct_total")
    base_tot = int(float(validation.get("baseline_total") or -1))
    cute_tot = int(float(validation.get("cute_total") or -1))

    cute_pt = prompt_counts.get("CUTE", -1)
    cute_ct = code_counts.get("CUTE", -1)
    b_pt = prompt_counts.get(bname, -1) if bname else -1
    b_ct = code_counts.get(bname, -1) if bname else -1

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown(
            '<div style="border:1px solid rgba(255,255,255,0.12);border-radius:14px;'
            'padding:14px 16px;background:rgba(255,255,255,0.03);">'
            "<strong>Baseline · normal tokenizer</strong><br/>"
            f"<span style='opacity:0.85;font-size:0.92rem'>{bname or '⚠ Install tiktoken for cl100k — `pip install tiktoken`'}"
            "</span></div>",
            unsafe_allow_html=True,
        )
        if b_pt >= 0 and b_ct >= 0:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Prompt tokens", b_pt)
            with m2:
                st.metric("Python tokens", b_ct)
            with m3:
                st.metric("Total tokens", b_pt + b_ct)
            st.metric(
                "Encode prompt+code (local)",
                f"{ms_b:.3f} ms/call",
                help="Mean CPU time to encode the same bundle as CUTE.",
            )
        else:
            st.info("Load **tiktoken** (cl100k) or GPT-2 baseline for a side-by-side.")

    with col_r:
        st.markdown(
            '<div style="border:1px solid rgba(129,140,248,0.35);border-radius:14px;'
            'padding:14px 16px;background:rgba(99,102,241,0.08);">'
            "<strong>CUTE · your tokenizer</strong><br/>"
            "<span style='opacity:0.85;font-size:0.92rem'>CUTETokenizerFast (your build)</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if cute_pt >= 0 and cute_ct >= 0:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Prompt tokens", cute_pt)
            with m2:
                st.metric("Python tokens", cute_ct)
            with m3:
                st.metric("Total tokens", cute_tot if cute_tot >= 0 else cute_pt + cute_ct)
            encode_ratio = (ms_c / ms_b) if ms_b > 0 and ms_c > 0 else None
            delta = (
                f"{encode_ratio:.2f}× wall time vs baseline encode"
                if encode_ratio is not None
                else None
            )
            st.metric(
                "Encode prompt+code (local)",
                f"{ms_c:.3f} ms/call",
                delta=delta,
                help="Wall time to encode the same text once (mean). Higher ms ⇒ slower on CPU.",
            )

    savings_num = float(savings) if savings is not None else None
    if savings_num is not None and base_tot > 0 and cute_tot >= 0:
        if savings_num > 0:
            st.success(
                f"**Token efficiency:** CUTE uses **{savings_num:.1f}% fewer** total tokens than **{bname}** "
                f"({base_tot} → {cute_tot}). Lower totals ⇒ lower cost when billed per token."
            )
        elif savings_num < 0:
            st.warning(
                f"**On this sample**, CUTE uses **{-savings_num:.1f}% more** total tokens than **{bname}** "
                f"({base_tot} baseline → {cute_tot} CUTE). "
                "That does **not** invalidate CUTE in general — compare on **tiktoken/cl100k**, "
                "train on a **code-heavy corpus**, and run **`benchmarks/compression.py`** on held-out files."
            )
        else:
            st.info("**Tie:** same total token count vs baseline for this text.")

    elif cute_tot >= 0 and base_tot > 0 and savings is None:
        st.caption("Compare **Total tokens**: lower is usually cheaper per-token billing.")

    if artifact_code and artifact_code.strip():
        st.markdown("##### Same Python · side-by-side (identical bytes; token IDs differ by tokenizer)")
        cc1, cc2 = st.columns(2, gap="medium")
        with cc1:
            st.caption(f"**Baseline context:** {bname or 'install tiktoken (`pip install tiktoken`)'} · totals above left")
            st.code(artifact_code, language="python")
        with cc2:
            st.caption("**CUTE context:** CUTETokenizerFast · totals above right")
            st.code(artifact_code, language="python")

    st.caption(
        "Local encode timing is CPU-only on your machine (cf. `benchmarks/latency.py`). "
        "Groq latency is separate."
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

          html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif;
          }
          .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 3rem !important;
            max-width: min(96vw, 1400px) !important;
          }
          /* Hero card */
          .cute-hero {
            padding: 1.35rem 1.5rem 1.25rem 1.5rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(99,102,241,0.14) 0%, rgba(14,165,233,0.10) 50%, rgba(16,185,129,0.10) 100%);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 18px 50px rgba(0,0,0,0.25);
            margin-bottom: 1.1rem;
          }
          .cute-hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.2;
          }
          .cute-hero p {
            margin: 0;
            opacity: 0.9;
            line-height: 1.45;
            font-size: 0.98rem;
          }
          .cute-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border-radius: 999px;
            padding: 0.2rem 0.7rem;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.65rem;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.06);
          }
          /* Metric strip */
          div[data-testid="stMetricValue"] {
            font-variant-numeric: tabular-nums;
          }
          /* Chat input polish */
          div[data-testid="stChatInput"] textarea {
            border-radius: 14px !important;
          }
          /* Code block font */
          pre, code, .stCodeBlock {
            font-family: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="CUTE · Tokenizer demo",
        page_icon="🐭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        cute_tokenizer = _import_cute_tokenizer()
    except ImportError as e:  # pragma: no cover - demo entrypoint
        st.error(
            "Could not import `cute_tokenizer`. From the repo root run `pip install -e .`, "
            "or ensure dependencies are installed and `src/cute_tokenizer/` exists."
        )
        st.exception(e)
        st.stop()

    _inject_styles()

    st.markdown(
        """
        <div class="cute-hero">
          <div class="cute-pill">Tokenizer validation · Groq · CUTE</div>
          <h1>Baseline vs CUTE — efficiency, speed, cost proxy</h1>
          <p>
            <strong>Groq</strong> writes Python; below, the page compares a <strong>normal</strong>
            tokenizer (tiktoken cl100k when installed) against <strong>your trained CUTE</strong>
            tokenizer on the <em>same</em> prompt + output — token totals, % savings, and local encode time.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar: tokenizer paths ---
    with st.sidebar:
        st.subheader("Tokenizer paths")
        default_dir = _default_tokenizer_dir()
        tokenizer_dir = st.text_input(
            "Directory with tokenizer artifacts",
            value=str(default_dir),
            help="Must contain tokenizer.json and cute_mapping.json",
        )
        path_dir = Path(tokenizer_dir).expanduser().resolve()
        t_json = path_dir / "tokenizer.json"
        m_json = path_dir / "cute_mapping.json"
        ok = t_json.is_file() and m_json.is_file()
        if ok:
            st.success("Found tokenizer.json and cute_mapping.json")
        else:
            st.warning(
                "Artifacts not found at that path. Build a tokenizer, set CUTE_TOKENIZER_DIR, "
                "or point to a folder with tokenizer.json and cute_mapping.json."
            )
        st.caption(f"cute-tokenizer `{cute_tokenizer.__version__}`")

        st.divider()
        st.subheader("Groq LLM")
        env_groq = (os.environ.get("GROQ_API_KEY") or "").strip()
        secrets_key = ""
        try:
            secrets_key = str(st.secrets.get("GROQ_API_KEY", "") or "").strip()
        except Exception:
            pass
        groq_key_input = st.text_input(
            "Groq API key (optional if set via env / secrets)",
            type="password",
            key="groq_api_key_field",
            help="Prefer: `$env:GROQ_API_KEY` (PowerShell) or `.streamlit/secrets.toml`. "
            "Never commit keys to git.",
        )
        groq_model = st.text_input(
            "Groq model id",
            value=GROQ_MODEL_DEFAULT,
            help="Example: openai/gpt-oss-120b",
        )
        use_stub = st.checkbox(
            "Offline: use stub code instead of Groq",
            value=False,
            help="No API calls; emits the deterministic placeholder Python.",
        )

        st.caption("Request size (on-demand tier ~8k TPM): limit history + completion tokens.")
        max_completion_tokens = st.number_input(
            "Max new tokens (completion)",
            min_value=256,
            max_value=8192,
            value=1536,
            step=256,
            help="Lower this if you hit 413 / rate limits. Default 1536 keeps room for prompt.",
        )
        max_prior_turns = st.slider(
            "Prior conversation turns to send",
            min_value=0,
            max_value=12,
            value=3,
            help="Each turn is one user + one assistant pair from chat history.",
        )
        max_prompt_est_tokens = st.number_input(
            "Estimated input token budget",
            min_value=1500,
            max_value=12000,
            value=5200,
            step=100,
            help="Rough limit on prompt side (heuristic). Drop if Groq still rejects.",
        )
        per_msg_chars = st.number_input(
            "Max chars per history message",
            min_value=2000,
            max_value=50000,
            value=8000,
            step=1000,
            help="Truncates long assistant code blocks sent back as context.",
        )

    effective_groq_key = env_groq or secrets_key or (groq_key_input or "").strip()

    # --- Session ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    encoders: dict[str, Callable[[str], list[int]]] | None = None
    if ok:
        try:
            encoders = _benchmark_encode_fns(str(t_json), str(m_json))
        except Exception as e:
            st.error("Failed to load tokenizer.")
            st.exception(e)

    # --- Render chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                src = msg.get("source", "groq")
                st.markdown(f"**{src}** · generated Python — **validation split below**")
                if msg.get("benchmark") and msg.get("validation"):
                    _render_validation_split(
                        prompt_counts=msg["benchmark"]["prompt"],
                        code_counts=msg["benchmark"]["code"],
                        validation=msg["validation"],
                        artifact_code=msg["content"],
                    )
                    _render_benchmark_table(
                        msg["benchmark"]["prompt"],
                        msg["benchmark"]["code"],
                    )
                elif msg.get("benchmark"):
                    _render_benchmark_table(
                        msg["benchmark"]["prompt"],
                        msg["benchmark"]["code"],
                    )
                if not (msg.get("benchmark") and msg.get("validation")):
                    st.code(msg["content"], language="python")

    user_prompt = st.chat_input("Describe the Python you want…")

    if user_prompt and encoders is not None:
        code: str
        source = "groq"
        raw_llm = ""

        if use_stub:
            code = _synthetic_python_artifact(user_prompt)
            source = "stub"
        elif not effective_groq_key:
            st.warning(
                "Set **GROQ_API_KEY** in your environment, add it to `.streamlit/secrets.toml`, "
                "or paste it in the sidebar. Keys must not be committed to source code."
            )
            st.stop()
        else:
            api_messages, ctx_note = _build_groq_messages(
                prior_chat=st.session_state.messages,
                new_user_text=user_prompt,
                max_prior_turns=int(max_prior_turns),
                per_message_char_cap=int(per_msg_chars),
                max_estimated_prompt_tokens=int(max_prompt_est_tokens),
            )
            if ctx_note:
                st.caption(ctx_note)

            with st.spinner(f"Calling Groq ({groq_model.strip() or GROQ_MODEL_DEFAULT})…"):
                try:
                    raw_llm = _groq_chat_completion(
                        effective_groq_key,
                        groq_model.strip() or GROQ_MODEL_DEFAULT,
                        api_messages,
                        max_tokens=int(max_completion_tokens),
                    )
                except Exception as e:
                    st.error("Groq request failed.")
                    st.exception(e)
                    st.stop()

            code = _extract_python_from_markdown(raw_llm)
            if not code.strip():
                code = "# (empty model response)\npass\n"

        prompt_counts = _encode_lengths(user_prompt, encoders)
        code_counts = _encode_lengths(code, encoders)
        in_tok = prompt_counts.get("CUTE", -1)
        out_tok = code_counts.get("CUTE", -1)
        validation_payload = _compute_validation_payload(
            encoders,
            user_prompt=user_prompt,
            code=code,
            prompt_counts=prompt_counts,
            code_counts=code_counts,
        )

        st.session_state.messages.append({"role": "user", "content": user_prompt})
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": code,
                "in_tok": in_tok,
                "out_tok": out_tok,
                "benchmark": {"prompt": prompt_counts, "code": code_counts},
                "validation": validation_payload,
                "source": source,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(f"**{source}** · generated Python — **validation split below**")
            _render_validation_split(
                prompt_counts=prompt_counts,
                code_counts=code_counts,
                validation=validation_payload,
                artifact_code=code,
            )
            if raw_llm and not use_stub:
                with st.expander("Raw LLM response"):
                    st.markdown(raw_llm)
            _render_benchmark_table(prompt_counts, code_counts)

    elif user_prompt and encoders is None:
        st.info("Load tokenizer files in the sidebar to enable encoding.")

    # --- Metrics row (live summary from last turn if present) ---
    last_asst = next(
        (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
        None,
    )
    if last_asst:
        st.markdown("##### Last turn snapshot")
        lc, rc = st.columns(2, gap="large")
        v = last_asst.get("validation") or {}
        bname = str(v.get("baseline_name") or "baseline")
        base_tot = int(float(v.get("baseline_total") or -1))
        cute_tot = int(float(v.get("cute_total") or -1))
        savings = v.get("savings_pct_total")

        with lc:
            st.caption(f"Baseline · {bname}" if bname else "Baseline")
            if base_tot >= 0:
                st.metric("Total tokens (prompt + Python)", base_tot)
                st.metric(
                    "Encode bundle (local)",
                    f"{float(v.get('bundle_ms_baseline') or 0):.3f} ms",
                )
            else:
                st.metric("Total tokens", "—")

        with rc:
            st.caption("CUTE · your tokenizer")
            if cute_tot >= 0:
                st.metric("Total tokens (prompt + Python)", cute_tot)
                st.metric(
                    "Encode bundle (local)",
                    f"{float(v.get('bundle_ms_cute') or 0):.3f} ms",
                )
                savings_num2 = float(savings) if savings is not None else None
                if savings_num2 is not None:
                    if savings_num2 > 0:
                        st.metric("Δ tokens (CUTE vs baseline)", f"{savings_num2:.1f}% fewer")
                    elif savings_num2 < 0:
                        st.metric("Δ tokens (CUTE vs baseline)", f"{-savings_num2:.1f}% more")
                    else:
                        st.metric("Δ tokens (CUTE vs baseline)", "tie")
            else:
                st.metric("Total tokens", "—")

        ratio = (
            last_asst["out_tok"] / last_asst["in_tok"]
            if last_asst["in_tok"]
            else float("nan")
        )
        st.caption(f"CUTE artifact/prompt ratio: **{ratio:.2f}×** (output length vs prompt length).")


if __name__ == "__main__":
    main()
