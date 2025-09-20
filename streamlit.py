# app.py — OpenPFx with Home landing page + Browse (multiselect) + Generate (LLM)
# Drop-in single file. Replace placeholder copy where noted.

import os
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(page_title="OpenPFx", page_icon="💬", layout="wide")

from jh_main.jh_pfx_prompts import (
    example,
    icd10_example,
    single_fewshot_icd10_labeling_prompt,
    baseline_zeroshot_prompt,
    writer_prompt,
    doctor_prompt,
    readability_checker_prompt,
    ICD10_LABELER_INSTRUCTION,
)

# Core LLM-backed functions implemented in your separate module
from jh_main.streamlit_calls import (
    suggest_icd10_code,      
    zeroshot_call,
    fewshot_call,
    agentic_conversation,
)

# ====== ROUTING (query param) ======
def _set_page(page: str):
    try:
        st.query_params["page"] = page  # Streamlit ≥1.31
    except Exception:
        st.experimental_set_query_params(page=page)  # legacy
    st.rerun()

def _get_page(default="home") -> str:
    try:
        qp = st.query_params
        val = qp.get("page")
        if isinstance(val, list):
            return (val[0] or default).lower()
        return (val or default).lower()
    except Exception:
        qp = st.experimental_get_query_params()
        return (qp.get("page", [default])[0]).lower()

# ====== TOP NAV ======
def _top_nav(active: str):
    def _btn(label, target):
        is_active = (active == target)
        style = (
            "padding:.5rem .9rem;border-radius:10px;border:1px solid #e5e7eb;"
            f"{'background:#111;color:#fff;' if is_active else 'background:#f0f2f6;color:#111;'}"
            "font-weight:600;cursor:pointer;"
        )
        st.markdown(
            f"""
            <form action="" method="get" style="display:inline;">
              <input type="hidden" name="page" value="{target}">
              <button type="submit" style="{style}">{label}</button>
            </form>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.title("OpenPFx")
        st.caption("Open source explanations of medical imaging findings to help patients understand their medical reports.")
    with c2:
        st.markdown("""
        <div class="pfx-nav">
        <form action="" method="get">
            <input type="hidden" name="page" value="home">
            <button type="submit" class="pfx-nav-btn">Home</button>
        </form>
        <form action="" method="get">
            <input type="hidden" name="page" value="browse">
            <button type="submit" class="pfx-nav-btn">Browse PFx</button>
        </form>
        <form action="" method="get">
            <input type="hidden" name="page" value="generate">
            <button type="submit" class="pfx-nav-btn">Generate</button>
        </form>
        </div>
        <style>
        .pfx-nav{
            display:flex;
            justify-content:flex-end;
            align-items:center;
            gap:8px;                 /* space between buttons */
        }
        .pfx-nav form{ margin:0; } /* remove default form margins */
        .pfx-nav-btn{
            padding:.5rem .9rem;
            border-radius:10px;
            border:1px solid #e5e7eb;
            background:#f0f2f6;
            color:#111;
            font-weight:600;
            cursor:pointer;
        }
        </style>
        """, unsafe_allow_html=True)
    st.divider()


# ====== STYLE ======
st.markdown(
    """
    <style>
      .pfx-card { background:#fff;border:1px solid #e5e7eb;border-radius:14px;
                  padding:18px 20px;min-height:120px;box-shadow:0 1px 3px rgba(0,0,0,.05);line-height:1.55; }
      .pfx-muted { color:#6b7280; }
      .pfx-meta { display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-top:12px; }
      .pfx-pill { border:1px solid #e5e7eb;border-radius:999px;padding:8px 12px;background:#fafafa;font-size:.92rem; }
      .pfx-toolbar a { text-decoration:none;background:#f0f2f6;padding:.55rem .9rem;border-radius:10px;border:1px solid #e5e7eb;font-weight:600;color:#111; }
      .pfx-toolbar { display:flex;gap:.5rem;justify-content:flex-end;margin-top:.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ====== GLOBAL HELPERS ======
def _safe_str(x) -> str:
    return ("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)).strip()

def _fmt_percent(val):
    try:
        f = float(val)
        return f"{(f*100) if 0 <= f <= 1 else f:.1f}%"
    except Exception:
        return _safe_str(val)

def _fmt_float(val):
    try:
        return f"{float(val):.1f}"
    except Exception:
        return _safe_str(val)

def _first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _copy_button(js_text: str, key: str, height: int = 60):
    txt = json.dumps(js_text)
    st_html(
        f"""<div style='margin-top:10px'>
              <button id='copy-pfx-btn-{key}' style='padding:8px 12px;border-radius:6px;border:1px solid #e5e7eb;background:#f0f2f6;cursor:pointer;font-weight:600;'>📋 Copy</button>
            </div>
            <script>
              (function(){{
                const btn = document.getElementById('copy-pfx-btn-{key}');
                const txt = {txt};
                if (btn) {{
                  btn.addEventListener('click', async () => {{
                    try {{ await navigator.clipboard.writeText(txt); }}
                    catch (e) {{
                      const ta=document.createElement('textarea'); ta.value=txt; document.body.appendChild(ta);
                      ta.select(); try {{ document.execCommand('copy'); }} catch(_){{
                      }} document.body.removeChild(ta);
                    }}
                    const msg=document.createElement('div');
                    msg.textContent='Copied!'; msg.style.cssText='position:fixed;bottom:24px;right:24px;background:#111;color:#fff;padding:6px 10px;border-radius:999px;font-size:12px;z-index:9999;';
                    document.body.appendChild(msg); setTimeout(()=>msg.remove(),1400);
                  }});
                }}
              }})();
            </script>""",
        height=height,
    )

REQUIRED_SCHEMA = [
    "finding",
    "ICD10_code",
    "PFx",
    "PFx_ICD10_code",
    "Flesch_Score",
    "accuracy",
]

def _ensure_schema(df: pd.DataFrame | dict | None) -> pd.DataFrame:
    """Ensure required columns exist, but preserve all other columns too."""
    if df is None:
        return pd.DataFrame(columns=REQUIRED_SCHEMA)

    if isinstance(df, dict):
        df = pd.DataFrame([df])

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame(columns=REQUIRED_SCHEMA)

    # Add any missing required columns
    for col in REQUIRED_SCHEMA:
        if col not in df.columns:
            df[col] = ""

    # Put required columns first; keep any extras afterward
    ordered_cols = [c for c in REQUIRED_SCHEMA if c in df.columns] + \
                   [c for c in df.columns if c not in REQUIRED_SCHEMA]
    return df[ordered_cols]

def _extract_pfx_text(df: pd.DataFrame | None) -> str:
    if df is None or "PFx" not in df.columns:
        return ""
    vals = [str(x).strip() for x in df["PFx"].fillna("").astype(str).tolist() if str(x).strip()]
    return "\n\n---\n".join(vals)

# ====== DATA LOAD / NORMALIZE ======
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

CSV_DIR = BASE_DIR / "Generated_PFx_CSVs"

WORKFLOW_FILES = {
    "Zero-shot":         CSV_DIR / "PFx_final - PFx_Zeroshot.csv",
    "Few-shot":          CSV_DIR / "PFx_final - PFx_Single_Fewshot.csv",
    "Multiple Few-shot": CSV_DIR / "PFx_final - PFx_Multiple_Few.csv",
    "Agentic":           CSV_DIR / "PFx_final - PFx_Agentic.csv",
}

LEGACY_FALLBACK = CSV_DIR / "pfx_source.csv"

@st.cache_data(show_spinner=False)
def _load_any_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, header=None)
        except Exception:
            return None

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for want in candidates:
        if want in lower_map:
            return lower_map[want]
    return None

def _normalize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # No-headers CSV fallback
    if all(str(c).startswith("Unnamed") for c in df.columns) and df.shape[1] >= 2:
        df = df.iloc[:, :6]
        df.columns = ["Finding", "PFx", "ICD10", "Accuracy", "Readability(FRES)", "FRES"][: df.shape[1]]
        for col in ["ICD10", "Accuracy", "Readability(FRES)", "FRES"]:
            if col not in df.columns:
                df[col] = None
        return df

    finding_col = _pick_col(df, ["finding", "name", "incidental finding", "finding_name", "title", "label"])
    pfx_col     = _pick_col(df, ["pfx", "explanation", "patient friendly explanation", "pfx_text", "answer", "output", "pf x"])
    icd_col     = _pick_col(df, ["icd10", "icd-10", "icd10_code", "icd code", "icd"])
    acc_col     = _pick_col(df, ["accuracy", "eval_accuracy", "is_correct", "correctness", "score"])
    read_col    = _pick_col(df, ["readability", "grade", "grade_level", "fkgl", "flesch_kincaid", "flesch-kincaid", "smog", "readability(fres)", "readability (fres)"])
    fres_col    = _pick_col(df, ["fres", "_0_flesch", "flesch reading ease", "flesch_reading_ease", "flesch reading-ease", "flesch score", "flesch"])

    cols = list(df.columns)
    if finding_col is None and len(cols) >= 1:
        finding_col = cols[0]
    if pfx_col is None and len(cols) >= 2:
        pfx_col = cols[1]

    out = pd.DataFrame({
        "Finding": df[finding_col].astype(str).str.strip() if finding_col else "",
        "PFx": df[pfx_col].astype(str) if pfx_col else "",
        "ICD10": df[icd_col].astype(str) if icd_col else None,
        "Accuracy": df[acc_col] if acc_col else None,
        "Readability(FRES)": df[read_col].astype(str) if read_col else None,
        "FRES": df[fres_col] if fres_col else None,
    })

    out = out.dropna(subset=["Finding"]).copy()
    out["Finding"] = out["Finding"].str.strip()
    out = out.drop_duplicates(subset=["Finding"], keep="first")
    return out

@st.cache_data(show_spinner=False)
def _load_all_workflows(workflow_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for wf, path in workflow_files.items():
        raw = _load_any_csv(path)
        if raw is not None:
            datasets[wf] = _normalize_dataframe(raw)
    if not datasets:
        legacy = _load_any_csv(LEGACY_FALLBACK)
        if legacy is not None:
            datasets["Zero-shot"] = _normalize_dataframe(legacy)
    return datasets

DATASETS = _load_all_workflows(WORKFLOW_FILES)

# ====== MODEL/READING OPTIONS ======
MODEL_OPTIONS = [
    "gpt-4o-2024-08-06",
    "gpt-4o-mini",
]
READING_LEVELS = [
    "PROFESSIONAL",
    "COLLEGE_GRADUATE",
    "COLLEGE",
    "TENTH_TO_TWELTH_GRADE",
    "EIGTH_TO_NINTH_GRADE",
    "SEVENTH_GRADE",
    "SIXTH_GRADE",
    "FIFTH_GRADE",
]

# ====== PAGES ======
def page_home():
    # optional: keep your existing _top_nav if you want the title only (no buttons)
    _top_nav("home")  # shows title/caption only in your current setup

    # ===== Hero =====
    st.header("Making medical reports understandable.")
    st.write(
        "OpenPFx turns complex radiology findings into **clear, patient-friendly explanations** "
        "that provide context without medical jargon."
    )
    st.caption(
        "Explanations are generated through multiple AI workflows, with the **agentic workflow "
        "outperforming others in testing for accuracy and readability**."
    )
    st.markdown(
        "- **Clarity, not jargon:** Plain-language explanations of imaging findings.\n"
        "- **Context you can trust:** Understand what was seen in your scan, explained simply.\n"
        "- **Transparent process:** Every explanation includes ICD-10 code, readability score, and an accuracy check."
    )
    st.divider()

    # ===== The Problem =====
    st.subheader("The Problem")
    st.write(
        "Radiology reports are written for doctors, not patients. When patients read them, the technical "
        "language around incidental findings can be confusing and overwhelming. OpenPFx makes these findings "
        "understandable by providing clear, patient-friendly context."
    )
    st.divider()

    # ===== The Solution =====
    st.subheader("The Solution")
    st.write(
        "**Patient-Friendly Explanations (PFx):** OpenPFx automatically generates accessible explanations of incidental findings. "
        "Each explanation provides general context in everyday language, alongside key details like coding and readability."
    )
    st.divider()

    # ===== Browse PFx =====
    st.subheader("Browse PFx")
    st.write(
        "Explore a growing library of **pre-generated explanations** for hundreds of incidental findings. "
        "These PFx were created through OpenPFx workflows and curated for patients and clinicians. "
        "Search by finding, compare workflows, and copy explanations for easy sharing."
    )
    c1, c2, c3 = st.columns([1, 1, 6])
    with c1:
        if st.button("Browse PFx →", type="primary", use_container_width=True, key="home_browse_btn"):
            _set_page("browse")
    st.divider()

    # ===== Generate Your Own =====
    st.subheader("Generate Your Own")
    st.write(
        "If your finding isn’t in the library, create a new explanation instantly. "
        "Enter the finding and choose a workflow to produce a patient-friendly version."
    )
    st.markdown(
        "> ⚠️ **ICD-10 note for this mode:** the suggested ICD-10 code may not always be correct. "
        "Codes are provided for general context only and should be confirmed with your clinician."
    )
    c4, c5, c6 = st.columns([1, 1, 6])
    with c4:
        if st.button("Generate PFx →", type="primary", use_container_width=True, key="home_generate_btn"):
            _set_page("generate")
    st.divider()

    # ===== Workflows Explained =====
    st.subheader("Workflows Explained")
    st.write("OpenPFx offers four approaches for generating explanations:")
    st.markdown(
        "- **Zero-shot** — Quick, direct explanation with no prior examples.\n"
        "- **Few-shot** — Uses reference examples to improve consistency.\n"
        "- **Multiple Few-shot** — Adds a broader set of examples for more stability.\n"
        "- **Agentic** — A multi-agent process with built-in checks for coding, accuracy, and readability. "
        "In testing, **this workflow outperformed all others**."
    )
    st.divider()

    # ===== What You’ll See =====
    st.subheader("What You’ll See")
    st.markdown(
        "- A **patient-friendly summary** of the finding\n"
        "- An **ICD-10 code** corresponding to the finding\n"
        "- A **readability score** (Flesch) so you know the language is approachable\n"
        "- An **accuracy check** for extra confidence"
    )
    st.divider()

    # ===== Global ICD-10 Disclaimer =====
    st.subheader("Important Note on ICD-10 Codes")
    st.write(
        "All ICD-10 codes shown in OpenPFx are **general** and are not specific to a patient’s individual case. "
        "They are provided for context only and should be confirmed with your clinician before use in medical or administrative decisions."
    )
    st.divider()

    # ===== Accuracy & Safety =====
    st.subheader("Accuracy & Safety")
    st.write(
        "OpenPFx provides **general context only**. It is not personalized medical advice and should never replace "
        "a conversation with your clinician."

        "All PFx are generated by AI and may contain inaccuracies or omissions."
    )
    st.markdown("> Always consult your doctor about what your imaging results mean for you.")
    st.divider()

    # ===== FAQ =====
    st.subheader("FAQ")
    with st.expander("What is an incidental finding?"):
        st.write("An unexpected result that appears on a scan but wasn’t related to the original reason for imaging.")
    with st.expander("What does OpenPFx provide?"):
        st.write(
            "Plain-language context around incidental findings — via a **Browse** library of pre-generated PFx and "
            "a **Generate Your Own** tool for on-demand explanations."
        )
    with st.expander("Why are there multiple workflows?"):
        st.write(
            "Each workflow represents a different approach. The **agentic** workflow has been the most accurate and "
            "readable in testing."
        )
    with st.expander("How reliable are the ICD-10 codes?"):
        st.write(
            "They’re general codes for context and may not match the specifics of your case. "
            "Always confirm with your clinician."
        )

    st.caption("© OpenPFx — built for patients and clinicians.")

def page_browse():
    _top_nav("browse")

    st.subheader("Browse Patient-Friendly Explanations")
    st.caption("PLACEHOLDER: pick a workflow and select one or more findings to view PFx.")

    if not DATASETS:
        st.error("No datasets found. Place your CSV files next to this app file.")
        return

    left, right = st.columns([1, 2], gap="large")

    with left:
        workflow_names = list(DATASETS.keys())
        workflow = st.selectbox("Workflow", workflow_names, index=0, key="browse_workflow")
        df = DATASETS[workflow]

        finding_col = _first_col(df, ["Finding", "finding"])
        if not finding_col:
            st.error("No 'Finding' column found in the selected dataset.")
            return

        options = sorted(df[finding_col].dropna().astype(str).unique().tolist())
        selected = st.multiselect("Select one or more findings", options, key="browse_findings")

    with right:
        st.subheader("Patient-Friendly Explanation")
        if not selected:
            st.markdown(
                "<div class='pfx-card pfx-muted'>PLACEHOLDER: Select findings on the left to view PFx.</div>",
                unsafe_allow_html=True,
            )
            return

        pfx_cols  = ["PFx", "pfx", "PFx_text"]
        icd_cols  = ["ICD10", "ICD10_code", "PFx_ICD10_code"]
        acc_cols  = ["Accuracy", "accuracy"]
        read_cols = ["Readability(FRES)", "Readability (FRES)", "FRES", "Flesch_Score"]

        for j, f in enumerate(selected):
            row_q = df[df[finding_col].astype(str) == str(f)]
            if row_q.empty:
                st.caption(f"⚠️ No row found for: {f}")
                continue
            row = row_q.iloc[0]

            # PFx text
            pfx_text = ""
            for c in pfx_cols:
                if c in df.columns:
                    pfx_text = _safe_str(row.get(c))
                    if pfx_text:
                        break

            # Meta fields
            icd10 = ""
            for c in icd_cols:
                if c in df.columns:
                    icd10 = _safe_str(row.get(c))
                    if icd10:
                        break

            acc_str = ""
            for c in acc_cols:
                if c in df.columns:
                    acc_str = _fmt_percent(row.get(c))
                    if acc_str:
                        break

            read_str = ""
            for c in read_cols:
                if c in df.columns:
                    val = row.get(c)
                    if c in ("FRES", "Flesch_Score", "Readability(FRES)", "Readability (FRES)"):
                        read_str = _fmt_float(val)
                    else:
                        read_str = _safe_str(val)
                    if read_str:
                        break

            st.markdown(f"### {f}")
            st.markdown(
                f"<div class='pfx-card'>{pfx_text if pfx_text else '<span class=\"pfx-muted\">PLACEHOLDER: PFx text missing for this item.</span>'}</div>",
                unsafe_allow_html=True,
            )
            if pfx_text:
                _copy_button(pfx_text, key=f"copy_browse_{j}")

            pills = []
            if icd10:   pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10}</div>")
            if acc_str: pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
            if read_str:pills.append(f"<div class='pfx-pill'><b>Readability (FRES):</b> {read_str}</div>")
            if pills:
                st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)

            if j < len(selected) - 1:
                st.divider()

def page_generate():
    _top_nav("generate")

    st.subheader("Generate Your Own PFx")
    st.caption("PLACEHOLDER: Choose model & workflow, enter a finding, optionally use auto ICD-10.")

    if "gen_panel_count" not in st.session_state:
        st.session_state.gen_panel_count = 1

    left, right = st.columns([1, 2], gap="large")

    # ---------- LEFT: inputs ----------
    with left:
        st.markdown("### Inputs")

        for i in range(st.session_state.gen_panel_count):
            st.markdown(f"#### Finding {i+1}")

            finding_key = f"gen_finding_{i}"
            icd_key     = f"gen_icd10_{i}"

            incidental_finding = st.text_input(
                "Incidental Finding",
                key=finding_key,
                placeholder="e.g., Hepatic hemangioma",
            )

            ai_model = st.selectbox(
                "Model",
                MODEL_OPTIONS,
                index=0,
                key=f"gen_model_{i}",
            )

            reading_level = st.selectbox(
                "Reading Level", READING_LEVELS, index=6, key=f"gen_reading_{i}"
            )

            workflow_options = ["All", "Zero-shot", "Few-shot", "Agentic"]
            workflow_choice = st.selectbox(
                "Workflow", workflow_options, index=0, key=f"gen_workflow_{i}"
            )

            # Auto-suggest ICD-10 BEFORE rendering widget
            if incidental_finding and not st.session_state.get(icd_key):
                try:
                    code = suggest_icd10_code(incidental_finding, ai_model)
                except Exception:
                    code = None
                if code:
                    st.session_state[icd_key] = code

            icd10_code = st.text_input(
                "ICD-10 Code (Optional)",
                key=icd_key,
                placeholder="e.g., D18.03",
            )

            if incidental_finding and st.session_state.get(icd_key):
                st.info(f'Auto-filled ICD-10: **{st.session_state[icd_key]}**')

            # Per-panel state
            if f"gen_df_{i}" not in st.session_state:
                st.session_state[f"gen_df_{i}"] = None
            if f"gen_pfx_{i}" not in st.session_state:
                st.session_state[f"gen_pfx_{i}"] = ""
            if f"gen_error_{i}" not in st.session_state:
                st.session_state[f"gen_error_{i}"] = ""

            # Generate
            if st.button("🚀 Generate PFx", type="primary", key=f"gen_btn_{i}"):
                try:
                    st.session_state[f"gen_error_{i}"] = None
                    st.session_state[f"gen_df_{i}"] = None
                    st.session_state[f"gen_pfx_{i}"] = ""

                    def _run_one(fn):
                        out = fn(incidental_finding, icd10_code, reading_level, ai_model)
                        return _ensure_schema(out)

                    if not _safe_str(incidental_finding):
                        raise ValueError("Please enter an Incidental Finding before generating.")

                    if workflow_choice == "Zero-shot":
                        df = _run_one(zeroshot_call)
                    elif workflow_choice == "Few-shot":
                        df = _run_one(fewshot_call)
                    elif workflow_choice == "Agentic":
                        df = _run_one(agentic_conversation)
                    elif workflow_choice == "All":
                        df_zero = _run_one(zeroshot_call);         df_zero["_workflow"] = "Zero-shot"
                        df_few  = _run_one(fewshot_call);          df_few["_workflow"]  = "Few-shot"
                        df_ag   = _run_one(agentic_conversation);  df_ag["_workflow"]   = "Agentic"
                        df = pd.concat([df_zero, df_few, df_ag], ignore_index=True)
                    else:
                        df = _ensure_schema(None)

                    st.session_state[f"gen_df_{i}"]  = df
                    st.session_state[f"gen_pfx_{i}"] = _extract_pfx_text(df)
                    if df.empty:
                        st.session_state[f"gen_error_{i}"] = "No results returned by the selected workflow(s)."

                except Exception as e:
                    st.session_state[f"gen_error_{i}"] = f"Error during generation: {e}"

        # Add / Reset
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("➕ Add another finding", use_container_width=True, key="gen_add"):
                st.session_state.gen_panel_count = min(st.session_state.get("gen_panel_count", 1) + 1, 10)
                st.rerun()
        with b2:
            if st.button("↺ Reset", use_container_width=True, key="gen_reset"):
                for k in list(st.session_state.keys()):
                    if k.startswith((
                        "gen_finding_", "gen_icd10_", "gen_reading_", "gen_workflow_",
                        "gen_model_", "gen_btn_", "gen_df_", "gen_pfx_", "gen_error_"
                    )):
                        del st.session_state[k]
                st.session_state.gen_panel_count = 1
                st.rerun()

    # ---------- RIGHT: outputs ----------
    with right:
        for i in range(st.session_state.gen_panel_count):
            wf_choice_i = st.session_state.get(f"gen_workflow_{i}", "Zero-shot")
            df_out = st.session_state.get(f"gen_df_{i}")
            err    = st.session_state.get(f"gen_error_{i}")

            if err:
                st.error(err)

            if wf_choice_i == "All":
                if isinstance(df_out, pd.DataFrame) and not df_out.empty and "_workflow" in df_out.columns:
                    order = ["Zero-shot", "Few-shot", "Agentic"]
                    for g in [g for g in order if g in df_out["_workflow"].unique()]:
                        row = df_out[df_out["_workflow"] == g].iloc[0]
                        finding_name = (st.session_state.get(f"gen_finding_{i}") or "").strip() or f"Finding {i+1}"
                        st.markdown(f"### {finding_name} — {g} Explanation")
                        pfx_text = (row.get("PFx") or "").strip()
                        st.markdown(f"<div class='pfx-card'>{pfx_text}</div>", unsafe_allow_html=True)
                        if pfx_text:
                            _copy_button(pfx_text, key=f"copy_all_{g}_{i}")

                        icd10   = (row.get("ICD10_code") or "").strip()
                        acc_str = _fmt_percent(row.get("accuracy"))
                        fres_str= _fmt_float(row.get("Flesch_Score"))

                        pills = []
                        if icd10:
                            pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10}</div>")
                        if acc_str:
                            pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
                        if fres_str:
                            pills.append(f"<div class='pfx-pill'><b>Flesch:</b> {fres_str}</div>")
                        if pills:
                            st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)


                    st.markdown("### Generation Details")
                    try:
                        st.download_button(
                            "Download results (CSV)",
                            data=df_out.to_csv(index=False).encode("utf-8"),
                            file_name=f"pfx_generated_{i+1}.csv",
                            mime="text/csv",
                            key=f"dl_all_{i}",
                        )
                    except Exception:
                        pass

            else:
                finding_name = (st.session_state.get(f"gen_finding_{i}") or "").strip() or f"Finding {i+1}"
                st.markdown(f"### {finding_name} — Explanation")
                pfx_text = (st.session_state.get(f"gen_pfx_{i}") or "").strip()
                st.markdown(
                    f"<div class='pfx-card'>{pfx_text if pfx_text else '<span class=\"pfx-muted\">PLACEHOLDER: Your PFx will appear here once generated.</span>'}</div>",
                    unsafe_allow_html=True,
                )
                if pfx_text:
                    _copy_button(pfx_text, key=f"copy_single_{i}")

                if isinstance(df_out, pd.DataFrame) and not df_out.empty:
                    row = df_out.iloc[0]
                    icd10   = (row.get("ICD10_code") or "").strip()
                    acc_str = _fmt_percent(row.get("accuracy"))
                    fres_str= _fmt_float(row.get("Flesch_Score"))

                    pills = []
                    if icd10:
                        pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10}</div>")
                    if acc_str:
                        pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
                    if fres_str:
                        pills.append(f"<div class='pfx-pill'><b>Flesch:</b> {fres_str}</div>")
                    if pills:
                        st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)

                    st.markdown("### Generation Details")
                    try:
                        st.download_button(
                            "Download results (CSV)",
                            data=df_out.to_csv(index=False).encode("utf-8"),
                            file_name=f"pfx_generated_{i+1}.csv",
                            mime="text/csv",
                            key=f"dl_single_{i}",
                        )
                    except Exception:
                        pass

# ====== ROUTER ======
_page = _get_page("home")
if _page == "home":
    page_home()
elif _page == "browse":
    page_browse()
elif _page == "generate":
    page_generate()
else:
    _set_page("home")