import pandas as pd
import re
import dotenv
import os
import streamlit as st
from pathlib import Path
import json
from streamlit.components.v1 import html as st_html

OPENAI_MODEL = os.getenv("OPENAI_MODEL")

from jh_pfx_prompts import example, icd10_example, single_fewshot_icd10_labeling_prompt, baseline_zeroshot_prompt, writer_prompt,doctor_prompt, readability_checker_prompt, ICD10_LABELER_INSTRUCTION

# import fewshot examples
df_fewshot = pd.read_csv('jh_main/pfx_fewshot_examples_college.csv')

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="PFx: Patient Friendly Explanations",
    page_icon="💬",
    layout="wide",
)

# ==========================
# Helpers
# ==========================
def _get_query_param(name: str, default: str = "") -> str:
    """Resilient way to read a query parameter across Streamlit versions."""
    try:
        qs = st.query_params  # new API (1.31+)
        val = qs.get(name)
        if isinstance(val, list):
            return val[0] if val else default
        return val if val is not None else default
    except Exception:
        try:
            qs = st.experimental_get_query_params()  # legacy API
            return (qs.get(name, [default]) or [default])[0]
        except Exception:
            return default

# Base directory for local CSVs (works both in `streamlit run` and notebooks)
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

# Files expected next to this script
WORKFLOW_FILES = {
    "Zero-shot":         BASE_DIR / "PFx_final - PFx_Zeroshot.csv",
    "Few-shot":          BASE_DIR / "PFx_final - PFx_Single_Fewshot.csv",
    "Multiple Few-shot": BASE_DIR / "PFx_final - PFx_Multiple_Few.csv",
    "Agentic":           BASE_DIR / "PFx_final - PFx_Agentic.csv",
}

LEGACY_FALLBACK = BASE_DIR / "pfx_source.csv"

# ==========================
# Session State (stable keys)
# ==========================
for k, v in {
    "gen_error": None,
    "generated_df": None,
    "generated_pfx": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

model_options = [
    "gpt-4o-2024-08-06", 
    "gpt-4o-mini",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano"
]

# Reading Level options (UI-only for now)
PROFESSIONAL = "PROFESSIONAL"
COLLEGE_GRADUATE = "COLLEGE_GRADUATE"
COLLEGE = "COLLEGE"
TENTH_TO_TWELTH_GRADE = "TENTH_TO_TWELTH_GRADE"
EIGTH_TO_NINTH_GRADE = "EIGTH_TO_NINTH_GRADE"
SEVENTH_GRADE = "SEVENTH_GRADE"
SIXTH_GRADE = "SIXTH_GRADE"
FIFTH_GRADE = "FIFTH_GRADE"

READING_LEVELS = [
    PROFESSIONAL,
    COLLEGE_GRADUATE,
    COLLEGE,
    TENTH_TO_TWELTH_GRADE,
    EIGTH_TO_NINTH_GRADE,
    SEVENTH_GRADE,
    SIXTH_GRADE,
    FIFTH_GRADE,
]

# ==========================
# Utilities for result handling
# ==========================
import pandas as pd

REQUIRED_SCHEMA = ["finding", "ICD10_code", "PFx", "PFx_ICD10_code"]

def _ensure_schema(df: pd.DataFrame | dict | None) -> pd.DataFrame:
    """Normalize any return (DataFrame/dict/None) into a DF with REQUIRED_SCHEMA."""
    if df is None:
        return pd.DataFrame(columns=REQUIRED_SCHEMA)
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return pd.DataFrame(columns=REQUIRED_SCHEMA)
    for col in REQUIRED_SCHEMA:
        if col not in df.columns:
            df[col] = ""
    return df[REQUIRED_SCHEMA]

def _extract_pfx_text(df: pd.DataFrame | None) -> str:
    if df is None or "PFx" not in df.columns:
        return ""
    vals = [str(x).strip() for x in df["PFx"].fillna("").astype(str).tolist() if str(x).strip()]
    return "\n\n---\n".join(vals)

# ==========================
# Data Loading + Normalization
# ==========================
@st.cache_data(show_spinner=False)
def load_any_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, header=None)
        except Exception:
            return None


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for want in candidates:
        if want in lower_map:
            return lower_map[want]
    return None


def normalize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # If it's a bare CSV exported without headers
    if all(str(c).startswith("Unnamed") for c in df.columns) and df.shape[1] >= 2:
        df = df.iloc[:, :6]
        df.columns = [
            "Finding",
            "PFx",
            "ICD10",
            "Accuracy",
            "Readability(FRES)",
            "FRES",
        ][: df.shape[1]]
        for col in ["ICD10", "Accuracy", "Readability(FRES)", "FRES"]:
            if col not in df.columns:
                df[col] = None
        return df

    # Flexible header matching
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
    out["Finding"] = out["Finding"].stripped if hasattr(out["Finding"], 'stripped') else out["Finding"].str.strip()
    out = out.drop_duplicates(subset=["Finding"], keep="first")
    return out


@st.cache_data(show_spinner=False)
def load_all_workflows(workflow_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for wf, path in workflow_files.items():
        raw = load_any_csv(path)
        if raw is not None:
            datasets[wf] = normalize_dataframe(raw)
    if not datasets:
        legacy = load_any_csv(LEGACY_FALLBACK)
        if legacy is not None:
            datasets["Zero-shot"] = normalize_dataframe(legacy)
    return datasets

# Load datasets once
DATASETS = load_all_workflows(WORKFLOW_FILES)

# ==========================
# Global Styles
# ==========================
st.markdown(
    """
    <style>
      .pfx-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px 20px; min-height: 160px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); line-height: 1.55; }
      .pfx-muted { color: #6b7280; }
      .pfx-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 12px; }
      .pfx-pill { border: 1px solid #e5e7eb; border-radius: 999px; padding: 8px 12px; background: #fafafa; font-size: 0.92rem; }
      .pfx-toolbar a { text-decoration:none; background:#f0f2f6; padding:0.55rem 0.9rem; border-radius:10px; border:1px solid #e5e7eb; font-weight:600; color:#111; }
      .pfx-toolbar { display:flex; gap:.5rem; justify-content:flex-end; margin-top:.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# Top Header
# ==========================
lcol, rcol = st.columns([1, 1], gap="small")
with lcol:
    st.title("PFx: Patient Friendly Explanations")
    st.markdown("- Choose a **workflow** (Zero-shot, Few-shot, Multiple Few-shot, Agentic), then a **finding**.")
with rcol:
    st.markdown(
        """
        <div class="pfx-toolbar">
            <a href="?page=home" target="_self">Home</a>
            <a href="?page=generate" target="_self">Generate Your Own</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================
# Page Router
# ==========================
page = _get_query_param("page", "home").strip().lower()

if "panel_count" not in st.session_state:
    st.session_state.panel_count = 1

# ---------- Shared UI bits ----------
def copy_button(js_text: str, key: str, height: int = 60):
    import json
    # Escape js_text safely for JS
    safe_js_text = json.dumps(js_text)

    st_html(
        f"""<div style='margin-top:10px'>
              <button id='copy-pfx-btn-{key}' style='padding:8px 12px;border-radius:6px;border:1px solid #e5e7eb;background:#f0f2f6;cursor:pointer;font-weight:600;'>📋 Copy PFx</button>
            </div>
            <script>
              (function(){{
                const btn = document.getElementById('copy-pfx-btn-{key}');
                const txt = {safe_js_text};
                if (btn) {{
                  btn.addEventListener('click', async () => {{
                    try {{
                      await navigator.clipboard.writeText(txt);
                    }} catch (e) {{
                      const ta = document.createElement('textarea');
                      ta.value = txt; document.body.appendChild(ta); ta.select();
                      try {{ document.execCommand('copy'); }} catch(_) {{}}
                      document.body.removeChild(ta);
                    }}
                    const msg = document.createElement('div');
                    msg.textContent = 'Copied to Clipboard!';
                    msg.style.cssText = 'position:fixed;bottom:24px;right:24px;background:#111;color:#fff;padding:6px 10px;border-radius:999px;font-size:12px;z-index:9999;box-shadow:0 4px 12px rgba(0,0,0,.15);';
                    document.body.appendChild(msg);
                    setTimeout(() => msg.remove(), 2000);
                  }});
                }}
              }})();
            </script>""",
        height=height,
    )



def render_home_panel(idx: int):
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Workflow & Finding")
        if not DATASETS:
            st.error("No datasets found. Please place the four CSV files next to this file.")
            return

        workflow_names = list(DATASETS.keys())
        workflow = st.selectbox("Select workflow", workflow_names, index=0, key=f"wf_{idx}")
        df = DATASETS[workflow]

        # find the 'finding' column across possible schemas
        finding_col = _first_col(df, ["Finding", "finding"])
        if not finding_col:
            st.error("No 'Finding' column found in the selected dataset.")
            return

        # build options cleanly
        options = sorted(df[finding_col].dropna().astype(str).unique().tolist())

        # true multiselect (no sentinel value)
        selected_findings = st.multiselect(
            "Select one or more findings",
            options,
            key=f"findings_{idx}",
        )

    with right:
        st.subheader("Patient-Friendly Explanation")

        if not selected_findings:
            st.markdown(
                "<div class='pfx-card pfx-muted'>Pick a workflow and one or more findings on the left to view the PFx.</div>",
                unsafe_allow_html=True,
            )
            return

        # column fallbacks for content/metrics
        pfx_cols = ["PFx", "pfx", "PFx_text"]
        icd_cols = ["ICD10", "ICD10_code", "PFx_ICD10_code"]
        acc_cols = ["Accuracy", "accuracy"]
        read_cols = ["Readability(FRES)", "Readability (FRES)", "FRES", "Flesch_Score"]

        for j, f in enumerate(selected_findings):
            # row lookup
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

            # meta fields
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
            # prefer readability value formatted to 1 decimal if numeric
            for c in read_cols:
                if c in df.columns:
                    val = row.get(c)
                    # If column name implies Flesch score, format as float; otherwise just show string
                    if c in ("FRES", "Flesch_Score", "Readability(FRES)", "Readability (FRES)"):
                        read_str = _fmt_float(val)
                    else:
                        read_str = _safe_str(val)
                    if read_str:
                        break

            # header for each selected finding
            st.markdown(f"### {f}")

            # PFx card
            st.markdown(
                f"<div class='pfx-card'>{pfx_text if pfx_text else '<span class=\"pfx-muted\">No PFx text found for this item.</span>'}</div>",
                unsafe_allow_html=True,
            )

            # copy button if PFx present
            if pfx_text:
                js_text = json.dumps(pfx_text)
                copy_button(js_text, key=f"home-{idx}-{j}")

            # pills
            pills = []
            if icd10:
                pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10}</div>")
            if acc_str:
                pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
            if read_str:
                pills.append(f"<div class='pfx-pill'><b>Readability (FRES):</b> {read_str}</div>")

            if pills:
                st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)
            else:
                st.caption("No advanced stats available for this entry.")

            if j < len(selected_findings) - 1:
                st.divider()

# ==========================
# GENERATE PAGE (LLM-INTEGRATED)
# ==========================

elif page == "generate":
    st.subheader("Generate Your Own PFx")
    st.caption("Select workflow, enter details, and generate a patient-friendly explanation.")

    # how many generate panels to render
    if "gen_panel_count" not in st.session_state:
        st.session_state.gen_panel_count = 1

    left, right = st.columns([1, 2], gap="large")

    # ---------- LEFT: inputs (multiple panels) ----------
    with left:
        st.markdown("### Inputs")

        for i in range(st.session_state.gen_panel_count):
            st.markdown(f"#### Finding {i+1}")

            incidental_finding = st.text_input(
                "Incidental Finding", key=f"gen_finding_{i}",
                placeholder="e.g., Hepatic hemangioma"
            )
            icd10_code = st.text_input(
                "ICD-10 Code", key=f"gen_icd10_{i}",
                placeholder="e.g., D18.03"
            )
            reading_level = st.selectbox(
                "Reading Level", READING_LEVELS, index=6, key=f"gen_reading_{i}"
            )
            workflow_options = ["Zero-shot", "Few-shot", "Agentic", "All"]
            workflow_choice = st.selectbox(
                "Workflow", workflow_options, index=0, key=f"gen_workflow_{i}"
            )
            ai_model = st.selectbox(
                "Model", model_options, index=0, key=f"gen_model_{i}"
            )

            # per-panel state
            if f"gen_df_{i}" not in st.session_state:
                st.session_state[f"gen_df_{i}"] = None
            if f"gen_pfx_{i}" not in st.session_state:
                st.session_state[f"gen_pfx_{i}"] = ""
            if f"gen_error_{i}" not in st.session_state:
                st.session_state[f"gen_error_{i}"] = ""

            # Generate
            if st.button("🚀 Generate PFx", type="primary", key=f"gen_btn_{i}"):
                st.session_state[f"gen_error_{i}"] = None
                st.session_state[f"gen_df_{i}"] = None
                st.session_state[f"gen_pfx_{i}"] = ""

                if not incidental_finding.strip():
                    st.session_state[f"gen_error_{i}"] = "Please enter an Incidental Finding before generating."
                else:
                    try:
                        from streamlit_calls import (
                            zeroshot_call, fewshot_call, agentic_conversation,
                        )

                        def _run_one(fn):
                            out = fn(incidental_finding, icd10_code, reading_level, ai_model)
                            return _ensure_schema(out)

                        if workflow_choice == "Zero-shot":
                            df = _run_one(zeroshot_call)

                        elif workflow_choice == "Few-shot":
                            df = _run_one(fewshot_call)

                        elif workflow_choice == "Agentic":
                            df = _run_one(agentic_conversation)

                        elif workflow_choice == "All":
                            # run all three and tag them so we can label in the UI
                            df_zero = _run_one(zeroshot_call);      df_zero["_workflow"] = "Zero-shot"
                            df_few  = _run_one(fewshot_call);       df_few["_workflow"]  = "Few-shot"
                            df_ag   = _run_one(agentic_conversation); df_ag["_workflow"]   = "Agentic"
                            df = pd.concat([df_zero, df_few, df_ag], ignore_index=True)
                        else:
                            df = _ensure_schema(None)

                        st.session_state[f"gen_df_{i}"]  = df
                        st.session_state[f"gen_pfx_{i}"] = _extract_pfx_text(df)
                        if df.empty:
                            st.session_state[f"gen_error_{i}"] = "No results returned by the selected workflow(s)."

                    except Exception as e:
                        st.session_state[f"gen_error_{i}"] = f"Error during generation: {e}"

    btn_cols = st.columns([1, 1, 6], gap="small")

    with btn_cols[0]:
        if st.button("➕ Add another finding", use_container_width=True, key="gen_add"):
            st.session_state.gen_panel_count = min(st.session_state.get("gen_panel_count", 1) + 1, 10)
            st.rerun()

    with btn_cols[1]:
        if st.button("↺ Reset", use_container_width=True, key="gen_reset"):
            for k in list(st.session_state.keys()):
                if k.startswith(("gen_finding_", "gen_icd10_", "gen_reading_", "gen_workflow_",
                                "gen_model_", "gen_btn_", "gen_df_", "gen_pfx_", "gen_error_")):
                    del st.session_state[k]
            st.session_state.gen_panel_count = 1
            st.rerun()

                

    # ---------- RIGHT: outputs (multiple panels) ----------
    with right:
        def _fmt_percent(v):
            if pd.isna(v): return ""
            try:
                f = float(v)
                return f"{f*100:.1f}%" if 0.0 <= f <= 1.0 else f"{f:.1f}%"
            except Exception:
                return str(v).strip()

        def _fmt_num(v):
            if pd.isna(v): return ""
            try:    return f"{float(v):.1f}"
            except: return str(v).strip()

        for i in range(st.session_state.gen_panel_count):
            wf_choice_i = st.session_state.get(f"gen_workflow_{i}", "Zero-shot")
            df_out = st.session_state.get(f"gen_df_{i}")
            err    = st.session_state.get(f"gen_error_{i}")

            if err:
                st.error(err)

            # "All" -> three stacked sections
            if wf_choice_i == "All":
                if isinstance(df_out, pd.DataFrame) and not df_out.empty:
                    # ensure order Zero-shot, Few-shot, Agentic if tags exist
                    order = ["Zero-shot", "Few-shot", "Agentic"]
                    if "_workflow" in df_out.columns:
                        groups = [g for g in order if g in df_out["_workflow"].unique()]
                        for g in groups:
                            row = df_out[df_out.get("_workflow") == g].iloc[0]
                            st.markdown(f"<div style='padding:4px 0;font-weight:600;font-size:1rem;color:#000;'>{g}</div>", unsafe_allow_html=True)
                            pfx_text = (row.get("PFx") or "").strip()
                            st.markdown(f"<div class='pfx-card'>{pfx_text}</div>", unsafe_allow_html=True)
                            # copy button
                            copy_button(json.dumps(pfx_text), key=f"copy_all_{g}_{i}")
                            # pills
                            icd10 = (row.get("ICD10_code") or "").strip()
                            acc_str = _fmt_percent(row.get("accuracy"))
                            fres_str = _fmt_num(row.get("Flesch_Score"))
                            pills = []
                            if icd10:   pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10}</div>")
                            if acc_str: pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
                            if fres_str:pills.append(f"<div class='pfx-pill'><b>Flesch:</b> {fres_str}</div>")
                            if pills: st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)
                            st.divider()
                    # CSV for this panel
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

            # Single workflow -> single card + pills
            else:
                st.markdown(f"### Patient-Friendly Explanation ({i+1})")
                pfx_text = (st.session_state.get(f"gen_pfx_{i}") or "").strip()
                card_html = f"<div class='pfx-card'>{pfx_text if pfx_text else '<span class=\"pfx-muted\">Your PFx will appear here once generated.</span>'}</div>"
                st.markdown(card_html, unsafe_allow_html=True)
                if pfx_text:
                    copy_button(json.dumps(pfx_text), key=f"copy_single_{i}")

                if isinstance(df_out, pd.DataFrame) and not df_out.empty:
                    row = df_out.iloc[0]
                    icd10   = (row.get("ICD10_code") or "").strip()
                    acc_str = _fmt_percent(row.get("accuracy"))
                    fres_str= _fmt_num(row.get("Flesch_Score"))
                    pills = []
                    if icd10:   pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10}</div>")
                    if acc_str: pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
                    if fres_str:pills.append(f"<div class='pfx-pill'><b>Flesch:</b> {fres_str}</div>")
                    if pills: st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)
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

# ==========================
# Unknown Page -> Fallback
# ==========================
else:
    st.info("Unknown page. Use the buttons above to navigate.")

import pandas as pd
import os
import textstat
from openai import OpenAI
CLIENT = OpenAI()
import json
import re
import requests
from dotenv import load_dotenv
import math
import unicodedata

# import fewshot examples
df_fewshot = pd.read_csv('jh_main/pfx_fewshot_examples_college.csv')

# import prompts 
from jh_pfx_prompts import example, icd10_example, baseline_zeroshot_prompt, single_fewshot_icd10_labeling_prompt

from autogen import LLMConfig
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import RoundRobinPattern
from autogen.agentchat.group import OnCondition, StringLLMCondition
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group import TerminateTarget

from pydantic import BaseModel, Field
from typing import Optional
from typing import Annotated

from call_functions import extract_json, label_icd10s, extract_json_gpt4o
from tools import calculate_fres

from typing import Annotated

# import necessary libraries 
import pandas as pd
import os
import textstat
from openai import OpenAI
import json
import re
import requests
from dotenv import load_dotenv
import math
import unicodedata

from jh_pfx_prompts import example, icd10_example, single_fewshot_icd10_labeling_prompt, baseline_zeroshot_prompt, writer_prompt,doctor_prompt, readability_checker_prompt, ICD10_LABELER_INSTRUCTION