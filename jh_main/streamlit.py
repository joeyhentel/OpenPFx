import pandas as pd
import re
import streamlit as st
from pathlib import Path
import json
from streamlit.components.v1 import html as st_html

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

model_options = [
    "gpt-3.5-turbo",
    "gpt-4o", 
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
    st.markdown(f"#### Finding {idx+1}")
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Workflow & Finding")
        if not DATASETS:
            st.error("No datasets found. Please place the four CSV files next to this file.")
            return
        workflow_names = list(DATASETS.keys())
        workflow = st.selectbox("Select workflow", workflow_names, index=0, key=f"wf_{idx}")
        df = DATASETS[workflow]
        options = df["Finding"].tolist()
        finding = st.selectbox("Select a finding", ["— Select —"] + options, index=0, key=f"finding_{idx}")
        finding = None if finding == "— Select —" else finding

    with right:
        st.subheader("Patient-Friendly Explanation")
        if finding:
            row = df.loc[df["Finding"] == finding].iloc[0]
            pfx_text = (row.get("PFx") or "").strip()
            st.markdown(
                f"<div class='pfx-card'>{pfx_text if pfx_text else '<span class=\\"pfx-muted\\">No PFx text found for this item.</span>'}</div>",
                unsafe_allow_html=True,
            )
            if pfx_text:
                js_text = json.dumps(pfx_text)
                copy_button(js_text, key=f"home-{idx}")

            icd10 = (row.get("ICD10") or "").strip()
            acc_val = row.get("Accuracy")
            acc_str = ""
            if pd.notna(acc_val):
                try:
                    f_acc = float(acc_val)
                    acc_str = f"{f_acc*100:.1f}%" if 0 <= f_acc <= 1 else f"{f_acc:.1f}%"
                except Exception:
                    acc_str = str(acc_val)

            read_key_options = ["Readability(FRES)", "Readability (FRES)"]
            read_str = ""
            for k in read_key_options:
                v = row.get(k)
                if v is not None and str(v).strip() != "":
                    read_str = str(v).strip()
                    break

            fres_val = row.get("FRES")
            fres_str = ""
            if pd.notna(fres_val):
                try:
                    fres_str = f"{float(fres_val):.1f}"
                except Exception:
                    fres_str = str(fres_val)

            pills = []
            if icd10:
                pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10}</div>")
            if acc_str:
                pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
            if read_str or fres_str:
                pills.append(f"<div class='pfx-pill'><b>Readability(FRES):</b> {read_str} {fres_str}</div>")

            if pills:
                st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)
            else:
                st.caption("No advanced stats available for this entry.")
        else:
            st.markdown("<div class='pfx-card pfx-muted'>Pick a workflow and finding on the left to view the PFx.</div>", unsafe_allow_html=True)


# ==========================
# HOME PAGE
# ==========================
if page in ("", "home"):
    for i in range(st.session_state.panel_count):
        render_home_panel(i)
        if i < st.session_state.panel_count - 1:
            st.divider()

    btn_cols = st.columns([1, 1, 6])
    with btn_cols[0]:
        if st.button("➕ Add another finding", use_container_width=True):
            st.session_state.panel_count = min(st.session_state.panel_count + 1, 10)
            st.rerun()
    with btn_cols[1]:
        if st.button("↺ Reset", use_container_width=True):
            keys_to_clear = [k for k in list(st.session_state.keys()) if k.startswith("wf_") or k.startswith("finding_")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.session_state.panel_count = 1
            st.rerun()

# ==========================
# GENERATE PAGE (LLM-INTEGRATED)
# ==========================

elif page == "generate":
    st.subheader("Generate Your Own PFx")
    st.caption("Select workflow, enter details, and generate a patient-friendly explanation.")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("### Inputs")
        incidental_finding = st.text_input("Incidental Finding", placeholder="e.g., Hepatic hemangioma")
        icd10_code = st.text_input("ICD-10 Code", placeholder="e.g., D18.03")
        reading_level = st.selectbox("Reading Level", READING_LEVELS, index=6)  # default to SIXTH_GRADE

        # Workflow selector for generation UI
        workflow_options = ["Zero-shot", "Few-shot", "Agentic", "All"]
        workflow_choice = st.selectbox("Workflow", workflow_options, index=0)

        ai_model = st.selectbox("Model", model_options, index = 0)

        # Generate button
        generate_clicked = st.button("🚀 Generate PFx", type="primary")

        # Initialize session state holders
        if "generated_pfx" not in st.session_state:
            st.session_state.generated_pfx = ""
        if "generated_df" not in st.session_state:
            st.session_state.generated_df = None
        if "gen_error" not in st.session_state:
            st.session_state.gen_error = ""

        if generate_clicked:
            if not incidental_finding:
                st.warning("Please enter an Incidental Finding before generating.")
            else:
                # Local import to avoid changing global imports
                try:
                    from streamlit_calls import zeroshot_call, fewshot_call, agentic_conversation
                except Exception as e:
                    st.session_state.gen_error = f"Couldn't import generation functions from streamlit_calls.py: {e}"
                    st.session_state.generated_pfx = ""
                    st.session_state.generated_df = None
                else:
                    try:
                        if workflow_choice == "Zero-shot":
                            df_result = zeroshot_call(incidental_finding, icd10_code, reading_level, ai_model)
                        elif workflow_choice == "Few-shot":
                            df_result = fewshot_call(incidental_finding, icd10_code, reading_level, ai_model)
                        elif workflow_choice == "Agentic":
                            df_result = agentic_conversation(incidental_finding, icd10_code, reading_level, ai_model)
                        else:  # "All" -> run all and concat
                            df_zero = zeroshot_call(incidental_finding, icd10_code, reading_level, ai_model)
                            df_few  = fewshot_call(incidental_finding, icd10_code, reading_level, ai_model)
                            df_ag   = agentic_conversation(incidental_finding, icd10_code, reading_level, ai_model)
                            import pandas as _pd  # local alias to avoid polluting namespace
                            parts = [d for d in [df_zero, df_few, df_ag] if d is not None]
                            df_result = _pd.concat(parts, ignore_index=True) if parts else None

                        # Persist results
                        if df_result is not None and hasattr(df_result, "empty") and not df_result.empty:
                            st.session_state.generated_df = df_result
                            # Prefer PFx column if present
                            try:
                                st.session_state.generated_pfx = str(df_result.iloc[0]["PFx"]).strip()
                            except Exception:
                                st.session_state.generated_pfx = ""
                            st.session_state.gen_error = ""
                        else:
                            st.session_state.generated_df = None
                            st.session_state.generated_pfx = ""
                            st.session_state.gen_error = "No results returned by the selected workflow(s)."
                    except Exception as e:
                        st.session_state.generated_df = None
                        st.session_state.generated_pfx = ""
                        st.session_state.gen_error = f"Error during generation: {e}"

    with right:
        st.markdown("### Patient-Friendly Explanation")
        if st.session_state.get("gen_error"):
            st.error(st.session_state.gen_error)

        pfx_text = (st.session_state.get("generated_pfx") or "").strip()
        card_html = (
            f"<div class='pfx-card'>{pfx_text if pfx_text else '<span class=\\"pfx-muted\\">Your PFx will appear here once generated.</span>'}</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)
        if pfx_text:
            js_text = json.dumps(pfx_text)
            copy_button(js_text, key="gen")

        # Show details table and allow download if available
        df_out = st.session_state.get("generated_df")
        if df_out is not None:
            st.markdown("### Generation Details")
            st.dataframe(df_out, use_container_width=True)
            try:
                csv_bytes = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("Download results (CSV)", data=csv_bytes, file_name="pfx_generated.csv", mime="text/csv")
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
