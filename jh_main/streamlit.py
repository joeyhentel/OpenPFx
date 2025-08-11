# streamlit.py
"""
PFx: Patient Friendly Explanations — single-page app

This page combines:
- **Browse PFx**: pick workflow → select finding → view PFx and Advanced stats
- **Generate Your Own**: choose workflow (Zero-shot, Few-shot, Agentic), enter Finding + ICD-10, click Generate → PFx card + Advanced stats

Notes
- CSVs expected next to this file (portable paths).
- Readability(FRES) pill shows Flesch Reading Ease (from column or computed).
- `textstat` is optional; a fallback FRES is computed if it isn't installed.
- For generation, set `OPENAI_API_KEY` in Streamlit Secrets or env. `OPENAI_MODEL` optional (default: gpt-4o-mini).
"""

from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# ==========================
# Page config & header
# ==========================
st.set_page_config(page_title="PFx: Patient Friendly Explanations", page_icon="💬", layout="wide")

lcol, rcol = st.columns([1, 1], gap="small")
with lcol:
    st.title("PFx: Patient Friendly Explanations")
    st.markdown(
        "- Choose a **workflow** (Zero-shot, Few-shot, Multiple Few-shot, Agentic), then a **finding**.\n"
        "- The PFx card displays the explanation; enable **Advanced stats** to see ICD-10, accuracy, and Readability (FRES).\n"
        "- Column names in your CSVs are auto-detected (e.g., `Finding`, `PFx/Explanation`, `ICD10`, `Accuracy`, `Readability`, `_0_flesch`)."
    )
with rcol:
    st.markdown("\u00A0")  # spacer to keep layout tidy

# ==========================
# Portable file configuration
# ==========================
BASE_DIR = Path(__file__).resolve().parent

WORKFLOW_FILES = {
    "Zero-shot":         BASE_DIR / "PFx_final - PFx_Zeroshot.csv",
    "Few-shot":          BASE_DIR / "PFx_final - PFx_Single_Fewshot.csv",
    "Multiple Few-shot": BASE_DIR / "PFx_final - PFx_Multiple_Few.csv",
    "Agentic":           BASE_DIR / "PFx_final - PFx_Agentic.csv",
}

LEGACY_FALLBACK = BASE_DIR / "pfx_source.csv"

# ==========================
# Data loading & normalization
# ==========================
@st.cache_data(show_spinner=False)
def load_any_csv(path: Path) -> Optional[pd.DataFrame]:
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


def normalize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    # No-header two+ cols
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
    read_col    = _pick_col(df, ["readability", "grade", "grade_level", "fkgl", "flesch_kincaid", "flesch-kincaid", "smog"])
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


datasets = load_all_workflows(WORKFLOW_FILES)

# ==========================
# Shared UI CSS
# ==========================
st.markdown(
    """
    <style>
    .pfx-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px 20px; min-height: 160px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); line-height: 1.55; }
    .pfx-muted { color: #6b7280; }
    .pfx-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 12px; }
    .pfx-pill { border: 1px solid #e5e7eb; border-radius: 999px; padding: 8px 12px; background: #fafafa; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================
# FRES helpers (textstat optional)
# ==========================
try:
    import textstat  # type: ignore
    HAS_TEXTSTAT = True
except Exception:
    HAS_TEXTSTAT = False

VOWEL_RE = re.compile(r"[aeiouy]+", re.I)


def _syllables(word: str) -> int:
    w = word.lower()
    if not w:
        return 0
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    if w.endswith("e"):
        w = w[:-1]
    groups = VOWEL_RE.findall(w)
    return max(1, len(groups))


def _fallback_fres(text: str) -> float:
    sentences = max(1, len(re.findall(r"[.!?]", text)) or 1)
    words_list = re.findall(r"\b[\w'-]+\b", text)
    words = max(1, len(words_list))
    syllables = sum(_syllables(w) for w in words_list) or 1
    return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)


def compute_fres(text: str) -> Optional[float]:
    try:
        if HAS_TEXTSTAT:
            return float(textstat.flesch_reading_ease(text))
        return float(_fallback_fres(text))
    except Exception:
        return None

# ==========================
# OpenAI helpers (generation)
# ==========================
from openai import OpenAI

try:
    from jh_pfx_prompts import (
        baseline_zeroshot_prompt,
        icd10_example,
        single_fewshot_icd10_labeling_prompt,
    )
except Exception:
    baseline_zeroshot_prompt = None
    icd10_example = None
    single_fewshot_icd10_labeling_prompt = None

FEWSHOT_EXAMPLES_CSV = BASE_DIR / "pfx_fewshot_examples_college.csv"


def get_openai_client() -> OpenAI:
    return OpenAI()


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None


def label_icd10s(pfx_output: str) -> Optional[Dict[str, Any]]:
    if single_fewshot_icd10_labeling_prompt is None or icd10_example is None:
        return None
    try:
        df_fewshot = pd.read_csv(FEWSHOT_EXAMPLES_CSV)
    except Exception:
        return None
    examples = ""
    for _, row in df_fewshot.iterrows():
        try:
            examples += icd10_example.format(**row)
        except Exception:
            continue
    prompt = single_fewshot_icd10_labeling_prompt.format(examples=examples, PFx=pfx_output)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are an ICD10 medical coder for incidental findings. Always respond with a valid JSON object containing the ICD-10 code and its explanation."},
            {"role": "system", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    return extract_json_from_text(content)


def generate_zeroshot_pfx(finding: str, reading_level: str = "6th grade") -> Dict[str, Any]:
    if baseline_zeroshot_prompt is None:
        return {"PFx": "(Error: baseline_zeroshot_prompt not available)"}
    prompt = baseline_zeroshot_prompt.format(Incidental_Finding=finding, Reading_Level=reading_level)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a medical professional rephrasing and explaining medical terminology to a patient in an understandable manner."},
            {"role": "system", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content if resp and resp.choices else ""
    data = extract_json_from_text(content) or {}
    pfx_text = data.get("PFx") if isinstance(data, dict) else None
    if not isinstance(pfx_text, str) or not pfx_text.strip():
        pfx_text = content.strip()
        data = {"PFx": pfx_text}
    return data

# ==========================
# UI — Tabs: Browse and Generate on one page
# ==========================

browse_tab, generate_tab = st.tabs(["Browse PFx", "Generate Your Own"])

with browse_tab:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Workflow & Finding")
        if not datasets:
            st.error("No datasets found. Place CSVs next to this file or adjust paths.")
            st.stop()
        workflow_names = list(datasets.keys())
        workflow = st.selectbox("Select workflow", workflow_names, index=0, key="wf_browse")
        df = datasets[workflow]
        options = df["Finding"].tolist()
        finding = st.selectbox("Select a finding", ["— Select —"] + options, index=0, key="finding_browse")
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
            show_stats = st.checkbox("Show advanced stats (ICD-10, accuracy, Readability(FRES))", value=False, key="adv_browse")
            if show_stats:
                icd10 = (row.get("ICD10") or "").strip()
                acc_val = row.get("Accuracy")
                acc_str = ""
                if pd.notna(acc_val):
                    try:
                        f_acc = float(acc_val)
                        acc_str = f"{f_acc*100:.1f}%" if 0 <= f_acc <= 1 else f"{f_acc:.1f}%"
                    except Exception:
                        acc_str = str(acc_val)
                read_str = (row.get("Readability(FRES)") or "").strip()
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
                    # Show either provided readability text or FRES number (or both)
                    label = read_str if read_str else fres_str
                    pills.append(f"<div class='pfx-pill'><b>Readability(FRES):</b> {label}</div>")
                if pills:
                    st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)
                else:
                    st.caption("No advanced stats available for this entry.")
        else:
            st.markdown("<div class='pfx-card pfx-muted'>Pick a workflow and finding on the left to view the PFx.</div>", unsafe_allow_html=True)

with generate_tab:
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Inputs")
        workflow_gen = st.selectbox("Workflow", ["Zero-shot", "Few-shot", "Agentic"], index=0, key="wf_gen")
        finding_gen = st.text_input("Finding name", value="", placeholder="e.g., Simple renal cyst", key="finding_gen")
        icd10_input = st.text_input("ICD-10 code", value="", placeholder="e.g., N28.1", key="icd10_gen")
        can_generate = bool(workflow_gen and finding_gen.strip() and icd10_input.strip())
        generate_clicked = st.button("Generate", disabled=not can_generate, key="btn_gen")

    with right:
        st.subheader("Patient-Friendly Explanation")
        if generate_clicked and can_generate:
            if workflow_gen == "Zero-shot":
                result = generate_zeroshot_pfx(finding_gen, reading_level="6th grade")
            elif workflow_gen == "Few-shot":
                st.info("Few-shot generation not wired yet. Falling back to Zero-shot.")
                result = generate_zeroshot_pfx(finding_gen, reading_level="6th grade")
            else:
                st.info("Agentic generation not wired yet. Falling back to Zero-shot.")
                result = generate_zeroshot_pfx(finding_gen, reading_level="6th grade")

            pfx_text = (result.get("PFx") or "").strip()
            if pfx_text:
                st.markdown(f"<div class='pfx-card'>{pfx_text}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='pfx-card pfx-muted'>No PFx text generated.</div>", unsafe_allow_html=True)

            show_stats_gen = st.checkbox("Show advanced stats (ICD-10, accuracy, Readability(FRES))", value=False, key="adv_gen")
            if show_stats_gen:
                icd10_user = icd10_input.strip()
                labeled = label_icd10s(pfx_text) or {}
                icd10_llm = labeled.get("ICD10") or labeled.get("ICD-10") or labeled.get("code") or ""
                match = (icd10_user.upper().strip() == str(icd10_llm).upper().strip()) if icd10_llm else False
                acc_str = "100.0%" if match else ("0.0%" if icd10_llm else "—")
                fres_val = compute_fres(pfx_text)
                fres_str = f"{fres_val:.1f}" if isinstance(fres_val, (float, int)) else ""
                pills = []
                if icd10_user:
                    pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10_user}</div>")
                if acc_str and acc_str != "—":
                    pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
                if fres_str:
                    pills.append(f"<div class='pfx-pill'><b>Readability(FRES):</b> {fres_str}</div>")
                if pills:
                    st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)
                else:
                    st.caption("No advanced stats available.")
        else:
            st.markdown("<div class='pfx-card pfx-muted'>Fill out all fields on the left and click Generate.</div>", unsafe_allow_html=True)