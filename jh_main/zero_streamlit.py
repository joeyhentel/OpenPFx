# zero_streamlit.py
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import textstat
from openai import OpenAI

# ----- Paths -----
BASE_DIR = Path(__file__).resolve().parent
FEWSHOT_EXAMPLES_CSV = BASE_DIR / "pfx_fewshot_examples_college.csv"

# ----- Prompts -----
try:
    from jh_pfx_prompts import (
        baseline_zeroshot_prompt,
        icd10_example,
        single_fewshot_icd10_labeling_prompt,
    )
except Exception as e:
    baseline_zeroshot_prompt = None
    icd10_example = None
    single_fewshot_icd10_labeling_prompt = None


# ==========================
# Helpers
# ==========================

def get_openai_client() -> OpenAI:
    """Create an OpenAI client. Expects OPENAI_API_KEY in env/secrets."""
    # Streamlit Cloud: add to Secrets. Locally: export or use .env.
    # If using python-dotenv, you can uncomment below 2 lines.
    # from dotenv import load_dotenv
    # load_dotenv()
    return OpenAI()


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a string that may contain fenced code blocks."""
    if not text:
        return None
    # Try fenced block first
    m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # Fallback: try to parse the entire text as JSON
    try:
        return json.loads(text)
    except Exception:
        return None


def _render_card_css():
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
# ICD-10 labeling (few-shot) helper
# ==========================

def label_icd10s(pfx_output: str) -> Optional[Dict[str, Any]]:
    """
    Takes a single PFx string and returns a labeled ICD-10 result as a dict.
    Uses the few-shot examples CSV and `single_fewshot_icd10_labeling_prompt`.
    """
    if single_fewshot_icd10_labeling_prompt is None or icd10_example is None:
        return None

    try:
        df_fewshot = pd.read_csv(FEWSHOT_EXAMPLES_CSV)
    except Exception:
        return None

    # Build up the few-shot examples for ICD-10 labeling
    pfx_icd10_fewshot_examples = ""
    for _, row in df_fewshot.iterrows():
        try:
            pfx_icd10_fewshot_examples += icd10_example.format(**row)
        except Exception:
            # If a row is missing fields referenced by the template, skip it
            continue

    # Generate the prompt for ICD-10 labeling
    prompt = single_fewshot_icd10_labeling_prompt.format(
        examples=pfx_icd10_fewshot_examples,
        PFx=pfx_output,
    )

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an ICD10 medical coder for incidental findings. "
                    "Always respond with a valid JSON object containing the ICD-10 code and its explanation."
                ),
            },
            {"role": "system", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content if resp and resp.choices else ""
    return extract_json_from_text(content)


# ==========================
# Zero-shot PFx generation
# ==========================

def generate_zeroshot_pfx(finding: str, reading_level: str = "6th grade") -> Dict[str, Any]:
    """
    Call the model with the baseline zero-shot PFx prompt.
    Returns a dict with keys: PFx (str), PFx_ICD10_code (optional)
    """
    if baseline_zeroshot_prompt is None:
        return {"PFx": "(Error: baseline_zeroshot_prompt not available)"}

    prompt = baseline_zeroshot_prompt.format(
        Incidental_Finding=finding,
        Reading_Level=reading_level,
    )

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are a medical professional rephrasing and explaining medical terminology to a patient in an understandable manner.",
            },
            {"role": "system", "content": prompt},
        ],
    )

    content = resp.choices[0].message.content if resp and resp.choices else ""
    data = extract_json_from_text(content) or {}

    # Ensure PFx string exists
    pfx_text = data.get("PFx") if isinstance(data, dict) else None
    if not isinstance(pfx_text, str) or not pfx_text.strip():
        # If the model didn't give JSON, just return the raw content
        pfx_text = content.strip()
        data = {"PFx": pfx_text}
    return data


# ==========================
# Public entry: render the Generate page
# ==========================

def render_generate_page():
    st.subheader("Generate Your Own")

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.write("Select a workflow and provide inputs.")
        workflow = st.selectbox(
            "Workflow",
            ["Zero-shot", "Few-shot", "Agentic"],
            index=0,
        )
        finding = st.text_input("Finding name", value="", placeholder="e.g., Simple renal cyst")
        icd10_input = st.text_input("ICD-10 code", value="", placeholder="e.g., N28.1")

        can_generate = bool(workflow and finding.strip() and icd10_input.strip())
        generate_clicked = st.button("Generate", type="primary", disabled=not can_generate)

    with right:
        st.subheader("Patient-Friendly Explanation")
        _render_card_css()

        if generate_clicked and can_generate:
            # Run selected workflow
            if workflow == "Zero-shot":
                result = generate_zeroshot_pfx(finding, reading_level="6th grade")
                pfx_text = (result.get("PFx") or "").strip()
            elif workflow == "Few-shot":
                st.info("Few-shot generation not wired yet. Falling back to Zero-shot.")
                result = generate_zeroshot_pfx(finding, reading_level="6th grade")
                pfx_text = (result.get("PFx") or "").strip()
            else:  # Agentic
                st.info("Agentic generation not wired yet. Falling back to Zero-shot.")
                result = generate_zeroshot_pfx(finding, reading_level="6th grade")
                pfx_text = (result.get("PFx") or "").strip()

            # Render PFx card
            if pfx_text:
                st.markdown(f"<div class='pfx-card'>{pfx_text}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='pfx-card pfx-muted'>No PFx text generated.</div>", unsafe_allow_html=True)

            # Advanced stats
            show_stats = st.checkbox("Show advanced stats (ICD-10, accuracy, Readability(FRES))", value=False)
            if show_stats:
                # ICD-10 from user input
                icd10_user = icd10_input.strip()

                # Label ICD-10 from PFx via LLM (optional)
                labeled = label_icd10s(pfx_text) or {}
                icd10_llm = labeled.get("ICD10") or labeled.get("ICD-10") or labeled.get("code") or ""

                # Accuracy as a match metric
                match = (icd10_user.upper().strip() == str(icd10_llm).upper().strip()) if icd10_llm else False
                acc_str = "100.0%" if match else ("0.0%" if icd10_llm else "—")

                # Readability (FRES)
                try:
                    fres_val = float(textstat.flesch_reading_ease(pfx_text))
                    fres_str = f"{fres_val:.1f}"
                except Exception:
                    fres_str = ""

                # Render pills (match existing style)
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
            st.markdown(
                "<div class='pfx-card pfx-muted'>Fill out all fields on the left and click Generate.</div>",
                unsafe_allow_html=True,
            )