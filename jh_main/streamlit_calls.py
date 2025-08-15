# streamlit_calls.py
from __future__ import annotations

import os
import json
import pandas as pd
import textstat

from openai import OpenAI
CLIENT = OpenAI()

from dotenv import load_dotenv
load_dotenv()

from typing import Optional
from pydantic import BaseModel, Field

# Local helpers / prompts
from call_functions import extract_json, label_icd10s, extract_json_gpt4o
from tools import calculate_fres
from jh_pfx_prompts import (
    example,
    icd10_example,
    single_fewshot_icd10_labeling_prompt,
    baseline_zeroshot_prompt,
    writer_prompt,
    doctor_prompt,
    readability_checker_prompt,
    ICD10_LABELER_INSTRUCTION,
    single_fewshot_prompt,
)

# AutoGen (agentic)
from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import RoundRobinPattern
from autogen.agentchat.group import OnCondition, StringLLMCondition, AgentTarget, TerminateTarget

# --- Environment ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o-2024-08-06"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Few-shot data path
FEWSHOT_PATH = "jh_main/pfx_fewshot_examples_college.csv"


def _safe_flesch(text: str) -> Optional[float]:
    """Compute Flesch score on a string; return None on failure."""
    try:
        return float(textstat.flesch_reading_ease(str(text or "")))
    except Exception:
        return None


def _three_char_match(a: str, b: str) -> bool:
    """Case-insensitive comparison of first 3 characters of two strings."""
    a = str(a or "")[:3].upper()
    b = str(b or "")[:3].upper()
    return bool(a and b and a == b)


def _row_accuracy(code_str: str, agent_code: str, pfx_icd10: str) -> Optional[float]:
    """Average of two boolean matches; None if insufficient data."""
    have_any = bool(str(agent_code or "") or str(pfx_icd10 or ""))
    code_str = str(code_str or "")
    if not code_str or not have_any:
        return None
    m1 = _three_char_match(code_str, agent_code)
    m2 = _three_char_match(code_str, pfx_icd10)
    return (int(m1) + int(m2)) / 2.0


# ==========================
# Zero-shot
# ==========================
def zeroshot_call(finding: str, code: str, grade_level: str, ai_model: str) -> pd.DataFrame:
    """
    Returns a one-row DataFrame with columns:
    finding, ICD10_code, PFx, PFx_ICD10_code, _0_agent_icd10_codes,
    Flesch_Score, _0_icd10_matches, _0_pfx_icd10_matches, accuracy
    """
    cols = [
        "finding", "ICD10_code", "PFx", "PFx_ICD10_code",
        "_0_agent_icd10_codes", "Flesch_Score",
        "_0_icd10_matches", "_0_pfx_icd10_matches", "accuracy"
    ]
    out = pd.DataFrame(columns=cols)

    prompt = baseline_zeroshot_prompt.format(
        Incidental_Finding=finding,
        Reading_Level=grade_level,
    )

    pfx_response = CLIENT.chat.completions.create(
        model=ai_model or OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a medical professional rephrasing and explaining medical terminology to a patient in an understandable manner."},
            {"role": "system", "content": prompt},
        ],
        stream=False,
    )

    extracted = extract_json(pfx_response.choices[0]) or {}
    pfx_text = str(extracted.get("PFx", "") or "")
    pfx_icd10 = str(extracted.get("PFx_ICD10_code", "") or "")

    # Label ICD-10 from the PFx text (scalar)
    try:
        agent_code = str(label_icd10s(pfx_text) or "")
    except Exception:
        agent_code = ""

    flesch = _safe_flesch(pfx_text)
    code_str = str(code or "")

    _0_icd10_matches = _three_char_match(code_str, agent_code)
    _0_pfx_icd10_matches = _three_char_match(code_str, pfx_icd10)
    acc = _row_accuracy(code_str, agent_code, pfx_icd10)

    out.loc[0] = {
        "finding": finding,
        "ICD10_code": code_str,
        "PFx": pfx_text,
        "PFx_ICD10_code": pfx_icd10,
        "_0_agent_icd10_codes": agent_code,
        "Flesch_Score": flesch,
        "_0_icd10_matches": _0_icd10_matches,
        "_0_pfx_icd10_matches": _0_pfx_icd10_matches,
        "accuracy": acc,
    }

    return out


# ==========================
# Few-shot
# ==========================
def fewshot_call(finding: str, code: str, grade_level: str, ai_model: str) -> pd.DataFrame:
    """
    Returns a one-row DataFrame with the same columns as zeroshot_call.
    """
    cols = [
        "finding", "ICD10_code", "PFx", "PFx_ICD10_code",
        "_0_agent_icd10_codes", "Flesch_Score",
        "_0_icd10_matches", "_0_pfx_icd10_matches", "accuracy"
    ]
    out = pd.DataFrame(columns=cols)

    if not os.path.exists(FEWSHOT_PATH):
        raise FileNotFoundError(f"Fewshot examples file not found at {FEWSHOT_PATH}. Please check the path.")
    df_fewshot = pd.read_csv(FEWSHOT_PATH)

    # Build few-shot examples block (use dict to format)
    pfx_fewshot_examples = ""
    for _, r in df_fewshot.iterrows():
        pfx_fewshot_examples += example.format(**r.to_dict())

    prompt = single_fewshot_prompt.format(
        Examples=pfx_fewshot_examples,
        Incidental_Finding=finding,
        Reading_Level=grade_level,
    )

    pfx_response = CLIENT.chat.completions.create(
        model=ai_model or OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a medical professional rephrasing and explaining medical terminology to a patient in an understandable manner."},
            {"role": "system", "content": prompt},
        ],
        stream=False,
    )

    extracted = extract_json(pfx_response.choices[0]) or {}
    pfx_text = str(extracted.get("PFx", "") or "")
    pfx_icd10 = str(extracted.get("PFx_ICD10_code", "") or "")

    try:
        agent_code = str(label_icd10s(pfx_text) or "")
    except Exception:
        agent_code = ""

    flesch = _safe_flesch(pfx_text)
    code_str = str(code or "")

    _0_icd10_matches = _three_char_match(code_str, agent_code)
    _0_pfx_icd10_matches = _three_char_match(code_str, pfx_icd10)
    acc = _row_accuracy(code_str, agent_code, pfx_icd10)

    out.loc[0] = {
        "finding": finding,
        "ICD10_code": code_str,
        "PFx": pfx_text,
        "PFx_ICD10_code": pfx_icd10,
        "_0_agent_icd10_codes": agent_code,
        "Flesch_Score": flesch,
        "_0_icd10_matches": _0_icd10_matches,
        "_0_pfx_icd10_matches": _0_pfx_icd10_matches,
        "accuracy": acc,
    }

    return out


# ==========================
# Agentic conversation
# ==========================
def agentic_conversation(finding: str, code: str, grade_level: str, ai_model: str) -> pd.DataFrame:
    """
    Returns a one-row DataFrame with the same columns as zeroshot_call.
    Uses AutoGen agents (writer, icd10_labeler, doctor, readability_checker).
    """

    class WriterOutput(BaseModel):
        finding: str = Field(..., description="Name of incidental finding")
        ICD10_Code: str = Field(..., description="The ICD-10 code for the incidental finding")
        PFx: str = Field(..., description="Patient-friendly explanation of the finding")

    class LabelerOutput(BaseModel):
        finding: str = Field(..., description="Name of incidental finding")
        ICD10_Code: str = Field(..., description="The ICD-10 code given by writer")
        PFx: str = Field(..., description="The patient-friendly explanation given by writer")
        PFx_ICD10_Code: str = Field(..., description="The ICD-10 code you determine based off of PFx")

    class DoctorReadabilityOutput(BaseModel):
        Verdict: str = Field(..., description="Overall judgment about the explanation, accurate or inaccurate")
        Explanation: Optional[str] = Field(None, description="Why the verdict was given")
        Improvements: Optional[str] = Field(None, description="Suggested changes to improve readability or accuracy")

    llm_config = LLMConfig(
        api_type="openai",
        model=ai_model or OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
    )

    writer_config = LLMConfig(
        api_type="openai",
        model=ai_model or OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        response_format=WriterOutput,
    )

    labeler_config = LLMConfig(
        api_type="openai",
        model=ai_model or OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        response_format=LabelerOutput,
    )

    doctor_config = LLMConfig(
        api_type="openai",
        model=ai_model or OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        response_format=DoctorReadabilityOutput,
    )

    readability_config = LLMConfig(
        api_type="openai",
        model=ai_model or OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        response_format=DoctorReadabilityOutput,
    )

    cols = [
        "finding", "ICD10_code", "PFx", "PFx_ICD10_code",
        "_0_agent_icd10_codes", "Flesch_Score",
        "_0_icd10_matches", "_0_pfx_icd10_matches", "accuracy"
    ]
    agent_results = pd.DataFrame(columns=cols)

    with llm_config:
        writer = ConversableAgent(
            name="writer",
            system_message=writer_prompt.format(Incidental_Finding=finding, Reading_Level=grade_level),
            llm_config=writer_config,
        )

        icd10_labeler = ConversableAgent(
            name="icd10_labeler",
            system_message=ICD10_LABELER_INSTRUCTION,
            llm_config=labeler_config,
            code_execution_config=False,
        )

        doctor = ConversableAgent(
            name="Doctor",
            system_message=doctor_prompt.format(Incidental_Finding=finding, ICD10_code=code),
            llm_config=doctor_config,
            code_execution_config=False,
        )

        readability_checker = ConversableAgent(
            name="Readability_Checker",
            system_message=readability_checker_prompt.format(reading_level=grade_level),
            llm_config=readability_config,
            code_execution_config=False,
            functions=[calculate_fres],
        )

        pattern = RoundRobinPattern(
            initial_agent=writer,
            agents=[writer, icd10_labeler, doctor, readability_checker],
        )

        writer.handoffs.set_after_work(AgentTarget(icd10_labeler))
        icd10_labeler.handoffs.set_after_work(AgentTarget(doctor))

        doctor.handoffs.add_llm_conditions([
            OnCondition(
                target=AgentTarget(readability_checker),
                condition=StringLLMCondition(prompt="If the response is medically accurate, send the response to the readability_checker."),
            ),
            OnCondition(
                target=AgentTarget(writer),
                condition=StringLLMCondition(prompt="""If the response is medically inaccurate or the original and PFx_ICD10 codes are significantly different,
send it back to the writer with an explanation and suggestions for improving medical accuracy."""),
            ),
        ])

        readability_checker.handoffs.add_llm_conditions([
            OnCondition(
                target=AgentTarget(writer),
                condition=StringLLMCondition(prompt="""If the response does not meet the desired reading level, send it back to the writer with guidance to improve readability."""),
            ),
            OnCondition(
                target=TerminateTarget(),
                condition=StringLLMCondition(prompt="If the response meets the readability criteria, terminate."),
            ),
        ])

        result, context, last_agent = initiate_group_chat(
            pattern=pattern,
            messages="Please play your specified role in generating a patient friendly explanation of an incidental MRI finding.",
            max_rounds=20,
        )

    chat = extract_json_gpt4o(result) or {}

    pfx_text = str(chat.get("PFx", "") or "")
    # Normalize to the same output key name used elsewhere
    pfx_icd10 = str(chat.get("PFx_ICD10_Code", chat.get("PFx_ICD10_code", "")) or "")

    try:
        agent_code = str(label_icd10s(pfx_text) or "")
    except Exception:
        agent_code = ""

    flesch = _safe_flesch(pfx_text)
    code_str = str(code or "")

    _0_icd10_matches = _three_char_match(code_str, agent_code)
    _0_pfx_icd10_matches = _three_char_match(code_str, pfx_icd10)
    acc = _row_accuracy(code_str, agent_code, pfx_icd10)

    agent_results.loc[0] = {
        "finding": finding,
        "ICD10_code": code_str,
        "PFx": pfx_text,
        "PFx_ICD10_code": pfx_icd10,
        "_0_agent_icd10_codes": agent_code,
        "Flesch_Score": flesch,
        "_0_icd10_matches": _0_icd10_matches,
        "_0_pfx_icd10_matches": _0_pfx_icd10_matches,
        "accuracy": acc,
    }

    return agent_results


__all__ = ["zeroshot_call", "fewshot_call", "agentic_conversation"]
