import re
import pandas as pd 
import os
import dotenv
from jh_pfx_prompts import example, icd10_example, single_fewshot_icd10_labeling_prompt, baseline_zeroshot_prompt, writer_prompt,doctor_prompt, readability_checker_prompt, ICD10_LABELER_INSTRUCTION
from openai import OpenAI
CLIENT = OpenAI()
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
import json
import unicodedata

# import fewshot examples
df_fewshot = pd.read_csv('jh_main/pfx_fewshot_examples_college.csv')

def label_icd10s(pfx_output):
    """
    Takes a single PFx response (string or JSON) and returns
    the ICD-10 code string (e.g., 'R93.0'). Returns '' if none found.
    """

    import math

    # ---- Normalize PFx text ----
    if isinstance(pfx_output, dict):
        pfx_text = str(pfx_output.get("PFx", "") or "").strip()
    else:
        pfx_text = str(pfx_output or "").strip()

    # ---- Build few-shot examples (handle NaNs) ----
    parts = []
    for _, row in df_fewshot.iterrows():
        mapping = {
            k: ("" if (isinstance(v, float) and math.isnan(v)) else v)
            for k, v in row.items()
        }
        parts.append(icd10_example.format(**mapping))
    pfx_icd10_fewshot_examples = "".join(parts)

    # ---- Compose prompt ----
    prompt = single_fewshot_icd10_labeling_prompt.format(
        examples=pfx_icd10_fewshot_examples,
        PFx=pfx_text
    )

    # ---- LLM call ----
    pfx_icd10_response = CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an ICD-10 medical coder for incidental findings. "
                    "Always respond with a valid JSON object containing the ICD-10 code and its explanation."
                ),
            },
            {
                "role": "user",  # send prompt here
                "content": prompt,
            },
        ],
        stream=False,
    )

    # ---- Extract JSON dict ----
    labeled_result = extract_json(pfx_icd10_response) or {}

    # ---- Return only the code string ----
    return str(labeled_result.get("ICD10_code", "") or "").strip()

import json, re

def _first_balanced_json(text: str) -> str:
    """Return the first balanced {...} JSON substring (ignores braces inside strings)."""
    in_str = False
    esc = False
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start:i+1]
    return ""

def extract_json_from_text(content: str) -> dict:
    """Parse JSON from model text. Supports ```json fences, ``` fences, or inline {..}."""
    if not content:
        return {}
    # 1) ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", content, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        # 2) generic ``` ... ```
        m = re.search(r"```\s*(\{.*?\})\s*```", content, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 3) first balanced JSON object inline
    candidate = _first_balanced_json(content)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return {}

def extract_json(openai_response) -> dict:
    """
    Robustly extract a JSON dict from an OpenAI Chat Completions response or from a raw string.
    Always returns a dict ({} on failure).
    """
    # If caller already passed a string, parse it directly.
    if isinstance(openai_response, str):
        return extract_json_from_text(openai_response)

    # Try to get message content from a Chat Completions response
    content = ""
    try:
        # OpenAI SDK objects
        content = getattr(openai_response.choices[0].message, "content", "") or ""
    except Exception:
        # Fallbacks: dict-like / other shapes
        try:
            choices = openai_response.get("choices", [])
            if choices and "message" in choices[0]:
                content = choices[0]["message"].get("content", "") or ""
        except Exception:
            content = ""

    return extract_json_from_text(content)

    
def extract_json_gpt4o(chat_result, verbose=False):
    messages = getattr(chat_result, "chat_history", None) or getattr(chat_result, "messages", [])

    for msg in reversed(messages):
        name = msg.get("name", "").lower()
        if name != "icd10_labeler":
            continue

        content = msg.get("content", "").strip()
        content = unicodedata.normalize("NFKC", content)

        if verbose:
            print(f"[DEBUG] Raw content from {name}:\n{content}")

        content = re.sub(r"```(?:json)?", "", content).strip("` \n")

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # fallback with simpler, safe regex
        json_candidates = re.findall(r"\{.*?\}", content, re.DOTALL)
        for candidate in json_candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        if verbose:
            print(f"[WARN] No valid JSON in {name}'s message.")
        return None

    print("[WARN] No message from 'icd10_labeler' found.")
    return None

