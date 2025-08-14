import re
import pandas as pd 
import os
import dotenv
from jh_pfx_prompts import example, icd10_example, single_fewshot_icd10_labeling_prompt, baseline_zeroshot_prompt, writer_prompt,doctor_prompt, readability_checker_prompt, ICD10_LABELER_INSTRUCTION
from openai import OpenAI
CLIENT = OpenAI()
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
import json

# import fewshot examples
df_fewshot = pd.read_csv('jh_main/pfx_fewshot_examples_college.csv')

def label_icd10s(pfx_output):
    """
    Takes a single PFx response (string or JSON) and returns
    a labeled ICD-10 result as a Python dictionary (or object).
    """

    # Build up the few-shot examples for ICD-10 labeling
    pfx_icd10_fewshot_examples = ""
    for i, row in df_fewshot.iterrows():
        pfx_icd10_fewshot_examples += icd10_example.format(**row)

    # Generate the prompt for ICD-10 labeling
    # (Adjust the '{PFx}' if pfx_output is a dictionary with a specific key you need)
    prompt = single_fewshot_icd10_labeling_prompt.format(
        examples=pfx_icd10_fewshot_examples,
        PFx=pfx_output  # or PFx=pfx_output['key'] if needed
    )

    # Call the model to get ICD-10 codes
    pfx_icd10_response = CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": "You are an ICD10 medical coder for incidental findings. Always respond with a valid JSON object containing the ICD-10 code and its explanation."
            },
            {
                "role": "system",
                "content": prompt
            }
        ],
        stream=False,
    )

    # Extract the JSON structure (or dictionary) from the LLM response
    labeled_result = extract_json(pfx_icd10_response.choices[0])

    return labeled_result

# extract the json from openai
def extract_json(openai_response):
    if openai_response:  # Ensure the response is not None
        try:
            # Extract content from response object
            content = openai_response.message.content
            
            # Search for JSON within the content
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)  # Parse JSON string to Python dict
            else:
                print("No JSON found in response content.")
                return None
        except AttributeError as e:
            print(f"Attribute error: {e}. Ensure the input is a valid response object.")
            return None
    else:
        return None
    
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

