from call_functions import extract_json, label_icd10s, extract_json_gpt4o
from tools import calculate_fres
import re
import pandas as pd
from jh_pfx_prompts import example, icd10_example, single_fewshot_icd10_labeling_prompt, baseline_zeroshot_prompt, writer_prompt,doctor_prompt, readability_checker_prompt, ICD10_LABELER_INSTRUCTION

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat import initiate_group_chat
from autogen.agentchat.group.patterns import RoundRobinPattern
from autogen.agentchat.group import OnCondition, StringLLMCondition
from autogen.agentchat.group import AgentTarget
from autogen.agentchat.group import TerminateTarget

from pydantic import BaseModel, Field
from typing import Optional
from typing import Annotated

# import necessary libraries 
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

OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# import fewshot examples
df_fewshot = pd.read_csv('jh_main/pfx_fewshot_examples_college.csv')

# calls LLM & creates dataframe with results
def zeroshot_call(finding, code, grade_level, ai_model):
    import pandas as pd
    import re

    # Prepare empty dataframe
    results_df = pd.DataFrame(columns=["finding", "ICD10_code", "PFx", "PFx_ICD10_code"])

    # Format the prompt
    prompt = baseline_zeroshot_prompt.format(
        Incidental_Finding=finding,
        Reading_Level=grade_level,
    )

    # Call the LLM
    pfx_response = CLIENT.chat.completions.create(
        model=ai_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a medical professional rephrasing and explaining medical terminology to a patient in an understandable manner."},
            {"role": "system", "content": prompt}
        ],
        stream=False,
    )

    # Extract structured JSON
    extracted_response = extract_json(pfx_response.choices[0]) or {}

    # Build row dictionary
    row_data = {
        "finding": finding,
        "ICD10_code": code,
        "PFx": extracted_response.get("PFx", ""),
        "PFx_ICD10_code": extracted_response.get("PFx_ICD10_code", "")
    }

    # Get agent-generated ICD10 code
    agent_code = label_icd10s(pfx_response)
    row_data["_0_agent_icd10_codes"] = agent_code

    # Compare only first three characters, forcing string conversion
    row_data["_0_icd10_matches"] = (
        str(row_data["ICD10_code"])[:3] == str(row_data["_0_agent_icd10_codes"])[:3]
    )
    row_data["_0_pfx_icd10_matches"] = (
        str(row_data["ICD10_code"])[:3] == str(row_data["PFx_ICD10_code"])[:3]
    )

    # Calculate accuracy
    row_data["accuracy"] = (
        int(row_data["_0_icd10_matches"]) + int(row_data["_0_pfx_icd10_matches"])
    ) / 2

    # Append to DataFrame
    results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)

    return results_df

# zeroshot prompts LLM & creates dataframe with results
def fewshot_call(finding, code, grade_level, ai_model):
    #reading levels
    PROFESSIONAL = "Professional"
    COLLEGE_GRADUATE = "College Graduate"
    COLLEGE = "College"
    TENTH_TO_TWELTH_GRADE = "10th to 12th grade"
    EIGTH_TO_NINTH_GRADE = "8th to 9th grade"
    SEVENTH_GRADE = "7th grade"
    SIXTH_GRADE = "6th grade"
    FIFTH_GRADE = "5th grade"
    N_A = "N/A"

    # import fewshot examples
    df_fewshot = pd.read_csv('pfx_fewshot_examples_college.csv')

    # import prompts 
    from jh_pfx_prompts import example, icd10_example, baseline_zeroshot_prompt, single_fewshot_icd10_labeling_prompt
    few_results_df = pd.DataFrame(columns=["finding", "ICD10_code", "PFx", "PFx_ICD10_code"])

    pfx_fewshot_examples = ""
    for i, row in df_fewshot.iterrows():
        pfx_fewshot_examples += example.format(**row) 

    prompt = single_fewshot_prompt.format(Examples = pfx_fewshot_examples, Incidental_Finding = row['Incidental_Finding'], Reading_Level = grade_level)

    pfx_response = CLIENT.chat.completions.create(
        model=ai_model,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a medical professional rephrasing and explaining medical terminology to a patient in an understandable manner."},
            {"role": "system", "content": prompt}
        ],
        stream=False,
    )

    extracted_response = extract_json(pfx_response.choices[0])

    few_results_df = {
        "finding": finding,
        "ICD10_code": code,
        "PFx": extracted_response.get("PFx", ""),
        "PFx_ICD10_code": extracted_response.get("PFx_ICD10_code", "")
    }

    agent_code = label_icd10s(pfx_response)

    few_results_df["_0_agent_icd10_codes"] = agent_code

    # Compare only the first three characters for accuracy
    few_results_df["_0_icd10_matches"] = (
        str(few_results_df["ICD10_code"])[:3] == str(few_results_df["_0_agent_icd10_codes"])[:3]
    )
    few_results_df["_0_pfx_icd10_matches"] = (
        str(few_results_df["ICD10_code"])[:3] == str(few_results_df["PFx_ICD10_code"])[:3]
    )

    few_results_df["accuracy"] = (
        few_results_df["_0_icd10_matches"] + few_results_df["_0_pfx_icd10_matches"]
    ) / 2

    return few_results_df


# agentic conversation & creates dataframe with results
def agentic_conversation(finding, code, grade_level, ai_model):
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
        model=ai_model,
        api_key=OPENAI_API_KEY,
    )

    writer_config=LLMConfig(
        api_type="openai",
        model=ai_model,
        api_key=OPENAI_API_KEY,
        response_format=WriterOutput,
    )

    labeler_config = LLMConfig(
        api_type="openai",
        model=ai_model,
        api_key=OPENAI_API_KEY,
        response_format=LabelerOutput,
    )

    doctor_config = LLMConfig(
        api_type="openai",
        model=ai_model,
        api_key=OPENAI_API_KEY,
        response_format=DoctorReadabilityOutput,
    )

    readability_config=LLMConfig(
        api_type="openai",
        model=ai_model,
        api_key=OPENAI_API_KEY,
        response_format=DoctorReadabilityOutput,
        
    )

    agent_results = pd.DataFrame(columns=["finding", "ICD10_code", "PFx", "PFx_ICD10_Code"])

    with llm_config:
        writer = ConversableAgent(
            name = "writer",
            system_message = writer_prompt.format(Incidental_Finding = finding, Reading_Level = grade_level),
            llm_config = writer_config,
        )
    
        icd10_labeler = ConversableAgent(
            name = "icd10_labeler",
            system_message = ICD10_LABELER_INSTRUCTION,
            llm_config = labeler_config,
            code_execution_config=False,
        )
    
        doctor = ConversableAgent( 
            name = "Doctor",
            system_message = doctor_prompt.format(Incidental_Finding = finding, ICD10_code = code),
            llm_config = doctor_config,
            code_execution_config=False,
        )
    
        readability_checker = ConversableAgent(
            name = "Readability_Checker",
            system_message = readability_checker_prompt.format(reading_level = grade_level),
            llm_config = readability_config,
            code_execution_config=False,
            functions=[calculate_fres],
        )
    
        pattern = RoundRobinPattern(
            initial_agent = writer,
            agents = [writer, icd10_labeler, doctor, readability_checker],
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
                condition=StringLLMCondition(prompt="""If the response is medically inaccuare or the original and pfx_icd10_codes are signifigantly different, 
                send the response back to the writer agent with an explanation of why it was sent back and suggestions for improvement in medical accuracy."""),
            ),
        ])

        readability_checker.handoffs.add_llm_conditions([
            OnCondition(
                target=AgentTarget(writer),
                condition=StringLLMCondition("""If the response does not meet the criteria for the desired reading level, send it back to the writer agent
                with an explanation of why it wasn't readable enough and suggestions for improving the readability."""),
            ),
            OnCondition(
              target=TerminateTarget(),
                condition=StringLLMCondition("If the response meets the readability criteria, send it to TerminateTarget."),
            ),
        ])
    

        result, context, last_agent = initiate_group_chat(
            pattern = pattern,
            messages = """Please play your specified role in generating a patient friendly explanation of an inicidental MRI finding.""",
            max_rounds = 20,
        )

        chat = extract_json_gpt4o(result)

        agent_results.loc[i] = {
        "finding": finding,
        "ICD10_code": code,
        "PFx": chat.get("PFx", ""),
        "PFx_ICD10_Code": chat.get("PFx_ICD10_Code", "")
        }

        agent_code = label_icd10s(pfx_response)

        agent_results["_0_agent_icd10_codes"] = agent_code

        # Compare only the first three characters for accuracy
        agent_results_["_0_icd10_matches"] = (
            str(agent_results["ICD10_code"])[:3] == str(agent_results["_0_agent_icd10_codes"])[:3]
        )
        agent_results["_0_pfx_icd10_matches"] = (
            str(agent_results["ICD10_code"])[:3] == str(agent_results["PFx_ICD10_code"])[:3]
        )

        agent_results["accuracy"] = (
            agent_results["_0_icd10_matches"] + agent_results["_0_pfx_icd10_matches"]
        ) / 2

        return agent_results

