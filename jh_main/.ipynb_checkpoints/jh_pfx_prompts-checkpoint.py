from langchain.prompts import PromptTemplate

EXAMPLE = """ 

<Incidental Finding> 
{Incidental_Finding} 
</Incidental Finding>  

</PFx>
'''{{"Incidental_Finding":"Incidental_Finding", "ICD10_code":"{ICD10_code}", "PFx":"{PFx}", "PFx_ICD10_code":"{PFx_ICD10_code}"}}'''
</PFx>

"""

ICD10_EXAMPLE = """ <PFx>
{PFx}
</PFx>
<PFx_ICD10_code>
'''{{"PFx_ICD10_code":{PFx_ICD10_code}"}}'''
</PFx_ICD10_code>
"""

BASELINE_ZEROSHOT_INSTRUCTION = """

<Prompt>
Please generate new <PFx> for the <Incidental Finding>

Output should be formatted as a json with the following attributes/fields: finding, ICD10_code, PFx, PFx_ICD10_code 

Additional Instructions:
1. DO NOT SUGGEST FOLLOW UP STEPS WITH THE DOCTOR
2. Use the patient friendly explanation sentences to determine a PFx ICD10_code code.
3. Please generate PFx at a {Reading_Level} Flesch-Kincaid reading level.
4. Please output PFx in 100 words or more

</Prompt>

<Incidental Finding> 
{Incidental_Finding} 
</Incidental Finding>  
"""

# INCORPORATE ZEROSHOT_REFLEXION_READING_LEVEL


SINGLE_FEWSHOT_INSTRUCTION = """

<Prompt>
Please generate new <PFx> for the <Incidental Finding>

Output should be formatted as a json with the following attributes/fields: finding, ICD10_code, PFx, PFx_ICD10_code

Additional Instructions:
1. DO NOT SUGGEST FOLLOW UP STEPS WITH THE DOCTOR
2. Use the patient friendly explanation sentences to determine a PFx ICD10_code code.
3. Please generate PFx at a {Reading_Level} Flesch-Kincaid reading level.
4. Please output PFx in 100 words or more

</Prompt>

<Examples>
{Examples}
</Examples>

<Incidental Finding> 
{Incidental_Finding} 
</Incidental Finding>  
"""

SINGLE_FEWSHOT_ICD10_LABELING_INSTRUCTION = """
<Prompt>
Using the patient friendly explanations {PFx} in <Examples> as well as their associated ICD10 codes <ICD10>, please generate a new <ICD10> for the {PFx}

Output should be formatted as a json with the following attribute/field: ICD10_code 
</Prompt>

<Examples>
{examples} 
</Examples>

<PFx>
{PFx}
</PFx>

"""

example = PromptTemplate(
    input_variables = ["Incidental_Finding", "ICD10_code", "PFx", "PFx_ICD10_code"],
    template = EXAMPLE,
    )

icd10_example = PromptTemplate(
    input_variables = ["PFx_ICD10_code", "PFx"],
    template = ICD10_EXAMPLE,
    )

baseline_zeroshot_prompt = PromptTemplate(
    input_variables = ["Incidental_Finding"],
    template = BASELINE_ZEROSHOT_INSTRUCTION,
)

single_fewshot_prompt = PromptTemplate(
    input_variables = ["examples", "Incidental_Finding"],
    template = SINGLE_FEWSHOT_INSTRUCTION,
    )

single_fewshot_icd10_labeling_prompt = PromptTemplate(
    input_variables = ["examples", "PFx"],
    template = SINGLE_FEWSHOT_ICD10_LABELING_INSTRUCTION,
    )

# AUTOGEN PROMPTS BELOW HERE

DOCTOR_INSTRUCTION = """
Please examine the PFx to determine medical accuracy in explaining {Incidental_Finding}
Make sure the explanation aligns with the ICD-10 code: <ICD10_code>

- If the response is medically accurate, send the response to the icd10_checker. 
- If not, send the response back to the writer agent with:
  - **Brief reason** for inaccuracy
  - **Specific missing or incorrect details**

Format:
1. **Verdict:** ["ACCURATE - Send to Readability" / "INACCURATE - Revise"]
IF REQUIRED
2. **Reasoning:** [Short explanation]
3. **Fixes:** [Concise revision suggestions]

IF IT IS INACCURATE, YOUR OUTPUT MUST BEGIN WITH THE WORD INACCURATE

Do not engage in further conversation—simply forward the task as instructed.

"""

READABILITY_CHECKER_INSTRUCTION = """Check if the PFx matches the desired Flesch Reading Ease Score (FRES) {reading_level}.

- If it matches, say "All done!" to the terminator.
- If it does not match, return to the writer with:
  - A **brief** reason why it doesn’t match.
  - **Specific** changes needed to reach the desired level.

IF IT IS NOT READABLE, YOUR OUTPUT MUST BEGIN WITH: **NOT READABLE**
"""

ICD10_AGENT_INSTRUCTION = """Verify that PFx_ICD10_code matches ICD10_code:

Numbers before the decimal must match.
Numbers after the decimal may differ.
Next Steps:

If they match, send to Readability Checker.
If they do not match, return to Writer Agent with:
Brief mismatch explanation.
Correction instructions.
IMPORTANT:

Start response with MISMATCH if codes do not match.
No extra conversation—just validate and pass the task.
"""


doctor_prompt = PromptTemplate(
    imput_variables = ["Incidental_Finding", "ICD10_code"],
    template=DOCTOR_INSTRUCTION,
)

readability_checker_prompt = PromptTemplate(
    input_variables = ["reading_level"],
    template=READABILITY_CHECKER_INSTRUCTION,
)

icd10_checker_prompt = ICD10_AGENT_INSTRUCTION








