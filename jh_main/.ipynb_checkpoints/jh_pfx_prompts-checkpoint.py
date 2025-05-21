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

WRITER_INSTRUCTION = """

<Prompt>
Please generate new <PFx> for the <Incidental Finding>

Output should be formatted as a json with the following attributes/fields: finding, ICD10_code, PFx

Additional Instructions:
1. DO NOT SUGGEST FOLLOW UP STEPS WITH THE DOCTOR
2. Maintain a formal, informative tone
3. If you choose to utilize an anology, do not use more than one. Maint
4. Please generate PFx at a {Reading_Level} Flesch-Kincaid reading level.
5. Please output PFx in 100 words or more

</Prompt>

<Incidental Finding> 
{Incidental_Finding} 
</Incidental Finding>  
"""

DOCTOR_INSTRUCTION = """
<Context>
You are a doctor tasked with evaluating medical accuracy of patient friendly explanations (PFx) of Incidental Findings
</Context>
<Prompt>
Please examine the PFx to determine medical accuracy in explaining {Incidental_Finding}
The explanation should adequately and specifically explain {Incidental_Finding}
Assure the explanation aligns with the ICD-10 code: {ICD10_code}
Examine the ICD10_code and the PFx_ICD10_code to determine the accuracy of the PFx. If they are completely different, the response could be inaccurate depending on what the codes represent.

- If the response is medically accurate, send the response to the readability_checker. 
- If not, send the response back to the writer agent with:
  - **Brief reason** for inaccuracy
  - **Specific missing or incorrect details**
</Prompt>
<Format>
1. **Verdict:** ["ACCURATE - Send to Readability" / "INACCURATE - Revise"]
IF REQUIRED
2. **Reasoning:** [Short explanation]
3. **Fixes:** [Concise revision suggestions]
IF IT IS INACCURATE, YOUR OUTPUT MUST BEGIN WITH THE WORD INACCURATE
</Format>
"""

READABILITY_CHECKER_INSTRUCTION = """
<Context>
You are an expert in the English language. You are familiar with the Flesch Reading Ease Score (FRES) metric to determine readability.
Here is the FRES calculation: 206.835 - 1.015 × (total words ÷ total sentences) - 84.6 × (total syllables ÷ total words) 
Here is the FRES Scale:
90 - 100 5th grade
80 - 90	6th grade
70 - 80	7th grade
60 - 70	8th & 9th 
50 - 60	10th to 12th grade
30 - 50	College
10 - 30	College graduate
0 - 10	Professional
</Context>
<Prompt>
Determine if the provided PFx matches the desired Flesch Reading Ease Score (FRES) {reading_level}.
Only provide the writer with advice, do not give sample revisions.
</Prompt>
<Format>
- If it does not match, return to the writer with:
  - A **brief** reason why it doesn’t match.
  - **Specific** changes needed to reach the desired level.
- If it matches, say "All done!"
IF IT DOES NOT MATCH, YOUR OUTPUT MUST BEGIN WITH: **NOT READABLE**
IF IT IS IN THE DESIRED RANGE, SAY "All done!"
</Format>
"""

ICD10_LABELER_INSTRUCTION = """"
<Context>
You are a medical professional knowledgable in ICD10 codes and their corresponding disorders. You're tasked with labelling a pateint friendly explanation with the ICD10 code of the described incidental finding.
</Context>
<Prompt>
Please label the provided response with the ICD10 code of the described incidental finding.
Add the following field to the provided response with the ICD10 code you identified: PFx_ICD10_code
This is your sole job, do not do anything else.
</Prompt>
<Format>
Output should be formatted as a json with the following attributes/fields: finding, ICD10_code, PFx, PFx_ICD10_code
</Format>
"""

writer_prompt = PromptTemplate (
    input_variables = ["Incidental_Finding", "Reading_Level"],
    template = WRITER_INSTRUCTION,
)

doctor_prompt = PromptTemplate(
    imput_variables = ["Incidental_Finding", "ICD10_code"],
    template=DOCTOR_INSTRUCTION,
)

readability_checker_prompt = PromptTemplate(
    input_variables = ["reading_level"],
    template=READABILITY_CHECKER_INSTRUCTION,
)








