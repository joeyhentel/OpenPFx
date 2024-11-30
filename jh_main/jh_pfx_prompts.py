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
1. Do not suggest follow-up steps with the doctor
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
1. Do not suggest follow-up steps with the doctor
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











