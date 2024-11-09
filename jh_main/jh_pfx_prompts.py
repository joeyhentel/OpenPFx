from langchain.prompts import promptTemplate

EXAMPLE = """ 

<Incidental Finding> 
{Incidental_Finding} 
</Incidental Finding>  

</PFx>
'''{{"Incidental_Finding":"Incidental_Finding", "ICD10_code":"{ICD10_code}", "PFx":"{PFx}", "PFx_ICD10_code":"{PFx_ICD10_code}"}}'''
</PFx>

"""

BASELINE_ZEROSHOT_PROMPT = """

<Prompt>
Please generate new <PFx> for the <Incidental Finding>

Output should be formatted as a json with the following attributes/fields: finding, ICD10_code, PFx, PFx_ICD10_code 

Additional Instructions:
1. Do not suggest follow-up steps with the doctor
2. Use the patient friendly explanation sentences to determine a PFx ICD10_code code.
3. Please generate PFx at a {Reading_Level} Flesch-Kincaid reading level.
4. Please output PFx in 100 words or more
5. Use a bedside manner 

</Prompt>

<Incidental Finding> 
{Incidental_Finding} 
</Incidental Finding>  
"""


# INCORPORATE ZEROSHOT_REFLEXION_READING_LEVEL


SINGLE_FEWSHOT_PROMPT = """

<Prompt>
Please generate new <PFx> for the <Incidental Finding>

Output should be formatted as a json with the following attributes/fields: finding, ICD10_code, PFx, PFx_ICD10_code

Additional Instructions:
1. Do not suggest follow-up steps with the doctor
2. Use the patient friendly explanation sentences to determine a PFx ICD10_code code.
3. Please generate PFx at a {Reading_Level} Flesch-Kincaid reading level.
4. Please output PFx in 100 words or more
5. Use a bedside manner 

</Prompt>

<Examples>
{Examples}
</Examples>

<Incidental Finding> 
{Incidental_Finding} 
</Incidental Finding>  
"""


MULTIPLE_FEWSHOT_PROMPT = """
Please generate 5 new <PFx> for the <Incidental Finding> 

Output should be formatted as a json with the following attributes/fields: finding, ICD10_code, PFx, PFx_ICD10_code 

Additional Instructions:
1. Do not suggest follow-up steps with the doctor
2. Use the patient friendly explanation sentences to determine a PFx ICD10_code code.
3. Please generate PFx at a {Reading_Level} Flesch-Kincaid reading level.
4. Please output PFx in 100 words or more
5. Use a bedside manner 

</Prompt>

<Examples>
{Examples}
</Examples>

<Incidental Finding> 
{Incidental_Finding} 
</Incidental Finding>  
"""










