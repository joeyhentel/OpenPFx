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

# zeroshot llm call 
def zeroshot_call(finding, code, grade_level)
    results_df = pd.DataFrame(columns = ["finding", "ICD10_code", "PFx", "PFx_ICD10_code"])

    prompt = baseline_zeroshot_prompt.format(Incidental_Finding = finding, Reading_Level = SIXTH_GRADE)
    
    pfx_response = CLIENT.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": "You are a medical profesional rephrasing and explaining medical terminology to a patient in an understandable manner."},
            {"role": "system", "content": prompt}
        ],
        stream=False,
    )

    extracted_response = extract_json(pfx_response.choices[0])

    results_df = {
        "finding": finding,
        "ICD10_code": code,
        "PFx": extracted_response.get("PFx", ""),
        "PFx_ICD10_code": extracted_response.get("PFx_ICD10_code", "")
    }

    agent_code = label_icd10s(response)

    results_df['_0_agent_icd10_codes'] = agent_code
    results_df["_0_icd10_matches"] = results_df.ICD10_code == results_df._0_agent_icd10_codes
    results_df["_0_pfx_icd10_matches"] = results_df.ICD10_code == results_df["PFx_ICD10_code"] 
    results_df["accuracy"] = (results_df._0_icd10_matches + results_df._0_pfx_icd10_matches) / 2
