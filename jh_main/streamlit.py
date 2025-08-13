import pandas as pd
import streamlit as st
from pathlib import Path
import json
from streamlit.components.v1 import html as st_html

# ==========================
# Page Config
# ==========================
st.set_page_config(
    page_title="PFx: Patient Friendly Explanations",
    page_icon="💬",
    layout="wide",
)

# ==========================
# Helpers
# ==========================
def _get_query_param(name: str, default: str = "") -> str:
    try:
        qs = st.query_params
        val = qs.get(name)
        if isinstance(val, list):
            return val[0] if val else default
        return val if val is not None else default
    except Exception:
        try:
            qs = st.experimental_get_query_params()
            return (qs.get(name, [default]) or [default])[0]
        except Exception:
            return default

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

WORKFLOW_FILES = {
    "Zero-shot": BASE_DIR / "PFx_final - PFx_Zeroshot.csv",
    "Few-shot": BASE_DIR / "PFx_final - PFx_Single_Fewshot.csv",
    "Multiple Few-shot": BASE_DIR / "PFx_final - PFx_Multiple_Few.csv",
    "Agentic": BASE_DIR / "PFx_final - PFx_Agentic.csv",
}

LEGACY_FALLBACK = BASE_DIR / "pfx_source.csv"

READING_LEVELS = [
    "PROFESSIONAL",
    "COLLEGE_GRADUATE",
    "COLLEGE",
    "TENTH_TO_TWELTH_GRADE",
    "EIGTH_TO_NINTH_GRADE",
    "SEVENTH_GRADE",
    "SIXTH_GRADE",
    "FIFTH_GRADE",
]

GENERATE_WORKFLOWS = ["Zero-shot", "Few-shot", "Agentic", "All"]

@st.cache_data(show_spinner=False)
def load_any_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, header=None)
        except Exception:
            return None

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for want in candidates:
        if want in lower_map:
            return lower_map[want]
    return None

def normalize_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if all(str(c).startswith("Unnamed") for c in df.columns) and df.shape[1] >= 2:
        df = df.iloc[:, :6]
        df.columns = ["Finding", "PFx", "ICD10", "Accuracy", "Readability(FRES)", "FRES"][: df.shape[1]]
        for col in ["ICD10", "Accuracy", "Readability(FRES)", "FRES"]:
            if col not in df.columns:
                df[col] = None
        return df
    finding_col = _pick_col(df, ["finding", "name", "incidental finding", "finding_name", "title", "label"])
    pfx_col = _pick_col(df, ["pfx", "explanation", "patient friendly explanation", "pfx_text", "answer", "output", "pf x"])
    icd_col = _pick_col(df, ["icd10", "icd-10", "icd10_code", "icd code", "icd"])
    acc_col = _pick_col(df, ["accuracy", "eval_accuracy", "is_correct", "correctness", "score"])
    read_col = _pick_col(df, ["readability", "grade", "grade_level", "fkgl", "flesch_kincaid", "flesch-kincaid", "smog", "readability(fres)", "readability (fres)"])
    fres_col = _pick_col(df, ["fres", "_0_flesch", "flesch reading ease", "flesch_reading_ease", "flesch reading-ease", "flesch score", "flesch"])
    cols = list(df.columns)
    if finding_col is None and len(cols) >= 1:
        finding_col = cols[0]
    if pfx_col is None and len(cols) >= 2:
        pfx_col = cols[1]
    out = pd.DataFrame({
        "Finding": df[finding_col].astype(str).str.strip() if finding_col else "",
        "PFx": df[pfx_col].astype(str) if pfx_col else "",
        "ICD10": df[icd_col].astype(str) if icd_col else None,
        "Accuracy": df[acc_col] if acc_col else None,
        "Readability(FRES)": df[read_col].astype(str) if read_col else None,
        "FRES": df[fres_col] if fres_col else None,
    })
    out = out.dropna(subset=["Finding"]).copy()
    out["Finding"] = out["Finding"].str.strip()
    out = out.drop_duplicates(subset=["Finding"], keep="first")
    return out

@st.cache_data(show_spinner=False)
def load_all_workflows(workflow_files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    datasets = {}
    for wf, path in workflow_files.items():
        raw = load_any_csv(path)
        if raw is not None:
            datasets[wf] = normalize_dataframe(raw)
    if not datasets:
        legacy = load_any_csv(LEGACY_FALLBACK)
        if legacy is not None:
            datasets["Zero-shot"] = normalize_dataframe(legacy)
    return datasets

DATASETS = load_all_workflows(WORKFLOW_FILES)

# ==========================
# GENERATE PAGE
# ==========================
page = _get_query_param("page", "home").strip().lower()
if page == "generate":
    st.subheader("Generate Your Own PFx")
    st.caption("UI only for now — wire your LLM call into the commented hook below.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown("### Inputs")
        workflow_choice = st.selectbox("Workflow", GENERATE_WORKFLOWS, index=0)
        incidental_finding = st.text_input("Incidental Finding", placeholder="e.g., Hepatic hemangioma")
        icd10_code = st.text_input("ICD-10 Code", placeholder="e.g., D18.03")
        reading_level = st.selectbox("Reading Level", READING_LEVELS, index=6)
        generate_clicked = st.button("🚀 Generate PFx", type="primary")
        if "generated_pfx" not in st.session_state:
            st.session_state.generated_pfx = ""
        if generate_clicked and not incidental_finding:
            st.warning("Please enter an Incidental Finding before generating.")
        # Example LLM hook
        # if generate_clicked and incidental_finding:
        #     pfx_text = your_llm_function(
        #         finding=incidental_finding,
        #         icd10=icd10_code,
        #         reading_level=reading_level,
        #         workflow=workflow_choice
        #     )
        #     st.session_state.generated_pfx = pfx_text
    with right:
        st.markdown("### Patient-Friendly Explanation")
        pfx_text = (st.session_state.get("generated_pfx") or "").strip()
        st.markdown(f"<div class='pfx-card'>{pfx_text if pfx_text else '<span class=\\"pfx-muted\\">Your PFx will appear here once generated.</span>'}</div>", unsafe_allow_html=True)
