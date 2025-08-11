# streamlit.py
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="PFx: Patient Friendly Explanations",
    page_icon="💬",
    layout="wide",
)

# ==========================
# Top bar: Title (left) + CTA button (right)
# ==========================
lcol, rcol = st.columns([1, 1], gap="small")
with lcol:
    st.title("PFx: Patient Friendly Explanations")
    st.markdown(
        "- Choose a **workflow** (Zero-shot, Few-shot, Multiple Few-shot, Agentic), then a **finding**.\n"
        "- The PFx card displays the explanation; enable **Advanced stats** to see ICD-10, accuracy, and Readability (FRES).\n"
with rcol:
    st.markdown(
        """
        <div style="display:flex; justify-content:flex-end; margin-top:0.5rem;">
            <a href="?page=generate" target="_self"
               style="text-decoration:none; background:#f0f2f6; padding:0.55rem 0.9rem; border-radius:10px; border:1px solid #e5e7eb; font-weight:600; color:#111;">
               Generate Your Own!
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==========================
# Simple router for the future "Generate" page
# ==========================
qs = st.query_params
page = qs.get("page", [""])[0] if isinstance(qs.get("page"), list) else qs.get("page", "")
if page == "generate":
    st.subheader("Generate Your Own (coming soon)")
    st.info("You clicked **Generate Your Own!** — I’ll wire this up once you share the specs.")
    st.stop()

# ==========================
# File configuration (portable: paths relative to this file)
# ==========================
BASE_DIR = Path(__file__).resolve().parent

WORKFLOW_FILES = {
    "Zero-shot":         BASE_DIR / "PFx_final - PFx_Zeroshot.csv",
    "Few-shot":          BASE_DIR / "PFx_final - PFx_Single_Fewshot.csv",
    "Multiple Few-shot": BASE_DIR / "PFx_final - PFx_Multiple_Few.csv",
    "Agentic":           BASE_DIR / "PFx_final - PFx_Agentic.csv",
}

LEGACY_FALLBACK = BASE_DIR / "pfx_source.csv"

# ==========================
# Data loading & normalization
# ==========================
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
        df.columns = ["Finding", "PFx", "ICD10", "Accuracy", "Readability", "FRES"][: df.shape[1]]
        for col in ["ICD10", "Accuracy", "Readability", "FRES"]:
            if col not in df.columns:
                df[col] = None
        return df

    finding_col = _pick_col(df, ["finding", "name", "incidental finding", "finding_name", "title", "label"])
    pfx_col = _pick_col(df, ["pfx", "explanation", "patient friendly explanation", "pfx_text", "answer", "output", "pf x"])
    icd_col = _pick_col(df, ["icd10", "icd-10", "icd10_code", "icd code", "icd"])
    acc_col = _pick_col(df, ["accuracy", "eval_accuracy", "is_correct", "correctness", "score"])
    read_col = _pick_col(df, ["readability", "grade", "grade_level", "fkgl", "flesch_kincaid", "flesch-kincaid", "smog"])
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
    datasets: dict[str, pd.DataFrame] = {}
    for wf, path in workflow_files.items():
        raw = load_any_csv(path)
        if raw is not None:
            datasets[wf] = normalize_dataframe(raw)
    if not datasets:
        legacy = load_any_csv(LEGACY_FALLBACK)
        if legacy is not None:
            datasets["Zero-shot"] = normalize_dataframe(legacy)
    return datasets

datasets = load_all_workflows(WORKFLOW_FILES)

# ==========================
# UI: Left controls / Right content
# ==========================
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Workflow & Finding")
    if not datasets:
        st.error("No datasets found. Please place the four CSV files next to this file.")
        st.stop()
    workflow_names = list(datasets.keys())
    workflow = st.selectbox("Select workflow", workflow_names, index=0, key="wf")
    df = datasets[workflow]
    options = df["Finding"].tolist()
    finding = st.selectbox("Select a finding", ["— Select —"] + options, index=0, key="finding")
    finding = None if finding == "— Select —" else finding

with right:
    st.subheader("Patient-Friendly Explanation")
    st.markdown(
        """
        <style>
        .pfx-card { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px 20px; min-height: 160px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); line-height: 1.55; }
        .pfx-muted { color: #6b7280; }
        .pfx-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; margin-top: 12px; }
        .pfx-pill { border: 1px solid #e5e7eb; border-radius: 999px; padding: 8px 12px; background: #fafafa; font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if finding:
        row = df.loc[df["Finding"] == finding].iloc[0]
        pfx_text = (row.get("PFx") or "").strip()
        st.markdown(f"<div class='pfx-card'>{pfx_text if pfx_text else '<span class=\\"pfx-muted\\">No PFx text found for this item.</span>'}</div>", unsafe_allow_html=True)
        show_stats = st.checkbox("Show advanced stats (ICD-10, accuracy, Readability(FRES))", value=False)
        if show_stats:
            icd10 = (row.get("ICD10") or "").strip()
            acc_val = row.get("Accuracy")
            acc_str = ""
            if pd.notna(acc_val):
                try:
                    f_acc = float(acc_val)
                    acc_str = f"{f_acc*100:.1f}%" if 0 <= f_acc <= 1 else f"{f_acc:.1f}%"
                except Exception:
                    acc_str = str(acc_val)
            read_str = (row.get("Readability (FRES)") or "").strip()
            fres_val = row.get("FRES")
            fres_str = ""
            if pd.notna(fres_val):
                try:
                    fres_str = f"{float(fres_val):.1f}"
                except Exception:
                    fres_str = str(fres_val)
            pills = []
            if icd10:
                pills.append(f"<div class='pfx-pill'><b>ICD-10:</b> {icd10}</div>")
            if acc_str:
                pills.append(f"<div class='pfx-pill'><b>Accuracy:</b> {acc_str}</div>")
            if read_str or fres_str:
                pills.append(f"<div class='pfx-pill'><b>Readability(FRES):</b> {read_str} {fres_str}</div>")
            if pills:
                st.markdown("<div class='pfx-meta'>" + "".join(pills) + "</div>", unsafe_allow_html=True)
            else:
                st.caption("No advanced stats available for this entry.")
    else:
        st.markdown("<div class='pfx-card pfx-muted'>Pick a workflow and finding on the left to view the PFx.</div>", unsafe_allow_html=True)
