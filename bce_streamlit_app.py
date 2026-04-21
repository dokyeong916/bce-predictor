import re
from typing import Optional, Tuple

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="BCE Demo App", page_icon="🧬", layout="wide")


UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(sequence: str) -> str:
    """Normalize protein sequence input."""
    sequence = sequence.upper().strip()
    sequence = re.sub(r"\s+", "", sequence)
    return sequence


@st.cache_data(show_spinner=False)
def fetch_uniprot_sequence(accession: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch a FASTA record from UniProt and return (header, sequence)."""
    accession = accession.strip()
    if not accession:
        return None, None

    url = UNIPROT_FASTA_URL.format(accession=accession)
    response = requests.get(url, timeout=20)

    if response.status_code != 200:
        return None, None

    lines = response.text.strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        return None, None

    header = lines[0][1:]
    sequence = "".join(lines[1:]).strip().upper()
    return header, sequence


def validate_sequence(sequence: str) -> Tuple[bool, str]:
    if not sequence:
        return False, "Sequence is empty."

    invalid = sorted(set(sequence) - VALID_AA)
    if invalid:
        return False, f"Invalid amino-acid characters found: {' '.join(invalid)}"

    if len(sequence) < 6:
        return False, "Sequence is too short. Use at least 6 amino acids."

    return True, "OK"


def toy_bce_predictor(sequence: str, window: int = 12, threshold: float = 0.58) -> pd.DataFrame:
    """A simple placeholder scoring function.

    This is NOT a real BCE model. It just gives you a working end-to-end demo
    until you replace it with a trained model.
    """
    rows = []
    hydrophilic = set("DEHKNQRSTY")

    for i in range(0, len(sequence) - window + 1):
        peptide = sequence[i:i + window]
        score = sum(aa in hydrophilic for aa in peptide) / window
        rows.append(
            {
                "start": i + 1,
                "end": i + window,
                "peptide": peptide,
                "score": round(score, 3),
                "prediction": "BCE-like" if score >= threshold else "non-BCE-like",
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["score", "start"], ascending=[False, True]).reset_index(drop=True)


def highlight_top_regions(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if df.empty:
        return df
    return df.head(top_n).copy()


st.title("🧬 B-cell Epitope Demo App")
st.caption("Starter app: manual protein sequence input + UniProt accession lookup + placeholder prediction")

with st.sidebar:
    st.header("Settings")
    window_size = st.slider("Peptide window size", min_value=6, max_value=30, value=12)
    score_threshold = st.slider("Demo threshold", min_value=0.0, max_value=1.0, value=0.58, step=0.01)
    top_n = st.slider("Top results to show", min_value=3, max_value=20, value=5)
    st.markdown("---")
    st.info("Later, replace the demo predictor with your trained BCE model.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Option 1: Load from UniProt")
    uniprot_id = st.text_input("UniProt accession", placeholder="Example: P01308")
    load_clicked = st.button("Fetch from UniProt")

    if load_clicked:
        with st.spinner("Fetching sequence from UniProt..."):
            header, fetched_seq = fetch_uniprot_sequence(uniprot_id)
        if fetched_seq:
            st.session_state["sequence"] = fetched_seq
            st.session_state["sequence_source"] = f"UniProt: {header}"
            st.success("Sequence loaded successfully.")
        else:
            st.error("Could not retrieve a sequence for that UniProt accession.")

with col2:
    st.subheader("Option 2: Paste a sequence manually")
    manual_sequence = st.text_area(
        "Protein sequence",
        value=st.session_state.get("sequence", ""),
        height=220,
        placeholder="Paste an amino-acid sequence here...",
    )

    if st.button("Use manual sequence"):
        st.session_state["sequence"] = manual_sequence
        st.session_state["sequence_source"] = "Manual input"
        st.success("Manual sequence saved.")

sequence = clean_sequence(st.session_state.get("sequence", manual_sequence))
source = st.session_state.get("sequence_source", "Not set")

st.markdown("---")
st.subheader("Current input")
st.write(f"**Source:** {source}")
st.write(f"**Length:** {len(sequence)} aa")

if sequence:
    st.code(sequence[:3000] + ("..." if len(sequence) > 3000 else ""), language=None)

predict_clicked = st.button("Run prediction")

if predict_clicked:
    ok, message = validate_sequence(sequence)
    if not ok:
        st.error(message)
    else:
        with st.spinner("Running demo predictor..."):
            results = toy_bce_predictor(sequence, window=window_size, threshold=score_threshold)
            top_hits = highlight_top_regions(results, top_n=top_n)

        st.success("Prediction complete.")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Windows scored", len(results))
        with metric_col2:
            st.metric("Top score", f"{results['score'].max():.3f}")

        st.subheader("Top candidate regions")
        st.dataframe(top_hits, use_container_width=True)

        st.subheader("All windows")
        st.dataframe(results, use_container_width=True)

        csv_data = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv_data,
            file_name="bce_demo_results.csv",
            mime="text/csv",
        )

st.markdown("---")
st.subheader("Next upgrades")
st.markdown(
    """
1. Replace `toy_bce_predictor()` with a real trained model.
2. Add known epitope lookup from IEDB.
3. Highlight predicted regions directly on the full sequence.
4. Deploy the app publicly.
"""
)

