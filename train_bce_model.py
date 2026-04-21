import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


def extract_aac_features(peptide: str):
    peptide = peptide.upper()
    length = len(peptide)
    return [peptide.count(aa) / length for aa in AA_ORDER]


# 아주 작은 예시 학습 데이터
# 나중에 IEDB 기반 데이터로 바꾸면 됨
positive_peptides = [
    "VEALYLVCGERG",
    "TPKTRREAEDLQ",
    "LQVGQVELGGGP",
    "SLYQLENYCN",
    "RGFFYTPKTRR",
]

negative_peptides = [
    "MALWMRLLPLLAL",
    "ALWGPDPAAAFV",
    "LCGSHLVEALYL",
    "GPLALEGSLQKR",
    "GIVEQCCTSICS",
]

X = []
y = []

for pep in positive_peptides:
    X.append(extract_aac_features(pep))
    y.append(1)

for pep in negative_peptides:
    X.append(extract_aac_features(pep))
    y.append(0)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(X, y)

bundle = {
    "model": model,
    "feature_type": "aac",
    "aa_order": AA_ORDER,
}

joblib.dump(bundle, "bce_model.joblib")
print("Saved model to bce_model.joblib")