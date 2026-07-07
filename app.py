# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

st.set_page_config(page_title="SLO Risk Modeler", layout="wide")

# -----------------------------
# Hardcoded Zambia dataset
# -----------------------------
data = {
    "Mine":["Lumwana","Trident","Mwambashi","Baluba","Lubambe",
            "Chambishi","Kansanshi","Chibuluma South","Muliashi North","Mufulira"],
    "Buildings":[0,9,6,3,4,2,1,7,8,5],
    "Roads":[1,7,8,9,5,0,3,4,6,2],
    "Amenities":[2,0,1,3,4,7,8,5,6,9],
    "Education":[2,0,1,3,5,6,8,7,4,9],
    "Health":[2,0,1,3,4,7,8,5,6,9],
    "Population_k":[11,3,8,11,41,54,72,46,28,121],
    "Composite_Score":[1.4,3.2,3.4,4.2,4.4,4.4,5.6,5.6,6.0,6.8]
}
df = pd.DataFrame(data)
df["pop_log"] = np.log1p(df["Population_k"])

#------------------------------------------------------
#2. Derive risk tiers from the composite score (paper's own scoring logic:
#    lower composite score = higher social vulnerability = higher SLO risk)
#------------------------------------------------------

def risk(score):
    if score <= 3.4:
        return "High"
    elif score <= 4.4:
        return "Medium"
    return "Low"

df["Risk"] = df["Composite_Score"].apply(risk)
features=["Buildings","Roads","Amenities","Education","Health","pop_log"]

# ---------------------------------------------------------------------------
# 3. Regression: predict the continuous composite score
#    (Leave-One-Out CV, mirroring Sarkheil et al.'s LOOCV protocol for
#    small-n ESG datasets)
# ---------------------------------------------------------------------------

X=df[features]
y=df["Composite_Score"]
yc=df["Risk"]

@st.cache_resource
def train():
    loo=LeaveOneOut()

    reg=RandomForestRegressor(n_estimators=300,max_depth=3,random_state=42)
    reg_pred=cross_val_predict(reg,X,y,cv=loo)
    mae=mean_absolute_error(y,reg_pred)
    rho,p=spearmanr(y,reg_pred)
#-----------------------------------------------------------------------
# 4. Classification: predict the risk tier (High/Medium/Low)
# ----------------------------------------------------------------------
    clf=RandomForestClassifier(n_estimators=300,max_depth=3,random_state=42)
    cls_pred=cross_val_predict(clf,X,yc,cv=loo)
    acc=accuracy_score(yc,cls_pred)
    cm=confusion_matrix(yc,cls_pred,labels=["High","Medium","Low"])

    reg.fit(X,y)
    clf.fit(X,yc)
    
#------------------------------------------------------------------------------
# 5. Feature importance (permutation importance on full-fit model, since
#    n is too small for a held-out importance split)
# -----------------------------------------------------------------------------
    perm=permutation_importance(clf,X,yc,n_repeats=30,random_state=42)
    imp=pd.DataFrame({"Feature":features,
                      "Importance":perm.importances_mean}).sort_values("Importance")

    return reg,clf,mae,rho,p,acc,cm,imp

reg,clf,mae,rho,p,acc,cm,imp=train()

#--------------------------------------------------------------------------
#Streamlit App display
# -------------------------------------------------------------------------

st.title("Social License to Operate (SLO) Risk Predictor")
st.write("Open-data proof-of-concept trained on published Zambia mine scores.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Validation (Leave-One-Out CV, n=10)")
    st.metric("LOOCV MAE", f"{mae:.2f}", help="Mean absolute error on composite score, scale 0–9")
    st.metric("Spearman ρ", f"{rho:.2f}", help="Rank correlation between predicted and actual score")
    st.metric("Classification Accuracy", f"{acc:.0%}",
               help="3-tier accuracy vs. 40% majority-class baseline")
    st.caption(f"Spearman p-value = {p:.3f}")
    if acc < 0.40:
        st.caption("🔻 Below the majority-class baseline — with only 10 mines, discrete "
                   "tiering is not yet reliable. Rank ordering (ρ) is the more trustworthy signal here.")
with c2:
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(cm, index=["High","Medium","Low"], columns=["High","Medium","Low"])
    fig_cm, ax_cm = plt.subplots(figsize=(3.5,3))
    im = ax_cm.imshow(cm_df, cmap="Blues")
    ax_cm.set_xticks(range(3)); ax_cm.set_xticklabels(cm_df.columns)
    ax_cm.set_yticks(range(3)); ax_cm.set_yticklabels(cm_df.index)
    for i in range(3):
        for j in range(3):
            ax_cm.text(j, i, cm_df.iloc[i,j], ha="center", va="center")
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

left, right = st.columns([1, 1])

with left:
    st.subheader("Published Zambia Dataset")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download Dataset",
                       df.to_csv(index=False),
                       "zambia_dataset.csv",
                       "text/csv")
with right:
    st.subheader("Transferability Screen (Hypothetical DRC Site)")
    b = st.slider("Buildings", 0, 9, 2)
    r = st.slider("Roads", 0, 9, 3)
    a = st.slider("Amenities", 0, 9, 1)
    e = st.slider("Education", 0, 9, 2)
    h = st.slider("Health", 0, 9, 1)
    pop = st.slider("Population (thousand)", 1, 200, 15)
    inp = pd.DataFrame([[b, r, a, e, h, np.log1p(pop)]], columns=features)
    score = reg.predict(inp)[0]
    tier = clf.predict(inp)[0]
    probs = dict(zip(clf.classes_, clf.predict_proba(inp)[0]))
    st.metric("Predicted Composite Score", f"{score:.2f}/9")
    if tier == "High":
        st.error("HIGH SLO DISRUPTION RISK")
    elif tier == "Medium":
        st.warning("MEDIUM SLO DISRUPTION RISK")
    else:
        st.success("LOW SLO DISRUPTION RISK")
    st.write("Class probabilities")
    st.json({k: round(float(v), 2) for k, v in probs.items()})
