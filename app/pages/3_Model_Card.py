"""Model card: how the model was built, how well it works, its limits and fairness checks."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Card · SecondLook", page_icon="📄", layout="wide")

from lib import applicant_X, footer, label_of, load_applicants, load_model, load_portfolio  # noqa: E402

booster, meta, explainer = load_model()
df = load_portfolio()
m = meta["metrics"]

st.title("📄 Model Card")
st.caption("A model card summarises what a model is for, how it was built and tested, and where it should not be trusted.")

st.subheader("Model details")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
| | |
|---|---|
| **Task** | Predict the probability that a loan applicant will have payment difficulties (default) |
| **Algorithm** | LightGBM gradient-boosted trees, {meta['n_trees']:,} trees, learning rate {meta['params']['learning_rate']} |
| **Inputs** | {len(meta['features'])} features: application data, engineered ratios, and SQL summaries of 6 credit-history tables |
| **Excluded on purpose** | Gender (kept only for the fairness audit), applicant ID |
| **Training data** | [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data): 246,008 applicants (80%) |
| **Test data** | 61,503 applicants (20%), stratified, never used in training |
| **Decision rule** | Approve if PD < {meta['cutoff']:.2f}; PD {meta['borderline_band'][0]:.2f}–{meta['borderline_band'][1]:.2f} flagged *under review* |
""")
with c2:
    st.markdown(f"""
| Metric (test set) | LightGBM | Logistic baseline |
|---|---|---|
| ROC AUC | **{m['test']['roc_auc']:.3f}** | {m['baseline_test']['roc_auc']:.3f} |
| PR AUC | **{m['test']['pr_auc']:.3f}** | {m['baseline_test']['pr_auc']:.3f} |
| Brier score (lower is better) | **{m['test']['brier']:.4f}** | {m['baseline_test']['brier']:.4f} |
| 5-fold CV AUC | {m['cv_auc_mean']:.3f} ± {m['cv_auc_std']:.3f} | |
| Precision / recall of declines at cutoff | {m['precision_at_cutoff']:.0%} / {m['recall_at_cutoff']:.0%} | |
| Approval rate / default rate of approved | {m['approval_rate']:.0%} / {m['default_rate_approved']:.1%} | |
""")
    pa = meta["profit_assumptions"]
    st.caption(f"Cutoff chosen to maximise profit under stated assumptions: +{pa['margin']:.0%} of the loan on repayment, "
               f"−{pa['lgd']:.0%} on default.")

st.subheader("What drives the model (SHAP)")


@st.cache_data(show_spinner="Computing SHAP importance...")
def shap_importance():
    ids = load_applicants()["SK_ID_CURR"].tolist()
    X = pd.concat([applicant_X(i) for i in ids])
    sv = explainer.shap_values(X)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=X.columns).sort_values(ascending=False).head(15)
    return pd.DataFrame({"Factor": [label_of(f) for f in imp.index], "Mean |SHAP|": imp.values})


imp = shap_importance()
fig = px.bar(imp.iloc[::-1], x="Mean |SHAP|", y="Factor", orientation="h", color_discrete_sequence=["#55A868"])
fig.update_layout(height=450, margin=dict(t=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("Average size of each factor's push on the score, over the 300 applicants in this demo. "
           "External credit scores dominate, followed by past repayment behaviour and loan structure.")

st.subheader("Fairness check (preview)")
st.markdown("Gender is **not** a model input. We still check whether outcomes differ between groups, because other features can act as proxies.")
approved = df["PD_LGBM"] < meta["cutoff"]
df = df.assign(Approved=approved, Age_group=pd.cut(df["AGE_YEARS"], [20, 30, 40, 50, 60, 70], labels=["20-29", "30-39", "40-49", "50-59", "60-69"]))


def fairness(by):
    g = df.groupby(by, observed=True)
    t = pd.DataFrame({
        "Applicants": g.size(),
        "Default rate": g["TARGET"].mean(),
        "Approval rate": g["Approved"].mean(),
        "Approval rate of repayers": df[df["TARGET"] == 0].groupby(by, observed=True)["Approved"].mean(),
        "Default rate of approved": df[df["Approved"]].groupby(by, observed=True)["TARGET"].mean(),
    })
    return t[t["Applicants"] >= 100]          # tiny groups (e.g. 2 'unknown' gender) give unreliable rates


fmt = {"Applicants": "{:,}", "Default rate": "{:.1%}", "Approval rate": "{:.1%}", "Approval rate of repayers": "{:.1%}",
       "Default rate of approved": "{:.1%}"}
fg = fairness(df["CODE_GENDER"].rename("Gender"))
fa = fairness(df["Age_group"].rename("Age group"))
st.markdown("**By gender**")
st.dataframe(fg.style.format(fmt), use_container_width=True)
st.markdown("**By age group**")
st.dataframe(fa.style.format(fmt), use_container_width=True)
st.markdown(f"""
**What this shows**
- Approval rates follow each group's actual default rate, and in **every** group the approved loans default less often than the group overall.
- **Equal-opportunity gap:** among applicants who actually **repaid**, approval rates still differ:
  **{fg.loc['M', 'Approval rate of repayers']:.0%}** for men vs **{fg.loc['F', 'Approval rate of repayers']:.0%}** for women, and
  **{fa['Approval rate of repayers'].iloc[0]:.0%}** for ages 20-29 vs **{fa['Approval rate of repayers'].iloc[-1]:.0%}** for ages 60-69.
  Good young and male borrowers are declined more often than good older and female ones.
- Gender is not a model input, but age is, and other features (job, employer, external scores) correlate with both. This is the kind of
  gap a **human second look** is meant to catch, and the Phase 2 fairness audit will test whether constraining age-related features
  narrows it without losing accuracy.
""")

st.subheader("Intended use and limitations")
st.markdown("""
- **Intended use:** decision *support* for loan officers reviewing consumer loan applications, especially borderline ones. A human makes the final decision.
- **Not for:** fully automated declines, or markets and products very different from the training data.
- **Data:** historical, anonymised Home Credit data; the competition does not disclose which country it comes from. Before real use, the model would need to be validated on local (e.g. Philippine) applicants.
- **Probabilities** are well calibrated on the test set, but calibration should be monitored over time as the applicant mix changes.
- **Explanations** (SHAP) describe *the model's* reasoning, not causes in the real world.
- **Letters** are drafted by Claude and checked by rules, then by an independent audit (a second Claude reads only the letter; Python checks its claims against the model's reasons). In testing it caught all four planted errors, but about half of its flags on real letters were false alarms and its verdicts vary between runs, so a loan officer reviews every letter.
- **Profit figures** use illustrative assumptions, adjustable on the dashboard.
""")
footer()
