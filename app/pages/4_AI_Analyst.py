"""AI Analyst: ask portfolio questions in plain English; Claude answers with SQL and SecondLook's own tools."""
import json

import streamlit as st

st.set_page_config(page_title="AI Analyst · SecondLook", page_icon="💬", layout="wide")

from lib import api_key_available, footer, load_applicants, load_model  # noqa: E402
from secondlook import agent as A  # noqa: E402

MAX_QUESTIONS = 10                 # per session, to protect the public API key
HISTORY_TURNS = 3                  # earlier question/answer pairs sent with a follow-up
EXAMPLES = [
    "Which persona has the most repayers among the applicants we don't approve?",
    "What is the approval rate for thin-file applicants compared with everyone else?",
    "Is the gap in approval rates between young and older repayers statistically real?",
    "For demo applicant 209138, what is the smallest loan change that would get them approved?",
]


@st.cache_resource(show_spinner=False)
def toolbox() -> A.Toolbox:
    booster, meta, explainer = load_model()
    return A.Toolbox(booster, meta, explainer, load_applicants())


def show_work(r: dict):
    n = len(r["steps"])
    with st.expander(f"How this was answered: {n} tool call{'s' if n != 1 else ''} · ${r['cost_usd']:.4f}"):
        for i, s in enumerate(r["steps"], 1):
            flag = " ⚠️ error, sent back to Claude" if s["is_error"] else ""
            st.markdown(f"**{i}. `{s['tool']}`**{flag}")
            if s["tool"] == "run_sql":
                st.code(s["input"]["query"], language="sql")
            else:
                st.code(json.dumps(s["input"]), language="json")
            out = s["result"]
            st.code(out if len(out) <= 1500 else out[:1500] + "\n... (cut for display)", language="text")
        st.caption(f"{r['model']} · {r['api_calls']} API call{'s' if r['api_calls'] != 1 else ''} · "
                   f"{r['input_tokens']:,} input + {r['output_tokens']:,} output tokens · ${r['cost_usd']:.4f} · {r['seconds']}s")


st.title("💬 AI Analyst")
st.caption("Ask a question about the test portfolio, the borrower personas or the fairness audit. Claude writes SQL on a "
           "read-only copy of the data and calls SecondLook's own tools (SHAP reasons, path to approval, fairness audit). "
           "Every answer shows the exact queries and tool calls behind it, and what it cost.")

chat = st.session_state.setdefault("analyst_chat", [])
asked = st.session_state.setdefault("analyst_asked", 0)
has_key = api_key_available()

with st.sidebar:
    st.header("Session")
    st.metric("Questions left", f"{MAX_QUESTIONS - asked} of {MAX_QUESTIONS}")
    spent = sum(m["result"]["cost_usd"] for m in chat if m.get("result"))
    st.caption(f"Spent this session: ${spent:.4f}. Each question costs about a tenth of a cent on Claude Haiku 5.5.")
    if st.button("Clear conversation", disabled=not chat):
        st.session_state["analyst_chat"] = []
        st.rerun()
    st.caption("Clearing the conversation does not reset the question limit.")
    with st.expander("What the data covers"):
        st.markdown("- **portfolio**: 61,503 test applicants: outcome, predicted risk, decision, gender, age band, "
                    "credit file, persona and path to approval\n- **personas**: the five borrower personas\n"
                    "- **fairness** and **fairness_gaps**: the audit with bootstrap intervals\n"
                    "- 300 demo applicants can be explained one by one")

for m in chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("result"):
            show_work(m["result"])

if not has_key:
    st.info("The AI Analyst needs a Claude API key. Add `ANTHROPIC_API_KEY` to `.env` locally or to Streamlit Secrets.")
elif asked >= MAX_QUESTIONS:
    st.warning(f"You've used all {MAX_QUESTIONS} questions for this session. This limit protects the public demo's API key.")

question = None
if has_key and asked < MAX_QUESTIONS:
    if not chat:
        st.markdown("**Try one of these:**")
        cols = st.columns(len(EXAMPLES))
        for c, ex in zip(cols, EXAMPLES):
            if c.button(ex, use_container_width=True):
                question = ex
    typed = st.chat_input("Ask about approvals, defaults, personas or fairness...")
    question = typed or question

if question:
    history = []
    for m in chat[-2 * HISTORY_TURNS:]:
        history.append({"role": m["role"], "content": m["content"]})
    chat.append({"role": "user", "content": question})
    st.session_state["analyst_asked"] = asked + 1
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Querying the portfolio..."):
            from secondlook.letters import _client
            try:
                r = A.run_agent(question, _client(), toolbox(), history=history).to_dict()
            except Exception as e:                                   # API down, key revoked, etc.
                r = None
                st.error(f"The AI Analyst couldn't answer: {e}")
        if r:
            st.markdown(r["answer"])
            show_work(r)
            chat.append({"role": "assistant", "content": r["answer"], "result": r})
    if r:
        st.rerun()                     # refresh the sidebar counters; on an error, keep the message on screen

footer()
