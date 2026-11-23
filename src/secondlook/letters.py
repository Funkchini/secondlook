"""Draft plain-language decision letters with Claude, grounded in the model's SHAP reasons.

Flow for one applicant:
  1. build_letter_input(): turn the SHAP explanation into a small, safe fact sheet
     (decision, loan details, top risk reasons and strengths; sensitive features removed)
  2. generate_letter(): ask Claude for a letter as structured JSON, including the codes
     of the reasons it cited
  3. check_letter(): rule-based guardrails (only allowed reasons, no sensitive traits,
     no jargon, tone matches the decision); one automatic retry if a check fails
  4. results are cached on disk, so the app and the demo never need a live API call
     for an applicant that was already drafted
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .config import PROCESSED_DIR, ROOT
from .explain import SENSITIVE, category_of, format_value

MODEL = "claude-haiku-5-5"        # chosen in the proposal: fast and low-cost for short letters
EFFORT = "low"
PROMPT_VERSION = "v5"     # v2: no invented comparisons; v3: no internal tool name, letter-friendly factors;
                          # v4: no job/employer; v5: refined guardrails (car age, comparisons)
CACHE_PATH = PROCESSED_DIR / "letter_cache.json"

# Haiku 5.5 list prices, USD per million tokens (prompts under 100K tokens)
PRICE_IN, PRICE_OUT = 0.10, 0.50

SYSTEM_PROMPT = """You draft decision letters for SecondLook, a tool used by loan officers at a consumer lender that serves people with little or no credit history. A loan officer reviews every draft before it is sent.

You receive a fact sheet about one application: the decision, the loan details, the main factors that RAISED the applicant's estimated risk, and the main factors that LOWERED it (their strengths). Each factor has a code, a plain-English label and the applicant's value.

Write a short letter to the applicant (120-220 words):
- Open with the decision in the first two sentences, clearly and kindly.
- For "decline" and "under_review" letters, explain the 2-3 most important risk factors in everyday words, and mention 1-2 genuine strengths.
- For "approve" letters, confirm the approval and mention 1-2 strengths; do not list risk factors.
- For "under_review", explain that the application is close to the approval threshold, so a loan officer will look at it personally, and that they may be contacted about adjusting the loan terms.
- For "decline", say what generally helps future applications (for example, a record of on-time payments), and that they can ask for the decision to be reviewed.
- Use only the facts in the fact sheet. Do not invent amounts, dates, scores, policies, names or contact details. Sign off as "The Lending Team".
- Describe each factor only by what its label and value say. Do not compare a factor with the current loan or with anything else, and do not claim why it matters for this particular person (for example, never say a past amount is "similar to" or "at the level of" this loan). You may add one short, general reason why lenders look at that kind of factor (for example, "on-time payments show a reliable repayment record").
- Prefer factors that are easy for an applicant to understand; you may skip a factor whose meaning would be unclear to them.
- Never mention or hint at age, gender, marital or family status, children, or any other personal characteristic.
- No technical terms: never say SHAP, model, algorithm, log-odds, probability of default, PD, score cutoff or threshold value.
- "SecondLook" is the lender's internal tool. Never mention it to the applicant.

In reasons_cited, list the codes of the risk factors you mention. In strengths_cited, list the codes of the strengths you mention. Only use codes from the fact sheet."""


class Letter(BaseModel):
    """Structured letter returned by Claude."""
    subject: str = Field(description="Letter subject line")
    body: str = Field(description="Full letter text, plain paragraphs, signed 'The Lending Team'")
    reasons_cited: list[str] = Field(description="Codes of risk factors mentioned in the letter")
    strengths_cited: list[str] = Field(description="Codes of strengths mentioned in the letter")


@dataclass
class LetterInput:
    applicant_id: int
    decision: str                       # approve | under_review | decline
    loan_amount: str
    regular_payment: str
    number_of_payments: str
    risk_factors: list[dict] = field(default_factory=list)   # {code, label, value}
    strengths: list[dict] = field(default_factory=list)

    def fact_sheet(self) -> str:
        return json.dumps(asdict(self), indent=2)


def select_demo_applicants(pd_by_id, cutoff: float, n_each: int = 10, seed: int = 42) -> list[int]:
    """A reproducible demo set: n_each applicants from each decision group (approve / under_review / decline).

    pd_by_id: pandas Series of predicted PD indexed by SK_ID_CURR.
    """
    groups = pd_by_id.map(lambda p: decision_for(p, cutoff))
    picked = []
    for g in ("approve", "under_review", "decline"):
        picked += pd_by_id[groups == g].sample(n_each, random_state=seed).index.tolist()
    return [int(i) for i in picked]


def decision_for(pd_hat: float, cutoff: float, band_width: float = 0.05) -> str:
    """approve below the cutoff; under_review just above it (SecondLook's borderline band); decline beyond."""
    if pd_hat < cutoff:
        return "approve"
    return "under_review" if pd_hat < cutoff + band_width else "decline"


# Real model factors that mean little to an applicant. The loan officer still sees them in the
# app; the letter simply doesn't use them (it cites a subset of the true reasons, never others).
LETTER_SKIP = {"REGION_POPULATION_RELATIVE", "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
               "PREV_AVG_AMT_APPLIED", "POS_N_MONTHS", "CC_N_MONTHS", "INST_N_PAYMENTS", "N_DOCUMENTS",
               "TOTALAREA_MODE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
               # job and employer: coded categories ("Industry: type 4") and, as a stated reason, easy to read as unfair
               "OCCUPATION_TYPE", "ORGANIZATION_TYPE"}


def _letter_friendly(feature: str) -> bool:
    building = feature.endswith("_AVG") and not feature.startswith(("EXT_", "PREV_", "CC_", "INST_", "BUR_", "POS_"))
    return not (feature in SENSITIVE or feature in LETTER_SKIP or building or feature.startswith("FLAG_DOCUMENT"))


def build_letter_input(applicant_id: int, explanation: dict, raw_row, cutoff: float,
                       n_reasons: int = 4, n_strengths: int = 3) -> LetterInput:
    """Keep the top letter-friendly reasons in each direction; sensitive features never reach the prompt."""
    def keep(table, n):
        t = table[table["feature"].map(_letter_friendly)].head(n)
        return [{"code": f, "label": l, "value": v} for f, l, v in zip(t["feature"], t["label"], t["value"])]

    all_rows = explanation["all"]
    raises = all_rows[all_rows["shap"] > 0].sort_values("shap", ascending=False)
    lowers = all_rows[all_rows["shap"] < 0].sort_values("shap")
    return LetterInput(
        applicant_id=int(applicant_id),
        decision=decision_for(explanation["pd"], cutoff),
        loan_amount=format_value("AMT_CREDIT", raw_row["AMT_CREDIT"]),
        regular_payment=format_value("AMT_ANNUITY", raw_row["AMT_ANNUITY"]),
        number_of_payments=format_value("CREDIT_TERM", raw_row["CREDIT_TERM"]),
        risk_factors=keep(raises, n_reasons),
        strengths=keep(lowers, n_strengths),
    )


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
JARGON = re.compile(r"\b(shap|log[- ]odds|algorithm|probability of default|PD|cut-?off|threshold of|machine learning)\b", re.I)
SENSITIVE_WORDS = re.compile(r"\b(age|aged|young|old(er)?|gender|male|female|married|marital|single|divorced|widow(ed)?|children|child|family status|pregnan\w*)\b", re.I)
POSITIVE_FOR_DECLINE = re.compile(r"\b(congratulat\w*|pleased to approve|has been approved|you('re| are) approved)\b", re.I)
# Comparisons the fact sheet never states (e.g. "longer than your past loans") - invented reasoning
COMPARISON = re.compile(r"\b((longer|shorter|higher|lower|larger|smaller|bigger|greater) than|similar to|at (this|the same) level)\b", re.I)
VEHICLE = re.compile(r"\b(car|vehicle)\b", re.I)
TOOL_NAME = re.compile(r"Second\s?Look")       # case-sensitive: "take a second look" is fine


def _personal_characteristic(body: str) -> str | None:
    """First sensitive word found, ignoring 'age'/'old' when the sentence is about a car (OWN_CAR_AGE is a real factor)."""
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        for m in SENSITIVE_WORDS.finditer(sentence):
            if m.group(0).lower() in ("age", "aged", "old", "older") and VEHICLE.search(sentence):
                continue
            return m.group(0)
    return None


def check_letter(letter: Letter, li: LetterInput) -> list[str]:
    """Return a list of problems (empty = passed)."""
    issues = []
    allowed_r = {f["code"] for f in li.risk_factors}
    allowed_s = {f["code"] for f in li.strengths}
    bad_r = set(letter.reasons_cited) - allowed_r
    bad_s = set(letter.strengths_cited) - allowed_s
    if bad_r:
        issues.append(f"cites risk factors not in the fact sheet: {sorted(bad_r)}")
    if bad_s:
        issues.append(f"cites strengths not in the fact sheet: {sorted(bad_s)}")
    if set(letter.reasons_cited + letter.strengths_cited) & SENSITIVE:
        issues.append("cites a sensitive feature")
    if word := _personal_characteristic(letter.body):
        issues.append(f"mentions a personal characteristic: '{word}'")
    if m := JARGON.search(letter.body):
        issues.append(f"uses technical jargon: '{m.group(0)}'")
    if TOOL_NAME.search(letter.subject + " " + letter.body):
        issues.append("mentions the internal tool name SecondLook")
    for m in COMPARISON.finditer(letter.body):
        # Ratio factors (loan vs goods price, loan or repayment vs income) ARE comparisons: allow "... than the price/income"
        compared_to = letter.body[m.end():m.end() + 30]
        if "than" in m.group(0).lower() and re.match(r"\s+(the\s+|your\s+)?(\w+\s+)?(price|income|cost)\b", compared_to, re.I):
            continue
        issues.append(f"possible invented comparison: '{m.group(0)}{compared_to[:20]}...' (describe factors only by their own value)")
        break
    if li.decision in ("decline", "under_review") and not letter.reasons_cited:
        issues.append("gives no reason for a decline / review")
    if li.decision in ("decline", "under_review") and POSITIVE_FOR_DECLINE.search(letter.body):
        issues.append("tone says approved but the decision is not an approval")
    if li.decision == "approve" and letter.reasons_cited:
        issues.append("approval letter lists risk factors")
    words = len(letter.body.split())
    if not 90 <= words <= 280:
        issues.append(f"length {words} words is outside 90-280")
    return issues


# ---------------------------------------------------------------------------
# Generation with cache
# ---------------------------------------------------------------------------
def _client() -> anthropic.Anthropic:
    load_dotenv(ROOT / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set. Add it to .env (see .env.example).")
    return anthropic.Anthropic()


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def cache_key(li: LetterInput) -> str:
    return f"{li.applicant_id}|{li.decision}|{PROMPT_VERSION}|{MODEL}"


def generate_letter(li: LetterInput, use_cache: bool = True, max_attempts: int = 2) -> dict:
    """Draft (or fetch from cache) a checked letter for one applicant.

    Returns a dict with the letter fields, the check results, attempts, token usage and cost.
    """
    cache = _load_cache() if use_cache else {}
    key = cache_key(li)
    if key in cache:
        return {**cache[key], "from_cache": True}

    client = _client()
    messages = [{"role": "user", "content": f"Fact sheet:\n{li.fact_sheet()}"}]
    usage_in = usage_out = 0
    start = time.time()
    for attempt in range(1, max_attempts + 1):
        response = client.messages.parse(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            output_config={"effort": EFFORT},
            messages=messages,
            output_format=Letter,
        )
        usage_in += response.usage.input_tokens
        usage_out += response.usage.output_tokens
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined to draft a letter for applicant {li.applicant_id}")
        letter = response.parsed_output
        issues = check_letter(letter, li)
        if not issues:
            break
        # Retry once, telling Claude exactly what to fix
        messages = messages + [
            {"role": "assistant", "content": letter.model_dump_json()},
            {"role": "user", "content": "Please fix these problems and return the full letter again: " + "; ".join(issues)},
        ]

    result = {
        **letter.model_dump(),
        "applicant_id": li.applicant_id, "decision": li.decision,
        "checks_passed": not issues, "issues": issues, "attempts": attempt,
        "input_tokens": usage_in, "output_tokens": usage_out,
        "cost_usd": round(usage_in * PRICE_IN / 1e6 + usage_out * PRICE_OUT / 1e6, 6),
        "seconds": round(time.time() - start, 1), "model": MODEL, "prompt_version": PROMPT_VERSION,
        "fact_sheet": asdict(li),
    }
    if use_cache:
        cache = _load_cache()
        cache[key] = result
        _save_cache(cache)
    return {**result, "from_cache": False}
