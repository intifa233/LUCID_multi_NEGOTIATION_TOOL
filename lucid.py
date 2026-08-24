# -*- coding: utf-8 -*-
"""
Flask backend server for the LUCID Qualtrics chat interface.

Acts as a proxy between the Qualtrics frontend JavaScript and the OpenAI API.
Handles CORS, fetches configuration from environment variables, makes API calls,
and returns responses. Includes a root endpoint to display deployment status
and the necessary Qualtrics URL.
"""
from flask import Flask, request, jsonify, make_response
import json
import os      # Used for accessing environment variables (API keys, config)
import requests # Used for making HTTP requests to the OpenAI API
import html
import re      # For parsing model JSON/code-fence responses
import yaml    # For loading per-condition negotiation prompts from prompts.yaml
from datetime import datetime, timezone  # For timestamping offer-trajectory entries

# Initialize the Flask application
app = Flask(__name__)

# --- Condition Prompts (prompts.yaml) ---

def _load_condition_prompts():
    """
    Loads prompts.yaml (the per-condition negotiation system prompts) once at cold
    start. This lets the Prosocial/Proself prompt text be edited and redeployed
    independently of the Qualtrics .qsf file - no re-import into Qualtrics needed,
    and no risk of breaking anything else in the survey while editing a prompt.

    Note: this is separate from, and does not affect, the issue-tracking/extraction
    prompt in _extract_issue_updates_from_message_llm() below - that one stays
    hardcoded here.

    Returns {} (feature silently disabled, falls back to whatever prompt the
    frontend sends) if the file is missing or malformed, so a bad/missing YAML
    file never takes the whole endpoint down.
    """
    try:
        prompts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prompts.yaml')
        with open(prompts_path, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        # Normalize condition keys (e.g. "Prosocial" -> "prosocial") for case-insensitive lookup
        return {str(k).strip().lower(): v for k, v in data.items()}
    except Exception as e:
        print(f"[WARN] Could not load prompts.yaml ({type(e).__name__}: {e}). Condition-based prompt override disabled.")
        return {}

CONDITION_PROMPTS = _load_condition_prompts()

def _default_issue_statuses():
    return [
        {'id': 'issue-1', 'label': 'Bonus', 'status': ''},
        {'id': 'issue-2', 'label': 'Job Assignment', 'status': ''},
        {'id': 'issue-3', 'label': 'Vacation Time', 'status': ''},
        {'id': 'issue-4', 'label': 'Starting Date', 'status': ''},
        {'id': 'issue-5', 'label': 'Moving Expense Coverage', 'status': ''},
        {'id': 'issue-6', 'label': 'Insurance Coverage', 'status': ''},
        {'id': 'issue-7', 'label': 'Salary', 'status': ''},
        {'id': 'issue-8', 'label': 'Location', 'status': ''},
    ]


def _extract_issue_updates_from_message_llm(message, openai_api_key):
    """
    Use an LLM to extract issue updates from a single assistant message.
    Returns a dict like {'issue-1': '4%', 'issue-7': '$84,000'}.
    Returns {} on any failure.
    """
    if not message or not openai_api_key:
        return {}

    def _norm_key(text):
        return ''.join(ch for ch in str(text).lower() if ch.isalnum())

    defaults = _default_issue_statuses()
    id_to_label = {item['id']: item['label'] for item in defaults}
    label_to_id = {_norm_key(item['label']): item['id'] for item in defaults}
    # Common label variants that appear in assistant messages
    label_aliases = {
        'movingexpense': 'issue-5',
        'movingexpensecovered': 'issue-5',
        'movingexpensecoverage': 'issue-5',
        'insurance': 'issue-6',
        'insurancecovered': 'issue-6',
        'insurancecoverage': 'issue-6',
        'startdate': 'issue-4',
        'jobassignmentdivision': 'issue-2',
    }
    label_to_id.update(label_aliases)

    prompt = (
        "You extract the current status of negotiation issues from an assistant transcript. "
        "The transcript may contain multiple assistant turns. "
        "For each issue, find the MOST RECENT concrete value mentioned anywhere in the transcript. "
        "Include ALL issues that have any concrete value mentioned — do NOT skip issues just because "
        "their value did not change between turns. "
        "The text may contain markdown (**bold**), bullet points, dashes, or compact formatting. "
        "Issue IDs and labels are: "
        "issue-1 Bonus, issue-2 Job Assignment, issue-3 Vacation Time, issue-4 Starting Date, "
        "issue-5 Moving Expense Coverage, issue-6 Insurance Coverage, issue-7 Salary, issue-8 Location. "
        "Return ONLY valid JSON in this exact shape: "
        "{\"updates\":[{\"id\":\"issue-1\",\"label\":\"Bonus\",\"status\":\"4%\"}]}. "
        "Use ids whenever possible. Preserve exact values (e.g., Division A, Plan E, August 1, $82,000, 60%). "
        "Do not include entries with empty status."
    )

    cleaned_message = str(message)
    cleaned_message = cleaned_message.replace('**', '')
    cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()

    payload = {
        'model': 'gpt-4o-mini',
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': cleaned_message}
        ],
        'temperature': 0.0,
        'max_tokens': 600,
        'response_format': {'type': 'json_object'}
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    try:
        resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=12)
        if resp.status_code != 200:
            print(f"[INFO] LLM issue-update extraction returned {resp.status_code}, skipping updates")
            return {}

        raw = resp.json()['choices'][0]['message']['content']
        content = str(raw).strip()

        if content.startswith('```'):
            content = re.sub(r'^```(?:json)?\s*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content)

        parsed = None
        try:
            parsed = json.loads(content)
        except Exception:
            obj_match = re.search(r'\{.*\}', content, re.DOTALL)
            if obj_match:
                parsed = json.loads(obj_match.group(0))

        if not isinstance(parsed, dict):
            return {}

        updates_list = parsed.get('updates')
        if isinstance(updates_list, dict):
            # Also accept shape: {"updates": {"issue-1": "4%", ...}}
            updates_list = [
                {'id': issue_id, 'status': status}
                for issue_id, status in updates_list.items()
            ]
        if not isinstance(updates_list, list):
            return {}

        updates = {}
        for item in updates_list:
            if not isinstance(item, dict):
                continue

            issue_id = str(item.get('id', '')).strip().lower()
            if issue_id not in id_to_label:
                # allow model to return label when id is missing
                issue_label = _norm_key(item.get('label', ''))
                issue_id = label_to_id.get(issue_label, '')

            if not issue_id:
                continue

            status = item.get('status', '')
            status = str(status).strip() if status is not None else ''
            if status:
                updates[issue_id] = status

        return updates

    except Exception as e:
        print(f"[INFO] LLM issue-update extraction exception: {e}")
        return {}


def _detect_first_concession_llm(user_message, openai_api_key):
    """
    Used only for the Prosocial condition's one-time "first concession" exception
    (see prompts.yaml [NEGOTIATION PROTOCOL]): judges whether the candidate's latest
    message offers a TRADE - conceding on one issue specifically to ask the recruiter
    to move on a different one - and if so, which issue they're asking the recruiter
    to move on. This is a separate, cheap classification call rather than something
    the model is asked to track itself, because "has the candidate ever conceded
    before this point in the conversation" requires scanning arbitrarily far back in
    history - exactly the kind of long-range self-tracking LLMs are unreliable at
    (worse than the round-counting problem solved elsewhere by injecting the round
    number directly).

    Also extracts conceded_issue_id/conceded_new_value - which issue the candidate is
    giving ground on and the specific new value they're offering - purely as a linguistic
    read of the message. This is deliberately NOT trusted on its own: the caller cross-
    checks it against RECRUITER_PAYOFF_TABLE before treating is_concession as true, because
    "sounds like a concession" and "is actually favorable to the recruiter" are different
    questions (e.g. a candidate offering an EARLIER start date reads like a concession but
    scores WORSE for the recruiter on the real payoff table - not a real concession at all).

    Returns {'is_concession': False, 'requested_issue_id': None, 'conceded_issue_id': None,
    'conceded_new_value': None} on any failure, or if no message/key was given, so this
    never blocks the main call.
    """
    empty_result = {
        'is_concession': False, 'requested_issue_id': None,
        'conceded_issue_id': None, 'conceded_new_value': None
    }
    if not user_message or not openai_api_key:
        return dict(empty_result)

    defaults = _default_issue_statuses()
    valid_issue_ids = {item['id'] for item in defaults}

    system_prompt = (
        "You analyze one message from a job candidate in a negotiation. Determine whether "
        "the candidate is offering a TRADE: willing to give ground / accept less on one "
        "issue, specifically in order to ask for movement on a DIFFERENT issue. This must "
        "be an explicit or clearly implied concession paired with a request, not just a "
        "one-sided ask with no give. "
        "Issue ids and labels are: issue-1 Bonus, issue-2 Job Assignment, issue-3 Vacation "
        "Time, issue-4 Starting Date, issue-5 Moving Expense Coverage, issue-6 Insurance "
        "Coverage, issue-7 Salary, issue-8 Location. "
        "Return ONLY valid JSON in this exact shape: "
        "{\"is_concession\": true, \"requested_issue_id\": \"issue-6\", "
        "\"conceded_issue_id\": \"issue-4\", \"conceded_new_value\": \"July 1\"}. "
        "requested_issue_id is the issue the candidate is asking the RECRUITER to move on "
        "or improve - use null if is_concession is false or the requested issue is unclear. "
        "conceded_issue_id/conceded_new_value describe what the candidate is giving ground "
        "on: the issue, and the specific concrete new value they're now offering on it (e.g. "
        "'July 1', 'Division C', '80%') - not a description. Use null for both if "
        "is_concession is false or this is unclear."
    )

    payload = {
        'model': 'gpt-4o-mini',  # fast/cheap model for a single lightweight classification
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': str(user_message)}
        ],
        'temperature': 0.0,  # deterministic classification
        'max_tokens': 150,
        'response_format': {'type': 'json_object'}
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    try:
        resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"[INFO] First-concession detection returned {resp.status_code}, skipping")
            return dict(empty_result)

        raw = resp.json()['choices'][0]['message']['content']
        parsed = json.loads(raw)
        is_concession = bool(parsed.get('is_concession'))
        requested_issue_id = parsed.get('requested_issue_id')
        if requested_issue_id not in valid_issue_ids:
            requested_issue_id = None
        conceded_issue_id = parsed.get('conceded_issue_id')
        if conceded_issue_id not in valid_issue_ids:
            conceded_issue_id = None
        conceded_new_value = parsed.get('conceded_new_value')
        conceded_new_value = str(conceded_new_value).strip() if conceded_new_value else None
        return {
            'is_concession': is_concession,
            'requested_issue_id': requested_issue_id,
            'conceded_issue_id': conceded_issue_id,
            'conceded_new_value': conceded_new_value
        }

    except Exception as e:
        print(f"[INFO] First-concession detection exception: {e}")
        return dict(empty_result)


def _detect_reciprocity_claim_llm(assistant_message, openai_api_key):
    """
    Analyzes the RECRUITER's own reply for a claimed reciprocity trigger: is it justifying a
    grant by crediting the candidate with having given something up or offered flexibility on
    a specific issue this round (e.g. "since you offered flexibility on starting earlier, I
    can reciprocate by...")? This is the same directional bug the first-concession check
    guards against (see _detect_first_concession_llm), but that check only ever runs once, at
    the one-time first-concession moment - this runs every round, for both conditions, because
    prompts.yaml's ordinary reciprocity rule (not the one-time exception) can invoke the same
    "sounds like a concession but isn't" framing at any point in the negotiation (e.g. an
    EARLIER start date reads like a concession but scores WORSE for the recruiter on the real
    payoff table).

    Returns {'claims_reciprocity': False, 'credited_issue_id': None, 'credited_value': None}
    on any failure, or if no message/key given, so this never blocks the main call.
    """
    empty_result = {'claims_reciprocity': False, 'credited_issue_id': None, 'credited_value': None}
    if not assistant_message or not openai_api_key:
        return dict(empty_result)

    defaults = _default_issue_statuses()
    valid_issue_ids = {item['id'] for item in defaults}

    system_prompt = (
        "You analyze one message from a job RECRUITER in a negotiation. Determine whether the "
        "recruiter is justifying a concession/grant by crediting the CANDIDATE with having "
        "given something up or offered flexibility on a specific issue (e.g. 'since you "
        "offered flexibility on starting earlier, I can...', 'to reciprocate your willingness "
        "to...'). This must be an explicit or clearly implied claim that the CANDIDATE moved "
        "or offered something specific on some issue - not just the recruiter unilaterally "
        "granting something with no such claim. "
        "Issue ids and labels are: issue-1 Bonus, issue-2 Job Assignment, issue-3 Vacation "
        "Time, issue-4 Starting Date, issue-5 Moving Expense Coverage, issue-6 Insurance "
        "Coverage, issue-7 Salary, issue-8 Location. "
        "Return ONLY valid JSON in this exact shape: "
        "{\"claims_reciprocity\": true, \"credited_issue_id\": \"issue-4\", "
        "\"credited_value\": \"June 15\"}. "
        "credited_issue_id/credited_value are the issue and the specific concrete value the "
        "recruiter is crediting to the candidate (e.g. 'June 15', 'Division C', '80%') - not "
        "a description. Use null for both if claims_reciprocity is false or unclear."
    )

    payload = {
        'model': 'gpt-4o-mini',  # fast/cheap model for a single lightweight classification
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': str(assistant_message)}
        ],
        'temperature': 0.0,  # deterministic classification
        'max_tokens': 150,
        'response_format': {'type': 'json_object'}
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    try:
        resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"[INFO] Reciprocity-claim detection returned {resp.status_code}, skipping")
            return dict(empty_result)

        raw = resp.json()['choices'][0]['message']['content']
        parsed = json.loads(raw)
        claims_reciprocity = bool(parsed.get('claims_reciprocity'))
        credited_issue_id = parsed.get('credited_issue_id')
        if credited_issue_id not in valid_issue_ids:
            credited_issue_id = None
        credited_value = parsed.get('credited_value')
        credited_value = str(credited_value).strip() if credited_value else None
        return {
            'claims_reciprocity': claims_reciprocity,
            'credited_issue_id': credited_issue_id,
            'credited_value': credited_value
        }

    except Exception as e:
        print(f"[INFO] Reciprocity-claim detection exception: {e}")
        return dict(empty_result)


# --- Hold-firm enforcement (rounds 1-N, both conditions) ---
# Shared by Prosocial and Proself - both prompts hold the same opening anchor and the
# same "don't move Salary/Vacation Time in the first HOLD_FIRM_ROUNDS rounds" rule (see
# [Starting point] / [CONCESSION PACING] in prompts.yaml). Keep these in sync with that
# file if the case's opening offer or hold-firm window ever changes.
HOLD_FIRM_ROUNDS = 3
HOLD_FIRM_ANCHOR = {'issue-7': '$84,000', 'issue-3': '10 days'}  # Salary, Vacation Time

# Full opening-offer baseline, all 8 issues - matches [Starting point] in prompts.yaml
# exactly (same values, both conditions). HOLD_FIRM_ANCHOR above only covers Salary/
# Vacation; checks that need a fallback "prior value" for the other six issues (the
# first-concession directional check and its grant safety net) use this instead. Keep in
# sync with prompts.yaml if the case's opening offer ever changes.
RECRUITER_OPENING_OFFER = {
    'issue-1': '4%', 'issue-2': 'Division B', 'issue-3': '10 days', 'issue-4': 'July 15',
    'issue-5': '70%', 'issue-6': 'Plan D', 'issue-7': '$84,000', 'issue-8': 'Atlanta',
}


def _parse_number(text):
    """
    Parses a loose numeric expression - '$86,000', '86000', '86k', '15 days', '10%' - into
    a plain int, or None if nothing parseable is found. Handles the 'k' thousands shorthand
    (e.g. '84k' -> 84000): a plain [\\d,]+ regex would otherwise truncate that to 84, which
    then silently nearest-matches against payoff-table figures in the tens of thousands and
    returns a badly wrong point value instead of failing loudly. Shared by _matches_anchor
    and _lookup_recruiter_points so both get this fix in one place.
    """
    if not text:
        return None
    m = re.search(r'([\d,]+(?:\.\d+)?)\s*([kK])?', str(text))
    if not m or not m.group(1):
        return None
    num = float(m.group(1).replace(',', ''))
    if m.group(2):
        num *= 1000
    return int(round(num))


def _matches_anchor(extracted_value, anchor_value):
    """
    Loose comparison for the hold-firm check: '$84,000' should match '84000'/'84,000'
    despite formatting differences from the LLM extraction, so this doesn't flag a false
    violation over punctuation alone. Falls back to a case-insensitive exact string match
    if either side has no parseable number.
    """
    extracted_num = _parse_number(extracted_value)
    anchor_num = _parse_number(anchor_value)
    if extracted_num is not None and anchor_num is not None:
        return extracted_num == anchor_num
    return str(extracted_value).strip().lower() == str(anchor_value).strip().lower()


# --- Recruiter payoff table (all 8 issues) ---
# Digitized from the same [PAYOFF SCHEDULE — RECRUITER] table in prompts.yaml (identical in
# both conditions). This is what lets CODE tell whether a given value is actually favorable
# or unfavorable to the recruiter, instead of trusting the model's own framing or just
# detecting "did the value change." Example of why that distinction matters: a candidate
# offering to start EARLIER sounds like a concession, but on this table later start dates
# score higher for the recruiter (Jun1=0 ... Aug1=2400) - "earlier" is actually a worse
# position for the recruiter, not a gift. Keep in sync with prompts.yaml if the case's
# payoff schedule ever changes.
RECRUITER_PAYOFF_TABLE = {
    'issue-1': {'kind': 'number', 'options': [('10%', 0), ('8%', 400), ('6%', 800), ('4%', 1200), ('2%', 1600)]},              # Bonus
    'issue-2': {'kind': 'letter', 'options': [('Division A', 0), ('Division B', -600), ('Division C', -1200), ('Division D', -1800), ('Division E', -2400)]},  # Job Assignment
    'issue-3': {'kind': 'number', 'options': [('25 days', 0), ('20 days', 1000), ('15 days', 2000), ('10 days', 3000), ('5 days', 4000)]},  # Vacation Time
    'issue-4': {'kind': 'date', 'options': [('June 1', 0), ('June 15', 600), ('July 1', 1200), ('July 15', 1800), ('August 1', 2400)]},     # Starting Date
    'issue-5': {'kind': 'number', 'options': [('100%', 0), ('90%', 200), ('80%', 400), ('70%', 600), ('60%', 800)]},           # Moving Expense Coverage
    'issue-6': {'kind': 'letter', 'options': [('Plan A', 0), ('Plan B', 800), ('Plan C', 1600), ('Plan D', 2400), ('Plan E', 3200)]},        # Insurance Coverage
    'issue-7': {'kind': 'number', 'options': [('$90,000', -6000), ('$88,000', -4500), ('$86,000', -3000), ('$84,000', -1500), ('$82,000', 0)]},  # Salary
    'issue-8': {'kind': 'city', 'options': [('New York', 0), ('Boston', 300), ('Chicago', 600), ('Atlanta', 900), ('San Francisco', 1200)]},  # Location
}

_CITY_ALIASES = {'new york': 'New York', 'ny': 'New York', 'nyc': 'New York', 'boston': 'Boston',
                  'chicago': 'Chicago', 'atlanta': 'Atlanta', 'san francisco': 'San Francisco', 'sf': 'San Francisco'}
_DATE_ALIASES = {'june 1': 'June 1', 'jun 1': 'June 1', 'june 15': 'June 15', 'jun 15': 'June 15',
                  'july 1': 'July 1', 'jul 1': 'July 1', 'july 15': 'July 15', 'jul 15': 'July 15',
                  'august 1': 'August 1', 'aug 1': 'August 1'}


def _lookup_recruiter_points(issue_id, raw_value):
    """
    Matches a free-text extracted value (e.g. "15 days", "Division B", "$86,000", "SF") to
    the closest option on RECRUITER_PAYOFF_TABLE for that issue and returns its point value,
    or None if it can't be matched at all (empty/unparseable/off-grid). Numeric issues
    (Bonus/Vacation/Moving Coverage/Salary) match by nearest number; Job Assignment/Insurance
    match by the A-E letter; Location/Starting Date match by keyword (handles common
    abbreviations like "SF"/"NY").
    """
    table = RECRUITER_PAYOFF_TABLE.get(issue_id)
    if not table or not raw_value:
        return None
    text = str(raw_value).strip()
    options = table['options']

    if table['kind'] == 'number':
        num = _parse_number(text)
        if num is None:
            return None
        best_pts, best_diff = None, None
        for label, pts in options:
            onum = _parse_number(label)
            diff = abs(onum - num)
            if best_diff is None or diff < best_diff:
                best_pts, best_diff = pts, diff
        return best_pts

    if table['kind'] == 'letter':
        m = re.search(r'\b([A-E])\b', text.upper())
        if not m:
            return None
        letter = m.group(1)
        for label, pts in options:
            if label.strip().upper().endswith(letter):
                return pts
        return None

    if table['kind'] == 'date':
        low = text.lower()
        # Longest alias first: "july 1" is a substring of "july 15", so checking short
        # aliases first would misidentify "July 15" as "July 1".
        for alias, canonical in sorted(_DATE_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if alias in low:
                for label, pts in options:
                    if label == canonical:
                        return pts
        return None

    if table['kind'] == 'city':
        low = text.lower()
        for alias, canonical in sorted(_CITY_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if alias in low:
                for label, pts in options:
                    if label == canonical:
                        return pts
        return None

    return None


def _compare_recruiter_value(issue_id, new_value, prior_value):
    """
    Returns 'better' / 'worse' / 'same' / 'unknown' - whether new_value scores higher (more
    favorable to the recruiter), lower, the same, or couldn't be compared at all, versus
    prior_value, using RECRUITER_PAYOFF_TABLE. This is the ground-truth check for "is this
    actually a concession" that pure string/anchor matching can't provide.
    """
    new_pts = _lookup_recruiter_points(issue_id, new_value)
    prior_pts = _lookup_recruiter_points(issue_id, prior_value)
    if new_pts is None or prior_pts is None:
        return 'unknown'
    if new_pts > prior_pts:
        return 'better'
    if new_pts < prior_pts:
        return 'worse'
    return 'same'


def _one_level_step(issue_id, current_value):
    """
    Returns the next grid option ONE level toward the CANDIDATE - i.e. the option with the
    next-lower recruiter point value - from current_value, or None if current_value can't be
    matched or is already at the grid's most candidate-favorable option. Caps the first-
    concession exception's "unconditional" grant to a single step instead of jumping straight
    to whatever the candidate specifically asked for (each issue only has 5 discrete grid
    values, and RECRUITER_PAYOFF_TABLE's option lists aren't consistently ordered by
    favorability - e.g. issue-2's list runs best-for-recruiter-first while issue-6's runs the
    opposite way - so this sorts by points rather than trusting list order).
    """
    table = RECRUITER_PAYOFF_TABLE.get(issue_id)
    if not table:
        return None
    current_pts = _lookup_recruiter_points(issue_id, current_value)
    if current_pts is None:
        return None
    sorted_options = sorted(table['options'], key=lambda kv: kv[1])
    pts_list = [pts for _, pts in sorted_options]
    try:
        idx = pts_list.index(current_pts)
    except ValueError:
        return None
    if idx == 0:
        return None  # already at the most candidate-favorable option on this issue's grid
    return sorted_options[idx - 1][0]


def _first_concession_grant_status(prior_statuses, assistant_updates, target_issue_id):
    """
    Evaluates a first-concession grant note (see /lucid Step 3) against the one-level cap
    (see _one_level_step - the grant is unconditional but capped to a single grid step, never
    jumped straight to the candidate's full ask). If target_issue_id is given, only that issue
    counts (the model was told exactly which one to grant); otherwise any of the six non-
    Salary/Vacation issues counts (the model had a free choice among them).

    Returns one of:
      ('ok', None)                    - some eligible issue moved exactly one level in the
                                         candidate's favor, or nothing could be verified either
                                         way (benefit of the doubt - letter/date/city matching
                                         can legitimately fail to resolve)
      ('overshoot', (issue_id, cap))  - an eligible issue moved, but past its one-level cap
      ('missing', None)               - nothing eligible moved at all
    """
    prior_by_id = {item['id']: item.get('status', '') for item in prior_statuses}
    candidate_ids = [target_issue_id] if target_issue_id else [
        item['id'] for item in _default_issue_statuses() if item['id'] not in ('issue-3', 'issue-7')
    ]
    saw_unknown = False
    overshoot = None
    for issue_id in candidate_ids:
        if issue_id not in assistant_updates:
            continue
        prior_val = prior_by_id.get(issue_id) or RECRUITER_OPENING_OFFER.get(issue_id)
        direction = _compare_recruiter_value(issue_id, assistant_updates[issue_id], prior_val)
        if direction == 'worse':  # worse for the recruiter = moved in the candidate's favor
            cap_value = _one_level_step(issue_id, prior_val)
            if cap_value is not None and _compare_recruiter_value(issue_id, assistant_updates[issue_id], cap_value) == 'worse':
                overshoot = (issue_id, cap_value)  # moved further than the one-level cap allows
                continue
            return ('ok', None)
        if direction == 'unknown':
            saw_unknown = True
    if saw_unknown:
        return ('ok', None)
    if overshoot:
        return ('overshoot', overshoot)
    return ('missing', None)


# --- Round-based concession pacing targets (both conditions) ---
# Turns prompts.yaml's [CONCESSION PACING] prose into an explicit, checkable requirement:
# by the round the schedule names as a deadline, Salary/Vacation Time must have moved at
# least this far - not just "described as expected" in a system prompt the model has to
# remember to apply many turns later. Built after testing showed relying on the prose alone
# was unreliable in BOTH directions across models: conceding too early (caught by the
# hold-firm check above), and never conceding at all even past the deadline (which nothing
# previously caught). Keep in sync with prompts.yaml if the schedule ever changes.
PROSELF_CONCESSION_ROUND = 9  # proself's schedule: "only in rounds 9-10" (or genuine walkaway risk, not modeled here)
PACING_STEP = {'issue-7': '$86,000', 'issue-3': '15 days'}  # the mandated step for both conditions' first move


def _pacing_target(condition_key, turn_number):
    """
    Returns {'issue-7': target_or_None, 'issue-3': target_or_None} - the minimum concession
    level (canonical grid value) the recruiter is required to have reached by this round.
    None means the opening anchor is still an acceptable position, no mandatory move yet.
    """
    if condition_key == 'prosocial' and turn_number >= 6:  # schedule: "$86,000 by round 6" / "15 days by round 6"
        return dict(PACING_STEP)
    if condition_key == 'proself' and turn_number >= PROSELF_CONCESSION_ROUND:
        return dict(PACING_STEP)
    return {'issue-7': None, 'issue-3': None}


def _call_openai_completion(messages, model, temperature, seed, openai_api_key, timeout=30):
    """
    Minimal OpenAI chat completion call used only for the hold-firm regeneration retry
    (see /lucid Step 5). Returns the generated text, or None on any failure - deliberately
    lightweight, since a failed retry just means the caller keeps the original reply
    rather than needing full error-handling parity with the main call in Step 4.
    """
    try:
        payload = {'model': model, 'messages': messages, 'temperature': temperature}
        if seed is not None:
            payload['seed'] = seed
        resp = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {openai_api_key}'},
            json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            print(f"[WARN] Hold-firm regeneration call returned {resp.status_code}")
            return None
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[WARN] Hold-firm regeneration call exception: {e}")
        return None


def _normalize_prior_issue_statuses(raw):
    """
    Coerce whatever issue-status snapshot the frontend sent back (echoed from a
    previous response, read out of Qualtrics Embedded Data) into the canonical
    8-slot list. Falls back to empty defaults if 'raw' is missing or malformed,
    so a first turn (or an older frontend that doesn't send this field yet)
    degrades gracefully instead of erroring.
    """
    defaults = _default_issue_statuses()
    if not isinstance(raw, list) or not raw:
        return defaults

    by_id = {}
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        issue_id = str(item.get('id') or '').strip().lower() or f'issue-{idx + 1}'
        by_id[issue_id] = str(item.get('status') or '').strip()

    return [
        {'id': item['id'], 'label': item['label'], 'status': by_id.get(item['id'], '')}
        for item in defaults
    ]


def apply_issue_updates(base_statuses, updates):
    """
    Overlay a {issue_id: status} updates dict onto a base 8-slot issue-status
    list, returning a new list. Unmentioned issues keep their prior value;
    only non-empty updates overwrite.
    """
    result = [dict(item) for item in base_statuses]
    id_to_index = {item['id']: idx for idx, item in enumerate(result)}
    for issue_id, status in (updates or {}).items():
        idx = id_to_index.get(issue_id)
        if idx is None or not status:
            continue
        result[idx]['status'] = status
    return result


def diff_issue_statuses(prior_statuses, new_statuses):
    """
    Compare two 8-slot issue-status lists and return only the issues whose
    value actually changed, e.g. {"issue-7": {"label": "Salary", "from": "$82,000", "to": "$85,000"}}.
    This is the per-round "what moved" signal that a single latest-value
    snapshot can't provide.
    """
    prior_by_id = {item['id']: (item.get('status') or '').strip() for item in prior_statuses}
    changed = {}
    for item in new_statuses:
        issue_id = item['id']
        old_val = prior_by_id.get(issue_id, '')
        new_val = (item.get('status') or '').strip()
        if new_val and new_val != old_val:
            changed[issue_id] = {'label': item.get('label', ''), 'from': old_val, 'to': new_val}
    return changed


def build_issue_trajectory_entry(turn_number, user_message, assistant_message,
                                  prior_statuses, user_updates, assistant_updates, merged_statuses):
    """
    Package one round's worth of multi-issue trade-off signal: who said what,
    what each side's message contributed, and what changed vs. the prior
    snapshot. Appending one of these per round (frontend-side) builds the full
    negotiation trajectory instead of only ever exposing the latest state.
    """
    return {
        'turn_number': turn_number,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'user_message_excerpt': (user_message or '')[:500],
        'assistant_message_excerpt': (assistant_message or '')[:500],
        'user_updates': user_updates or {},
        'assistant_updates': assistant_updates or {},
        'changed_from_prior': diff_issue_statuses(prior_statuses, merged_statuses),
        'issues': merged_statuses,
    }

# --- Configuration & CORS ---

def get_allowed_origins_config():
    """
    Reads the ALLOWED_ORIGINS environment variable and parses it into a list.
    Defaults to allowing all origins ('*') if the variable is not set.
    Uses print for logging and visible in Vercel Function Logs.
    """
    origins_str = os.getenv('ALLOWED_ORIGINS')
    print(f"[DEBUG ENV] Raw ALLOWED_ORIGINS: '{origins_str}'") # Vercel Log

    if not origins_str:
        # Default to wildcard if environment variable is missing or empty
        print("[WARN ENV] ALLOWED_ORIGINS not set. Defaulting CORS to allow all ('*').") # Vercel Log
        return ['*']

    # Parse comma-separated list, removing empty strings and stripping whitespace
    allowed_list = [origin.strip() for origin in origins_str.split(',') if origin.strip()]
    print(f"[DEBUG ENV] Parsed ALLOWED_ORIGINS: {allowed_list}") # Vercel Log
    return allowed_list

@app.before_request
def handle_preflight():
    """
    Handles CORS preflight (OPTIONS) requests specifically for the /lucid endpoint.
    Checks the request's Origin header against the ALLOWED_ORIGINS config
    and returns appropriate CORS headers if allowed, or a 403 if denied.
    Echoes back requested headers. Only adds 'Access-Control-Allow-Credentials'
    when needed and with the value 'true'.
    """
    # Intercept only OPTIONS requests targetting the main API endpoint
    # UPDATED: Changed request.method.upper() == 'OPTIONS' to request.method == 'OPTIONS' (Flask normalizes it)
    if request.method == 'OPTIONS' and request.path == '/lucid':
        print(f"[INFO] Intercepting OPTIONS request for {request.path}") # Vercel Log
        origin = request.headers.get('Origin') # Get the origin of the requesting domain
        allowed_origins = get_allowed_origins_config() # Fetch the configured allowed origins

        print(f"[DEBUG PREFLIGHT] Request Origin: '{origin}'") # Vercel Log
        print(f"[DEBUG PREFLIGHT] Checking against Allowed: {allowed_origins}") # Vercel Log

        ac_allow_origin = None # Initialize
        send_credentials = False # Initialize

        # ---- decide origin & credentials ------------------------
        if '*' in allowed_origins:
            ac_allow_origin = '*'
            send_credentials = False        # wildcard ⇒ no creds
            print("[DEBUG PREFLIGHT] Policy: Allowed Wildcard (*), Credentials False") # Vercel Log
        elif origin and origin in allowed_origins: # Added check for origin existence
            ac_allow_origin = origin
            send_credentials = True
            print(f"[DEBUG PREFLIGHT] Policy: Allowed Specific Origin ({origin}), Credentials True") # Vercel Log
        else:
            # Origin not allowed by configuration
            print(f"[WARN] Preflight origin '{origin}' denied by policy for /lucid.") # Vercel Log
            return make_response('Origin not permitted for CORS preflight', 403)

        # ---- echo back ALL requested headers --------------------
        # Retrieve the headers the browser wants to send in the actual request
        req_hdrs = request.headers.get(
            'Access-Control-Request-Headers', ''
        )  # e.g. "X-Requested-With,Content-Type" or just "Content-Type" etc.
        print(f"[DEBUG PREFLIGHT] Access-Control-Request-Headers received: '{req_hdrs}'") # Vercel Log

        # Construct the response for the preflight request (204 No Content)
        res = make_response('', 204)

        # Build the core CORS headers
        cors_headers = {
            'Access-Control-Allow-Origin': ac_allow_origin,
            'Access-Control-Allow-Methods': 'POST, OPTIONS', # Allowed methods for the actual request
            # Allow the headers the browser requested, default to Content-Type if none specified
            'Access-Control-Allow-Headers': req_hdrs if req_hdrs else 'Content-Type',
            'Access-Control-Max-Age': '86400' # Cache preflight response for 1 day
        }

        # --- Add Allow-Credentials header ONLY if needed and with value 'true' ---
        if send_credentials:
            cors_headers['Access-Control-Allow-Credentials'] = 'true'
            print("[DEBUG PREFLIGHT] Adding Access-Control-Allow-Credentials: true") # Vercel Log
        else:
             print("[DEBUG PREFLIGHT] Not adding Access-Control-Allow-Credentials header") # Vercel Log

        # Update response headers
        res.headers.update(cors_headers)

        print(f"[INFO] Preflight OK for /lucid. Sending 204 with headers: {dict(res.headers)}") # Vercel Log
        return res

    # If not an OPTIONS request for /lucid, proceed to the actual route function
    pass

# --- Application Routes ---

@app.route('/')
def hello_world():
    """
    Root endpoint (/). Primarily serves as a status check and provides a helpful
    HTML page displaying the correct URL needed for the Qualtrics setup,
    if deployed on Vercel (detects via VERCEL_URL env var).
    Also handles basic CORS headers for GET requests to the root.
    """
    print("[INFO] Root route '/' accessed.") # Vercel Log
    origin = request.headers.get('Origin')
    allowed_origins = get_allowed_origins_config()

    # Attempt to get the Vercel deployment URL from environment variables
    # --- Determine the correct backend URL using the incoming request context ---
    backend_url_for_qualtrics = "[Error determining backend URL from request]" # Default/fallback
    backend_url_base = "Unknown"
    try:
        # request.url_root gives "scheme://host:port/" - reflects how user accessed page
        # Strip the trailing '/' and append our specific endpoint path.
        backend_url_base = request.url_root.rstrip('/')
        backend_url_for_qualtrics = f"{backend_url_base}/lucid"
        backend_url_for_qualtrics = html.escape(backend_url_for_qualtrics) # Escape for safety
        print(f"[DEBUG URL] Derived base from request.url_root: {backend_url_base}")
        print(f"[DEBUG URL] Constructed Backend URL for Qualtrics: {backend_url_for_qualtrics}")
    except Exception as e:
        print(f"[ERROR URL] Failed to derive URL from request.url_root: {e}")

    # --- Format the displayed allowed origins ---
    # (Ensure allowed_origins is defined earlier in the function)
    escaped_origins_list = [html.escape(o) for o in allowed_origins]
    if escaped_origins_list == ['*']:
        allowed_origins_display = "<code>*</code> (Any origin - less secure)"
    else:
        allowed_origins_display = ", ".join(f"<code>{o}</code>" for o in escaped_origins_list)
        if not allowed_origins_display:
             allowed_origins_display = "<i>None specified (CORS likely misconfigured/denied)</i>"

    # --- Generate Simplified HTML Page ---
    # Uses only: backend_url_for_qualtrics, allowed_origins_display
    display_html = f"""
    <!DOCTYPE html><html lang="en">
    <head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>LUCID Backend Deployed</title>
    <style>
        body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif; padding: 20px; line-height: 1.6; background-color: #f8f9fa; color: #212529; }}
        .container {{ max-width: 750px; margin: 40px auto; padding: 35px; border: 1px solid #dee2e6; border-radius: 8px; background-color: #ffffff; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }}
        h1 {{ color: #0d6efd; border-bottom: 2px solid #0d6efd; padding-bottom: 10px; margin-bottom: 20px; }}
        h2 {{ color: #495057; margin-top: 30px; border-bottom: 1px solid #ced4da; padding-bottom: 8px;}}
        code {{ background-color: #e9ecef; padding: 0.2em 0.5em; border-radius: 4px; font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 0.9em; color: #d63384;}}
        .url-box {{ background-color: #f1f3f5; padding: 12px 18px; border: 1px solid #adb5bd; border-radius: 5px; font-family: "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; word-wrap: break-word; margin-bottom: 15px; font-size: 1.05em; color: #0b5ed7; }}
        button {{ padding: 10px 18px; cursor: pointer; border-radius: 5px; border: none; background-color: #0d6efd; color: white; font-size: 15px; transition: background-color 0.2s ease; }}
        button:hover {{ background-color: #0b5ed7; }}
        .copied-message {{ color: #198754; font-weight: bold; display: none; margin-left: 10px;}}
        .important {{ background-color: #fff3cd; border: 1px solid #ffeeba; color: #664d03; padding: 15px 20px; border-radius: 5px; margin-top: 20px; }}
        .important code {{ background-color: #fde7a0; color: #664d03; }}
        ul {{ margin-top: 10px; padding-left: 20px; }} li {{ margin-bottom: 5px; }}
        p {{ margin-bottom: 1rem; }}
    </style>
    </head>
    <body><div class="container">
        <h1>LUCID Backend Successfully Deployed!</h1>

        <h2>Next Step: Configure Qualtrics</h2>
        <p>To connect your Qualtrics survey to this backend:</p>
        <ol>
            <li><strong>Copy the full Backend URL below.</strong> This URL should reflect the main production domain when accessed via production. Use this URL for the <code>LUCIDBackendURL</code> Embedded Data field in Qualtrics.</li>
            <li>In your Qualtrics Survey Flow, create or update the Embedded Data field named <code>LUCIDBackendURL</code> and paste this URL as its value.</li>
        </ol>
        <p><strong>Backend URL (Value for <code>LUCIDBackendURL</code>):</strong></p>
        <div id="qualtricsUrlBox" class="url-box">{backend_url_for_qualtrics}</div>
        <button onclick="copyUrl()">Copy Backend URL</button>
        <span id="copiedMsg" class="copied-message">Copied!</span>
    </div>
    <script>function copyUrl() {{ const urlText = document.getElementById('qualtricsUrlBox').innerText; navigator.clipboard.writeText(urlText).then(() => {{ const msg = document.getElementById('copiedMsg'); msg.style.display = 'inline'; setTimeout(() => {{ msg.style.display = 'none'; }}, 2500); }}).catch(err => {{ console.error('Failed to copy: ', err); alert('Failed to copy URL.'); }}); }}</script>
    </body></html>
    """

    # Create Flask response object with the HTML
    resp = make_response(display_html)
    resp.headers['Content-Type'] = 'text/html' # Set correct MIME type

    # Apply basic CORS headers for the root route as well (GET requests usually simpler)
    origin_to_send = None
    send_credentials_get = False # Renamed variable to avoid conflict
    if '*' in allowed_origins:
        origin_to_send = '*'
    elif origin and origin in allowed_origins:
        origin_to_send = origin
        send_credentials_get = True # Allow credentials if specific origin matches
    if origin_to_send:
        resp.headers['Access-Control-Allow-Origin'] = origin_to_send
        resp.headers['Vary'] = 'Origin'
        # Only add credentials header if needed and true
        if send_credentials_get:
            resp.headers['Access-Control-Allow-Credentials'] = 'true'
    return resp

@app.route('/lucid', methods=['POST'])
def lucid():
    """
    Main API endpoint (/lucid).
    Receives chat messages and configuration from Qualtrics frontend via POST request.
    Validates request origin using CORS settings.
    Calls the OpenAI Chat Completions API.
    Returns the AI's response or an error message in JSON format.
    Includes necessary CORS headers on the response, including 'Access-Control-Allow-Credentials'
    only when appropriate and with the value 'true'.
    """
    # --- Step 1: CORS Check for POST request ---
    origin = request.headers.get('Origin')
    allowed_origins = get_allowed_origins_config()
    print(f"[DEBUG POST /lucid] Request Origin: '{origin}' vs Allowed: {allowed_origins}") # Vercel Log

    origin_to_send = None # Header value for Access-Control-Allow-Origin
    # 'allow_credentials_post' will determine if the 'Access-Control-Allow-Credentials' header is sent
    allow_credentials_post = False # Default to false, set true only for specific allowed origins
    is_request_allowed = False # Flag to track if request passes CORS check

    # Determine if the request origin is permitted
    if '*' in allowed_origins:
        origin_to_send = '*'
        is_request_allowed = True
        allow_credentials_post = False # Cannot use credentials with wildcard
        print("[DEBUG POST /lucid] Policy: Allowed Wildcard (*), Credentials False") # Vercel Log
    elif origin and origin in allowed_origins:
        origin_to_send = origin
        is_request_allowed = True
        allow_credentials_post = True # Allow credentials for specific origins
        print(f"[DEBUG POST /lucid] Policy: Allowed Specific Origin ({origin}), Credentials True") # Vercel Log
    else:
        # Origin is not in the allowed list (and not wildcard)
        is_request_allowed = False
        print(f"[DEBUG POST /lucid] Policy: Denied Origin ({origin})") # Vercel Log

    # If CORS check fails, return 403 Forbidden immediately
    if not is_request_allowed:
        print(f"[WARN] POST to /lucid denied for origin: {origin}.") # Vercel Log
        error_resp = make_response(jsonify({'error': 'Forbidden', 'message': 'Origin not permitted.'}), 403)
        # Add CORS headers even on error where possible, though browser might ignore on 403
        if origin_to_send:
             error_resp.headers['Access-Control-Allow-Origin'] = origin_to_send
             error_resp.headers['Vary'] = 'Origin'
             # Only add credentials header if needed and true
             if allow_credentials_post:
                 error_resp.headers['Access-Control-Allow-Credentials'] = 'true'
        return error_resp
    # --- End CORS Check ---

    # --- Step 2: Process Request Body ---
    print(f"[INFO] ------ Entered lucid function from allowed origin: {origin} ------") # Vercel Log
    post_data = request.data # Get raw request body
    print(f"[INFO /lucid] Received {len(post_data)} bytes.") # Vercel Log

    response_data = {} # Dictionary to hold the JSON response data
    status_code = 500  # Default to Internal Server Error

    try:
        # Decode body as UTF-8 and parse JSON
        body = json.loads(post_data.decode('utf-8'))

        # --- Step 3: Get and Check for API Key ---
        # UPDATED: Check for both uppercase and lowercase env var names
        openai_api_key = (
            os.getenv('OPENAI_API_KEY') or  # Vercel / production (Screaming Snake Case)
            os.getenv('openai_api_key')     # legacy/local (lower snake case)
        )

        # Basic check/log for the API key (without exposing the key itself)
        if isinstance(openai_api_key, str) and len(openai_api_key) > 7:
            print(f"[DIAGNOSTIC /lucid] API Key Found (Length: {len(openai_api_key)}).") # Vercel Log
        elif not openai_api_key:
            print("[CRITICAL DIAGNOSTIC /lucid] Neither os.getenv('OPENAI_API_KEY') nor os.getenv('openai_api_key') returned a value!") # Vercel Log

        # --- Check if API Key is actually present ---
        if not openai_api_key:
            print('[CRITICAL /lucid] OpenAI API key not found in environment variables (checked OPENAI_API_KEY and openai_api_key).') # Vercel Log
            # Set error response if key is missing
            response_data = {'error': 'Configuration Error', 'message':'OpenAI API key not configured on server.'}
            status_code = 500 # Indicate server configuration error
        else:
            # API Key found, proceed to extract data and call OpenAI

            # Extract parameters sent from Qualtrics frontend
            model = body.get('model', 'gpt-4o') # Use model from request, default to gpt-4o if not sent (JS usually sends its default)
            messages = body.get('messages', []) # Get message history array
            temp_from_frontend = body.get('temperature') # Get optional temperature
            seed_from_frontend = body.get('seed') # Get optional seed

            # Validate messages list (must not be empty)
            if not messages or not isinstance(messages, list):
                print("[WARN /lucid] Invalid or empty 'messages' list received.") # Vercel Log
                response_data = {'error': 'Bad Request', 'message': 'Messages list is missing, empty, or invalid.'}
                status_code = 400 # Bad Request
            else:
                # --- Required: resolve the system prompt from prompts.yaml via condition ---
                # prompts.yaml is now the single source of truth for what the AI recruiter
                # says. The LUCIDPromptInitial value in the .qsf is no longer used for this -
                # the frontend may still send it as messages[0], but it's overwritten below.
                # This endpoint requires a recognized `condition` field with a matching
                # prompts.yaml entry, and returns an error instead of silently falling back
                # to a stale or missing prompt.
                condition = body.get('condition')
                condition_key = str(condition).strip().lower() if condition else None
                condition_prompt = CONDITION_PROMPTS.get(condition_key) if condition_key else None

                if not condition_prompt:
                    print(f"[ERROR /lucid] No prompt found for condition='{condition}'. Loaded conditions: {sorted(CONDITION_PROMPTS.keys())}") # Vercel Log
                    response_data = {
                        'error': 'Configuration Error',
                        'message': f"No prompt configured for condition '{condition}'. Check that the frontend sends a valid 'condition' and that prompts.yaml has a matching entry."
                    }
                    status_code = 400 # Bad Request
                else:
                    # Use the prompts.yaml text as the system message: replace messages[0] if
                    # it's already a system message, otherwise prepend one.
                    if isinstance(messages[0], dict) and messages[0].get('role') == 'system':
                        messages[0] = dict(messages[0], content=condition_prompt.get('initial_prompt', ''))
                    else:
                        messages = [{'role': 'system', 'content': condition_prompt.get('initial_prompt', '')}] + messages
                    print(f"[INFO /lucid] Using prompts.yaml system prompt for condition='{condition_key}'") # Vercel Log

                    # Process temperature (use value from frontend if valid, otherwise default to 1.0)
                    used_temperature = 1.0 # Default temperature
                    if temp_from_frontend is not None:
                        try:
                            parsed_temp = float(temp_from_frontend)
                            if 0.0 <= parsed_temp <= 2.0: used_temperature = parsed_temp
                            else: print(f"[WARN /lucid] Temp '{parsed_temp}' out of range, using default.") # Vercel Log
                        except (ValueError, TypeError): print(f"[WARN /lucid] Invalid temp format ('{temp_from_frontend}'), using default.") # Vercel Log
                    print(f"[INFO /lucid] Using temperature: {used_temperature}") # Vercel Log

                    # Process seed (use value from frontend if valid, otherwise default to None)
                    used_seed = None # Default: OpenAI handles randomness
                    if seed_from_frontend is not None:
                        try: used_seed = int(seed_from_frontend)
                        except (ValueError, TypeError): print(f"[WARN /lucid] Invalid seed format ('{seed_from_frontend}'), using default (None).") # Vercel Log
                    print(f"[INFO /lucid] Using seed: {used_seed}") # Vercel Log

                    # Determine which round this submission is. Used both to tell the model
                    # where it actually is in the negotiation (below) and later to timestamp
                    # the offer-trajectory entry. Prefer what the frontend sends (it tracks
                    # this authoritatively via turnNumber); fall back to counting completed
                    # assistant turns already in the transcript if that's missing.
                    turn_number = body.get('turn_number')
                    if not isinstance(turn_number, int):
                        turn_number = sum(1 for m in messages if m.get('role') == 'assistant') + 1
                    print(f"[INFO /lucid] Current round: {turn_number}") # Vercel Log

                    # Find the candidate's latest message. Needed below for the Prosocial
                    # first-concession check, and reused again later for issue-status extraction.
                    latest_user_message = ''
                    for msg in reversed(messages):
                        if msg.get('role') == 'user':
                            latest_user_message = str(msg.get('content', ''))
                            break

                    # The accumulated package as of last round (frontend echoes this back, same
                    # as issue_statuses elsewhere). Computed here (rather than only later, where
                    # the older code path did) so the pacing-target check in Step 5 can compare
                    # against it too - reused again in the issue-tracking block below.
                    prior_issue_statuses = _normalize_prior_issue_statuses(body.get('issue_statuses'))

                    # This round's required minimum concession level, if any - see
                    # _pacing_target(). Computed once, used both for the round note (Step 4)
                    # and the post-hoc enforcement check (Step 5).
                    pacing_target = _pacing_target(condition_key, turn_number)

                    # --- Prosocial-only: one-time "first concession" exception ---
                    # See prompts.yaml [NEGOTIATION PROTOCOL]: the first time the candidate offers
                    # ANY concession - on any issue, regardless of what they're asking for in
                    # return - Prosocial should reward it with a one-time trust-building gesture.
                    # The TRIGGER (has a concession happened yet) and the GRANT (what gets given
                    # for free) are deliberately decoupled: the trigger fires and consumes the
                    # exception on any detected concession, but the grant itself is never Salary or
                    # Vacation Time (issue-7/issue-3) - if that's what the candidate specifically
                    # asked for, the model picks a different issue to give instead, rather than the
                    # exception just not firing. Whether this has already happened is tracked via a
                    # flag the frontend echoes back each round (prosocial_first_concession_used)
                    # rather than asked of the model, for the same reason turn_number is computed
                    # here instead of self-counted: "has this ever happened before in this
                    # conversation" is exactly the kind of long-range state an LLM can't reliably
                    # track on its own.
                    prosocial_first_concession_used = bool(body.get('prosocial_first_concession_used'))
                    first_concession_note = None
                    # Set only when the grant targets one specific, known issue (the
                    # candidate's own ask was grantable outright) - lets the safety net
                    # below (Step 5) verify compliance against that exact issue instead of
                    # just trusting the note was followed. Stays None for the "pick any
                    # alternate issue" cases, where the model has a free choice.
                    first_concession_target_issue = None
                    if condition_key == 'prosocial' and not prosocial_first_concession_used and latest_user_message:
                        concession_check = _detect_first_concession_llm(latest_user_message, openai_api_key)
                        # Don't just trust the classifier's "is_concession" framing - cross-
                        # check the specific issue/value it says the candidate is giving
                        # ground on against the real payoff table. "Sounds like a concession"
                        # and "is actually favorable to the recruiter" are different questions
                        # (e.g. an EARLIER start date reads like a concession but scores WORSE
                        # for the recruiter on the real table - not a real concession at all).
                        # Only overrides on a confirmed 'worse' reading; 'same'/'unknown' get
                        # the benefit of the doubt since the extraction can legitimately fail
                        # to match (letter/date/city text is looser than plain numbers).
                        if concession_check.get('is_concession'):
                            conceded_issue_id = concession_check.get('conceded_issue_id')
                            conceded_new_value = concession_check.get('conceded_new_value')
                            if conceded_issue_id and conceded_new_value:
                                prior_by_id_cc = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                                conceded_prior_value = prior_by_id_cc.get(conceded_issue_id) or RECRUITER_OPENING_OFFER.get(conceded_issue_id)
                                if _compare_recruiter_value(conceded_issue_id, conceded_new_value, conceded_prior_value) == 'worse':
                                    print(f"[INFO /lucid] First-concession classifier flagged {conceded_issue_id}->{conceded_new_value} as a concession, but the payoff table says it's WORSE for the recruiter - overriding to not-a-concession") # Vercel Log
                                    concession_check['is_concession'] = False
                        if concession_check.get('is_concession'):
                            requested_issue_id = concession_check.get('requested_issue_id')
                            in_hold_firm_window = turn_number <= HOLD_FIRM_ROUNDS
                            prosocial_first_concession_used = True  # consumed either way - see comment above
                            if requested_issue_id and requested_issue_id not in ('issue-3', 'issue-7'):
                                # Requested issue is grantable outright - but capped to ONE
                                # grid step toward the candidate, not a jump straight to
                                # whatever they specifically asked for (see _one_level_step).
                                first_concession_target_issue = requested_issue_id
                                requested_issue_label = next(
                                    (item['label'] for item in _default_issue_statuses() if item['id'] == requested_issue_id),
                                    requested_issue_id
                                )
                                requested_prior_by_id = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                                requested_prior_value = requested_prior_by_id.get(requested_issue_id) or RECRUITER_OPENING_OFFER.get(requested_issue_id)
                                one_level_value = _one_level_step(requested_issue_id, requested_prior_value)
                                if one_level_value:
                                    first_concession_note = (
                                        f"[System note: this is the candidate's first concession this negotiation. "
                                        f"Per your one-time first-concession exception, move {requested_issue_label} "
                                        f"ONE step in the candidate's favor this reply - specifically to "
                                        f"{one_level_value} - unconditionally. Do NOT jump straight to whatever "
                                        f"they specifically asked for, even if it's less generous than their "
                                        f"request - one step only. Do NOT accept whatever concession they offered "
                                        f"in return, even though they offered it - explicitly tell them it isn't "
                                        f"needed, and leave every other issue exactly at its current value this "
                                        f"round.]"
                                    )
                                else:
                                    # Already at the most candidate-favorable grid value for
                                    # this issue - nothing left to give on it specifically.
                                    first_concession_note = (
                                        f"[System note: this is the candidate's first concession this negotiation, "
                                        f"but {requested_issue_label} is already at its most candidate-favorable "
                                        f"value - there's nothing left to move there. As a one-time goodwill "
                                        f"gesture instead, pick ONE of your other issues and move it ONE step in "
                                        f"the candidate's favor, unconditionally, even if they haven't "
                                        f"specifically asked for it.]"
                                    )
                                    first_concession_target_issue = None
                                print(f"[INFO /lucid] Prosocial first-concession exception triggered on {requested_issue_label}") # Vercel Log
                            elif requested_issue_id in ('issue-3', 'issue-7') and in_hold_firm_window:
                                # Genuinely can't grant the literal ask yet (still in the
                                # hold-firm window) - fall back to an alternate-issue gift.
                                # This is the ONLY case where the hold-firm framing below is
                                # actually true.
                                first_concession_note = (
                                    f"[System note: this is the candidate's first concession this negotiation, "
                                    f"but you cannot move on Salary or Vacation Time right now (still in your "
                                    f"hold-firm window). As a one-time goodwill gesture instead, pick ONE of "
                                    f"your other issues (Bonus, Job Assignment, Insurance Coverage, Starting "
                                    f"Date, Moving Expense Coverage, or Location) and move it ONE step in the "
                                    f"candidate's favor, unconditionally, in this reply, even if they haven't "
                                    f"specifically asked for it - explain you can't move on salary/vacation yet "
                                    f"but want to show good faith. Do NOT jump straight to their ideal value on "
                                    f"whatever issue you pick - one step only. Do NOT move Salary or Vacation "
                                    f"Time.]"
                                )
                                print("[INFO /lucid] Prosocial first-concession exception triggered (hold-firm window - granting an alternate issue instead)") # Vercel Log
                            else:
                                # Either the classifier couldn't tell which issue was
                                # requested, or it was Salary/Vacation but the hold-firm
                                # window has already passed - "still in your hold-firm
                                # window" would be a false claim in either case (this was a
                                # bug: the old code used that wording unconditionally here).
                                # Still keep the grant itself off Salary/Vacation via this
                                # one-time exception specifically (that's governed by the
                                # normal pacing schedule instead), but explain why in terms
                                # that are actually true.
                                first_concession_note = (
                                    f"[System note: this is the candidate's first concession this negotiation. "
                                    f"As a one-time goodwill gesture, pick ONE of your other issues (Bonus, Job "
                                    f"Assignment, Insurance Coverage, Starting Date, Moving Expense Coverage, or "
                                    f"Location) and move it ONE step in the candidate's favor, unconditionally, "
                                    f"in this reply, even if they haven't specifically asked for it. Do NOT jump "
                                    f"straight to their ideal value on whatever issue you pick - one step only. "
                                    f"Keep handling Salary and Vacation Time through your normal concession "
                                    f"schedule separately - this one-time gift is on a different issue.]"
                                )
                                print("[INFO /lucid] Prosocial first-concession exception triggered (unclear/out-of-window request - granting an alternate issue instead)") # Vercel Log

                    # --- Step 4: Call OpenAI API ---
                    openai_url = 'https://api.openai.com/v1/chat/completions'
                    headers = {
                        'Content-Type': 'application/json',
                        'Authorization': f'Bearer {openai_api_key}' # Use API key for authorization
                    }
                    # Tell the model what round it's on rather than making it count turns in its
                    # own context (unreliable, especially with reinforcement-prompt system messages
                    # interspersed) - the system prompt's concession-pacing schedule ("hold rounds
                    # 1-3, move by round 6, ...") is only usable if the model has ground truth on
                    # where it is. During the hold-firm window itself, spell the rule out explicitly
                    # here rather than trust it to be recalled correctly from the initial system
                    # prompt alone - same reasoning as injecting the round number in the first place.
                    # Appended fresh each call, right after the latest user message for maximum
                    # salience - NOT persisted back to the frontend's conversation history, so it
                    # never pollutes the stored transcript or duplicates across turns.
                    if turn_number <= HOLD_FIRM_ROUNDS:
                        round_note = (
                            f"[System note: this is round {turn_number} of the negotiation, still within "
                            f"your hold-firm window (rounds 1-{HOLD_FIRM_ROUNDS}). You must NOT move Salary "
                            f"or Vacation Time away from your opening anchor ({HOLD_FIRM_ANCHOR['issue-7']} / "
                            f"{HOLD_FIRM_ANCHOR['issue-3']}) this round, no matter what the candidate offers "
                            f"or asks for - hold firm on those two issues specifically. You may discuss, "
                            f"concede on, or trade any of your other issues freely.]"
                        )
                    else:
                        round_note = f"[System note: this is round {turn_number} of the negotiation. Pace your concessions accordingly, per your instructions.]"
                    # Prosocial-only: [ROUND 1 — INFORMATION EXCHANGE] in the system prompt is
                    # static (sent every call, unlike this note), so nothing in the prompt content
                    # itself tells the model that step is already done once round 1 has passed - it
                    # can just keep re-doing it every round. Make that explicit here instead of
                    # relying on the model to infer it from the round number.
                    if condition_key == 'prosocial' and turn_number > 1:
                        round_note += (
                            " You already completed your round-1 priority-gathering step in an "
                            "earlier message - do not ask the candidate to restate their priorities "
                            "again this round. Move the negotiation forward on the actual package "
                            "instead, unless they bring up something new."
                        )
                    # If this round has a mandatory concession deadline (see _pacing_target), check
                    # the ACCUMULATED package so far (not just what's mentioned this round) and name
                    # explicitly whatever hasn't been reached yet - turns the schedule into a
                    # concrete requirement instead of prose the model has to remember to apply many
                    # turns after reading it.
                    if any(pacing_target.values()):
                        prior_by_id = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                        still_needed = []
                        for issue_id, target in pacing_target.items():
                            if not target:
                                continue
                            current = prior_by_id.get(issue_id) or HOLD_FIRM_ANCHOR[issue_id]
                            if _compare_recruiter_value(issue_id, current, target) == 'better':
                                label = 'Salary' if issue_id == 'issue-7' else 'Vacation Time'
                                still_needed.append(f"{label} to {target}")
                        if still_needed:
                            round_note += (
                                f" Per your concession schedule, by this round you are REQUIRED to "
                                f"have moved {' and '.join(still_needed)} (you have not reached this "
                                f"yet) - do so in this reply, even if the candidate hasn't "
                                f"specifically asked for it."
                            )
                    messages_for_api = messages + [{'role': 'system', 'content': round_note}]
                    if first_concession_note:
                        # Same ephemeral treatment as the round-number note above - fresh each call,
                        # never persisted back into the stored conversation history.
                        messages_for_api.append({'role': 'system', 'content': first_concession_note})
                    # Construct payload for OpenAI
                    data_payload = {
                        'model': model,
                        'messages': messages_for_api,
                        'temperature': used_temperature
                    }
                    # Only include seed if one was provided and valid
                    if used_seed is not None:
                        data_payload['seed'] = used_seed

                    print(f"[INFO /lucid] Calling OpenAI API (model: {model}). Payload keys: {list(data_payload.keys())}") # Vercel Log

                    # Make the POST request to OpenAI with a timeout
                    response_openai = requests.post(openai_url, headers=headers, json=data_payload, timeout=30)
                    openai_status = response_openai.status_code
                    openai_response_text = response_openai.text # Get raw text for potential error logging
                    print(f"[INFO /lucid] OpenAI response status: {openai_status}") # Vercel Log

                    # --- Step 5: Process OpenAI Response ---
                    if openai_status == 200:
                        # Successful call
                        print("[INFO /lucid] Successfully processed OpenAI response.") # Vercel Log
                        try:
                            # Parse the JSON response from OpenAI
                            resp_json = response_openai.json()
                            # Extract the generated text content safely
                            generated_text = resp_json['choices'][0]['message']['content']

                            # --- Hold-firm safety net (rounds 1-HOLD_FIRM_ROUNDS, both conditions) ---
                            # The round_note above asks nicely; this is the enforcement layer. Extract
                            # what the reply actually says about Salary/Vacation Time and, if it moved
                            # either away from the anchor during the hold-firm window, regenerate once
                            # with an explicit correction rather than let a premature concession reach
                            # the participant. assistant_updates is reused below for issue tracking
                            # either way, so this isn't wasted extraction work.
                            assistant_updates = _extract_issue_updates_from_message_llm(generated_text, openai_api_key)
                            if turn_number <= HOLD_FIRM_ROUNDS:
                                violated = [
                                    issue_id for issue_id, anchor in HOLD_FIRM_ANCHOR.items()
                                    if issue_id in assistant_updates and not _matches_anchor(assistant_updates[issue_id], anchor)
                                ]
                                if violated:
                                    violated_labels = ['Salary' if i == 'issue-7' else 'Vacation Time' for i in violated]
                                    print(f"[WARN /lucid] Hold-firm violation on {violated_labels} in round {turn_number}, regenerating") # Vercel Log
                                    correction_note = (
                                        f"[System note: your previous draft reply moved on "
                                        f"{' and '.join(violated_labels)}, which violates your hold-firm window "
                                        f"(rounds 1-{HOLD_FIRM_ROUNDS}). Write your reply again: keep Salary at "
                                        f"{HOLD_FIRM_ANCHOR['issue-7']} and Vacation Time at {HOLD_FIRM_ANCHOR['issue-3']} "
                                        f"unchanged this round. You may still respond to the candidate and move any "
                                        f"other issue.]"
                                    )
                                    retry_text = _call_openai_completion(
                                        messages_for_api + [{'role': 'system', 'content': correction_note}],
                                        model, used_temperature, used_seed, openai_api_key
                                    )
                                    if retry_text:
                                        generated_text = retry_text
                                        assistant_updates = _extract_issue_updates_from_message_llm(generated_text, openai_api_key)
                                        still_violated = [
                                            issue_id for issue_id, anchor in HOLD_FIRM_ANCHOR.items()
                                            if issue_id in assistant_updates and not _matches_anchor(assistant_updates[issue_id], anchor)
                                        ]
                                        if still_violated:
                                            print(f"[WARN /lucid] Hold-firm still violated on {still_violated} after regeneration - keeping it, not retrying again") # Vercel Log
                                    else:
                                        print("[WARN /lucid] Hold-firm regeneration call failed - keeping original (violating) reply") # Vercel Log

                            # --- Pacing-deadline safety net (rounds with a mandatory concession
                            # target - see _pacing_target) ---
                            # The opposite failure mode from the hold-firm check above: instead of
                            # conceding too early, the model just never gets around to conceding at
                            # all, even past its own schedule's deadline. Compare the ACCUMULATED
                            # package (prior rounds + this reply) against the required minimum using
                            # the real payoff table (RECRUITER_PAYOFF_TABLE), not string matching -
                            # "did the value change" isn't the same question as "is this actually a
                            # concession" (e.g. an earlier start date sounds like a candidate
                            # concession but scores WORSE for the recruiter on the real table).
                            elif any(pacing_target.values()):
                                accumulated = apply_issue_updates(prior_issue_statuses, assistant_updates)
                                accumulated_by_id = {item['id']: item.get('status', '') for item in accumulated}
                                under_conceded = [
                                    issue_id for issue_id, target in pacing_target.items()
                                    if target and _compare_recruiter_value(
                                        issue_id, accumulated_by_id.get(issue_id) or HOLD_FIRM_ANCHOR[issue_id], target
                                    ) == 'better'
                                ]
                                if under_conceded:
                                    targets_desc = ', '.join(
                                        f"{'Salary' if i == 'issue-7' else 'Vacation Time'} to {pacing_target[i]}"
                                        for i in under_conceded
                                    )
                                    print(f"[WARN /lucid] Pacing violation - required concession(s) not yet reached in round {turn_number} ({targets_desc}), regenerating") # Vercel Log
                                    correction_note = (
                                        f"[System note: your previous draft reply did not move {targets_desc}, "
                                        f"which your concession schedule requires by this round. Write your reply "
                                        f"again: move {targets_desc} in this reply, even if the candidate hasn't "
                                        f"specifically asked for it. You may still respond to the candidate and "
                                        f"address any other issue.]"
                                    )
                                    retry_text = _call_openai_completion(
                                        messages_for_api + [{'role': 'system', 'content': correction_note}],
                                        model, used_temperature, used_seed, openai_api_key
                                    )
                                    if retry_text:
                                        generated_text = retry_text
                                        assistant_updates = _extract_issue_updates_from_message_llm(generated_text, openai_api_key)
                                        recheck = apply_issue_updates(prior_issue_statuses, assistant_updates)
                                        recheck_by_id = {item['id']: item.get('status', '') for item in recheck}
                                        still_under = [
                                            i for i in under_conceded
                                            if _compare_recruiter_value(i, recheck_by_id.get(i) or HOLD_FIRM_ANCHOR[i], pacing_target[i]) == 'better'
                                        ]
                                        if still_under:
                                            print(f"[WARN /lucid] Pacing still not met on {still_under} after regeneration - keeping it, not retrying again") # Vercel Log
                                    else:
                                        print("[WARN /lucid] Pacing regeneration call failed - keeping original (under-conceded) reply") # Vercel Log

                            # --- First-concession grant safety net (Prosocial only, whenever
                            # the exception fired this round) ---
                            # Independent of the two checks above (this can fire any round,
                            # including inside the hold-firm window, so it isn't chained onto
                            # their if/elif). The note above asks for a specific, ONE-LEVEL
                            # gift; this verifies the reply actually gave it - and gave no
                            # more than that - using the real payoff table rather than
                            # trusting the model followed through.
                            if first_concession_note:
                                grant_status, grant_info = _first_concession_grant_status(
                                    prior_issue_statuses, assistant_updates, first_concession_target_issue
                                )
                                if grant_status != 'ok':
                                    if grant_status == 'overshoot':
                                        overshoot_issue_id, cap_value = grant_info or (None, None)
                                        overshoot_label = next(
                                            (item['label'] for item in _default_issue_statuses() if item['id'] == overshoot_issue_id),
                                            overshoot_issue_id
                                        )
                                        grant_instruction = (
                                            f"your one-time gift moved {overshoot_label} further than the single "
                                            f"grid step this exception allows - scale it back to exactly {cap_value}, "
                                            f"not the candidate's full ask"
                                        )
                                    elif first_concession_target_issue:
                                        grant_label = next(
                                            (item['label'] for item in _default_issue_statuses() if item['id'] == first_concession_target_issue),
                                            first_concession_target_issue
                                        )
                                        grant_instruction = f"grant your one-time, one-step gift on {grant_label}"
                                    else:
                                        grant_instruction = (
                                            "pick ONE of your other issues (Bonus, Job Assignment, Insurance Coverage, "
                                            "Starting Date, Moving Expense Coverage, or Location) and grant your "
                                            "one-time, one-step gift on it"
                                        )
                                    print(f"[WARN /lucid] First-concession grant not honored ({grant_status}), regenerating") # Vercel Log
                                    correction_note = (
                                        f"[System note: your previous draft reply did not correctly grant your "
                                        f"one-time first-concession gift. Write your reply again: {grant_instruction}, "
                                        f"unconditionally, in this reply.]"
                                    )
                                    retry_text = _call_openai_completion(
                                        messages_for_api + [{'role': 'system', 'content': correction_note}],
                                        model, used_temperature, used_seed, openai_api_key
                                    )
                                    if retry_text:
                                        generated_text = retry_text
                                        assistant_updates = _extract_issue_updates_from_message_llm(generated_text, openai_api_key)
                                        recheck_status, _ = _first_concession_grant_status(
                                            prior_issue_statuses, assistant_updates, first_concession_target_issue
                                        )
                                        if recheck_status != 'ok':
                                            print(f"[WARN /lucid] First-concession grant still not honored ({recheck_status}) after regeneration - keeping it, not retrying again") # Vercel Log
                                    else:
                                        print("[WARN /lucid] First-concession regeneration call failed - keeping original reply") # Vercel Log

                            # --- Reciprocity-claim safety net (every round, both conditions) ---
                            # The checks above only cover the one-time first-concession moment
                            # and the Salary/Vacation hold-firm/pacing windows. The same
                            # directional bug (an earlier start date reads like a candidate
                            # concession but scores WORSE for the recruiter on the real payoff
                            # table) can also happen any time the model invokes prompts.yaml's
                            # ordinary reciprocity rule organically - e.g. "Since you offered
                            # flexibility on starting earlier, I can reciprocate by...". Whenever
                            # the reply credits the candidate with a specific concession, verify
                            # that claim against RECRUITER_PAYOFF_TABLE before letting the
                            # reciprocal grant stand.
                            reciprocity_check = _detect_reciprocity_claim_llm(generated_text, openai_api_key)
                            if reciprocity_check.get('claims_reciprocity'):
                                credited_issue_id = reciprocity_check.get('credited_issue_id')
                                credited_value = reciprocity_check.get('credited_value')
                                if credited_issue_id and credited_value:
                                    credited_prior_by_id = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                                    credited_prior_value = credited_prior_by_id.get(credited_issue_id) or RECRUITER_OPENING_OFFER.get(credited_issue_id)
                                    if _compare_recruiter_value(credited_issue_id, credited_value, credited_prior_value) == 'worse':
                                        credited_label = next(
                                            (item['label'] for item in _default_issue_statuses() if item['id'] == credited_issue_id),
                                            credited_issue_id
                                        )
                                        print(f"[WARN /lucid] Reciprocity claim invalid - {credited_issue_id}->{credited_value} is NOT favorable to the recruiter, regenerating") # Vercel Log
                                        correction_note = (
                                            f"[System note: your previous draft reply credited the candidate with a "
                                            f"concession on {credited_label} ({credited_value}) and reciprocated based "
                                            f"on that - but per your payoff schedule, that value is NOT actually "
                                            f"favorable to you compared to your current position on {credited_label}, "
                                            f"so it isn't a real concession. Write your reply again: do not treat that "
                                            f"as a concession or reciprocate based on it. You may still make a move "
                                            f"this reply if it's justified some other way (your own concession "
                                            f"schedule, or a genuine concession the candidate made elsewhere), but "
                                            f"not this one.]"
                                        )
                                        retry_text = _call_openai_completion(
                                            messages_for_api + [{'role': 'system', 'content': correction_note}],
                                            model, used_temperature, used_seed, openai_api_key
                                        )
                                        if retry_text:
                                            generated_text = retry_text
                                            assistant_updates = _extract_issue_updates_from_message_llm(generated_text, openai_api_key)
                                            recheck = _detect_reciprocity_claim_llm(generated_text, openai_api_key)
                                            still_invalid = False
                                            if recheck.get('claims_reciprocity'):
                                                rc_issue = recheck.get('credited_issue_id')
                                                rc_value = recheck.get('credited_value')
                                                if rc_issue and rc_value:
                                                    rc_prior = credited_prior_by_id.get(rc_issue) or RECRUITER_OPENING_OFFER.get(rc_issue)
                                                    still_invalid = _compare_recruiter_value(rc_issue, rc_value, rc_prior) == 'worse'
                                            if still_invalid:
                                                print("[WARN /lucid] Reciprocity claim still invalid after regeneration - keeping it, not retrying again") # Vercel Log
                                        else:
                                            print("[WARN /lucid] Reciprocity-claim regeneration call failed - keeping original reply") # Vercel Log

                            # Prepare the successful response data for Qualtrics frontend
                            response_data = {
                                'generated_text': generated_text,
                                'used_temperature': used_temperature, # Echo back parameters used
                                # Echoed back every round regardless of condition (stays False for
                                # Proself, which never touches this flag) so the frontend can persist
                                # it and send it back next round - see the first-concession block above.
                                'prosocial_first_concession_used': prosocial_first_concession_used
                            }
                            # --- Multi-issue offer tracking (per-round, both speakers) ---
                            # The frontend echoes back the last snapshot it persisted (body['issue_statuses'])
                            # so each round only needs to look at THIS round's two new messages (cheap, constant
                            # cost per turn) instead of re-scanning the whole growing transcript. We extract the
                            # human's proposal and the AI's reply separately so trade-offs from either side are
                            # captured, then diff the merged result against the prior snapshot to see what moved.
                            try:
                                # prior_issue_statuses was already computed earlier (Step 4), so it's
                                # available for both the pacing-target check above and here.
                                user_updates = (
                                    _extract_issue_updates_from_message_llm(latest_user_message, openai_api_key)
                                    if latest_user_message else {}
                                )
                                # assistant_updates was already computed above (post hold-firm check),
                                # against the FINAL generated_text (post-regeneration if that happened).

                                # The displayed/tracked "current offer" snapshot reflects only what the AI
                                # actually said or agreed to - NOT the participant's unilateral ask. A user
                                # proposing "$95,000" shouldn't make $95,000 show up as the current offer on
                                # the panel (or in the trajectory's `issues` state) unless the AI mentioned it
                                # back. user_updates is still extracted and recorded on the trajectory entry
                                # below (so what the participant asked for is never lost for analysis) - it
                                # just doesn't feed the merged/displayed snapshot.
                                merged_statuses = apply_issue_updates(prior_issue_statuses, assistant_updates)

                                # turn_number was already computed above (Step 4) so the round-context
                                # note sent to the model and the trajectory entry logged here agree.
                                response_data['issue_statuses'] = merged_statuses
                                response_data['issue_trajectory_entry'] = build_issue_trajectory_entry(
                                    turn_number, latest_user_message, generated_text,
                                    prior_issue_statuses, user_updates, assistant_updates, merged_statuses
                                )
                            except Exception as e:
                                print(f"[WARN /lucid] Failed to extract issue_statuses/trajectory: {e}")
                                response_data['issue_statuses'] = []
                                response_data['issue_trajectory_entry'] = None
                            if used_seed is not None:
                                response_data['used_seed'] = used_seed # Echo back seed if used

                            status_code = 200 # OK
                        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
                            # Handle cases where OpenAI gives 200 but response format is unexpected
                            print(f"[ERROR /lucid] OpenAI response format unexpected (Status 200): {openai_response_text} - Error: {e}") # Vercel Log
                            response_data = {'error': 'Internal Server Error', 'message': 'Invalid response format from AI service.'}
                            status_code = 500
                    else:
                        # Handle error responses from OpenAI (non-200 status)
                        print(f"[ERROR DIAGNOSTIC /lucid] OpenAI API Error ({openai_status}): {openai_response_text}") # Vercel Log
                        # Try to extract a cleaner error message from OpenAI's response JSON
                        error_details = openai_response_text
                        try:
                           error_json = response_openai.json()
                           if 'error' in error_json and 'message' in error_json['error']:
                               error_details = error_json['error']['message']
                        except json.JSONDecodeError:
                            pass # Use raw text if parsing fails
                        response_data = {'error': f'AI Service Error ({openai_status})', 'message': error_details}
                        # Use OpenAI's status code if it's a standard error, otherwise default to 500
                        status_code = openai_status if openai_status < 600 else 500

    # --- Step 6: Handle Exceptions during Request Processing ---
    except requests.exceptions.Timeout:
        print("[ERROR /lucid] Request to OpenAI timed out.") # Vercel Log
        response_data = {'error': 'Gateway Timeout', 'message': 'Request to AI service timed out.'}
        status_code = 504 # Gateway Timeout
    except requests.exceptions.RequestException as e:
        # Handle network errors connecting to OpenAI
        print(f"[ERROR /lucid] Network error connecting to OpenAI: {e}") # Vercel Log
        response_data = {'error': 'Service Unavailable', 'message': 'Network error connecting to AI service.'}
        status_code = 503 # Service Unavailable
    except json.JSONDecodeError:
        # Handle invalid JSON sent from the frontend
        print(f"[ERROR /lucid] Invalid JSON received from client.") # Vercel Log
        response_data = {'error': 'Bad Request', 'message': 'Invalid JSON format in request body.'}
        status_code = 400 # Bad Request
    except Exception as e:
        # Catch-all for any other unexpected errors
        print(f"[ERROR /lucid] Unexpected server error: {e.__class__.__name__}: {e}") # Vercel Log
        # Consider logging the full traceback here if possible in production
        import traceback
        traceback.print_exc() # Print traceback to logs
        response_data = {'error': 'Internal Server Error', 'message': f'An unexpected error occurred processing the request.'}
        status_code = 500

    # --- Step 7: Create and Return Final Flask Response ---
    final_response = make_response(jsonify(response_data), status_code)

    # Add required CORS headers to the actual response
    final_response.headers['Access-Control-Allow-Origin'] = origin_to_send
    final_response.headers['Vary'] = 'Origin' # Important for caching proxies

    # UPDATED: Only add Access-Control-Allow-Credentials header if it should be 'true'
    if allow_credentials_post: # This boolean reflects the decision made earlier
        final_response.headers['Access-Control-Allow-Credentials'] = 'true'
        print("[DEBUG POST /lucid] Adding Access-Control-Allow-Credentials: true to final response") # Vercel Log
    else:
        print("[DEBUG POST /lucid] Not adding Access-Control-Allow-Credentials header to final response") # Vercel Log


    final_response.headers['Content-Type'] = 'application/json' # Ensure correct content type

    print(f"[INFO /lucid] Responding with status code: {status_code}") # Vercel Log
    return final_response

# --- Main Execution Block (for local development) ---
if __name__ == '__main__':
    # This block only runs when the script is executed directly (e.g., `python lucid_api.py`)
    # It's ignored when run by a WSGI server like Vercel's Python runtime.
    print("[INFO] Starting Flask development server...")

    # Optional: Set environment variables locally for testing
    # os.environ['OPENAI_API_KEY'] = 'YOUR_LOCAL_TEST_KEY_HERE' # Use uppercase for testing
    # os.environ['ALLOWED_ORIGINS'] = '*' # Example: Allow all for local testing
    # os.environ['VERCEL_URL'] = 'localhost:8080' # Example for testing the root page

    # Run the Flask development server
    # Debug mode is controlled via the FLASK_DEBUG environment variable (DO NOT enable in production)
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    local_port = int(os.getenv('PORT', 8080)) # Use PORT env var if set, otherwise default to 8080
    app.run(debug=debug_mode, port=local_port, host='0.0.0.0') # Host 0.0.0.0 makes it accessible on network
