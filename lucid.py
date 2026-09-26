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
import time    # For the short backoff sleep in _post_openai_with_retry
from datetime import datetime, timezone  # For timestamping offer-trajectory entries

# Initialize the Flask application
app = Flask(__name__)

# --- Retry wrapper for OpenAI calls ---

# Status codes OpenAI itself uses for "try again, this wasn't your fault": rate-limited
# (429) or momentarily overloaded/unavailable (500/502/503/504). Confirmed live in
# production: a 503 that arrived AND resolved in under 100ms - nothing to do with our
# own prompt content or timeout budgets, just a transient upstream hiccup that a plain
# retry would have papered over.
_TRANSIENT_OPENAI_STATUSES = (429, 500, 502, 503, 504)


def _post_openai_with_retry(url, headers, payload, timeout, max_retries=2, backoff_seconds=(1, 2)):
    """
    POSTs to the OpenAI API, retrying on failures that are cheap and worth retrying:
    a transient HTTP status (_TRANSIENT_OPENAI_STATUSES above) or a
    requests.exceptions.ConnectionError (network-level failure, also typically fast).
    Up to max_retries retries (3 attempts total by default), with a short sleep
    between attempts.

    Deliberately does NOT retry on requests.exceptions.Timeout - a timeout already
    means the full timeout budget (25s/45s) was spent once; retrying would double
    that single call's worst-case latency and stack badly with the per-round
    safety-net chain (see /lucid Step 5), for a failure mode the generous per-call
    timeouts already exist to absorb. Timeout still propagates to the caller
    unchanged, same as before this wrapper existed.

    Returns the final requests.Response - either a 200, or the last non-transient/
    still-failing response after retries are exhausted. Callers keep handling
    non-200 status codes exactly as they did before; this wrapper only changes
    what happens before a status code reaches them.
    """
    attempt = 0
    while True:
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.exceptions.ConnectionError as e:
            if attempt >= max_retries:
                raise
            wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
            print(f"[WARN] OpenAI call raised {type(e).__name__}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})") # Vercel Log
            time.sleep(wait)
            attempt += 1
            continue

        if resp.status_code not in _TRANSIENT_OPENAI_STATUSES or attempt >= max_retries:
            return resp

        wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
        print(f"[WARN] OpenAI call returned {resp.status_code} (transient), retrying in {wait}s (attempt {attempt + 1}/{max_retries})") # Vercel Log
        time.sleep(wait)
        attempt += 1

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
        'model': 'gpt-5.6',
        'messages': [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': cleaned_message}
        ],
        # gpt-5.6 rejects any 'temperature' other than its default (1), so it's omitted
        # here rather than pinned to 0 - the old gpt-4o-mini extractor used temperature 0
        # specifically for determinism; gpt-5.6 has no equivalent knob, so this call is no
        # longer guaranteed deterministic (measured, empirically more reliable regardless -
        # see the extractor-swap test findings). Also uses 'max_completion_tokens', not
        # 'max_tokens' - gpt-5.6 rejects that key outright. Sized well above typical usage
        # (measured ~100-250 total) because gpt-5.6 spends invisible reasoning_tokens out of
        # this same budget before any visible JSON - too tight a cap can exhaust it on
        # reasoning alone and return empty content (observed live at max_completion_tokens=150
        # on the classifiers below; this call already had headroom, widened further to match).
        'max_completion_tokens': 900,
        'response_format': {'type': 'json_object'}
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    try:
        # 25s (was 12s) - gpt-5.6 spends invisible reasoning_tokens before any visible
        # output, so this call runs slower/more variable than the old gpt-4o-mini one this
        # timeout was originally sized for. This call IS caught locally (returns {} below on
        # any exception), so a timeout here degrades gracefully rather than 500ing - but too
        # tight a timeout still means losing this round's extraction unnecessarily often.
        resp = _post_openai_with_retry('https://api.openai.com/v1/chat/completions', headers, payload, timeout=25)
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


def _detect_first_concession_llm(user_message, openai_api_key, current_offer_statuses=None, prior_assistant_message=None):
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

    current_offer_statuses (optional, the recruiter's current 8-issue package - same shape
    as _default_issue_statuses()) is given to the classifier as grounding context. Without
    it, a candidate message like "I can take a later start date" (agreeing to accept
    whatever's already on the table, without naming a value of their own) gets extracted as
    a vague conceded_new_value like "a later date" - which RECRUITER_PAYOFF_TABLE can't
    match, so the caller's payoff cross-check can neither confirm nor deny it and defaults
    to trusting the classification as unverified. Grounded with the current offer,
    the classifier can instead report the concrete value actually on the table (e.g. "July
    15"), so the caller's 'same' check (see _compare_recruiter_value) can correctly catch
    that this is acquiescence, not a fresh concession.

    prior_assistant_message (optional, the RECRUITER's own immediately preceding reply) is
    given so this same call can also answer a second, distinct question - piggybacked here
    rather than as a separate API call since the candidate's message is already being sent -
    whether the candidate's message is simply ACCEPTING a conditional trade the recruiter
    itself proposed last round (e.g. recruiter said "if you accept X, I'll offer Y", candidate
    now says "ok deal"). Found live in production: without this, "ok deal" doesn't look like a
    new concession to this classifier (nothing is being conceded, just accepted), so the
    general "no free concession" round-note prescription told the model it couldn't move
    anything - even though the recruiter's own prior offer was what was being accepted, not a
    freebie the candidate was fishing for. Distinct from is_concession: a message can accept a
    prior offer without conceding anything new itself.

    accepted_issue_id/accepted_value (only meaningful when accepts_prior_offer is true)
    extract WHAT the recruiter's own preceding message actually promised - the specific
    issue and value on the recruiter's side of that trade (e.g. the recruiter said "I'll
    raise the bonus to 6%", not what the candidate gave up for it) - piggybacked onto this
    same call since prior_assistant_message is already being read for accepts_prior_offer.
    Used by the caller to deterministically verify the promised trade actually landed in
    the reply that follows, rather than trusting the round-note prescription alone (see
    _prior_offer_landed_status) - the same "prescribe AND verify" pattern used everywhere
    else in this file, closing the gap where a later safety net's regeneration could
    otherwise silently drop what was promised with nothing checking it actually shipped.

    accepted_counterpart_issue_id/accepted_counterpart_value (also only meaningful when
    accepts_prior_offer is true) extract the OTHER side of that same trade - what the
    RECRUITER's preceding message asked the CANDIDATE to give up/accept in exchange (e.g.
    if the recruiter said "I can offer July 1 in exchange for Division A", accepted_value
    is 'July 1' (issue-4, what the candidate gets) and accepted_counterpart_value is
    'Division A' (issue-2, what the candidate gives up). Used by the caller to verify,
    when the candidate merely ACCEPTS a trade the recruiter itself proposed rather than
    volunteering a fresh concession, whether they're genuinely paying real value for it -
    found live: a candidate accepting "Division A for July 1" pays a real, payoff-table-
    verifiable cost, but is_concession alone never catches this (the classifier correctly
    reads pure acceptance as not a fresh concession), so without this the Prosocial
    one-time first-concession gift could never fire in a negotiation where the recruiter
    always proposes the specific trade and the candidate only ever confirms it.

    Returns {'is_concession': False, 'requested_issue_id': None, 'conceded_issue_id': None,
    'conceded_new_value': None, 'accepts_prior_offer': False, 'accepted_issue_id': None,
    'accepted_value': None, 'accepted_counterpart_issue_id': None,
    'accepted_counterpart_value': None} on any failure, or if no message/key was given, so
    this never blocks the main call.
    """
    empty_result = {
        'is_concession': False, 'requested_issue_id': None,
        'conceded_issue_id': None, 'conceded_new_value': None,
        'accepts_prior_offer': False, 'accepted_issue_id': None, 'accepted_value': None,
        'accepted_counterpart_issue_id': None, 'accepted_counterpart_value': None
    }
    if not user_message or not openai_api_key:
        return dict(empty_result)

    defaults = _default_issue_statuses()
    valid_issue_ids = {item['id'] for item in defaults}

    current_offer_note = ""
    if current_offer_statuses:
        lines = [
            f"{item['label']} ({item['id']}): {item['status']}"
            for item in current_offer_statuses if item.get('status')
        ]
        if lines:
            current_offer_note = (
                "\nThe recruiter's CURRENT offer on file (for grounding only) is:\n"
                + "\n".join(lines)
                + "\nIf the candidate does not name a specific new value of their own on an "
                "issue but is simply agreeing to accept what's already listed above for that "
                "issue, set conceded_new_value to that SAME listed value (not a vague "
                "description like 'a later date' or 'more flexible') - accepting the status "
                "quo is not a fresh concession, and the exact value is needed to verify that."
            )

    prior_offer_note = ""
    if prior_assistant_message:
        prior_offer_note = (
            "\nThe RECRUITER's own immediately preceding message (for judging "
            "accepts_prior_offer only) was:\n" + str(prior_assistant_message)
        )

    system_prompt = (
        "You analyze one message from a job candidate in a negotiation. Determine TWO "
        "separate things. "
        "(1) is_concession: is the candidate offering a TRADE - willing to give ground / "
        "accept less on one issue, specifically in order to ask for movement on a DIFFERENT "
        "issue? This must be an explicit or clearly implied concession paired with a "
        "request, not just a one-sided ask with no give. "
        "(2) accepts_prior_offer: is the candidate simply ACCEPTING a conditional trade the "
        "RECRUITER itself proposed in its preceding message shown below (e.g. the recruiter "
        "said something like 'if you accept X, I'll offer Y', and the candidate's message "
        "here is an agreement like 'ok deal', 'that works', 'I accept', 'sounds good')? "
        "This is true ONLY if the recruiter's preceding message actually named a specific "
        "conditional trade AND the candidate's message here clearly agrees to it - not for a "
        "generic pleasant reply with no specific trade to accept. A message can be true for "
        "accepts_prior_offer while is_concession is false (accepting isn't conceding "
        "something new) - they are independent, check both. "
        "Issue ids and labels are: issue-1 Bonus, issue-2 Job Assignment, issue-3 Vacation "
        "Time, issue-4 Starting Date, issue-5 Moving Expense Coverage, issue-6 Insurance "
        "Coverage, issue-7 Salary, issue-8 Location. "
        "Return ONLY valid JSON in this exact shape: "
        "{\"is_concession\": true, \"requested_issue_id\": \"issue-6\", "
        "\"conceded_issue_id\": \"issue-4\", \"conceded_new_value\": \"July 1\", "
        "\"accepts_prior_offer\": false, \"accepted_issue_id\": null, "
        "\"accepted_value\": null, \"accepted_counterpart_issue_id\": null, "
        "\"accepted_counterpart_value\": null}. "
        "requested_issue_id is the issue the candidate is asking the RECRUITER to move on "
        "or improve - use null if is_concession is false or the requested issue is unclear. "
        "conceded_issue_id/conceded_new_value describe what the candidate is giving ground "
        "on: the issue, and the specific concrete new value they're now offering on it (e.g. "
        "'July 1', 'Division C', '80%') - not a description. Use null for both if "
        "is_concession is false or this is unclear. "
        "accepted_issue_id/accepted_value describe the SPECIFIC issue and value the RECRUITER "
        "itself promised in its preceding message that the candidate is now accepting (e.g. "
        "if the recruiter's preceding message was 'if you accept X, I'll raise the bonus to "
        "6%', and the candidate accepts, accepted_issue_id is issue-1 and accepted_value is "
        "'6%' - the thing the RECRUITER offered to give, not what the candidate gave up for "
        "it). Use null for both if accepts_prior_offer is false, or if the recruiter's "
        "preceding message didn't name one specific concrete value for its own side of the "
        "trade. "
        "accepted_counterpart_issue_id/accepted_counterpart_value describe the OTHER side of "
        "that SAME trade - the specific issue and value the recruiter's preceding message "
        "asked the CANDIDATE to give up or accept in return (e.g. if the recruiter's "
        "preceding message was 'I can offer a July 1 start if you accept Division A instead "
        "of Division B', accepted_issue_id/accepted_value is issue-4/'July 1' (what the "
        "candidate gets) and accepted_counterpart_issue_id/accepted_counterpart_value is "
        "issue-2/'Division A' (what the candidate gives up) - the candidate's own accepting "
        "message doesn't need to repeat this value itself, it's read from the recruiter's "
        "preceding message, same as accepted_issue_id/accepted_value). Use null for both if "
        "accepts_prior_offer is false, or if the recruiter's preceding message was a one-"
        "sided offer with nothing named on the candidate's side."
        + current_offer_note
        + prior_offer_note
    )

    payload = {
        'model': 'gpt-5.6',  # single lightweight classification call
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': str(user_message)}
        ],
        # gpt-5.6 rejects any 'temperature' but its default (1) - no equivalent to the old
        # gpt-4o-mini call's temperature=0 determinism guarantee. 'max_completion_tokens',
        # not 'max_tokens' - gpt-5.6 rejects that key outright. 150 was too tight: gpt-5.6
        # spends invisible reasoning_tokens out of this same budget before any visible JSON,
        # and 150 was observed live exhausting entirely on reasoning, returning empty content.
        'max_completion_tokens': 450,
        'response_format': {'type': 'json_object'}
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    try:
        # 25s (was 10s) - gpt-5.6 spends invisible reasoning_tokens before any visible
        # output; this call IS caught locally (degrades gracefully to the empty default on
        # any exception), but too tight a timeout means losing this classification more
        # often than necessary.
        resp = _post_openai_with_retry('https://api.openai.com/v1/chat/completions', headers, payload, timeout=25)
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
        accepts_prior_offer = bool(parsed.get('accepts_prior_offer'))
        accepted_issue_id = parsed.get('accepted_issue_id')
        if accepted_issue_id not in valid_issue_ids:
            accepted_issue_id = None
        accepted_value = parsed.get('accepted_value')
        accepted_value = str(accepted_value).strip() if accepted_value else None
        accepted_counterpart_issue_id = parsed.get('accepted_counterpart_issue_id')
        if accepted_counterpart_issue_id not in valid_issue_ids:
            accepted_counterpart_issue_id = None
        accepted_counterpart_value = parsed.get('accepted_counterpart_value')
        accepted_counterpart_value = str(accepted_counterpart_value).strip() if accepted_counterpart_value else None
        return {
            'is_concession': is_concession,
            'requested_issue_id': requested_issue_id,
            'conceded_issue_id': conceded_issue_id,
            'conceded_new_value': conceded_new_value,
            'accepts_prior_offer': accepts_prior_offer,
            'accepted_issue_id': accepted_issue_id,
            'accepted_value': accepted_value,
            'accepted_counterpart_issue_id': accepted_counterpart_issue_id,
            'accepted_counterpart_value': accepted_counterpart_value
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
        "recruiter justifies a concession/grant by crediting the CANDIDATE with having ALREADY "
        "given something up or offered flexibility on a specific issue THIS ROUND (e.g. 'since "
        "you offered flexibility on starting earlier, I can...', 'to reciprocate your "
        "willingness to...', 'thank you for coming down to X, so I will...'). This must "
        "describe the candidate's move as something that has ALREADY happened in the "
        "conversation - an explicit or clearly implied claim about the past, not a "
        "hypothetical. "
        "Do NOT mark this true for a conditional or hypothetical offer where the recruiter "
        "proposes what THEY would do IF the candidate gives something in the future or in "
        "return (e.g. 'I am willing to move to 15 days if we can find a way to balance this "
        "elsewhere', 'I could offer X in exchange for Y', 'would you consider Y so that I can "
        "offer X'). That is the recruiter proposing their OWN future move, not a claim that "
        "the candidate already conceded anything - mark claims_reciprocity false for these, "
        "even if a specific value is mentioned. "
        "CRITICAL: when a sentence has the shape 'since/because you did/accepted A, I will do "
        "B', credited_issue_id/credited_value describe A - what the CANDIDATE is being said to "
        "have already given up or accepted - NEVER B, which is the recruiter's OWN reciprocal "
        "action being granted TO the candidate, not something credited FROM the candidate. For "
        "example, in 'Since you accepted 60% moving expense coverage, I'll raise the bonus to "
        "6%', the candidate is credited with the 60% moving coverage (issue-5) - the 6% bonus "
        "(issue-1) is the recruiter's own grant and must NEVER be used as credited_issue_id/"
        "credited_value, no matter how confidently or specifically it's stated. "
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
        'model': 'gpt-5.6',  # single lightweight classification call
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': str(assistant_message)}
        ],
        # gpt-5.6 rejects any 'temperature' but its default (1) - no equivalent to the old
        # gpt-4o-mini call's temperature=0 determinism guarantee. 'max_completion_tokens',
        # not 'max_tokens' - gpt-5.6 rejects that key outright. 150 was too tight: gpt-5.6
        # spends invisible reasoning_tokens out of this same budget before any visible JSON,
        # and 150 was observed live exhausting entirely on reasoning, returning empty content.
        'max_completion_tokens': 450,
        'response_format': {'type': 'json_object'}
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    try:
        # 25s (was 10s) - gpt-5.6 spends invisible reasoning_tokens before any visible
        # output; this call IS caught locally (degrades gracefully to the empty default on
        # any exception), but too tight a timeout means losing this classification more
        # often than necessary.
        resp = _post_openai_with_retry('https://api.openai.com/v1/chat/completions', headers, payload, timeout=25)
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


def _detect_unconfirmed_recap_values_llm(assistant_message, openai_api_key):
    """
    Analyzes the RECRUITER's own reply for issues where the "Current package" recap shows a
    NEW value that the reply's own PROSE (the text before the recap) frames as a
    conditional or hypothetical PROPOSAL - something the recruiter says it COULD do, or is
    asking the candidate whether they'd accept - rather than something already confirmed,
    applied, or agreed to this round.

    Found live: the model's own recap doesn't reliably distinguish "I've applied X" from "I
    could offer X if you accept Y - would that work for you?" - a reply can ask the
    candidate to confirm a brand-new trade in its prose while the SAME reply's recap table
    already shows that trade as if it were in effect. Observed for both directions in the
    same live reply: a gain to the candidate (salary bumped up) and a loss (bonus cut down),
    neither one actually agreed to yet.

    Every other safety net in this file verifies WHETHER a move is grounded (a real
    concession, an already-promised trade, a confirmed gift) - this is the only one that
    asks a different question: is the recap even claiming something has HAPPENED that the
    reply's own words say hasn't happened yet? That can't be answered by the payoff table
    (a hypothetical $86,000 and a confirmed $86,000 are the identical value), so unlike most
    checks in this file this one has to trust an LLM's read of the prose - kept narrow
    (flag only what's clearly conditional, default to NOT flagging when unsure) since a
    false positive here reverts a value the recruiter actually meant to confirm.

    Returns {'unconfirmed_issue_ids': []} on any failure, or if no message/key given, so
    this never blocks the main call.
    """
    empty_result = {'unconfirmed_issue_ids': []}
    if not assistant_message or not openai_api_key:
        return dict(empty_result)

    defaults = _default_issue_statuses()
    valid_issue_ids = {item['id'] for item in defaults}

    system_prompt = (
        "You analyze one message from a job RECRUITER in a negotiation. The message ends "
        "with a \"Current package:\" recap listing a value for each issue. Determine which "
        "issues, if any, have a value in that recap that the message's OWN PROSE (the text "
        "BEFORE the recap) frames as a conditional or hypothetical PROPOSAL - something the "
        "recruiter says it COULD do, or is asking the candidate whether they would accept - "
        "NOT something already confirmed, applied, or agreed to this round. "
        "Signs of a conditional/hypothetical proposal (SHOULD be flagged): 'I could offer X "
        "if you accept Y', 'would that work for you?', 'let me know if...', 'please "
        "confirm', 'does that work', or any sentence ending in a question about whether the "
        "candidate accepts a NEW trade. "
        "Signs something IS confirmed (do NOT flag it): 'Agreed', 'I've applied', a plain "
        "declarative statement with no question or conditional 'if' attached describing this "
        "issue specifically, or the candidate's own prior message already having accepted "
        "it. Also do NOT flag an issue whose recap value is unchanged from what it already "
        "was - only a NEWLY proposed value can be unconfirmed. "
        "Issue ids and labels are: issue-1 Bonus, issue-2 Job Assignment, issue-3 Vacation "
        "Time, issue-4 Starting Date, issue-5 Moving Expense Coverage, issue-6 Insurance "
        "Coverage, issue-7 Salary, issue-8 Location. "
        "Return ONLY valid JSON in this exact shape: "
        "{\"unconfirmed_issue_ids\": [\"issue-7\", \"issue-1\"]}. Empty list if every value "
        "in the recap is confirmed/applied, or if you are unsure - only include an issue "
        "when the prose clearly frames it as conditional/pending, not already agreed."
    )

    payload = {
        'model': 'gpt-5.6',  # single lightweight classification call
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': str(assistant_message)}
        ],
        # Same gpt-5.6 accommodations as the other classifiers in this file (no temperature
        # override, max_completion_tokens not max_tokens, sized to survive invisible
        # reasoning_tokens before any visible JSON).
        'max_completion_tokens': 450,
        'response_format': {'type': 'json_object'}
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {openai_api_key}'
    }

    try:
        resp = _post_openai_with_retry('https://api.openai.com/v1/chat/completions', headers, payload, timeout=25)
        if resp.status_code != 200:
            print(f"[INFO] Unconfirmed-recap-value detection returned {resp.status_code}, skipping")
            return dict(empty_result)

        raw = resp.json()['choices'][0]['message']['content']
        parsed = json.loads(raw)
        ids = parsed.get('unconfirmed_issue_ids')
        if not isinstance(ids, list):
            ids = []
        ids = [str(i) for i in ids if str(i) in valid_issue_ids]
        return {'unconfirmed_issue_ids': ids}

    except Exception as e:
        print(f"[INFO] Unconfirmed-recap-value detection exception: {e}")
        return dict(empty_result)


# The negotiation's total round count - matches the qsf's LUCIDRoundLimit embedded-data
# value. lucid.py has no way to read that value itself (it isn't sent in the request body),
# so this is kept in sync by hand; used only to prescribe final-round behavior (see the
# round_note construction in /lucid) - if the qsf's round limit ever changes, update this too.
TOTAL_ROUNDS = 10

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
        upper_text = text.upper()
        # Require the label word itself (Division/Plan) right before the letter - a bare
        # \b([A-E])\b would also match an ordinary English word that happens to be a single
        # A-E letter, most commonly the article "a" (e.g. "a nicer plan" would otherwise
        # false-positive-match "Plan A"). The extraction prompt always preserves this format
        # (e.g. "Division A", "Plan E"), so requiring it doesn't lose real matches.
        m = re.search(r'\b(?:DIVISION|PLAN)\s+([A-E])\b', upper_text)
        if not m:
            # Fall back to a bare letter only when the ENTIRE extracted value reduces to
            # just that letter (e.g. extraction returned "C" on its own, no prefix) - safe
            # because it then can't coincidentally be an ordinary word in a longer sentence.
            stripped = upper_text.strip()
            if stripped in ('A', 'B', 'C', 'D', 'E'):
                m = re.match(r'([A-E])$', stripped)
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


# Salary (issue-7) and Vacation Time (issue-3) are never in this list - those are governed
# by the normal hold-firm/pacing schedule instead, never this one-time exception.
_ALTERNATE_GIFT_ISSUE_LABELS = {
    'issue-1': 'Bonus', 'issue-2': 'Job Assignment', 'issue-4': 'Starting Date',
    'issue-5': 'Moving Expense Coverage', 'issue-6': 'Insurance Coverage', 'issue-8': 'Location',
}


def _normalize_exclude_ids(exclude_issue_ids):
    """Accepts None, a single issue id string, or an iterable of issue ids - always returns
    a set, so callers of _alternate_gift_issue_list_text / _first_concession_grant_status
    can pass either shape without special-casing. Needed once the alternate gift had to
    exclude TWO issues at once (both the conceded and the requested side of a package
    trade), not just the single accepted_issue_id case this was originally written for."""
    if not exclude_issue_ids:
        return set()
    if isinstance(exclude_issue_ids, str):
        return {exclude_issue_ids}
    return {i for i in exclude_issue_ids if i}


def _alternate_gift_issue_list_text(exclude_issue_ids=None):
    """
    Builds the human-readable list of issues offered in Prosocial's "pick ONE of your other
    issues" first-concession fallback prompt text, optionally excluding exclude_issue_ids
    (a single issue id or an iterable of them).

    Exists so this text stays in sync with _first_concession_grant_status's own candidate
    pool (see its exclude_issue_ids param) - found live: when the exception fires via a
    candidate accepting a trade the recruiter itself already promised (accepted_issue_id is
    set), that issue is excluded from the valid-grant scan since it's already being
    fulfilled this round, not an eligible NEW gift - but the prompt text used to always list
    all six issues regardless, including the one just excluded (e.g. "Starting Date" still
    offered as a choice even though accepted_issue_id is Starting Date). If the model had
    ever taken that offered choice, the verification would reject it as ungranted and force
    an unnecessary regeneration - the instruction and the check need to agree on the same
    pool. Extended to accept multiple exclusions once a package trade (conceded issue AND
    requested issue) both needed to be kept out of the same alternate-gift pool.
    """
    excluded = _normalize_exclude_ids(exclude_issue_ids)
    eligible = [label for iid, label in _ALTERNATE_GIFT_ISSUE_LABELS.items() if iid not in excluded]
    if not eligible:
        return ''
    if len(eligible) == 1:
        return eligible[0]
    return ', '.join(eligible[:-1]) + ', or ' + eligible[-1]


def _first_concession_grant_status(prior_statuses, assistant_updates, target_issue_id, exclude_issue_ids=None):
    """
    Evaluates a first-concession grant note (see /lucid Step 3) against the one-level cap
    (see _one_level_step - the grant is unconditional but capped to a single grid step, never
    jumped straight to the candidate's full ask). If target_issue_id is given, only that issue
    counts (the model was told exactly which one to grant); otherwise any of the six non-
    Salary/Vacation issues counts (the model had a free choice among them) - EXCEPT
    exclude_issue_ids, if given (a single issue id or an iterable of them): issue(s) already
    accounted for elsewhere this round - round_concession_check's accepted_issue_id (when the
    trigger came from accepting the recruiter's own prior trade), or the conceded/requested
    pair of a package trade (see the first-concession exception's Case 1/2 logic in /lucid
    Step 3) - rather than something extra. Found live: without this, a candidate accepting
    "Division A for July 1" got the July 1 date it was ALREADY promised double-credited as
    the one-time gift too, since it's the only issue that moved in the candidate's favor
    this round in the generic "any of the six alternates" scan - the gift needs to be
    something genuinely additional, not the trade that was already happening.

    Returns a 3-tuple (status, info, unverified_note):
      ('ok', (issue_id, value), None)       - some eligible issue was CONFIRMED to move exactly
                                               one level in the candidate's favor; info names
                                               exactly what was granted (used by the caller to
                                               build a deterministic grant announcement)
      ('ok', (issue_id, value), "<reason>") - allowed to stand, but that direction couldn't be
                                               verified either way (letter/date/city matching can
                                               legitimately fail to resolve); info still names the
                                               issue/value that was present, so a grant
                                               announcement can still be built - unverified_note
                                               is a short human-readable reason, surfaced
                                               separately by the caller rather than silently
                                               treated the same as a confirmed match
      ('overshoot', (issue_id, cap), None)  - an eligible issue moved, but past its one-level cap
      ('missing', None, None)               - nothing eligible moved at all
    """
    excluded = _normalize_exclude_ids(exclude_issue_ids)
    prior_by_id = {item['id']: item.get('status', '') for item in prior_statuses}
    candidate_ids = [target_issue_id] if target_issue_id else [
        item['id'] for item in _default_issue_statuses()
        if item['id'] not in ('issue-3', 'issue-7') and item['id'] not in excluded
    ]
    unknown_match = None
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
            return ('ok', (issue_id, assistant_updates[issue_id]), None)
        if direction == 'unknown':
            unknown_match = (issue_id, assistant_updates[issue_id])
    if unknown_match:
        unverified_note = (
            f"First-concession grant check (target={target_issue_id or 'any of the 6 alternates'}): "
            f"could not verify {unknown_match[0]} -> '{unknown_match[1]}' against the payoff table "
            f"(unknown) - allowed the grant to stand without confirming the one-level cap was honored."
        )
        return ('ok', unknown_match, unverified_note)
    if overshoot:
        return ('overshoot', overshoot, None)
    return ('missing', None, None)


def _first_concession_grant_status_with_fallback(prior_statuses, assistant_updates, target_issue_id, raw_text, exclude_issue_ids=None):
    """
    Wraps _first_concession_grant_status with one deterministic recovery attempt: if the
    result is 'missing' specifically because target_issue_id is absent from assistant_updates
    entirely, fall back to a plain regex match against raw_text's "Current package:" recap
    line for that issue's label - no extra LLM call. Addresses a real, reproduced
    gpt-4o-mini extraction failure mode (found in testing): it can silently drop a key from
    its JSON output even when the value is stated plainly and unambiguously in the recap
    (e.g. "Insurance Coverage: Plan C" present verbatim, nothing else said about it) - unlike
    the separate "extraction sides with prose over recap" pattern documented elsewhere, this
    isn't a disagreement to resolve, the key is just missing. Since we already know exactly
    which single issue to look for (the model was told exactly which one to grant), a plain
    line match is enough - no need to trust another LLM call to get it right either.

    exclude_issue_ids is passed straight through to _first_concession_grant_status (see
    there) - only relevant when target_issue_id is None.

    Returns (status, info, unverified_note, assistant_updates) - assistant_updates comes
    back unchanged unless the fallback recovers a value AND that recovery confirms status
    'ok', in which case a patched copy is returned so the recovered value also feeds forward
    into this round's issue-status tracking, not just this one check.
    """
    status, info, note = _first_concession_grant_status(prior_statuses, assistant_updates, target_issue_id, exclude_issue_ids)
    if status != 'missing' or not target_issue_id or target_issue_id in assistant_updates:
        return status, info, note, assistant_updates
    label = next((item['label'] for item in _default_issue_statuses() if item['id'] == target_issue_id), None)
    if not label:
        return status, info, note, assistant_updates
    cleaned = str(raw_text or '').replace('**', '')
    matches = re.findall(rf'{re.escape(label)}\s*:\s*([^\n]+)', cleaned, flags=re.IGNORECASE)
    if not matches:
        return status, info, note, assistant_updates
    patched_updates = dict(assistant_updates)
    patched_updates[target_issue_id] = matches[-1].strip()
    patched_status, patched_info, patched_note = _first_concession_grant_status(prior_statuses, patched_updates, target_issue_id, exclude_issue_ids)
    if patched_status == 'ok':
        print(f"[INFO /lucid] Extraction dropped {target_issue_id} entirely - recovered via recap regex fallback: {patched_updates[target_issue_id]!r}") # Vercel Log
        return patched_status, patched_info, patched_note, patched_updates
    return status, info, note, assistant_updates


def _package_trade_status(prior_statuses, assistant_updates, conceded_issue_id, conceded_value, requested_issue_id, request_blocked):
    """
    Verifies the all-or-nothing "package trade" side of the first-concession exception
    (see /lucid Step 3) actually landed as prescribed, deterministically - the same
    "prescribe AND verify" pattern used everywhere else in this file, since the round_note
    prescription alone can go unfollowed or get silently undone by a later regeneration.

    Case 1 (request_blocked=False, i.e. the requested side was grantable this round):
    the conceded issue must land at conceded_value or better for the RECRUITER (the
    recruiter is entitled to at least what was offered - over-taking is fine, under-taking
    isn't), AND the requested issue must move exactly one grid step toward the candidate
    (reuses _first_concession_grant_status's own one-level-cap logic, targeted
    specifically at requested_issue_id).

    Case 2 (request_blocked=True): the WHOLE trade was declined, so the conceded issue
    must NOT move from its prior value either - applying only the candidate's offered
    concession while declining what they asked for in return would be a one-sided freebie,
    not what was prescribed.

    Returns {'conceded_wrong': (issue_id, label, expected_value) | None,
    'requested_missing': (issue_id, label, one_level_target) | None} - both None means no
    violation found.
    """
    violations = {'conceded_wrong': None, 'requested_missing': None}
    if not conceded_issue_id or not conceded_value:
        return violations
    prior_by_id = {item['id']: item.get('status', '') for item in prior_statuses}
    conceded_prior_value = prior_by_id.get(conceded_issue_id) or RECRUITER_OPENING_OFFER.get(conceded_issue_id)
    conceded_label = next((item['label'] for item in _default_issue_statuses() if item['id'] == conceded_issue_id), conceded_issue_id)
    conceded_actual = assistant_updates.get(conceded_issue_id) or conceded_prior_value

    if request_blocked:
        direction = _compare_recruiter_value(conceded_issue_id, conceded_actual, conceded_prior_value)
        if direction not in ('same', 'unknown'):
            # it moved (in either direction) when the whole trade should have been declined
            violations['conceded_wrong'] = (conceded_issue_id, conceded_label, conceded_prior_value)
        return violations

    # Case 1: conceded side should land at conceded_value or better for the RECRUITER (the
    # recruiter is entitled to at least what the candidate offered to give up - over-taking
    # is fine, under-taking is the violation).
    direction = _compare_recruiter_value(conceded_issue_id, conceded_actual, conceded_value)
    if direction == 'worse':  # worse for the recruiter than what was offered = under-delivered
        violations['conceded_wrong'] = (conceded_issue_id, conceded_label, conceded_value)

    # Requested side should move exactly one grid step - reuse the existing grant-status
    # checker with a specific target, same overshoot/missing handling it already has.
    if requested_issue_id:
        req_status, req_info, _ = _first_concession_grant_status(prior_statuses, assistant_updates, requested_issue_id)
        if req_status != 'ok':
            requested_label = next((item['label'] for item in _default_issue_statuses() if item['id'] == requested_issue_id), requested_issue_id)
            requested_prior_value = prior_by_id.get(requested_issue_id) or RECRUITER_OPENING_OFFER.get(requested_issue_id)
            one_level_value = _one_level_step(requested_issue_id, requested_prior_value)
            if one_level_value:
                violations['requested_missing'] = (requested_issue_id, requested_label, one_level_value)
    return violations


def _prior_offer_landed_status(assistant_updates, accepted_issue_id, accepted_value, raw_text=None):
    """
    Verifies that a specific value the recruiter promised in its own prior conditional offer
    - which the candidate has now accepted, per _detect_first_concession_llm's
    accepted_issue_id/accepted_value - actually landed in THIS round's reply, rather than
    trusting the round-note prescription alone. Closes a real gap found in testing: the
    prescription can tell the model to apply the trade, but a LATER safety net's own
    regeneration (sampling fresh from stale context, not editing the current draft) can
    silently drop it while fixing something unrelated - nothing else re-checks that the
    specific promised value actually shipped, only that nothing moved for free.

    Same regex-fallback trick as _first_concession_grant_status_with_fallback: if
    assistant_updates is missing the issue entirely, fall back to a plain line match against
    raw_text's recap before giving up - no extra LLM call.

    Returns (status, actual_value):
      ('na', None)      - accepted_issue_id/accepted_value weren't given (nothing to verify)
      ('ok', value)      - the issue landed at exactly the promised value, or something even
                           more favorable to the candidate (over-delivering isn't a violation)
      ('under', value)   - landed, but at a value WORSE for the candidate than promised
      ('missing', None)  - the issue never showed up in assistant_updates or the recap at all
      ('unknown', value) - landed at some value, but direction against the promise couldn't
                           be verified either way (matching this file's usual "unknown ->
                           don't force a regen on ambiguity" handling elsewhere)
    """
    if not accepted_issue_id or not accepted_value:
        return 'na', None
    actual_value = assistant_updates.get(accepted_issue_id)
    if not actual_value and raw_text:
        label = next((item['label'] for item in _default_issue_statuses() if item['id'] == accepted_issue_id), None)
        if label:
            cleaned = str(raw_text).replace('**', '')
            matches = re.findall(rf'{re.escape(label)}\s*:\s*([^\n]+)', cleaned, flags=re.IGNORECASE)
            if matches:
                actual_value = matches[-1].strip()
    if not actual_value:
        return 'missing', None
    direction = _compare_recruiter_value(accepted_issue_id, actual_value, accepted_value)
    if direction in ('same', 'worse'):  # matches the promise, or over-delivers - both fine
        return 'ok', actual_value
    if direction == 'unknown':
        return 'unknown', actual_value
    return 'under', actual_value  # 'better' for the recruiter than promised = under-delivered


def _value_mentioned_in_prose(raw_text, value):
    """
    Whether a specific value (e.g. "6%", "Plan C", "$86,000") is mentioned in the reply's
    PROSE - the part before the "Current package" recap block - rather than only in the
    recap table itself. Cheap, deterministic, no extra LLM call: strips the recap off,
    normalizes away $/commas/case on both sides, and does a plain substring search.

    Used to narrow the final audit's "genuine concession this round" / "accepts prior
    offer" exemptions (see Rule 3 below): either flag being true only proves ONE specific
    issue actually got conceded/reciprocated/promised - a real trade found live where Alex
    correctly traded moving-expense-for-salary, and the recap silently ALSO bumped Bonus
    from 4% to 6% with zero mention anywhere in the text ("6%" doesn't appear in the
    prose at all). The recruiter is allowed to reciprocate on a different low-priority
    issue instead of the one requested/accepted (prompts.yaml says so explicitly, and
    other tests rely on a narrated-but-different-issue move like "I've also improved your
    insurance to Plan C" surviving untouched) - checking the VALUE rather than the issue's
    formal label matches how people actually talk ("insurance" or "the vacation days",
    not "Insurance Coverage"/"Vacation Time"), while still catching a value that's truly
    never mentioned anywhere.
    """
    if not raw_text or not value:
        return False
    cleaned = str(raw_text).replace('**', '')
    prose = re.split(r'current\s+package\s*:', cleaned, maxsplit=1, flags=re.IGNORECASE)[0]
    norm_prose = re.sub(r'[,$]', '', prose).lower()
    norm_value = re.sub(r'[,$]', '', str(value)).strip().lower()
    return bool(norm_value) and norm_value in norm_prose


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

# Prosocial's schedule describes a SECOND, conditional step beyond PACING_STEP: "...then
# toward $88,000 by round 8-9 if the candidate is engaging constructively... and to 20 days
# if it's needed in round 8-10" - this is prosocial's own hard limit (see prompts.yaml
# CRITICAL RULES 4), offered as available rather than required, and only when earned. Proself
# has no equivalent: its hard limit ($86,000/15 days) already equals PACING_STEP, so there's
# nothing further to stretch toward - "do not go further even under pressure" is explicit in
# its own schedule. Keep in sync with prompts.yaml if either schedule changes.
PROSOCIAL_STRETCH_ROUND = 8
PROSOCIAL_STRETCH_STEP = {'issue-7': '$88,000', 'issue-3': '20 days'}


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


def _pacing_stretch_target(condition_key, turn_number):
    """
    Returns {'issue-7': target_or_None, 'issue-3': target_or_None} - prosocial's OPTIONAL
    deeper step (PROSOCIAL_STRETCH_STEP), available from PROSOCIAL_STRETCH_ROUND onward.
    Unlike _pacing_target, this is never mandatory - the caller only offers it to the model
    when genuine_concession_this_round is also true this round (the deterministic stand-in
    for "engaging constructively": avoids trusting the model's own subjective read, same
    reasoning as everywhere else this session preferred a payoff-table check over a
    self-judged one). Always None for proself - see the comment on PROSOCIAL_STRETCH_STEP.
    """
    if condition_key == 'prosocial' and turn_number >= PROSOCIAL_STRETCH_ROUND:
        return dict(PROSOCIAL_STRETCH_STEP)
    return {'issue-7': None, 'issue-3': None}


def _call_openai_completion(messages, model, temperature, seed, openai_api_key, timeout=45):
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
        resp = _post_openai_with_retry(
            'https://api.openai.com/v1/chat/completions',
            {'Content-Type': 'application/json', 'Authorization': f'Bearer {openai_api_key}'},
            payload, timeout=timeout
        )
        if resp.status_code != 200:
            print(f"[WARN] Hold-firm regeneration call returned {resp.status_code}")
            return None
        return resp.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[WARN] Hold-firm regeneration call exception: {e}")
        return None


def _build_edit_retry_messages(messages_for_api, current_draft, correction_note):
    """
    Builds the message list for a regeneration retry that EDITS the current draft in place,
    rather than resampling fresh from the stale original context with only a description of
    what's wrong. Used by every regeneration call site in /lucid (hold-firm, pacing, first-
    concession, reciprocity, final audit).

    Found repeatedly across this file's own history: a regeneration that never sees its own
    previous attempt has to reconstruct the ENTIRE reply from the correction note's
    description alone - it doesn't know what "everything else" actually said, so it's free
    to (and empirically does) silently change or drop content the note never mentioned. This
    is the root cause of the "one safety net's fix gets undone by a later one's regeneration"
    pattern documented throughout this file (final audit's re-audit loop and Rule 4/6's
    backstops exist specifically to catch the fallout of this, after the fact).

    Passing the draft back as the most recent assistant turn, with an explicit "edit this
    text, minimal change" instruction, lets the model treat it as literal text to revise
    rather than a rough description to reconstruct from memory - the same token cost as
    before (the model still writes out a full reply either way; this only adds a few hundred
    cheap-to-process input tokens, not output tokens, so it doesn't meaningfully add latency),
    but far more likely to actually preserve everything not specifically called out.

    Does NOT change anything about which checks exist, their order, or the final audit's
    2-attempt cap - only how each individual regeneration call is constructed.
    """
    return messages_for_api + [
        {'role': 'assistant', 'content': current_draft},
        {'role': 'system', 'content': (
            correction_note + " This is a MINIMAL EDIT to your own draft reply shown "
            "immediately above - copy it verbatim except for the specific change(s) just "
            "described. Do not rewrite, rephrase, reorganize, or otherwise touch anything "
            "else."
        )}
    ]


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
            model = body.get('model', 'gpt-5.6') # Use model from request, default to gpt-5.6 if not sent (JS usually sends its default)
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
                    # gpt-5.x models reject any temperature value other than the default (1) -
                    # the API 400s outright on e.g. 0.7, rather than clamping it itself. Force
                    # it here so a stale/manual temperature request never breaks the call.
                    if model.startswith('gpt-5') and used_temperature != 1.0:
                        print(f"[INFO /lucid] Model '{model}' only supports temperature=1 - overriding requested {used_temperature}") # Vercel Log
                        used_temperature = 1.0
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

                    # The recruiter's own most recent reply (last round's, not this round's -
                    # that hasn't been generated yet). Needed so the concession classifier can
                    # also judge whether latest_user_message is simply ACCEPTING a conditional
                    # trade proposed there (see _detect_first_concession_llm's
                    # accepts_prior_offer).
                    prior_assistant_message = ''
                    for msg in reversed(messages):
                        if msg.get('role') == 'assistant':
                            prior_assistant_message = str(msg.get('content', ''))
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
                    # Prosocial's OPTIONAL deeper step, if this round is late enough to offer
                    # it - see _pacing_stretch_target(). Only actually offered to the model
                    # below when a genuine concession also happens this same round.
                    pacing_stretch_target = _pacing_stretch_target(condition_key, turn_number)

                    # --- Prosocial-only: one-time "first concession" exception ---
                    # See prompts.yaml [NEGOTIATION PROTOCOL]: the first time the candidate offers
                    # ANY concession - on any issue - Prosocial rewards it with TWO separate
                    # things: (1) the candidate's own proposed exchange (conceded issue <-> what
                    # they asked for) resolves as an all-or-nothing PACKAGE - if the requested
                    # issue can be granted this round (anything except Salary/Vacation Time while
                    # still in the hold-firm window), BOTH sides apply; if it can't, NEITHER side
                    # applies (see Case 1/2 below); (2) unconditionally, regardless of how (1)
                    # resolves, a one-time trust-building gift also lands on a THIRD issue,
                    # distinct from both sides of the exchange. Whether this has already happened
                    # is tracked via a flag the frontend echoes back each round
                    # (prosocial_first_concession_used) rather than asked of the model, for the
                    # same reason turn_number is computed here instead of self-counted: "has this
                    # ever happened before in this conversation" is exactly the kind of long-range
                    # state an LLM can't reliably track on its own.
                    prosocial_first_concession_used = bool(body.get('prosocial_first_concession_used'))
                    first_concession_note = None
                    # Set only when the alternate GIFT targets one specific, known issue (rare -
                    # currently unused, kept for parity with the grant-check's own target_issue_id
                    # param). Stays None for the normal "pick any alternate issue" case, where the
                    # model has a free choice among the eligible pool.
                    first_concession_target_issue = None
                    # The all-or-nothing package trade's own state (Case 1/2 below) - set when the
                    # exception fires, read by the safety net in Step 5 to verify it actually
                    # landed (or correctly didn't). None/False when the exception doesn't fire.
                    first_concession_package_case = None  # 'granted' | 'blocked' | None
                    first_concession_requested_issue_id = None
                    # Collects every place this round where a payoff-table check couldn't
                    # confirm or deny something and fell back to trusting an LLM's own
                    # judgment as unverified - the first-concession cross-check, the
                    # reciprocity-claim safety net, and the one-level-cap grant check.
                    # Returned to the frontend as its own field (never folded into
                    # generated_text, so it's invisible in the chat itself) so these
                    # unverified edge cases can be surfaced/audited separately from the
                    # negotiation reply proper.
                    unverified_trust_notes = []
                    # Set only when the first-concession safety net (Step 5) CONFIRMS a grant
                    # actually happened this round - a short, deterministic announcement of
                    # exactly what was granted, shown to the candidate as its own separate chat
                    # bubble by the frontend, per request (see response_data below). Stays None
                    # otherwise, including when the exception fired but the grant never got
                    # confirmed (still missing/overshooting after the retry).
                    first_concession_announcement = None

                    # --- General "genuine concession this round" check (BOTH conditions,
                    # every round) ---
                    # Reused for three things: (1) the round_note prescription below (Step 4) -
                    # tells the model, BEFORE it drafts a reply, whether it has any
                    # justification to move something unconditionally this round, rather than
                    # only checking after the fact whether its own text claimed one; (2) the
                    # general "no free concession" safety net (Step 5), which - unlike the
                    # older reciprocity-claim check - doesn't depend on the model's reply
                    # narrating a reciprocity claim at all, so it also catches silently moving
                    # something with no claim; (3) Prosocial's one-time first-concession
                    # exception, right below. Cross-checks the classifier's "is_concession"
                    # framing against the real payoff table, same as always: "sounds like a
                    # concession" and "is actually favorable to the recruiter" are different
                    # questions (e.g. an EARLIER start date reads like a concession but scores
                    # WORSE for the recruiter). Also covers 'same' (acquiescing to the
                    # recruiter's own current position isn't giving anything up either). Only
                    # 'unknown' gets the benefit of the doubt (not counted as genuine, but not
                    # ruled out either) since extraction can legitimately fail to match.
                    genuine_concession_this_round = False
                    genuine_concession_label = None
                    genuine_concession_value = None
                    # Unified issue-id for whichever path set genuine_concession_this_round -
                    # is_concession's own conceded_issue_id, or accepts_prior_offer's
                    # accepted_counterpart_issue_id. Lets downstream code (the first-concession
                    # exception's package-trade logic) reference ONE consistent variable
                    # regardless of which path triggered it.
                    genuine_concession_issue_id = None
                    round_concession_check = {
                        'is_concession': False, 'requested_issue_id': None,
                        'conceded_issue_id': None, 'conceded_new_value': None,
                        'accepts_prior_offer': False, 'accepted_issue_id': None,
                        'accepted_value': None, 'accepted_counterpart_issue_id': None,
                        'accepted_counterpart_value': None
                    }
                    if latest_user_message:
                        # Fill in RECRUITER_OPENING_OFFER for any issue prior_issue_statuses
                        # hasn't recorded yet (e.g. round 1, before any assistant_updates have
                        # been captured) so the classifier always sees the full current
                        # package, not a partially-blank one.
                        current_offer_for_classifier = [
                            dict(item, status=item.get('status') or RECRUITER_OPENING_OFFER.get(item['id'], ''))
                            for item in prior_issue_statuses
                        ]
                        round_concession_check = _detect_first_concession_llm(latest_user_message, openai_api_key, current_offer_for_classifier, prior_assistant_message)
                        if round_concession_check.get('is_concession'):
                            conceded_issue_id = round_concession_check.get('conceded_issue_id')
                            conceded_new_value = round_concession_check.get('conceded_new_value')
                            if conceded_issue_id and conceded_new_value:
                                prior_by_id_cc = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                                conceded_prior_value = prior_by_id_cc.get(conceded_issue_id) or RECRUITER_OPENING_OFFER.get(conceded_issue_id)
                                conceded_direction = _compare_recruiter_value(conceded_issue_id, conceded_new_value, conceded_prior_value)
                                if conceded_direction == 'better':
                                    genuine_concession_this_round = True
                                    genuine_concession_value = conceded_new_value
                                    genuine_concession_issue_id = conceded_issue_id
                                    genuine_concession_label = next(
                                        (item['label'] for item in _default_issue_statuses() if item['id'] == conceded_issue_id),
                                        conceded_issue_id
                                    )
                                elif conceded_direction in ('worse', 'same'):
                                    print(f"[INFO /lucid] Concession classifier flagged {conceded_issue_id}->{conceded_new_value} as a concession, but the payoff table says it's {conceded_direction.upper()} (not better) for the recruiter - not treated as genuine") # Vercel Log
                                else:  # unknown
                                    unverified_trust_notes.append(
                                        f"Concession check (round {turn_number}): classifier said the candidate "
                                        f"conceded {conceded_issue_id} -> '{conceded_new_value}', but that value "
                                        f"couldn't be matched against the payoff table (unknown) - not counted as "
                                        f"a confirmed genuine concession, but not ruled out either."
                                    )
                            else:
                                unverified_trust_notes.append(
                                    f"Concession check (round {turn_number}): classifier said is_concession=true "
                                    f"but didn't name a specific conceded issue/value to verify against the "
                                    f"payoff table."
                                )

                        # Accepting a trade the RECRUITER itself proposed can also be a
                        # genuine concession - the candidate is still paying real value, just
                        # for a number the recruiter named first rather than one they proposed
                        # themselves. Found live: a candidate who only ever confirms trades
                        # the recruiter initiates ("okay, I can accept Division A") never
                        # trips is_concession (correctly - they didn't volunteer anything), so
                        # without this check genuine_concession_this_round - and therefore
                        # Prosocial's one-time first-concession gift - could never fire in a
                        # negotiation shaped that way. Only counts if not already confirmed
                        # genuine above, and only when the recruiter's own prior message named
                        # a real, payoff-table-verifiable cost on the candidate's side (not
                        # just what the candidate receives).
                        if not genuine_concession_this_round and round_concession_check.get('accepts_prior_offer'):
                            counterpart_issue_id = round_concession_check.get('accepted_counterpart_issue_id')
                            counterpart_value = round_concession_check.get('accepted_counterpart_value')
                            if counterpart_issue_id and counterpart_value:
                                prior_by_id_cp = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                                counterpart_prior_value = prior_by_id_cp.get(counterpart_issue_id) or RECRUITER_OPENING_OFFER.get(counterpart_issue_id)
                                counterpart_direction = _compare_recruiter_value(counterpart_issue_id, counterpart_value, counterpart_prior_value)
                                if counterpart_direction == 'better':
                                    genuine_concession_this_round = True
                                    genuine_concession_value = counterpart_value
                                    genuine_concession_issue_id = counterpart_issue_id
                                    genuine_concession_label = next(
                                        (item['label'] for item in _default_issue_statuses() if item['id'] == counterpart_issue_id),
                                        counterpart_issue_id
                                    )
                                    print(f"[INFO /lucid] Candidate accepted a real cost ({counterpart_issue_id}->{counterpart_value}) as part of accepting the recruiter's own prior trade - counted as a genuine concession") # Vercel Log
                                elif counterpart_direction == 'unknown':
                                    unverified_trust_notes.append(
                                        f"Concession check (round {turn_number}): classifier said the candidate "
                                        f"accepted a prior trade costing them {counterpart_issue_id} -> "
                                        f"'{counterpart_value}', but that value couldn't be matched against the "
                                        f"payoff table (unknown) - not counted as a confirmed genuine concession, "
                                        f"but not ruled out either."
                                    )
                                # 'worse'/'same' silently not counted - same as the is_concession
                                # path above, no unverified-trust note needed since the payoff
                                # table gave a confident, negative answer.

                    # --- Prosocial-only: one-time "first concession" exception ---
                    # See prompts.yaml [NEGOTIATION PROTOCOL]: the first time the candidate
                    # offers a GENUINE concession (per the check above), Prosocial rewards it -
                    # see the introductory comment above for the Case 1/2 package-trade design.
                    if condition_key == 'prosocial' and not prosocial_first_concession_used and genuine_concession_this_round:
                        prosocial_first_concession_used = True  # consumed either way - see comment above

                        # The conceded side - whichever path (is_concession's own
                        # conceded_issue_id, or accepts_prior_offer's accepted_counterpart_*) set
                        # genuine_concession_this_round above.
                        conceded_issue_id = genuine_concession_issue_id
                        conceded_value = genuine_concession_value
                        conceded_label = genuine_concession_label

                        # The Case 1/2 package-trade framework below only applies when the
                        # candidate freshly PROPOSED a concession-for-request trade this round
                        # (is_concession's own path) - there's a real requested_issue_id to
                        # grant or decline. It does NOT apply when genuine_concession_this_round
                        # came from accepting a trade the recruiter itself already promised
                        # (accepts_prior_offer's accepted_counterpart_* path): there, nothing
                        # was freshly requested - the candidate already received what was
                        # promised (verified separately by Rule 4 / _prior_offer_landed_status),
                        # and what they paid (the counterpart) is simply the fact of accepting
                        # it, not something to grant/decline here. That path just gets the
                        # separate one-time gift, nothing else.
                        if round_concession_check.get('is_concession'):
                            requested_issue_id = round_concession_check.get('requested_issue_id')
                            in_hold_firm_window = turn_number <= HOLD_FIRM_ROUNDS

                            # Salary/Vacation can never be granted through this ONE-TIME
                            # exception while still in the hold-firm window - that's the only
                            # thing that can block the requested side; past hold-firm,
                            # Salary/Vacation are treated like any other issue here (this
                            # exception may nudge them one step early, ahead of their normal
                            # pacing deadline - a deliberate one-time allowance, not a bug). An
                            # unclear/missing requested_issue_id also counts as blocked -
                            # nothing specific to grant.
                            first_concession_requested_issue_id = requested_issue_id
                            request_blocked = (not requested_issue_id) or (
                                requested_issue_id in ('issue-3', 'issue-7') and in_hold_firm_window
                            )

                            # The alternate gift must come from a THIRD issue - never the
                            # conceded side, never the requested side (whether or not it's
                            # actually grantable) - it needs to be genuinely additional, not
                            # double-crediting either half of the candidate's own proposed trade.
                            gift_exclude_ids = {conceded_issue_id, requested_issue_id}
                            gift_note = (
                                f"Separately - regardless of how the trade above resolves - as a "
                                f"one-time goodwill gesture, pick ONE of your other issues "
                                f"({_alternate_gift_issue_list_text(gift_exclude_ids)}) and move it "
                                f"ONE step in the candidate's favor, unconditionally, in this reply, "
                                f"even if they haven't specifically asked for it. Do NOT jump "
                                f"straight to their ideal value on whatever issue you pick - one "
                                f"step only."
                            )

                            if not request_blocked:
                                # Case 1: the requested side CAN be granted this round - the
                                # candidate's own proposed exchange resolves as a real trade,
                                # both sides apply. Capped to one grid step on the requested
                                # side, same as the gift - never straight to the full ask.
                                first_concession_package_case = 'granted'
                                requested_issue_label = next(
                                    (item['label'] for item in _default_issue_statuses() if item['id'] == requested_issue_id),
                                    requested_issue_id
                                )
                                requested_prior_by_id = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                                requested_prior_value = requested_prior_by_id.get(requested_issue_id) or RECRUITER_OPENING_OFFER.get(requested_issue_id)
                                one_level_value = _one_level_step(requested_issue_id, requested_prior_value)
                                if one_level_value:
                                    first_concession_note = (
                                        f"[System note: this is the candidate's first concession this "
                                        f"negotiation - they offered {conceded_label} at {conceded_value} "
                                        f"in exchange for {requested_issue_label}. Apply BOTH sides of "
                                        f"that trade in this reply, unconditionally: set {conceded_label} "
                                        f"to exactly {conceded_value}, and move {requested_issue_label} "
                                        f"ONE step in the candidate's favor - specifically to "
                                        f"{one_level_value} - not straight to whatever they specifically "
                                        f"asked for, even if it's less generous than their request. "
                                        f"{gift_note}]"
                                    )
                                else:
                                    # Requested issue already at its most candidate-favorable
                                    # grid value - nothing left to move there, but the conceded
                                    # side is a real, payoff-table-verified benefit to the
                                    # recruiter with no reason to decline it, so it still
                                    # applies on its own.
                                    first_concession_note = (
                                        f"[System note: this is the candidate's first concession this "
                                        f"negotiation - they offered {conceded_label} at {conceded_value} "
                                        f"in exchange for {requested_issue_label}, but {requested_issue_label} "
                                        f"is already at its most candidate-favorable value - there's "
                                        f"nothing left to move there. Still apply their offered "
                                        f"concession: set {conceded_label} to exactly {conceded_value} "
                                        f"in this reply, unconditionally. {gift_note}]"
                                    )
                                print(f"[INFO /lucid] Prosocial first-concession exception triggered - package trade granted ({conceded_label} for {requested_issue_label})") # Vercel Log
                            else:
                                # Case 2: the requested side CANNOT be granted this round
                                # (Salary/Vacation still in hold-firm, or the ask was unclear) -
                                # decline the WHOLE trade as a package: the conceded side does
                                # NOT apply either, even though it was offered. The separate
                                # one-time gift still fires.
                                first_concession_package_case = 'blocked'
                                if requested_issue_id in ('issue-3', 'issue-7'):
                                    block_reason = "Salary/Vacation Time is still in your hold-firm window"
                                    blocked_label = 'Salary' if requested_issue_id == 'issue-7' else 'Vacation Time'
                                    trade_desc = f"{conceded_label} at {conceded_value} in exchange for {blocked_label}"
                                else:
                                    block_reason = "it wasn't clear which specific issue they were asking you to move on"
                                    trade_desc = f"{conceded_label} at {conceded_value} in exchange for something else"
                                first_concession_note = (
                                    f"[System note: this is the candidate's first concession this "
                                    f"negotiation - they offered {trade_desc}, but you cannot grant "
                                    f"that side of the trade right now ({block_reason}). Decline the "
                                    f"WHOLE trade as a package this reply: do NOT apply their offered "
                                    f"concession on {conceded_label} either - explain you can't agree "
                                    f"to the full trade yet, and leave {conceded_label} exactly at its "
                                    f"current value this round. {gift_note}]"
                                )
                                print("[INFO /lucid] Prosocial first-concession exception triggered - package trade declined (requested side blocked)") # Vercel Log
                        else:
                            # accepts_prior_offer path: nothing freshly requested this round,
                            # nothing to grant/decline - just the separate one-time gift, on a
                            # third issue distinct from both what was promised (accepted_issue_id)
                            # and what they paid for it (conceded_issue_id, the counterpart).
                            gift_exclude_ids = {round_concession_check.get('accepted_issue_id'), conceded_issue_id}
                            first_concession_note = (
                                f"[System note: this is the candidate's first concession this "
                                f"negotiation. As a one-time goodwill gesture, pick ONE of your "
                                f"other issues ({_alternate_gift_issue_list_text(gift_exclude_ids)}) "
                                f"and move it ONE step in the candidate's favor, unconditionally, "
                                f"in this reply, even if they haven't specifically asked for it. Do "
                                f"NOT jump straight to their ideal value on whatever issue you pick "
                                f"- one step only.]"
                            )
                            print("[INFO /lucid] Prosocial first-concession exception triggered - accepted a prior trade at real cost, alternate gift only") # Vercel Log
                        if first_concession_note:
                                # A separate, deterministic message announcing exactly what got
                                # granted (built from the real payoff table, not the model's own
                                # wording - see Step 5 below) will be shown to the candidate as
                                # its own chat bubble, right before this reply. Tell the model
                                # not to write its own prose announcing/explaining the specific
                                # gift, so the two don't say the same thing twice - it should
                                # still reflect the correct value in its "Current package:"
                                # recap (that's a different, required part of every reply) and
                                # otherwise continue the rest of the negotiation normally.
                                first_concession_note += (
                                    " A separate message announcing this exact gift will be shown "
                                    "to the candidate automatically, right before this reply - so "
                                    "do NOT write your own sentence announcing or explaining this "
                                    "specific gift in your reply text (e.g. don't say things like "
                                    "'I'll give you a free bump on X'). Still include the correct "
                                    "value in your \"Current package:\" recap as usual, and "
                                    "continue the rest of your reply normally."
                                )

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
                    # General "no free concession" prescription (both conditions, every round) -
                    # tells the model BEFORE it drafts a reply whether it has any justification to
                    # move something unconditionally, using the genuine_concession_this_round
                    # result computed above, rather than only catching an ungrounded move after
                    # the fact (see the matching safety net in Step 5). Same "prescribe, don't just
                    # describe" principle that fixed the concession-pacing schedule's reliability
                    # earlier - prose alone ("never give for free") wasn't being followed
                    # consistently; a concrete, round-specific verdict is. Skipped when the
                    # first-concession exception is already granting something this round - that's
                    # its own, separately-instructed exception and this note would contradict it.
                    if not first_concession_note:
                        # Independent checks, not mutually exclusive - a single message can
                        # both concede something new AND accept a trade the recruiter proposed
                        # last round (e.g. "the 60% moving trade is good, and I can also do
                        # August 1 if you improve vacation"). Both notes get appended when both
                        # are true, so the model isn't only told about one and left to guess
                        # about the other - found this could otherwise happen: the classifier
                        # sometimes reads "accepting a trade that lowers the candidate's own
                        # value" as itself a concession, so genuine_concession_this_round and
                        # accepts_prior_offer can both be true from the same message.
                        noted_something = False
                        if genuine_concession_this_round:
                            round_note += (
                                f" The candidate genuinely conceded on {genuine_concession_label} "
                                f"this round (now at {genuine_concession_value}, verified against "
                                f"your payoff schedule). Per your negotiation protocol, you may "
                                f"reciprocate with a proportional move on AT MOST ONE other issue in "
                                f"this reply - do not move anything beyond that without further "
                                f"justification."
                            )
                            noted_something = True
                        if round_concession_check.get('accepts_prior_offer'):
                            # Found live in production: the candidate's message (e.g. "ok deal")
                            # doesn't itself concede anything new, so genuine_concession_this_round
                            # is correctly false here - but it IS accepting a conditional trade the
                            # recruiter proposed in its own previous reply, and that's grounds
                            # enough to follow through, not a reason to hold back.
                            round_note += (
                                " The candidate appears to be accepting the conditional trade you "
                                "proposed in your OWN previous message (something like \"if you "
                                "accept X, I'll offer Y\"). If your previous reply named a specific "
                                "conditional trade, apply that exact trade now, unconditionally, in "
                                "this reply's \"Current package\" recap - do not ask them to "
                                "reconfirm or revert what you already offered. If your previous "
                                "reply did NOT actually name a specific conditional trade, treat "
                                "this the same as no genuine concession this round instead."
                            )
                            noted_something = True
                        if not noted_something:
                            round_note += (
                                " The candidate did NOT make a genuine concession this round (per "
                                "your payoff schedule) - per your negotiation protocol, you may NOT "
                                "move any issue unconditionally in this reply. You may still move "
                                "Salary/Vacation Time if your concession schedule separately requires "
                                "it this round (see above), and you may still propose a move "
                                "CONDITIONALLY (asking for something specific in return), but do not "
                                "agree to or grant anything outright."
                            )
                        # Prosocial's optional deeper step (see PROSOCIAL_STRETCH_STEP), rounds
                        # 8-10: pacing takes priority in this window specifically, so this is
                        # mentioned regardless of whether a genuine concession happened THIS
                        # round - not gated on securing something back either (unlike the
                        # ordinary "at most one other issue" reciprocity above, which is still
                        # about issues OTHER than Salary/Vacation).
                        if any(pacing_stretch_target.values()):
                            prior_by_id_stretch = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                            stretch_still_available = []
                            for issue_id, target in pacing_stretch_target.items():
                                if not target:
                                    continue
                                current = prior_by_id_stretch.get(issue_id) or HOLD_FIRM_ANCHOR[issue_id]
                                if _compare_recruiter_value(issue_id, current, target) == 'better':
                                    label = 'Salary' if issue_id == 'issue-7' else 'Vacation Time'
                                    stretch_still_available.append(f"{label} to {target}")
                            if stretch_still_available:
                                round_note += (
                                    f" This is also within your rounds 8-10 window: you may move "
                                    f"further toward {' and '.join(stretch_still_available)} in this "
                                    f"reply if you judge it appropriate, without needing a fresh "
                                    f"concession this specific round to justify it - pacing takes "
                                    f"priority here."
                                )
                    # Final-round prescription (both conditions) - the candidate gets no further
                    # turn after this one, so a conditional trade proposed here ("if you accept X,
                    # I'll do Y") can never actually be confirmed: the round limit is enforced by
                    # the frontend before a next message ever reaches this endpoint, so an
                    # acceptance like "ok deal" sent after this round is never processed at all -
                    # found live in production (see the round-10 screenshot this note was added
                    # for). Prescribing this rather than reacting to it after the fact, same
                    # principle as the rest of this round-note block.
                    if turn_number >= TOTAL_ROUNDS:
                        round_note += (
                            f" This is the FINAL round (round {turn_number} of {TOTAL_ROUNDS}) - the "
                            f"candidate will not get another turn to respond after this one. Do NOT "
                            f"propose a new conditional trade that would need their confirmation next "
                            f"round (e.g. \"if you accept X, I'll do Y\") - there is no next round for "
                            f"them to confirm it, and it would go unresolved. Present your definitive "
                            f"final package instead: either hold your current position firmly, or if "
                            f"you're willing to make one last move, apply it directly and "
                            f"unconditionally in this reply's \"Current package\" recap - do not leave "
                            f"anything pending on the candidate's acceptance."
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

                    # Make the POST request to OpenAI with a timeout. 45s (was 30s) - gpt-5.6
                    # spends invisible reasoning_tokens before any visible output and runs
                    # measurably slower/more variable than gpt-4o; a timeout here isn't caught
                    # locally, it propagates to the outer handler as a generic 500, which the
                    # frontend shows as "couldn't be sent" - found live in production.
                    # _post_openai_with_retry additionally retries once/twice on a transient
                    # OpenAI-side error (429/500/502/503/504) before giving up - found live in
                    # production too: a 503 that arrived and resolved in under 100ms, nothing
                    # to do with our own timeout budget, that a plain retry would have absorbed.
                    response_openai = _post_openai_with_retry(openai_url, headers, data_payload, timeout=45)
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
                                        _build_edit_retry_messages(messages_for_api, generated_text, correction_note),
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
                                        _build_edit_retry_messages(messages_for_api, generated_text, correction_note),
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

                            # --- First-concession safety net (Prosocial only, whenever the
                            # exception fired this round) ---
                            # Independent of the two checks above (this can fire any round,
                            # including inside the hold-firm window, so it isn't chained onto
                            # their if/elif). Verifies TWO things the note above asked for,
                            # using the real payoff table rather than trusting the model
                            # followed through: (1) the alternate one-time gift landed, on a
                            # specific ONE-LEVEL step, on a THIRD issue distinct from the
                            # package trade; (2) the package trade itself resolved as
                            # prescribed - both sides applied if granted (Case 1), or neither
                            # side applied if the requested side was blocked (Case 2). Both
                            # checked together so a single regeneration can fix whichever (or
                            # both) went wrong, rather than fighting each other across two
                            # separate single-shot nets.
                            # Defaults so _audit_final_state's Rule 3 below can safely reference
                            # these even on a non-first-concession round (where the block below
                            # never runs and never assigns them).
                            grant_status = None
                            grant_info = None
                            if first_concession_note:
                                # Excludes accepted_issue_id (the accepts_prior_offer path's own
                                # already-being-fulfilled issue) AND both sides of THIS round's
                                # package trade (conceded/requested) from the "pick any other
                                # issue" pool below - the gift needs to be genuinely additional,
                                # not double-crediting anything already accounted for elsewhere.
                                gift_exclude_ids = {
                                    round_concession_check.get('accepted_issue_id'),
                                    genuine_concession_issue_id,
                                    first_concession_requested_issue_id,
                                }
                                grant_status, grant_info, grant_unverified_note, assistant_updates = _first_concession_grant_status_with_fallback(
                                    prior_issue_statuses, assistant_updates, first_concession_target_issue, generated_text,
                                    exclude_issue_ids=gift_exclude_ids
                                )
                                if grant_unverified_note:
                                    unverified_trust_notes.append(grant_unverified_note)
                                package_violations = _package_trade_status(
                                    prior_issue_statuses, assistant_updates, genuine_concession_issue_id,
                                    genuine_concession_value, first_concession_requested_issue_id,
                                    first_concession_package_case == 'blocked'
                                )
                                if grant_status != 'ok' or package_violations['conceded_wrong'] or package_violations['requested_missing']:
                                    instruction_parts = []
                                    if grant_status == 'overshoot':
                                        overshoot_issue_id, cap_value = grant_info or (None, None)
                                        overshoot_label = next(
                                            (item['label'] for item in _default_issue_statuses() if item['id'] == overshoot_issue_id),
                                            overshoot_issue_id
                                        )
                                        instruction_parts.append(
                                            f"your one-time gift moved {overshoot_label} further than the single "
                                            f"grid step this exception allows - scale it back to exactly {cap_value}, "
                                            f"not the candidate's full ask"
                                        )
                                    elif grant_status != 'ok':
                                        if first_concession_target_issue:
                                            grant_label = next(
                                                (item['label'] for item in _default_issue_statuses() if item['id'] == first_concession_target_issue),
                                                first_concession_target_issue
                                            )
                                            instruction_parts.append(f"grant your one-time, one-step gift on {grant_label}")
                                        else:
                                            instruction_parts.append(
                                                f"pick ONE of your other issues "
                                                f"({_alternate_gift_issue_list_text(gift_exclude_ids)}) "
                                                f"and grant your one-time, one-step gift on it"
                                            )
                                    if package_violations['conceded_wrong']:
                                        cw_issue_id, cw_label, cw_target = package_violations['conceded_wrong']
                                        if first_concession_package_case == 'blocked':
                                            instruction_parts.append(
                                                f"the requested side of the candidate's proposed trade couldn't be "
                                                f"granted this round, so the WHOLE trade is declined - revert "
                                                f"{cw_label} back to exactly {cw_target}, do not apply their offered "
                                                f"concession on it"
                                            )
                                        else:
                                            instruction_parts.append(
                                                f"set {cw_label} to exactly {cw_target}, matching what the candidate "
                                                f"actually offered - not a smaller move than that"
                                            )
                                    if package_violations['requested_missing']:
                                        rm_issue_id, rm_label, rm_target = package_violations['requested_missing']
                                        instruction_parts.append(
                                            f"the candidate's trade was granted, so move {rm_label} ONE step to "
                                            f"exactly {rm_target} as promised - not the candidate's full ask, but "
                                            f"not left unmoved either"
                                        )
                                    print(f"[WARN /lucid] First-concession exception not honored (gift={grant_status}, package_case={first_concession_package_case}, violations={package_violations}), regenerating") # Vercel Log
                                    correction_note = (
                                        f"[System note: your previous draft reply did not correctly handle your "
                                        f"one-time first-concession exception. Write your reply again: "
                                        f"{'; also, '.join(instruction_parts)}, unconditionally, in this reply.]"
                                    )
                                    retry_text = _call_openai_completion(
                                        _build_edit_retry_messages(messages_for_api, generated_text, correction_note),
                                        model, used_temperature, used_seed, openai_api_key
                                    )
                                    if retry_text:
                                        generated_text = retry_text
                                        assistant_updates = _extract_issue_updates_from_message_llm(generated_text, openai_api_key)
                                        recheck_status, recheck_info, recheck_unverified_note, assistant_updates = _first_concession_grant_status_with_fallback(
                                            prior_issue_statuses, assistant_updates, first_concession_target_issue, generated_text,
                                            exclude_issue_ids=gift_exclude_ids
                                        )
                                        if recheck_unverified_note:
                                            unverified_trust_notes.append(recheck_unverified_note)
                                        recheck_package_violations = _package_trade_status(
                                            prior_issue_statuses, assistant_updates, genuine_concession_issue_id,
                                            genuine_concession_value, first_concession_requested_issue_id,
                                            first_concession_package_case == 'blocked'
                                        )
                                        if recheck_status != 'ok' or recheck_package_violations['conceded_wrong'] or recheck_package_violations['requested_missing']:
                                            print(f"[WARN /lucid] First-concession exception still not honored after regeneration (gift={recheck_status}, violations={recheck_package_violations}) - keeping it, not retrying again") # Vercel Log
                                        else:
                                            # Regeneration fixed it - use the post-regen result
                                            # below to build the grant announcement, not the
                                            # original (non-compliant) attempt.
                                            grant_status, grant_info = recheck_status, recheck_info
                                    else:
                                        print("[WARN /lucid] First-concession regeneration call failed - keeping original reply") # Vercel Log

                                # grant_status/grant_info now reflect the FINAL state (either the
                                # first check, if it was already compliant, or the post-regen
                                # recheck above). Only build an announcement when a specific
                                # issue/value was actually confirmed granted - never invent one
                                # for a still-missing/overshooting grant.
                                if grant_status == 'ok' and grant_info:
                                    granted_issue_id, granted_value = grant_info
                                    granted_label = next(
                                        (item['label'] for item in _default_issue_statuses() if item['id'] == granted_issue_id),
                                        granted_issue_id
                                    )
                                    # Built entirely from known facts (the confirmed issue/value),
                                    # not the model's own phrasing - shown to the candidate as its
                                    # own separate chat bubble by the frontend (see
                                    # response_data below), per request.
                                    first_concession_announcement = (
                                        f"Thanks for your concession — as a reciprocal gesture, I'd "
                                        f"like to give you a free move on {granted_label}: "
                                        f"{granted_value}, no need to give anything in return."
                                    )

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
                            # reciprocal grant stand. Also covers 'same': crediting the candidate
                            # for merely accepting the recruiter's own already-standing position
                            # (nothing actually moved) isn't a concession either - e.g. the
                            # candidate asks for June 1, then says "I can take a later starting
                            # date" without naming a value; that's acquiescing to July 15 (already
                            # on the table), not a fresh concession worth reciprocating.
                            reciprocity_check = _detect_reciprocity_claim_llm(generated_text, openai_api_key)
                            if reciprocity_check.get('claims_reciprocity'):
                                credited_issue_id = reciprocity_check.get('credited_issue_id')
                                credited_value = reciprocity_check.get('credited_value')
                                # Cross-check against round_concession_check's OWN accepted_issue_id
                                # - already independently grounded from the recruiter's PRIOR
                                # message (see _detect_first_concession_llm), not read from this
                                # reply. If the reciprocity classifier is crediting the candidate
                                # with that SAME issue, it's almost certainly the same directional
                                # misread this prompt was patched for before (crediting the
                                # recruiter's OWN grant TO the candidate as if conceded FROM them) -
                                # just in an implicit framing the earlier patch didn't cover (e.g.
                                # "we have agreement on X in exchange for Y... I can also..." rather
                                # than an explicit "since/because" sentence). Don't trust it: this
                                # round's actual grant is already verified separately (Rule 4 /
                                # _prior_offer_landed_status below), it isn't something to revert.
                                if credited_issue_id and credited_issue_id == round_concession_check.get('accepted_issue_id'):
                                    print(f"[INFO /lucid] Reciprocity claim ({credited_issue_id}->{credited_value}) matches this round's own accepted_issue_id - likely crediting the recruiter's own already-promised grant back to the candidate, not trusting it") # Vercel Log
                                    credited_issue_id = None
                                    credited_value = None
                                if credited_issue_id and credited_value:
                                    credited_prior_by_id = {item['id']: item.get('status', '') for item in prior_issue_statuses}
                                    credited_prior_value = credited_prior_by_id.get(credited_issue_id) or RECRUITER_OPENING_OFFER.get(credited_issue_id)
                                    credited_direction = _compare_recruiter_value(credited_issue_id, credited_value, credited_prior_value)
                                    if credited_direction in ('worse', 'same'):
                                        credited_label = next(
                                            (item['label'] for item in _default_issue_statuses() if item['id'] == credited_issue_id),
                                            credited_issue_id
                                        )
                                        print(f"[WARN /lucid] Reciprocity claim invalid - {credited_issue_id}->{credited_value} is {credited_direction.upper()} (not better) for the recruiter, regenerating") # Vercel Log
                                        # Name the EXACT value to revert to, rather than an abstract
                                        # "don't treat that as a concession" - a concrete target is
                                        # more likely to actually change the reply than a vague
                                        # prohibition (observed in testing: vague corrections often
                                        # came back with the reply byte-identical to the original).
                                        correction_note = (
                                            f"[System note: your previous draft reply credited the candidate with a "
                                            f"concession on {credited_label} ({credited_value}) and reciprocated based "
                                            f"on that - but per your payoff schedule, that value is NOT actually "
                                            f"favorable to you compared to your current position on {credited_label}, "
                                            f"so it isn't a real concession. Write your reply again: revert "
                                            f"{credited_label} back to exactly {credited_prior_value} in this reply, "
                                            f"and do not reciprocate based on that claim. You may still make a move "
                                            f"on a DIFFERENT issue this reply if it's justified some other way (your "
                                            f"own concession schedule, or a genuine concession the candidate made "
                                            f"elsewhere), but not on {credited_label}.]"
                                        )
                                        retry_text = _call_openai_completion(
                                            _build_edit_retry_messages(messages_for_api, generated_text, correction_note),
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
                                                    still_invalid = _compare_recruiter_value(rc_issue, rc_value, rc_prior) in ('worse', 'same')
                                            if still_invalid:
                                                print("[WARN /lucid] Reciprocity claim still invalid after regeneration - keeping it, not retrying again") # Vercel Log
                                        else:
                                            print("[WARN /lucid] Reciprocity-claim regeneration call failed - keeping original reply") # Vercel Log
                                    elif credited_direction == 'unknown':
                                        unverified_trust_notes.append(
                                            f"Reciprocity claim (round {turn_number}): reply credited the candidate "
                                            f"with {credited_issue_id} -> '{credited_value}', but that value couldn't "
                                            f"be matched against the payoff table (unknown) - allowed the reciprocal "
                                            f"grant to stand."
                                        )
                                else:
                                    unverified_trust_notes.append(
                                        f"Reciprocity claim (round {turn_number}): reply claimed reciprocity but "
                                        f"didn't credit a specific issue/value to verify - allowed to stand."
                                    )

                            # --- Final audit safety net (both conditions, every round) - runs
                            # LAST, after hold-firm/pacing, grant, and reciprocity ---
                            # Merges FIVE invariants into one final, comprehensive check against
                            # whatever generated_text/assistant_updates stand at this point, after
                            # every earlier safety net's regeneration:
                            #   1. hold-firm (rounds 1-HOLD_FIRM_ROUNDS): Salary/Vacation must
                            #      still exactly match HOLD_FIRM_ANCHOR - no exceptions, not even
                            #      a genuine concession this round excuses it.
                            #   2. pacing minimum: once this round is at/past a pacing deadline,
                            #      Salary/Vacation must still be at least at pacing_target.
                            #   3. no free concession: no OTHER issue may have moved in the
                            #      candidate's favor without a genuine concession this round -
                            #      runs on a first-concession round too, exempting only the
                            #      CONFIRMED gift issue (not the whole round - see
                            #      _first_concession_grant_status's exclude_issue_ids and the
                            #      grant_status == 'ok' check below).
                            #   4. promised trade landed: if the candidate accepted a specific
                            #      trade the recruiter itself promised, verify the exact promised
                            #      value actually shipped (see _prior_offer_landed_status).
                            #   5. no unconfirmed proposal shipped as applied: the recap must not
                            #      show a value that this reply's own prose frames as still
                            #      pending the candidate's acceptance (see
                            #      _detect_unconfirmed_recap_values_llm) - the only rule here that
                            #      can't be verified against the payoff table, since a hypothetical
                            #      value and a confirmed value are identical there.
                            # Rules 1 and 2 exist here IN ADDITION TO the dedicated hold-firm and
                            # pacing-deadline safety nets above (which only ever run once, first)
                            # because a LATER safety net's regeneration (grant/reciprocity, each
                            # independently resampling from the original stale context) can
                            # silently undo either fix - observed in testing for both.
                            #
                            # Re-audits fully after every regeneration (up to 2 attempts total)
                            # instead of only rechecking the issues originally flagged - also
                            # found in testing: a regeneration meant to fix one thing can
                            # volunteer an entirely different, previously-clean violation that a
                            # narrower recheck would miss entirely.
                            def _audit_final_state(current_assistant_updates):
                                accumulated = apply_issue_updates(prior_issue_statuses, current_assistant_updates)
                                accumulated_by_id = {item['id']: item.get('status', '') for item in accumulated}
                                prior_by_id = {item['id']: item.get('status', '') for item in prior_issue_statuses}

                                # Rule 1: hold-firm - hard constraint, no exemptions, runs every
                                # round regardless of first_concession_note.
                                hf_violations = []
                                if turn_number <= HOLD_FIRM_ROUNDS:
                                    for issue_id, anchor in HOLD_FIRM_ANCHOR.items():
                                        current_val = accumulated_by_id.get(issue_id) or RECRUITER_OPENING_OFFER.get(issue_id)
                                        if not _matches_anchor(current_val, anchor):
                                            label = 'Salary' if issue_id == 'issue-7' else 'Vacation Time'
                                            hf_violations.append((issue_id, label, anchor))

                                # Rule 2: pacing minimum - hard constraint once its deadline round
                                # is reached, also unconditional on first_concession_note. Hold-firm
                                # rounds excluded (rule 1 already governs those).
                                pc_violations = []
                                if turn_number > HOLD_FIRM_ROUNDS:
                                    for issue_id, target in pacing_target.items():
                                        if not target:
                                            continue
                                        current_val = accumulated_by_id.get(issue_id) or HOLD_FIRM_ANCHOR.get(issue_id)
                                        if _compare_recruiter_value(issue_id, current_val, target) == 'better':
                                            label = 'Salary' if issue_id == 'issue-7' else 'Vacation Time'
                                            pc_violations.append((issue_id, label, target))

                                # Rule 3: no free concession - runs every round, including a
                                # first-concession round (see note above): used to be skipped
                                # entirely whenever first_concession_note was set, on the
                                # assumption that round's one free move was always exactly the
                                # confirmed gift issue - but found live that a LATER safety net's
                                # own regeneration (triggered by something unrelated, e.g. the
                                # reciprocity-claim check) can resample fresh and volunteer
                                # brand-new, completely ungrounded moves on OTHER issues (a real
                                # case: Vacation Time and Bonus both moved with zero justification
                                # in a regenerated draft, entirely unrelated to the actual
                                # first-concession gift), and a blanket round-wide skip meant
                                # nothing would ever catch those. Skips issues already flagged by
                                # rule 1/2 to avoid double-flagging.
                                ug_moves = []
                                already_flagged_ids = {v[0] for v in hf_violations} | {v[0] for v in pc_violations}
                                for item in _default_issue_statuses():
                                    issue_id = item['id']
                                    if issue_id in already_flagged_ids:
                                        continue
                                    new_val = accumulated_by_id.get(issue_id) or RECRUITER_OPENING_OFFER.get(issue_id)
                                    old_val = prior_by_id.get(issue_id) or RECRUITER_OPENING_OFFER.get(issue_id)
                                    if _compare_recruiter_value(issue_id, new_val, old_val) != 'worse':
                                        continue  # didn't move in the candidate's favor
                                    pacing_step = pacing_target.get(issue_id)
                                    if pacing_step and _compare_recruiter_value(issue_id, new_val, pacing_step) != 'worse':
                                        continue  # within what the pacing schedule itself mandates this round
                                    stretch_step = pacing_stretch_target.get(issue_id)
                                    if stretch_step and _compare_recruiter_value(issue_id, new_val, stretch_step) != 'worse':
                                        continue  # within Prosocial's optional stretch ceiling (rounds
                                        # 8-10 only - pacing_stretch_target is None everywhere else) -
                                        # pacing takes priority in this window, not gated on a fresh
                                        # concession this specific round
                                    if first_concession_note and grant_status == 'ok' and grant_info and issue_id == grant_info[0]:
                                        continue  # the CONFIRMED one-time first-concession gift
                                        # issue for this round - already verified separately by the
                                        # grant-status check above, not an ungrounded move
                                    # A genuine concession or an accepted prior offer only ever
                                    # justifies ONE specific issue moving for free - not a blanket
                                    # pass for every issue that happens to move this round. Found
                                    # live: a real moving-expense-for-salary trade correctly
                                    # exempted Salary, but the same blanket flag also silently let
                                    # Bonus jump from 4% to 6% with zero mention anywhere in the
                                    # text. Exempt this issue only if it's the one the round's
                                    # classifier actually ties to the trade, OR the reply's own
                                    # prose names it (the recruiter reciprocating on a DIFFERENT
                                    # issue than requested is legitimate per prompts.yaml, but
                                    # should be narrated, not slipped into the recap silently).
                                    if genuine_concession_this_round and (
                                        issue_id == round_concession_check.get('requested_issue_id')
                                        or _value_mentioned_in_prose(generated_text, new_val)
                                    ):
                                        continue
                                    if round_concession_check.get('accepts_prior_offer') and (
                                        issue_id == round_concession_check.get('accepted_issue_id')
                                        or _value_mentioned_in_prose(generated_text, new_val)
                                    ):
                                        continue  # candidate accepted a trade the recruiter itself
                                        # already proposed - not a free giveaway, it's the
                                        # recruiter following through on its own prior offer
                                    ug_moves.append((issue_id, item['label'], old_val))

                                # Rule 4: if the candidate accepted a specific trade the
                                # recruiter itself promised last round (per
                                # round_concession_check's accepted_issue_id/accepted_value),
                                # verify that exact value actually landed - the other three
                                # rules only ever check "did something move for FREE", never
                                # "did something PROMISED actually ship". Found in testing: a
                                # safety net elsewhere in this round can regenerate from stale
                                # context and silently drop the promised value while fixing
                                # something unrelated, and nothing else catches that.
                                po_violations = []
                                accepted_issue_id = round_concession_check.get('accepted_issue_id')
                                accepted_value = round_concession_check.get('accepted_value')
                                if accepted_issue_id and accepted_value:
                                    po_status, po_actual = _prior_offer_landed_status(
                                        current_assistant_updates, accepted_issue_id, accepted_value, generated_text
                                    )
                                    if po_status in ('under', 'missing'):
                                        po_label = next(
                                            (item['label'] for item in _default_issue_statuses() if item['id'] == accepted_issue_id),
                                            accepted_issue_id
                                        )
                                        po_violations.append((accepted_issue_id, po_label, accepted_value))

                                # Rule 5: does the recap claim something has HAPPENED that
                                # this reply's own prose frames as still pending the
                                # candidate's acceptance? Unlike rules 1-4, this can't be
                                # verified against the payoff table (a hypothetical value
                                # and a confirmed value are identical there) - it needs an
                                # LLM's own read of the prose (see
                                # _detect_unconfirmed_recap_values_llm). Found live: a reply
                                # asked "would that trade work for you?" about a brand-new
                                # salary+bonus trade, but the SAME reply's recap already
                                # showed both values as if applied - one favorable to the
                                # candidate (missed by Rule 3, which only checks favorable
                                # moves and had already exempted it via the "mentioned in
                                # prose" check - mentioned isn't the same as confirmed) and
                                # one unfavorable (Rule 3 never even looks at those).
                                uc_violations = []
                                already_flagged_for_uc = already_flagged_ids | {v[0] for v in ug_moves}
                                unconfirmed_check = _detect_unconfirmed_recap_values_llm(generated_text, openai_api_key)
                                for issue_id in unconfirmed_check.get('unconfirmed_issue_ids', []):
                                    if issue_id in already_flagged_for_uc:
                                        continue
                                    new_val = accumulated_by_id.get(issue_id) or RECRUITER_OPENING_OFFER.get(issue_id)
                                    old_val = prior_by_id.get(issue_id) or RECRUITER_OPENING_OFFER.get(issue_id)
                                    if new_val == old_val:
                                        continue  # nothing actually changed on this issue - nothing to revert
                                    # If this move doesn't go beyond what the recruiter's OWN
                                    # concession schedule already mandates by this round, it's
                                    # not something the candidate needs to confirm - it's
                                    # unilateral and due regardless. Found live: a reply bundled
                                    # a pacing-MANDATORY Salary/Vacation move together with an
                                    # optional conditional ask (a bonus cut) in one "if you
                                    # accept" sentence - the classifier correctly read the whole
                                    # sentence as conditional, but reverting Salary/Vacation to
                                    # satisfy that then violated Rule 2's pacing minimum for this
                                    # round, and the two rules' regenerations couldn't converge.
                                    # Same exemption Rule 3 already uses for the same reason.
                                    pacing_step = pacing_target.get(issue_id)
                                    if pacing_step and _compare_recruiter_value(issue_id, new_val, pacing_step) != 'worse':
                                        continue
                                    stretch_step = pacing_stretch_target.get(issue_id)
                                    if stretch_step and _compare_recruiter_value(issue_id, new_val, stretch_step) != 'worse':
                                        continue
                                    uc_label = next(
                                        (item['label'] for item in _default_issue_statuses() if item['id'] == issue_id),
                                        issue_id
                                    )
                                    uc_violations.append((issue_id, uc_label, old_val))

                                # Rule 6: mirrors the single-shot first-concession safety net
                                # (Step 5) - the package-trade side of the first-concession
                                # exception (Case 1/2: granted -> both sides land; blocked ->
                                # neither side does) actually landed as prescribed. Exists for
                                # the same reason Rule 4 backstops the "accepted prior offer"
                                # single-shot check: a LATER safety net's own regeneration can
                                # silently undo what an earlier one already fixed.
                                pkg_violations = []
                                if first_concession_package_case:
                                    pkg_check = _package_trade_status(
                                        prior_issue_statuses, current_assistant_updates, genuine_concession_issue_id,
                                        genuine_concession_value, first_concession_requested_issue_id,
                                        first_concession_package_case == 'blocked'
                                    )
                                    if pkg_check['conceded_wrong']:
                                        pkg_violations.append(('conceded',) + pkg_check['conceded_wrong'])
                                    if pkg_check['requested_missing']:
                                        pkg_violations.append(('requested',) + pkg_check['requested_missing'])

                                return hf_violations, pc_violations, ug_moves, po_violations, uc_violations, pkg_violations

                            hold_firm_violations, pacing_violations, ungrounded_moves, prior_offer_violations, unconfirmed_violations, package_trade_violations = _audit_final_state(assistant_updates)
                            final_audit_attempts = 0
                            while (hold_firm_violations or pacing_violations or ungrounded_moves or prior_offer_violations or unconfirmed_violations or package_trade_violations) and final_audit_attempts < 2:
                                final_audit_attempts += 1
                                note_parts = []
                                if package_trade_violations:
                                    for kind, pv_issue_id, pv_label, pv_target in package_trade_violations:
                                        if kind == 'conceded':
                                            print(f"[WARN /lucid] First-concession package trade's conceded side not resolved correctly in the final check ({pv_label} should be {pv_target}), regenerating (attempt {final_audit_attempts})") # Vercel Log
                                            note_parts.append(
                                                f"the first-concession package trade means {pv_label} must be "
                                                f"exactly {pv_target} in this reply - set it, unconditionally"
                                            )
                                        else:  # 'requested'
                                            print(f"[WARN /lucid] First-concession package trade's requested side didn't land in the final check ({pv_label} should be {pv_target}), regenerating (attempt {final_audit_attempts})") # Vercel Log
                                            note_parts.append(
                                                f"the first-concession package trade was granted, so {pv_label} "
                                                f"must move to exactly {pv_target} in this reply - not the "
                                                f"candidate's full ask, but not left unmoved either"
                                            )
                                if prior_offer_violations:
                                    targets_desc = ', '.join(f"{label} to exactly {value}" for _, label, value in prior_offer_violations)
                                    print(f"[WARN /lucid] Promised trade didn't land in the final check ({targets_desc}), regenerating (attempt {final_audit_attempts})") # Vercel Log
                                    note_parts.append(
                                        f"the candidate accepted the conditional trade you proposed in "
                                        f"your own previous message, but your draft reply still doesn't "
                                        f"reflect it - set {targets_desc} in this reply's \"Current "
                                        f"package\" recap, unconditionally, exactly as you promised"
                                    )
                                if ungrounded_moves:
                                    targets_desc = ', '.join(f"{label} back to {old_val}" for _, label, old_val in ungrounded_moves)
                                    print(f"[WARN /lucid] Ungrounded free concession(s) with no genuine candidate concession this round ({targets_desc}), regenerating (attempt {final_audit_attempts})") # Vercel Log
                                    note_parts.append(
                                        f"the candidate did not make a genuine concession this round, but "
                                        f"your previous draft reply moved {targets_desc} anyway. Per your "
                                        f"negotiation protocol, never move an issue for free - revert "
                                        f"{targets_desc} in this reply. You may still propose a move "
                                        f"conditionally, asking for something specific in return, but do "
                                        f"not grant it outright"
                                    )
                                if unconfirmed_violations:
                                    targets_desc = ', '.join(f"{label} back to {old_val}" for _, label, old_val in unconfirmed_violations)
                                    print(f"[WARN /lucid] Recap shows a not-yet-accepted proposal as if applied ({targets_desc}), regenerating (attempt {final_audit_attempts})") # Vercel Log
                                    note_parts.append(
                                        f"your previous draft reply's own prose framed a NEW trade on "
                                        f"{', '.join(label for _, label, _ in unconfirmed_violations)} as a "
                                        f"proposal still pending the candidate's acceptance (e.g. asking "
                                        f"\"would that work for you?\"), but the \"Current package\" recap "
                                        f"already showed it as applied - revert {targets_desc} in this "
                                        f"reply's recap. You may still PROPOSE the trade in your prose, "
                                        f"conditionally, but do not apply it in the recap until the "
                                        f"candidate actually accepts it in a future message"
                                    )
                                if hold_firm_violations:
                                    violated_labels = ', '.join(label for _, label, _ in hold_firm_violations)
                                    targets_desc = ', '.join(f"{label} to exactly {anchor}" for _, label, anchor in hold_firm_violations)
                                    print(f"[WARN /lucid] Hold-firm violation surviving to the final check ({targets_desc}), regenerating (attempt {final_audit_attempts})") # Vercel Log
                                    note_parts.append(
                                        f"your previous draft reply still has {violated_labels} moved away "
                                        f"from its hold-firm anchor - this violates your hold-firm window "
                                        f"(rounds 1-{HOLD_FIRM_ROUNDS}), which allows NO exceptions, not even "
                                        f"for a genuine concession this round. Set {targets_desc} in this "
                                        f"reply, unconditionally"
                                    )
                                if pacing_violations:
                                    targets_desc = ', '.join(f"{label} to at least {target}" for _, label, target in pacing_violations)
                                    print(f"[WARN /lucid] Pacing minimum not met in the final check ({targets_desc}), regenerating (attempt {final_audit_attempts})") # Vercel Log
                                    note_parts.append(
                                        f"your previous draft reply still has not reached {targets_desc}, "
                                        f"which your concession schedule mandates by this round - move "
                                        f"{targets_desc} in this reply, even if the candidate hasn't "
                                        f"specifically asked for it"
                                    )
                                correction_note = "[System note: " + "; also, ".join(note_parts) + ".]"
                                retry_text = _call_openai_completion(
                                    _build_edit_retry_messages(messages_for_api, generated_text, correction_note),
                                    model, used_temperature, used_seed, openai_api_key
                                )
                                if not retry_text:
                                    print("[WARN /lucid] Final-check regeneration call failed - keeping current reply") # Vercel Log
                                    break
                                generated_text = retry_text
                                assistant_updates = _extract_issue_updates_from_message_llm(generated_text, openai_api_key)
                                # Full re-audit, not just a recheck of the issues flagged above -
                                # this is what catches a regeneration volunteering a NEW violation.
                                hold_firm_violations, pacing_violations, ungrounded_moves, prior_offer_violations, unconfirmed_violations, package_trade_violations = _audit_final_state(assistant_updates)

                            if hold_firm_violations or pacing_violations or ungrounded_moves or prior_offer_violations or unconfirmed_violations or package_trade_violations:
                                still_bad = (
                                    [label for _, label, _ in hold_firm_violations]
                                    + [label for _, label, _ in pacing_violations]
                                    + [label for _, label, _ in ungrounded_moves]
                                    + [label for _, label, _ in prior_offer_violations]
                                    + [label for _, label, _ in unconfirmed_violations]
                                    + [label for _, _, label, _ in package_trade_violations]
                                )
                                print(f"[WARN /lucid] Still not resolved on {still_bad} after {final_audit_attempts} final-check regeneration(s) - keeping it, not retrying again") # Vercel Log

                            # Prepare the successful response data for Qualtrics frontend
                            response_data = {
                                'generated_text': generated_text,
                                'used_temperature': used_temperature, # Echo back parameters used
                                # Echoed back every round regardless of condition (stays False for
                                # Proself, which never touches this flag) so the frontend can persist
                                # it and send it back next round - see the first-concession block above.
                                'prosocial_first_concession_used': prosocial_first_concession_used,
                                # Every "couldn't verify via payoff table, trusted the LLM's own
                                # judgment" edge case this round (see unverified_trust_notes above) -
                                # always present, empty list when none fired this round. Kept
                                # completely separate from generated_text so the frontend can log it
                                # (e.g. into its own Embedded Data field) without ever rendering it
                                # into the visible chat transcript.
                                'unverified_trust_notes': unverified_trust_notes,
                                # Only set when the first-concession safety net confirmed a grant
                                # actually happened this round (see above) - a short, deterministic
                                # announcement of exactly what was granted, meant to be shown as
                                # its own separate chat bubble, before generated_text, rather than
                                # folded into it.
                                'first_concession_announcement': first_concession_announcement
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
