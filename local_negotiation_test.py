# -*- coding: utf-8 -*-
"""
Standalone local test harness for the LUCID recruiter prompts (prompts.yaml).

Lets you chat with the Prosocial or Proself recruiter AI directly from the terminal,
without needing Flask, Vercel, or Qualtrics running. Reuses the exact same round-number
injection, Prosocial first-concession detection/one-level cap, and reciprocity-claim
safety net as lucid.py's real /lucid endpoint (so pacing/exception behavior is faithful
to production), but skips everything that's only for data collection - no offer-
trajectory logging, no Embedded Data. This is purely for eyeballing "does the prompt
actually behave the way we designed it" - nothing here is saved anywhere.

Setup:
    1. Put your OpenAI API key in .env (already gitignored):
           OPENAI_API_KEY=sk-...
    2. Run:
           python3 local_negotiation_test.py
    3. Pick a condition, then type candidate messages. Type "quit" to stop.

Optional env vars (set in .env or the shell):
    LUCID_TEST_MODEL         - defaults to gpt-5.6 (matches lucid.py's default)
    LUCID_TEST_TEMPERATURE   - defaults to 1.0 (matches lucid.py's default; gpt-5.x models
                               reject anything but 1.0, so leave this alone unless you
                               override LUCID_TEST_MODEL to something else too)
"""
import os
import sys
import json
import requests

import lucid  # reuses CONDITION_PROMPTS and _detect_first_concession_llm from the real backend


def _load_dotenv(path='.env'):
    """Minimal .env loader (KEY=VALUE per line) - avoids adding python-dotenv as a
    dependency just for this local test script. Does not override already-set env vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def call_openai(messages, api_key, model, temperature):
    # Mirrors lucid.py's own main-completion call: goes through _post_openai_with_retry so
    # this local harness sees the same transient-503-retry behavior as production, instead
    # of raising on the first hiccup.
    resp = lucid._post_openai_with_retry(
        'https://api.openai.com/v1/chat/completions',
        {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        {'model': model, 'messages': messages, 'temperature': temperature},
        timeout=45  # matches lucid.py's /lucid endpoint - gpt-5.6 runs slower/more variable
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:500]}")
    return resp.json()['choices'][0]['message']['content']


def main():
    _load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("No OPENAI_API_KEY found. Put it in .env as OPENAI_API_KEY=sk-... and try again.")
        sys.exit(1)

    model = os.getenv('LUCID_TEST_MODEL', 'gpt-5.6')
    temperature = float(os.getenv('LUCID_TEST_TEMPERATURE', '1.0'))

    conditions = sorted(lucid.CONDITION_PROMPTS.keys())
    if not conditions:
        print("prompts.yaml didn't load any conditions - check the file and try again.")
        sys.exit(1)

    print(f"Available conditions: {', '.join(conditions)}")
    condition_key = ''
    while condition_key not in conditions:
        condition_key = input("Which condition? ").strip().lower()

    system_prompt = lucid.CONDITION_PROMPTS[condition_key].get('initial_prompt', '')
    messages = [{'role': 'system', 'content': system_prompt}]

    turn_number = 0
    first_concession_used = False
    # Full 8-issue accumulated state, same shape lucid.py tracks via issue_statuses (starts
    # blank, same as a real round-1 request with no prior issue_statuses echoed back yet).
    # Needed for the pacing-target check (Salary/Vacation) and now also the first-concession
    # payoff-table checks, which need a prior value for whichever issue was conceded/granted,
    # not just Salary/Vacation.
    current_statuses = lucid._default_issue_statuses()

    print(f"\n--- Testing '{condition_key}' (model={model}, temperature={temperature}) ---")
    print("Type a candidate message and press Enter. Type 'quit' to stop.\n")

    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break
        if user_message.lower() in ('quit', 'exit'):
            break
        if not user_message:
            continue

        messages.append({'role': 'user', 'content': user_message})
        turn_number += 1
        # Collects every "couldn't verify via payoff table, trusted the LLM's own judgment"
        # edge case this round - printed as its own separate block below, apart from Alex's
        # reply, so these unverified cases are easy to spot while testing.
        unverified_trust_notes = []
        # Set only when the first-concession grant safety net confirms a grant actually
        # happened this round - a deterministic announcement (built from the confirmed
        # issue/value, not the model's own wording), printed as its own separate line below.
        first_concession_announcement = None

        # --- Same general "genuine concession this round" check as lucid.py's /lucid
        # endpoint (BOTH conditions, every round; computed BEFORE the round note so its
        # result can be prescribed into that note below). Reused for: (1) the round_note
        # prescription right after this, (2) the general "no free concession" safety net
        # further down, (3) Prosocial's one-time first-concession exception. ---
        genuine_concession_this_round = False
        genuine_concession_label = None
        genuine_concession_value = None
        round_concession_check = {'is_concession': False, 'requested_issue_id': None,
                                   'conceded_issue_id': None, 'conceded_new_value': None,
                                   'accepts_prior_offer': False, 'accepted_issue_id': None,
                                   'accepted_value': None, 'accepted_counterpart_issue_id': None,
                                   'accepted_counterpart_value': None}
        # Ground the classifier with the recruiter's full current package (falling back to
        # RECRUITER_OPENING_OFFER for anything not yet recorded) so a vague candidate message
        # like "I can take a later start date" gets extracted as the concrete value actually
        # on the table, not a vague description the payoff check below can't match.
        current_offer_for_classifier = [
            dict(item, status=item['status'] or lucid.RECRUITER_OPENING_OFFER.get(item['id'], ''))
            for item in current_statuses
        ]
        # Same as lucid.py's /lucid endpoint: the recruiter's own last reply (before this
        # turn's, which hasn't been generated yet), so the classifier can also judge whether
        # user_message is accepting a conditional trade proposed there.
        prior_assistant_message = ''
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                prior_assistant_message = str(msg.get('content', ''))
                break
        round_concession_check = lucid._detect_first_concession_llm(
            user_message, api_key, current_offer_for_classifier, prior_assistant_message
        )
        # Cross-check the classifier's framing against the real payoff table before trusting
        # it - "sounds like a concession" isn't the same as "actually favorable to the
        # recruiter" (e.g. an earlier start date reads like a concession but scores worse for
        # the recruiter on the real table). Also covers 'same': accepting a value identical to
        # the recruiter's own current position (e.g. "I can take a later starting date" with no
        # new value of their own, when the recruiter's anchor never moved) is acquiescence, not
        # a fresh concession.
        if round_concession_check.get('is_concession'):
            conceded_issue_id = round_concession_check.get('conceded_issue_id')
            conceded_new_value = round_concession_check.get('conceded_new_value')
            if conceded_issue_id and conceded_new_value:
                prior_by_id_cc = {item['id']: item['status'] for item in current_statuses}
                conceded_prior_value = prior_by_id_cc.get(conceded_issue_id) or lucid.RECRUITER_OPENING_OFFER.get(conceded_issue_id)
                conceded_direction = lucid._compare_recruiter_value(conceded_issue_id, conceded_new_value, conceded_prior_value)
                if conceded_direction == 'better':
                    genuine_concession_this_round = True
                    genuine_concession_value = conceded_new_value
                    genuine_concession_label = next(
                        (item['label'] for item in lucid._default_issue_statuses() if item['id'] == conceded_issue_id),
                        conceded_issue_id
                    )
                elif conceded_direction in ('worse', 'same'):
                    print(f"  [concession classifier flagged {conceded_issue_id}->{conceded_new_value} as a concession, but payoff table says it's {conceded_direction.upper()} (not better) for the recruiter - not treated as genuine]")
                else:  # unknown
                    unverified_trust_notes.append(
                        f"Concession check: classifier said the candidate conceded "
                        f"{conceded_issue_id} -> '{conceded_new_value}', but that value couldn't be "
                        f"matched against the payoff table (unknown) - not counted as a confirmed "
                        f"genuine concession, but not ruled out either."
                    )
            else:
                unverified_trust_notes.append(
                    "Concession check: classifier said is_concession=true but didn't name a "
                    "specific conceded issue/value to verify against the payoff table."
                )

        # Mirrors lucid.py: accepting a trade the RECRUITER itself proposed can also be a
        # genuine concession - the candidate is still paying real value, just for a number
        # the recruiter named first. Only counts if not already confirmed genuine above, and
        # only when the recruiter's own prior message named a real, payoff-table-verifiable
        # cost on the candidate's side.
        if not genuine_concession_this_round and round_concession_check.get('accepts_prior_offer'):
            counterpart_issue_id = round_concession_check.get('accepted_counterpart_issue_id')
            counterpart_value = round_concession_check.get('accepted_counterpart_value')
            if counterpart_issue_id and counterpart_value:
                prior_by_id_cp = {item['id']: item['status'] for item in current_statuses}
                counterpart_prior_value = prior_by_id_cp.get(counterpart_issue_id) or lucid.RECRUITER_OPENING_OFFER.get(counterpart_issue_id)
                counterpart_direction = lucid._compare_recruiter_value(counterpart_issue_id, counterpart_value, counterpart_prior_value)
                if counterpart_direction == 'better':
                    genuine_concession_this_round = True
                    genuine_concession_value = counterpart_value
                    genuine_concession_label = next(
                        (item['label'] for item in lucid._default_issue_statuses() if item['id'] == counterpart_issue_id),
                        counterpart_issue_id
                    )
                    print(f"  [candidate accepted a real cost ({counterpart_issue_id}->{counterpart_value}) as part of accepting the recruiter's own prior trade - counted as a genuine concession]")
                elif counterpart_direction == 'unknown':
                    unverified_trust_notes.append(
                        f"Concession check: classifier said the candidate accepted a prior trade "
                        f"costing them {counterpart_issue_id} -> '{counterpart_value}', but that "
                        f"value couldn't be matched against the payoff table (unknown) - not "
                        f"counted as a confirmed genuine concession, but not ruled out either."
                    )

        # --- Same ephemeral round-number note as lucid.py's /lucid endpoint (strengthened
        # during the hold-firm window, rounds 1-HOLD_FIRM_ROUNDS) ---
        if turn_number <= lucid.HOLD_FIRM_ROUNDS:
            round_note = (
                f"[System note: this is round {turn_number} of the negotiation, still within "
                f"your hold-firm window (rounds 1-{lucid.HOLD_FIRM_ROUNDS}). You must NOT move Salary "
                f"or Vacation Time away from your opening anchor ({lucid.HOLD_FIRM_ANCHOR['issue-7']} / "
                f"{lucid.HOLD_FIRM_ANCHOR['issue-3']}) this round, no matter what the candidate offers "
                f"or asks for - hold firm on those two issues specifically. You may discuss, "
                f"concede on, or trade any of your other issues freely.]"
            )
        else:
            round_note = f"[System note: this is round {turn_number} of the negotiation. Pace your concessions accordingly, per your instructions.]"
        # Same "stop repeating the round-1 priority question" reminder as lucid.py's /lucid endpoint
        if condition_key == 'prosocial' and turn_number > 1:
            round_note += (
                " You already completed your round-1 priority-gathering step in an "
                "earlier message - do not ask the candidate to restate their priorities "
                "again this round. Move the negotiation forward on the actual package "
                "instead, unless they bring up something new."
            )
        # Same pacing-deadline prescription as lucid.py's /lucid endpoint: if this round has a
        # mandatory concession target, name explicitly whatever hasn't been reached yet in the
        # accumulated package (not just what's mentioned this round).
        pacing_target = lucid._pacing_target(condition_key, turn_number)
        # Prosocial's OPTIONAL deeper step, if this round is late enough to offer it - see
        # lucid._pacing_stretch_target(). Only actually offered to the model below when a
        # genuine concession also happens this same round.
        pacing_stretch_target = lucid._pacing_stretch_target(condition_key, turn_number)
        if any(pacing_target.values()):
            current_by_id = {item['id']: item['status'] for item in current_statuses}
            still_needed = []
            for issue_id, target in pacing_target.items():
                if not target:
                    continue
                current = current_by_id.get(issue_id) or lucid.HOLD_FIRM_ANCHOR[issue_id]
                if lucid._compare_recruiter_value(issue_id, current, target) == 'better':
                    label = 'Salary' if issue_id == 'issue-7' else 'Vacation Time'
                    still_needed.append(f"{label} to {target}")
            if still_needed:
                round_note += (
                    f" Per your concession schedule, by this round you are REQUIRED to have "
                    f"moved {' and '.join(still_needed)} (you have not reached this yet) - do "
                    f"so in this reply, even if the candidate hasn't specifically asked for it."
                )
        # Same general "no free concession" prescription as lucid.py's /lucid endpoint (both
        # conditions, every round) - tells the model BEFORE it drafts a reply whether it has
        # any justification to move something unconditionally, using the
        # genuine_concession_this_round result computed above. Skipped when the first-
        # concession exception is about to grant something this round (computed here, before
        # that block runs, using the same trigger condition it uses) - that's its own,
        # separately-instructed exception and this note would contradict it.
        first_concession_will_fire = (
            condition_key == 'prosocial' and not first_concession_used and genuine_concession_this_round
        )
        if not first_concession_will_fire:
            # Independent checks, not mutually exclusive - a single message can both concede
            # something new AND accept a trade the recruiter proposed last round. Both notes
            # get appended when both are true - same as lucid.py's /lucid endpoint.
            noted_something = False
            if genuine_concession_this_round:
                round_note += (
                    f" The candidate genuinely conceded on {genuine_concession_label} this round "
                    f"(now at {genuine_concession_value}, verified against your payoff schedule). "
                    f"Per your negotiation protocol, you may reciprocate with a proportional move "
                    f"on AT MOST ONE other issue in this reply - do not move anything beyond that "
                    f"without further justification."
                )
                noted_something = True
            if round_concession_check.get('accepts_prior_offer'):
                # Same as lucid.py's /lucid endpoint: the candidate's message doesn't concede
                # anything new itself, but it IS accepting a conditional trade the recruiter
                # proposed in its own previous reply - grounds to follow through, not to hold back.
                round_note += (
                    " The candidate appears to be accepting the conditional trade you proposed "
                    "in your OWN previous message (something like \"if you accept X, I'll offer "
                    "Y\"). If your previous reply named a specific conditional trade, apply that "
                    "exact trade now, unconditionally, in this reply's \"Current package\" recap "
                    "- do not ask them to reconfirm or revert what you already offered. If your "
                    "previous reply did NOT actually name a specific conditional trade, treat "
                    "this the same as no genuine concession this round instead."
                )
                noted_something = True
            if not noted_something:
                round_note += (
                    " The candidate did NOT make a genuine concession this round (per your "
                    "payoff schedule) - per your negotiation protocol, you may NOT move any issue "
                    "unconditionally in this reply. You may still move Salary/Vacation Time if "
                    "your concession schedule separately requires it this round (see above), and "
                    "you may still propose a move CONDITIONALLY (asking for something specific in "
                    "return), but do not agree to or grant anything outright."
                )
            # Prosocial's optional deeper step, rounds 8-10: pacing takes priority in this
            # window specifically, so this is mentioned regardless of whether a genuine
            # concession happened THIS round - not gated on securing something back either.
            if any(pacing_stretch_target.values()):
                prior_by_id_stretch = {item['id']: item['status'] for item in current_statuses}
                stretch_still_available = []
                for issue_id, target in pacing_stretch_target.items():
                    if not target:
                        continue
                    current = prior_by_id_stretch.get(issue_id) or lucid.HOLD_FIRM_ANCHOR[issue_id]
                    if lucid._compare_recruiter_value(issue_id, current, target) == 'better':
                        label = 'Salary' if issue_id == 'issue-7' else 'Vacation Time'
                        stretch_still_available.append(f"{label} to {target}")
                if stretch_still_available:
                    round_note += (
                        f" This is also within your rounds 8-10 window: you may move further "
                        f"toward {' and '.join(stretch_still_available)} in this reply if you "
                        f"judge it appropriate, without needing a fresh concession this specific "
                        f"round to justify it - pacing takes priority here."
                    )
        # Same final-round prescription as lucid.py's /lucid endpoint - the candidate gets no
        # further turn after this one, so a conditional trade proposed here can never actually
        # be confirmed.
        if turn_number >= lucid.TOTAL_ROUNDS:
            round_note += (
                f" This is the FINAL round (round {turn_number} of {lucid.TOTAL_ROUNDS}) - the "
                f"candidate will not get another turn to respond after this one. Do NOT propose "
                f"a new conditional trade that would need their confirmation next round (e.g. "
                f"\"if you accept X, I'll do Y\") - there is no next round for them to confirm "
                f"it, and it would go unresolved. Present your definitive final package instead: "
                f"either hold your current position firmly, or if you're willing to make one "
                f"last move, apply it directly and unconditionally in this reply's \"Current "
                f"package\" recap - do not leave anything pending on the candidate's acceptance."
            )
        messages_for_api = messages + [{'role': 'system', 'content': round_note}]

        # --- Same Prosocial-only first-concession exception as lucid.py's /lucid endpoint.
        # Trigger (a genuine concession) and grant (never Salary/Vacation) are decoupled: the
        # exception consumes on any genuine concession, but if the candidate specifically
        # asked for Salary/Vacation, the model is told to grant a different issue instead. ---
        first_concession_target_issue = None
        if first_concession_will_fire:
            if True:  # extra nesting kept only so the block below didn't need re-indenting
                requested_issue_id = round_concession_check.get('requested_issue_id')
                in_hold_firm_window = turn_number <= lucid.HOLD_FIRM_ROUNDS
                first_concession_used = True  # consumed either way
                if requested_issue_id and requested_issue_id not in ('issue-3', 'issue-7'):
                    # Grantable outright - but capped to ONE grid step toward the candidate,
                    # not a jump straight to whatever they specifically asked for.
                    first_concession_target_issue = requested_issue_id
                    requested_issue_label = next(
                        (item['label'] for item in lucid._default_issue_statuses() if item['id'] == requested_issue_id),
                        requested_issue_id
                    )
                    requested_prior_by_id = {item['id']: item['status'] for item in current_statuses}
                    requested_prior_value = requested_prior_by_id.get(requested_issue_id) or lucid.RECRUITER_OPENING_OFFER.get(requested_issue_id)
                    one_level_value = lucid._one_level_step(requested_issue_id, requested_prior_value)
                    if one_level_value:
                        note = (
                            f"[System note: this is the candidate's first concession this negotiation. "
                            f"Per your one-time first-concession exception, move {requested_issue_label} "
                            f"ONE step in the candidate's favor this reply - specifically to "
                            f"{one_level_value} - unconditionally. Do NOT jump straight to whatever they "
                            f"specifically asked for, even if it's less generous than their request - one "
                            f"step only. Do NOT accept whatever concession they offered in return, even "
                            f"though they offered it - explicitly tell them it isn't needed, and leave "
                            f"every other issue exactly at its current value this round.]"
                        )
                    else:
                        note = (
                            f"[System note: this is the candidate's first concession this negotiation, "
                            f"but {requested_issue_label} is already at its most candidate-favorable "
                            f"value - there's nothing left to move there. As a one-time goodwill gesture "
                            f"instead, pick ONE of your other issues and move it ONE step in the "
                            f"candidate's favor, unconditionally, even if they haven't specifically asked "
                            f"for it.]"
                        )
                        first_concession_target_issue = None
                    print(f"  [first-concession exception triggered on {requested_issue_label}]")
                elif requested_issue_id in ('issue-3', 'issue-7') and in_hold_firm_window:
                    # Only case where the hold-firm framing is actually true.
                    note = (
                        f"[System note: this is the candidate's first concession this negotiation, "
                        f"but you cannot move on Salary or Vacation Time right now (still in your "
                        f"hold-firm window). As a one-time goodwill gesture instead, pick ONE of "
                        f"your other issues ({lucid._alternate_gift_issue_list_text(round_concession_check.get('accepted_issue_id'))}) "
                        f"and move it ONE step in the "
                        f"candidate's favor, unconditionally, in this reply, even if they haven't "
                        f"specifically asked for it - explain you can't move on salary/vacation yet "
                        f"but want to show good faith. Do NOT jump straight to their ideal value on "
                        f"whatever issue you pick - one step only. Do NOT move Salary or Vacation "
                        f"Time.]"
                    )
                    print("  [first-concession exception triggered (hold-firm window - granting an alternate issue instead)]")
                else:
                    # Classifier's ask was unclear, or it was Salary/Vacation but the
                    # hold-firm window already passed - don't claim "still in your
                    # hold-firm window" when that isn't true.
                    note = (
                        f"[System note: this is the candidate's first concession this negotiation. "
                        f"As a one-time goodwill gesture, pick ONE of your other issues "
                        f"({lucid._alternate_gift_issue_list_text(round_concession_check.get('accepted_issue_id'))}) "
                        f"and move it ONE step in the candidate's favor, unconditionally, in "
                        f"this reply, even if they haven't specifically asked for it. Do NOT jump "
                        f"straight to their ideal value on whatever issue you pick - one step only. "
                        f"Keep handling Salary and Vacation Time through your normal concession "
                        f"schedule separately - this one-time gift is on a different issue.]"
                    )
                    print("  [first-concession exception triggered (unclear/out-of-window request - granting an alternate issue instead)]")
                # A separate, deterministic message announcing exactly what got granted will
                # be printed on its own line below, once the grant is confirmed - so tell the
                # model not to write its own prose announcing the specific gift.
                note += (
                    " A separate message announcing this exact gift will be shown to the "
                    "candidate automatically, right before this reply - so do NOT write your "
                    "own sentence announcing or explaining this specific gift in your reply "
                    "text. Still include the correct value in your \"Current package:\" recap "
                    "as usual, and continue the rest of your reply normally."
                )
                messages_for_api.append({'role': 'system', 'content': note})
                first_concession_note_fired = True
            else:
                first_concession_note_fired = False
        else:
            first_concession_note_fired = False

        try:
            reply = call_openai(messages_for_api, api_key, model, temperature)
        except Exception as e:
            print(f"  [error calling OpenAI: {e}]")
            continue

        # --- Same enforcement layers as lucid.py's /lucid endpoint: regenerate once if the
        # reply either conceded too early (hold-firm window) or not enough (pacing deadline) ---
        assistant_updates = lucid._extract_issue_updates_from_message_llm(reply, api_key)
        if turn_number <= lucid.HOLD_FIRM_ROUNDS:
            violated = [
                issue_id for issue_id, anchor in lucid.HOLD_FIRM_ANCHOR.items()
                if issue_id in assistant_updates and not lucid._matches_anchor(assistant_updates[issue_id], anchor)
            ]
            if violated:
                violated_labels = ['Salary' if i == 'issue-7' else 'Vacation Time' for i in violated]
                print(f"  [hold-firm violation on {violated_labels} in round {turn_number}, regenerating]")
                correction_note = (
                    f"[System note: your previous draft reply moved on {' and '.join(violated_labels)}, "
                    f"which violates your hold-firm window (rounds 1-{lucid.HOLD_FIRM_ROUNDS}). Write your "
                    f"reply again: keep Salary at {lucid.HOLD_FIRM_ANCHOR['issue-7']} and Vacation Time at "
                    f"{lucid.HOLD_FIRM_ANCHOR['issue-3']} unchanged this round. You may still respond to the "
                    f"candidate and move any other issue.]"
                )
                retry_text = lucid._call_openai_completion(
                    messages_for_api + [{'role': 'system', 'content': correction_note}],
                    model, temperature, None, api_key
                )
                if retry_text:
                    reply = retry_text
                    assistant_updates = lucid._extract_issue_updates_from_message_llm(reply, api_key)
                else:
                    print("  [regeneration call failed - keeping original (violating) reply]")
        elif any(pacing_target.values()):
            accumulated = lucid.apply_issue_updates(current_statuses, assistant_updates)
            accumulated_by_id = {item['id']: item['status'] for item in accumulated}
            under_conceded = [
                issue_id for issue_id, target in pacing_target.items()
                if target and lucid._compare_recruiter_value(
                    issue_id, accumulated_by_id.get(issue_id) or lucid.HOLD_FIRM_ANCHOR[issue_id], target
                ) == 'better'
            ]
            if under_conceded:
                targets_desc = ', '.join(
                    f"{'Salary' if i == 'issue-7' else 'Vacation Time'} to {pacing_target[i]}"
                    for i in under_conceded
                )
                print(f"  [pacing violation - required concession(s) not yet reached in round {turn_number} ({targets_desc}), regenerating]")
                correction_note = (
                    f"[System note: your previous draft reply did not move {targets_desc}, which your "
                    f"concession schedule requires by this round. Write your reply again: move "
                    f"{targets_desc} in this reply, even if the candidate hasn't specifically asked for "
                    f"it. You may still respond to the candidate and address any other issue.]"
                )
                retry_text = lucid._call_openai_completion(
                    messages_for_api + [{'role': 'system', 'content': correction_note}],
                    model, temperature, None, api_key
                )
                if retry_text:
                    reply = retry_text
                    assistant_updates = lucid._extract_issue_updates_from_message_llm(reply, api_key)
                else:
                    print("  [regeneration call failed - keeping original (under-conceded) reply]")

        # --- Same first-concession grant safety net as lucid.py's /lucid endpoint: verify
        # the reply actually granted the gift AND capped it at one level, regenerate once if not ---
        # Defaults so the Rule 3 mirror below can safely reference these even on a
        # non-first-concession round.
        grant_status = None
        grant_info = None
        if first_concession_note_fired:
            # Excludes accepted_issue_id (mirrors lucid.py): if the exception fired via the
            # candidate accepting a trade the recruiter itself already promised, that issue
            # is already being fulfilled this round, not an eligible NEW gift.
            grant_status, grant_info, grant_unverified_note, assistant_updates = lucid._first_concession_grant_status_with_fallback(
                current_statuses, assistant_updates, first_concession_target_issue, reply,
                exclude_issue_id=round_concession_check.get('accepted_issue_id')
            )
            if grant_unverified_note:
                unverified_trust_notes.append(grant_unverified_note)
            if grant_status != 'ok':
                if grant_status == 'overshoot':
                    overshoot_issue_id, cap_value = grant_info or (None, None)
                    overshoot_label = next(
                        (item['label'] for item in lucid._default_issue_statuses() if item['id'] == overshoot_issue_id),
                        overshoot_issue_id
                    )
                    grant_instruction = (
                        f"your one-time gift moved {overshoot_label} further than the single grid step "
                        f"this exception allows - scale it back to exactly {cap_value}, not the "
                        f"candidate's full ask"
                    )
                elif first_concession_target_issue:
                    grant_label = next(
                        (item['label'] for item in lucid._default_issue_statuses() if item['id'] == first_concession_target_issue),
                        first_concession_target_issue
                    )
                    grant_instruction = f"grant your one-time, one-step gift on {grant_label}"
                else:
                    grant_instruction = (
                        f"pick ONE of your other issues "
                        f"({lucid._alternate_gift_issue_list_text(round_concession_check.get('accepted_issue_id'))}) "
                        f"and grant your one-time, one-step gift on it"
                    )
                print(f"  [first-concession grant not honored ({grant_status}), regenerating]")
                correction_note = (
                    f"[System note: your previous draft reply did not correctly grant your one-time "
                    f"first-concession gift. Write your reply again: {grant_instruction}, "
                    f"unconditionally, in this reply.]"
                )
                retry_text = lucid._call_openai_completion(
                    messages_for_api + [{'role': 'system', 'content': correction_note}],
                    model, temperature, None, api_key
                )
                if retry_text:
                    reply = retry_text
                    assistant_updates = lucid._extract_issue_updates_from_message_llm(reply, api_key)
                    recheck_status, recheck_info, recheck_unverified_note, assistant_updates = lucid._first_concession_grant_status_with_fallback(
                        current_statuses, assistant_updates, first_concession_target_issue, reply,
                        exclude_issue_id=round_concession_check.get('accepted_issue_id')
                    )
                    if recheck_unverified_note:
                        unverified_trust_notes.append(recheck_unverified_note)
                    if recheck_status != 'ok':
                        print(f"  [first-concession grant still not honored ({recheck_status}) after regeneration - keeping it, not retrying again]")
                    else:
                        grant_status, grant_info = recheck_status, recheck_info
                else:
                    print("  [first-concession regeneration call failed - keeping original reply]")

            # grant_status/grant_info now reflect the FINAL state - build the deterministic
            # announcement only when a specific issue/value was actually confirmed granted.
            if grant_status == 'ok' and grant_info:
                granted_issue_id, granted_value = grant_info
                granted_label = next(
                    (item['label'] for item in lucid._default_issue_statuses() if item['id'] == granted_issue_id),
                    granted_issue_id
                )
                first_concession_announcement = (
                    f"Thanks for your concession — as a reciprocal gesture, I'd like to give you "
                    f"a free move on {granted_label}: {granted_value}, no need to give anything "
                    f"in return."
                )

        # --- Same reciprocity-claim safety net as lucid.py's /lucid endpoint: every round,
        # both conditions - verify any "since you offered X, I'll reciprocate" claim against
        # the real payoff table before letting the reciprocal grant stand ---
        reciprocity_check = lucid._detect_reciprocity_claim_llm(reply, api_key)
        if reciprocity_check.get('claims_reciprocity'):
            credited_issue_id = reciprocity_check.get('credited_issue_id')
            credited_value = reciprocity_check.get('credited_value')
            # Mirrors lucid.py: cross-check against round_concession_check's OWN
            # accepted_issue_id - if the classifier is crediting the candidate with that SAME
            # issue, it's almost certainly the same directional misread (crediting the
            # recruiter's own grant TO the candidate as if conceded FROM them), just in an
            # implicit framing. Don't trust it.
            if credited_issue_id and credited_issue_id == round_concession_check.get('accepted_issue_id'):
                print(f"  [reciprocity claim ({credited_issue_id}->{credited_value}) matches this round's own accepted_issue_id - not trusting it]")
                credited_issue_id = None
                credited_value = None
            if credited_issue_id and credited_value:
                credited_prior_by_id = {item['id']: item['status'] for item in current_statuses}
                credited_prior_value = credited_prior_by_id.get(credited_issue_id) or lucid.RECRUITER_OPENING_OFFER.get(credited_issue_id)
                credited_direction = lucid._compare_recruiter_value(credited_issue_id, credited_value, credited_prior_value)
                if credited_direction in ('worse', 'same'):
                    credited_label = next(
                        (item['label'] for item in lucid._default_issue_statuses() if item['id'] == credited_issue_id),
                        credited_issue_id
                    )
                    print(f"  [reciprocity claim invalid - {credited_issue_id}->{credited_value} is {credited_direction.upper()} (not better) for the recruiter, regenerating]")
                    # Name the EXACT value to revert to, rather than an abstract "don't treat
                    # that as a concession" - a concrete target is more likely to actually
                    # change the reply than a vague prohibition.
                    correction_note = (
                        f"[System note: your previous draft reply credited the candidate with a "
                        f"concession on {credited_label} ({credited_value}) and reciprocated based on "
                        f"that - but per your payoff schedule, that value is NOT actually favorable to "
                        f"you compared to your current position on {credited_label}, so it isn't a real "
                        f"concession. Write your reply again: revert {credited_label} back to exactly "
                        f"{credited_prior_value} in this reply, and do not reciprocate based on that "
                        f"claim. You may still make a move on a DIFFERENT issue this reply if it's "
                        f"justified some other way (your own concession schedule, or a genuine "
                        f"concession the candidate made elsewhere), but not on {credited_label}.]"
                    )
                    retry_text = lucid._call_openai_completion(
                        messages_for_api + [{'role': 'system', 'content': correction_note}],
                        model, temperature, None, api_key
                    )
                    if retry_text:
                        reply = retry_text
                        assistant_updates = lucid._extract_issue_updates_from_message_llm(reply, api_key)
                        recheck = lucid._detect_reciprocity_claim_llm(reply, api_key)
                        still_invalid = False
                        if recheck.get('claims_reciprocity'):
                            rc_issue = recheck.get('credited_issue_id')
                            rc_value = recheck.get('credited_value')
                            if rc_issue and rc_value:
                                rc_prior = credited_prior_by_id.get(rc_issue) or lucid.RECRUITER_OPENING_OFFER.get(rc_issue)
                                still_invalid = lucid._compare_recruiter_value(rc_issue, rc_value, rc_prior) in ('worse', 'same')
                        if still_invalid:
                            print("  [reciprocity claim still invalid after regeneration - keeping it, not retrying again]")
                    else:
                        print("  [reciprocity-claim regeneration call failed - keeping original reply]")
                elif credited_direction == 'unknown':
                    unverified_trust_notes.append(
                        f"Reciprocity claim: reply credited the candidate with {credited_issue_id} -> "
                        f"'{credited_value}', but that value couldn't be matched against the payoff "
                        f"table (unknown) - allowed the reciprocal grant to stand."
                    )
            else:
                unverified_trust_notes.append(
                    "Reciprocity claim: reply claimed reciprocity but didn't credit a specific "
                    "issue/value to verify - allowed to stand."
                )

        # --- Same final-audit safety net as lucid.py's /lucid endpoint (both conditions,
        # every round) - runs LAST, after grant/reciprocity. Merges FIVE invariants:
        #   1. hold-firm (rounds 1-HOLD_FIRM_ROUNDS): Salary/Vacation must still exactly
        #      match HOLD_FIRM_ANCHOR - no exceptions, not even a genuine concession. Runs
        #      every round, REGARDLESS of first_concession_will_fire - the exception never
        #      grants directly on Salary/Vacation, so there's no legitimate conflict.
        #   2. pacing minimum: once at/past a pacing deadline, Salary/Vacation must still
        #      be at least at pacing_target. Also unconditional on first_concession_will_fire.
        #   3. no free concession: no OTHER issue may have moved in the candidate's favor
        #      without a genuine concession this round - runs on a first-concession round
        #      too now, exempting only the CONFIRMED gift issue (not the whole round).
        #   4. promised trade landed: if the candidate accepted a trade the recruiter
        #      itself promised, verify the exact promised value actually shipped.
        #   5. no unconfirmed proposal shipped as applied: the recap must not show a value
        #      the reply's own prose frames as still pending acceptance - the only rule
        #      here an LLM has to judge, since a hypothetical and a confirmed value are
        #      identical on the payoff table.
        # Rules 1/2 exist here IN ADDITION to the dedicated hold-firm/pacing safety nets
        # above (which only ever run once) because a LATER safety net's regeneration
        # (grant/reciprocity) can silently undo either fix - observed in testing for both.
        # Re-audits FULLY after every regeneration (up to 2 attempts) instead of only
        # rechecking the originally-flagged issues - also observed in testing: a
        # regeneration meant to fix one thing can volunteer an entirely different,
        # previously-clean violation that a narrower recheck would miss.
        def _audit_final_state(current_assistant_updates):
            accumulated = lucid.apply_issue_updates(current_statuses, current_assistant_updates)
            accumulated_by_id = {item['id']: item['status'] for item in accumulated}
            prior_by_id = {item['id']: item['status'] for item in current_statuses}

            hf_violations = []
            if turn_number <= lucid.HOLD_FIRM_ROUNDS:
                for issue_id, anchor in lucid.HOLD_FIRM_ANCHOR.items():
                    current_val = accumulated_by_id.get(issue_id) or lucid.RECRUITER_OPENING_OFFER.get(issue_id)
                    if not lucid._matches_anchor(current_val, anchor):
                        label = 'Salary' if issue_id == 'issue-7' else 'Vacation Time'
                        hf_violations.append((issue_id, label, anchor))

            pc_violations = []
            if turn_number > lucid.HOLD_FIRM_ROUNDS:
                for issue_id, target in pacing_target.items():
                    if not target:
                        continue
                    current_val = accumulated_by_id.get(issue_id) or lucid.HOLD_FIRM_ANCHOR.get(issue_id)
                    if lucid._compare_recruiter_value(issue_id, current_val, target) == 'better':
                        label = 'Salary' if issue_id == 'issue-7' else 'Vacation Time'
                        pc_violations.append((issue_id, label, target))

            # Mirrors lucid.py: runs every round, including a first-concession round - used
            # to be skipped entirely whenever first_concession_note_fired, but a LATER safety
            # net's regeneration (reciprocity) can resample fresh and volunteer brand-new,
            # completely ungrounded moves unrelated to the confirmed gift, and a blanket
            # round-wide skip meant nothing would ever catch those.
            ug_moves = []
            already_flagged_ids = {v[0] for v in hf_violations} | {v[0] for v in pc_violations}
            for item in lucid._default_issue_statuses():
                issue_id = item['id']
                if issue_id in already_flagged_ids:
                    continue
                new_val = accumulated_by_id.get(issue_id) or lucid.RECRUITER_OPENING_OFFER.get(issue_id)
                old_val = prior_by_id.get(issue_id) or lucid.RECRUITER_OPENING_OFFER.get(issue_id)
                if lucid._compare_recruiter_value(issue_id, new_val, old_val) != 'worse':
                    continue  # didn't move in the candidate's favor
                pacing_step = pacing_target.get(issue_id)
                if pacing_step and lucid._compare_recruiter_value(issue_id, new_val, pacing_step) != 'worse':
                    continue  # within what the pacing schedule itself mandates this round
                stretch_step = pacing_stretch_target.get(issue_id)
                if stretch_step and lucid._compare_recruiter_value(issue_id, new_val, stretch_step) != 'worse':
                    continue  # within Prosocial's optional stretch ceiling (rounds 8-10 only) -
                    # pacing takes priority in this window, not gated on a fresh concession
                if first_concession_note_fired and grant_status == 'ok' and grant_info and issue_id == grant_info[0]:
                    continue  # the CONFIRMED one-time first-concession gift issue this round -
                    # already verified separately, not an ungrounded move
                # Mirrors lucid.py: a genuine concession / accepted prior offer only ever
                # justifies ONE specific issue moving for free, not every issue that moves
                # this round - narrowed to the classifier's own tied issue, or any issue
                # whose new value is actually mentioned in the reply's prose (not just the
                # recap table).
                if genuine_concession_this_round and (
                    issue_id == round_concession_check.get('requested_issue_id')
                    or lucid._value_mentioned_in_prose(reply, new_val)
                ):
                    continue
                if round_concession_check.get('accepts_prior_offer') and (
                    issue_id == round_concession_check.get('accepted_issue_id')
                    or lucid._value_mentioned_in_prose(reply, new_val)
                ):
                    continue  # candidate accepted a trade the recruiter itself already
                    # proposed - not a free giveaway, it's the recruiter following through
                ug_moves.append((issue_id, item['label'], old_val))

            # Rule 4: same as lucid.py's /lucid endpoint - if the candidate accepted a
            # specific trade the recruiter itself promised last round, verify that exact
            # value actually landed, not just that nothing moved for free.
            po_violations = []
            accepted_issue_id = round_concession_check.get('accepted_issue_id')
            accepted_value = round_concession_check.get('accepted_value')
            if accepted_issue_id and accepted_value:
                po_status, po_actual = lucid._prior_offer_landed_status(
                    current_assistant_updates, accepted_issue_id, accepted_value, reply
                )
                if po_status in ('under', 'missing'):
                    po_label = next(
                        (item['label'] for item in lucid._default_issue_statuses() if item['id'] == accepted_issue_id),
                        accepted_issue_id
                    )
                    po_violations.append((accepted_issue_id, po_label, accepted_value))

            # Rule 5: mirrors lucid.py - does the recap claim something has HAPPENED that
            # this reply's own prose frames as still pending the candidate's acceptance?
            # Can't be verified against the payoff table (a hypothetical value and a
            # confirmed value are identical there) - needs an LLM read of the prose.
            uc_violations = []
            already_flagged_for_uc = already_flagged_ids | {v[0] for v in ug_moves}
            unconfirmed_check = lucid._detect_unconfirmed_recap_values_llm(reply, api_key)
            for issue_id in unconfirmed_check.get('unconfirmed_issue_ids', []):
                if issue_id in already_flagged_for_uc:
                    continue
                new_val = accumulated_by_id.get(issue_id) or lucid.RECRUITER_OPENING_OFFER.get(issue_id)
                old_val = prior_by_id.get(issue_id) or lucid.RECRUITER_OPENING_OFFER.get(issue_id)
                if new_val == old_val:
                    continue  # nothing actually changed on this issue - nothing to revert
                # Mirrors lucid.py: if this move doesn't go beyond what the recruiter's OWN
                # concession schedule already mandates by this round, it isn't something the
                # candidate needs to confirm - same exemption Rule 3 already uses.
                pacing_step = pacing_target.get(issue_id)
                if pacing_step and lucid._compare_recruiter_value(issue_id, new_val, pacing_step) != 'worse':
                    continue
                stretch_step = pacing_stretch_target.get(issue_id)
                if stretch_step and lucid._compare_recruiter_value(issue_id, new_val, stretch_step) != 'worse':
                    continue
                uc_label = next(
                    (item['label'] for item in lucid._default_issue_statuses() if item['id'] == issue_id),
                    issue_id
                )
                uc_violations.append((issue_id, uc_label, old_val))

            return hf_violations, pc_violations, ug_moves, po_violations, uc_violations

        hold_firm_violations, pacing_violations, ungrounded_moves, prior_offer_violations, unconfirmed_violations = _audit_final_state(assistant_updates)
        final_audit_attempts = 0
        while (hold_firm_violations or pacing_violations or ungrounded_moves or prior_offer_violations or unconfirmed_violations) and final_audit_attempts < 2:
            final_audit_attempts += 1
            note_parts = []
            if prior_offer_violations:
                targets_desc = ', '.join(f"{label} to exactly {value}" for _, label, value in prior_offer_violations)
                print(f"  [promised trade didn't land in the final check ({targets_desc}), regenerating (attempt {final_audit_attempts})]")
                note_parts.append(
                    f"the candidate accepted the conditional trade you proposed in your "
                    f"own previous message, but your draft reply still doesn't reflect it "
                    f"- set {targets_desc} in this reply's \"Current package\" recap, "
                    f"unconditionally, exactly as you promised"
                )
            if ungrounded_moves:
                targets_desc = ', '.join(f"{label} back to {old_val}" for _, label, old_val in ungrounded_moves)
                print(f"  [ungrounded free concession(s) with no genuine candidate concession this round ({targets_desc}), regenerating (attempt {final_audit_attempts})]")
                note_parts.append(
                    f"the candidate did not make a genuine concession this round, but "
                    f"your previous draft reply moved {targets_desc} anyway. Per your "
                    f"negotiation protocol, never move an issue for free - revert "
                    f"{targets_desc} in this reply. You may still propose a move "
                    f"conditionally, asking for something specific in return, but do "
                    f"not grant it outright"
                )
            if hold_firm_violations:
                violated_labels = ', '.join(label for _, label, _ in hold_firm_violations)
                targets_desc = ', '.join(f"{label} to exactly {anchor}" for _, label, anchor in hold_firm_violations)
                print(f"  [hold-firm violation surviving to the final check ({targets_desc}), regenerating (attempt {final_audit_attempts})]")
                note_parts.append(
                    f"your previous draft reply still has {violated_labels} moved away "
                    f"from its hold-firm anchor - this violates your hold-firm window "
                    f"(rounds 1-{lucid.HOLD_FIRM_ROUNDS}), which allows NO exceptions, not "
                    f"even for a genuine concession this round. Set {targets_desc} in "
                    f"this reply, unconditionally"
                )
            if pacing_violations:
                targets_desc = ', '.join(f"{label} to at least {target}" for _, label, target in pacing_violations)
                print(f"  [pacing minimum not met in the final check ({targets_desc}), regenerating (attempt {final_audit_attempts})]")
                note_parts.append(
                    f"your previous draft reply still has not reached {targets_desc}, "
                    f"which your concession schedule mandates by this round - move "
                    f"{targets_desc} in this reply, even if the candidate hasn't "
                    f"specifically asked for it"
                )
            if unconfirmed_violations:
                targets_desc = ', '.join(f"{label} back to {old_val}" for _, label, old_val in unconfirmed_violations)
                print(f"  [recap shows a not-yet-accepted proposal as if applied ({targets_desc}), regenerating (attempt {final_audit_attempts})]")
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
            note_parts.append("keep everything else in your reply exactly as it is")
            correction_note = "[System note: " + "; also, ".join(note_parts) + ".]"
            retry_text = lucid._call_openai_completion(
                messages_for_api + [{'role': 'system', 'content': correction_note}],
                model, temperature, None, api_key
            )
            if not retry_text:
                print("  [final-check regeneration call failed - keeping current reply]")
                break
            reply = retry_text
            assistant_updates = lucid._extract_issue_updates_from_message_llm(reply, api_key)
            hold_firm_violations, pacing_violations, ungrounded_moves, prior_offer_violations, unconfirmed_violations = _audit_final_state(assistant_updates)

        if hold_firm_violations or pacing_violations or ungrounded_moves or prior_offer_violations or unconfirmed_violations:
            still_bad = (
                [label for _, label, _ in hold_firm_violations]
                + [label for _, label, _ in pacing_violations]
                + [label for _, label, _ in ungrounded_moves]
                + [label for _, label, _ in prior_offer_violations]
                + [label for _, label, _ in unconfirmed_violations]
            )
            print(f"  [still not resolved on {still_bad} after {final_audit_attempts} final-check regeneration(s) - keeping it, not retrying again]")

        current_statuses = lucid.apply_issue_updates(current_statuses, assistant_updates)
        messages.append({'role': 'assistant', 'content': reply})
        # Printed as its own separate line, before the main reply, mirroring the two-bubble
        # split the real Qualtrics frontend renders - not folded into "Alex: ...".
        if first_concession_announcement:
            print(f"\nAlex (gift, round {turn_number}): {first_concession_announcement}")
        print(f"\nAlex (round {turn_number}): {reply}\n")
        # Printed as its own separate block, apart from Alex's reply, per request - so any
        # unverified-trust edge case is easy to spot while testing.
        if unverified_trust_notes:
            print(f"--- unverified trust (round {turn_number}) ---")
            for note in unverified_trust_notes:
                print(f"  * {note}")
            print("---")


if __name__ == '__main__':
    main()
