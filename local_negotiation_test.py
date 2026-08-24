# -*- coding: utf-8 -*-
"""
Standalone local test harness for the LUCID recruiter prompts (prompts.yaml).

Lets you chat with the Prosocial or Proself recruiter AI directly from the terminal,
without needing Flask, Vercel, or Qualtrics running. Reuses the exact same round-number
injection and Prosocial first-concession detection as lucid.py's real /lucid endpoint
(so pacing/exception behavior is faithful to production), but skips everything that's
only for data collection - no issue-status extraction, no offer-trajectory logging, no
Embedded Data. This is purely for eyeballing "does the prompt actually behave the way
we designed it" - nothing here is saved anywhere.

Setup:
    1. Put your OpenAI API key in .env (already gitignored):
           OPENAI_API_KEY=sk-...
    2. Run:
           python3 local_negotiation_test.py
    3. Pick a condition, then type candidate messages. Type "quit" to stop.

Optional env vars (set in .env or the shell):
    LUCID_TEST_MODEL         - defaults to gpt-4o (matches lucid.py's default)
    LUCID_TEST_TEMPERATURE   - defaults to 1.0 (matches lucid.py's default)
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
    resp = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'},
        json={'model': model, 'messages': messages, 'temperature': temperature},
        timeout=30
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
        messages_for_api = messages + [{'role': 'system', 'content': round_note}]

        # --- Same Prosocial-only first-concession exception as lucid.py's /lucid endpoint.
        # Trigger (any concession) and grant (never Salary/Vacation) are decoupled: the
        # exception consumes on ANY detected concession, but if the candidate specifically
        # asked for Salary/Vacation, the model is told to grant a different issue instead. ---
        first_concession_target_issue = None
        if condition_key == 'prosocial' and not first_concession_used:
            check = lucid._detect_first_concession_llm(user_message, api_key)
            # Cross-check the classifier's framing against the real payoff table before
            # trusting it - "sounds like a concession" isn't the same as "actually favorable
            # to the recruiter" (e.g. an earlier start date reads like a concession but
            # scores worse for the recruiter on the real table).
            if check.get('is_concession'):
                conceded_issue_id = check.get('conceded_issue_id')
                conceded_new_value = check.get('conceded_new_value')
                if conceded_issue_id and conceded_new_value:
                    prior_by_id_cc = {item['id']: item['status'] for item in current_statuses}
                    conceded_prior_value = prior_by_id_cc.get(conceded_issue_id) or lucid.RECRUITER_OPENING_OFFER.get(conceded_issue_id)
                    if lucid._compare_recruiter_value(conceded_issue_id, conceded_new_value, conceded_prior_value) == 'worse':
                        print(f"  [first-concession classifier flagged {conceded_issue_id}->{conceded_new_value} as a concession, but payoff table says it's WORSE for the recruiter - overriding to not-a-concession]")
                        check['is_concession'] = False
            if check.get('is_concession'):
                requested_issue_id = check.get('requested_issue_id')
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
                        f"your other issues (Bonus, Job Assignment, Insurance Coverage, Starting "
                        f"Date, Moving Expense Coverage, or Location) and move it ONE step in the "
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
                        f"As a one-time goodwill gesture, pick ONE of your other issues (Bonus, Job "
                        f"Assignment, Insurance Coverage, Starting Date, Moving Expense Coverage, or "
                        f"Location) and move it ONE step in the candidate's favor, unconditionally, in "
                        f"this reply, even if they haven't specifically asked for it. Do NOT jump "
                        f"straight to their ideal value on whatever issue you pick - one step only. "
                        f"Keep handling Salary and Vacation Time through your normal concession "
                        f"schedule separately - this one-time gift is on a different issue.]"
                    )
                    print("  [first-concession exception triggered (unclear/out-of-window request - granting an alternate issue instead)]")
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
        if first_concession_note_fired:
            grant_status, grant_info = lucid._first_concession_grant_status(
                current_statuses, assistant_updates, first_concession_target_issue
            )
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
                        "pick ONE of your other issues (Bonus, Job Assignment, Insurance Coverage, "
                        "Starting Date, Moving Expense Coverage, or Location) and grant your one-time, "
                        "one-step gift on it"
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
                    recheck_status, _ = lucid._first_concession_grant_status(
                        current_statuses, assistant_updates, first_concession_target_issue
                    )
                    if recheck_status != 'ok':
                        print(f"  [first-concession grant still not honored ({recheck_status}) after regeneration - keeping it, not retrying again]")
                else:
                    print("  [first-concession regeneration call failed - keeping original reply]")

        current_statuses = lucid.apply_issue_updates(current_statuses, assistant_updates)
        messages.append({'role': 'assistant', 'content': reply})
        print(f"\nAlex (round {turn_number}): {reply}\n")


if __name__ == '__main__':
    main()
