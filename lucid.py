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
import re      # For regex-based offer extraction

# Initialize the Flask application
app = Flask(__name__)

# --- Offer Extraction Functions ---

def _empty_offer_result(message, role='assistant'):
    return {
        'has_offer': False,
        'role': role,
        'raw_message': message,
        'keywords': [],
        'numeric_values': [],
        'offer_phrases': [],
        'extraction_method': 'ai_only'
    }

def extract_offers_from_message(message, role='user'):
    """
    Extracts negotiation offers from a message using AI analysis.
    Uses AI-only extraction. If the AI call fails or no API key is configured,
    returns an empty result so only model-derived offers are surfaced.
    
    Returns a dictionary with extracted offer components:
    {
        'has_offer': bool,
        'raw_message': str,
        'offer_summary': str,
        'numeric_values': list,
        'keywords': list,
        'extraction_method': 'ai'
    }
    """
    try:
        # Get OpenAI API key
        openai_api_key = (
            os.getenv('OPENAI_API_KEY') or
            os.getenv('openai_api_key')
        )
        
        if not openai_api_key:
            print("[INFO] OpenAI API key not available, skipping offer extraction")
            return _empty_offer_result(message, role)
        
        # Call OpenAI to extract offers
        openai_url = 'https://api.openai.com/v1/chat/completions'
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}'
        }
        
        # System prompt to instruct LLM on offer extraction
        system_prompt = """You are an expert negotiation analyst. Extract and summarize any negotiation offers, proposals, bids, or counter-offers from the given message.

Respond in JSON format with:
{
    "has_offer": true/false,
    "offer_summary": "brief summary of the offer or empty string",
    "numeric_values": ["$500", "20%", etc],
    "keywords": ["offer", "propose", "price", etc],
    "negotiation_elements": ["any special terms, conditions, or constraints mentioned"]
}

Focus on understanding the intent and meaning, not just keyword matching. Consider implicit offers and nuanced language."""
        
        payload = {
            'model': 'gpt-4o-mini',  # Use a faster model for quick extraction
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': message}
            ],
            'temperature': 0.3,  # Lower temp for more consistent extraction
            'max_tokens': 500
        }
        
        response_openai = requests.post(openai_url, headers=headers, json=payload, timeout=10)
        
        if response_openai.status_code == 200:
            resp_json = response_openai.json()
            response_text = resp_json['choices'][0]['message']['content']
            
            # Parse the JSON response from the AI
            extracted_data = json.loads(response_text)
            
            # Ensure all expected fields are present
            result = {
                'has_offer': extracted_data.get('has_offer', False),
                'role': role,
                'raw_message': message,
                'offer_summary': extracted_data.get('offer_summary', ''),
                'numeric_values': extracted_data.get('numeric_values', []),
                'keywords': extracted_data.get('keywords', []),
                'offer_phrases': [extracted_data.get('offer_summary', '')] if extracted_data.get('offer_summary') else [],
                'extraction_method': 'ai'
            }
            
            return result
        else:
            # If API call fails, fall back to rule-based
            print(f"[INFO] OpenAI API error ({response_openai.status_code}), skipping offer extraction")
            return _empty_offer_result(message, role)
            
    except (json.JSONDecodeError, KeyError, requests.exceptions.RequestException, Exception) as e:
        # Any error → return an empty result so only AI-derived offers are shown
        print(f"[INFO] Error in AI extraction ({type(e).__name__}), skipping offer extraction: {e}")
        return _empty_offer_result(message, role)

def get_latest_offers(conversation_history):
    """
    Analyzes conversation history to extract the latest offers from both user and assistant.
    Returns a structured summary of the most recent offers from each side.
    """
    assistant_offers = []
    
    for msg in conversation_history:
        if msg.get('role') == 'assistant':
            offer_data = extract_offers_from_message(msg.get('content', ''), 'assistant')
            if offer_data['has_offer']:
                assistant_offers.append(offer_data)
    
    # Get the latest offer from each side
    latest_assistant_offer = assistant_offers[-1] if assistant_offers else None
    
    return {
        'user_latest_offer': None,
        'assistant_latest_offer': latest_assistant_offer,
        'user_offer_count': 0,
        'assistant_offer_count': len(assistant_offers),
    }


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
        "You extract structured negotiation issue updates from ONE assistant message. "
        "The message may contain markdown (**bold**), numbering, bullet points, or compact formatting. "
        "Identify ONLY issues explicitly updated in this message, and ignore issues not updated here. "
        "Issue IDs and labels are: "
        "issue-1 Bonus, issue-2 Job Assignment, issue-3 Vacation Time, issue-4 Starting Date, "
        "issue-5 Moving Expense Coverage, issue-6 Insurance Coverage, issue-7 Salary, issue-8 Location. "
        "Return ONLY valid JSON in this exact shape: "
        "{\"updates\":[{\"id\":\"issue-1\",\"label\":\"Bonus\",\"status\":\"4%\"}]}. "
        "Use ids whenever possible. Preserve exact values from the message (e.g., Division A, Plan E, August 1, $82,000, 60%). "
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
        'max_tokens': 400,
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


def extract_issue_statuses_from_history(conversation_history, openai_api_key=None):
    """
    Analyze assistant messages in order and incrementally update issue slots.
    Only issues explicitly updated in a message are changed; all other slots are kept
    as-is. This prevents unrelated slots from being wiped when a new message only
    discusses one or two issues.

    Returned format:
    [
      {"id": "issue-1", "label": "Bonus", "status": "Open"},
      ... (8 items)
    ]
    """
    defaults = _default_issue_statuses()
    merged = [dict(item) for item in defaults]
    id_to_index = {item['id']: idx for idx, item in enumerate(merged)}

    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY') or os.getenv('openai_api_key')

    try:
        for msg in conversation_history:
            if msg.get('role') != 'assistant':
                continue

            content = msg.get('content', '')

            # LLM-only extraction.
            updates = _extract_issue_updates_from_message_llm(content, openai_api_key)

            for issue_id, status in updates.items():
                idx = id_to_index.get(issue_id)
                if idx is None:
                    continue
                if status:
                    merged[idx]['status'] = status

        return merged

    except Exception as e:
        print(f"[INFO] Exception during issue-status extraction: {e}")
        return defaults

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

                # --- Step 4: Call OpenAI API ---
                openai_url = 'https://api.openai.com/v1/chat/completions'
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {openai_api_key}' # Use API key for authorization
                }
                # Construct payload for OpenAI
                data_payload = {
                    'model': model,
                    'messages': messages,
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

                        # Extract offers from the conversation INCLUDING the new AI response (NEGOTIATION FEATURE)
                        # This ensures offers from the AI's latest response are immediately visible
                        messages_with_ai_response = messages + [{'role': 'assistant', 'content': generated_text}]
                        offers_data = get_latest_offers(messages_with_ai_response)

                        # Prepare the successful response data for Qualtrics frontend
                        response_data = {
                            'generated_text': generated_text,
                            'used_temperature': used_temperature, # Echo back parameters used
                            'offers': offers_data  # Include extracted offer data
                        }
                        # Also extract issue statuses (AI-driven when API key present; otherwise defaults)
                        try:
                            issue_statuses = extract_issue_statuses_from_history(messages_with_ai_response, openai_api_key)
                            response_data['issue_statuses'] = issue_statuses
                        except Exception as e:
                            print(f"[WARN /lucid] Failed to extract issue_statuses: {e}")
                            response_data['issue_statuses'] = []
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
