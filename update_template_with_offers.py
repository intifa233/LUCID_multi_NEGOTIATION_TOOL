#!/usr/bin/env python3
"""
Helper script to add issue status display HTML and JavaScript to the LUCID Negotiation Template.
Properly handles JSON escaping when modifying the .qsf file.
"""
import json
import sys
import os

# Path to the template file — defaults to the local directory; override via command-line argument
TEMPLATE_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "LUCID_Negotiation_Template.qsf")

# Load the template
with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find QID106 (the LUCID_TOOL question)
for element in data['SurveyElements']:
    if (element.get('Element') == 'SQ' and 
        element.get('PrimaryAttribute') == 'QID106'):
        
        payload = element['Payload']
        
        # ===== UPDATE QUESTION TEXT (HTML) =====
        old_html = payload['QuestionText']
        
        # The new layout will have two side-by-side columns at the root of chat-container.
        # Left column: Messages + Chat Input.
        # Right column: 8-issue Sidebar.
        
        issues_sidebar_html = '''
<div id="issue-status-sidebar" style="width: 280px; overflow-y: auto; box-sizing: border-box; padding-left: 10px; border-left: 1px solid #eaeaea;">
<div id="issue-status-panel" style="display:block;">
<div style="font-weight: bold; margin-bottom: 10px;">📋 AI offer</div>
<div id="issue-status-list" style="display: flex; flex-direction: column; gap: 10px;">
<div class="issue-status-card" data-issue-id="issue-1" style="padding: 10px; background-color: #fafafa; border: 1px solid #cfd8dc; border-radius: 6px;">
<div style="font-weight: bold; margin-bottom: 4px;">Bonus</div><div class="issue-status-value"></div>
</div>
<div class="issue-status-card" data-issue-id="issue-2" style="padding: 10px; background-color: #fafafa; border: 1px solid #cfd8dc; border-radius: 6px;">
<div style="font-weight: bold; margin-bottom: 4px;">Job Assignment</div><div class="issue-status-value"></div>
</div>
<div class="issue-status-card" data-issue-id="issue-3" style="padding: 10px; background-color: #fafafa; border: 1px solid #cfd8dc; border-radius: 6px;">
<div style="font-weight: bold; margin-bottom: 4px;">Vacation Time</div><div class="issue-status-value"></div>
</div>
<div class="issue-status-card" data-issue-id="issue-4" style="padding: 10px; background-color: #fafafa; border: 1px solid #cfd8dc; border-radius: 6px;">
<div style="font-weight: bold; margin-bottom: 4px;">Starting Date</div><div class="issue-status-value"></div>
</div>
<div class="issue-status-card" data-issue-id="issue-5" style="padding: 10px; background-color: #fafafa; border: 1px solid #cfd8dc; border-radius: 6px;">
<div style="font-weight: bold; margin-bottom: 4px;">Moving Expense Coverage</div><div class="issue-status-value"></div>
</div>
<div class="issue-status-card" data-issue-id="issue-6" style="padding: 10px; background-color: #fafafa; border: 1px solid #cfd8dc; border-radius: 6px;">
<div style="font-weight: bold; margin-bottom: 4px;">Insurance Coverage</div><div class="issue-status-value"></div>
</div>
<div class="issue-status-card" data-issue-id="issue-7" style="padding: 10px; background-color: #fafafa; border: 1px solid #cfd8dc; border-radius: 6px;">
<div style="font-weight: bold; margin-bottom: 4px;">Salary</div><div class="issue-status-value"></div>
</div>
<div class="issue-status-card" data-issue-id="issue-8" style="padding: 10px; background-color: #fafafa; border: 1px solid #cfd8dc; border-radius: 6px;">
<div style="font-weight: bold; margin-bottom: 4px;">Location</div><div class="issue-status-value"></div>
</div>
</div>
</div>
</div>'''

        # We will parse the old_html carefully. We know it starts with <div id="chat-container" ...>
        # To establish a strict two-column layout:
        
        # 1. Update chat-container to be a flex-row container
        new_html = old_html.replace(
            '<div id="chat-container" style="display: flex; flex-direction: column;',
            '<div id="chat-container" style="display: flex; flex-direction: row;'
        )
        
        # 2. Extract message-container inner HTML
        msg_start = new_html.find('<div id="message-container"')
        # It usually ends right when <div id="offers-sidebar" or <div id="issue-status-sidebar" starts, 
        # but let's grab to the end of its div.
        # But wait, there is a container wrapper: <div style="display: flex; gap: 15px; flex: 1; min-height: 0;">
        # We should just hardcode the entire structure to be safe and clean!

        message_container_html = '<div id="message-container" style="flex: 1; overflow-y: auto; overflow-x: hidden; padding: 10px; box-sizing: border-box; border-radius: 5px; background-color: #f9f9f9; border: 1px solid #e0e0e0;">&nbsp;</div>'
        
        form_html = '''<form id="chat-form" style="display: flex; align-items: stretch; gap: 10px; width: 100%; margin-top: auto; padding-top: 10px;"><textarea rows="2" placeholder="Type your message..." maxlength="1024" id="message-input" style="flex: 1; resize: none; border: 1px solid #ccc; border-radius: 4px; padding: 8px; font-family: inherit; font-size: inherit;"></textarea><button type="submit" class="sbutton"><svg xmlns="http://www.w3.org/2000/svg" width="1.5em" viewBox="0 0 24 24" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" stroke="currentColor" height="1.5em" fill="none"><polyline points="18 15 12 9 6 15"></polyline></svg></button></form>'''

        full_new_structure = f'''<div id="chat-container" style="display: flex; flex-direction: row; width: 98%; height: 650px; margin: 0 auto; padding: 15px; background-color: white; border-radius: 10px; box-sizing: border-box; gap: 15px;">
    <!-- LEFT COLUMN: Messages and Form -->
    <div id="chat-left-column" style="display: flex; flex-direction: column; flex: 1; min-width: 0; gap: 10px;">
        {message_container_html}
        {form_html}
    </div>
    <!-- RIGHT COLUMN: Sidebar -->
    {issues_sidebar_html}
</div>'''

        new_html = full_new_structure

        payload['QuestionText'] = new_html
        payload['QuestionText_Unsafe'] = new_html  # Update both fields
        
        # ===== UPDATE QUESTION JS =====
        old_js = payload['QuestionJS']
        
        # Add issue status rendering code after saveHistoriesToEmbeddedData()
        issues_js = '''
             // === NEGOTIATION ISSUE STATUS DISPLAY ===
             const issueStatusDefaults = [
                 { id: 'issue-1', label: 'Bonus', status: '' },
                 { id: 'issue-2', label: 'Job Assignment', status: '' },
                 { id: 'issue-3', label: 'Vacation Time', status: '' },
                 { id: 'issue-4', label: 'Starting Date', status: '' },
                 { id: 'issue-5', label: 'Moving Expense Coverage', status: '' },
                 { id: 'issue-6', label: 'Insurance Coverage', status: '' },
                 { id: 'issue-7', label: 'Salary', status: '' },
                 { id: 'issue-8', label: 'Location', status: '' }
             ];

             function normalizeIssueStatuses(rawStatuses) {
                 const parsedStatuses = [];

                 if (Array.isArray(rawStatuses)) {
                     rawStatuses.forEach((issue, index) => {
                         if (typeof issue === 'string') {
                             parsedStatuses.push({
                                 id: 'issue-' + (index + 1),
                                 label: 'Issue ' + (index + 1),
                                 status: issue || ''
                             });
                         } else if (issue && typeof issue === 'object') {
                             parsedStatuses.push({
                                 id: issue.id || ('issue-' + (index + 1)),
                                 label: issue.label || issue.name || ('Issue ' + (index + 1)),
                                 status: issue.status || issue.value || issue.current_status || ''
                             });
                         }
                     });
                 } else if (rawStatuses && typeof rawStatuses === 'object') {
                     Object.keys(rawStatuses).forEach((key, index) => {
                         const issue = rawStatuses[key];
                         if (issue && typeof issue === 'object') {
                             parsedStatuses.push({
                                 id: issue.id || key || ('issue-' + (index + 1)),
                                 label: issue.label || issue.name || key || ('Issue ' + (index + 1)),
                                 status: issue.status || issue.value || issue.current_status || ''
                             });
                         } else {
                             parsedStatuses.push({
                                 id: key || ('issue-' + (index + 1)),
                                 label: key || ('Issue ' + (index + 1)),
                                 status: issue ? String(issue) : ''
                             });
                         }
                     });
                 }

                 while (parsedStatuses.length < 8) {
                     const nextIndex = parsedStatuses.length + 1;
                     parsedStatuses.push({
                         id: 'issue-' + nextIndex,
                         label: 'Issue ' + nextIndex,
                        status: ''
                     });
                 }

                 return parsedStatuses.slice(0, 8);
             }

             function renderIssueStatuses(statusItems) {
                 const issueList = document.getElementById('issue-status-list');
                 const issuePanel = document.getElementById('issue-status-panel');
                 if (!issueList) {
                     return;
                 }

                 issueList.innerHTML = '';
                 statusItems.forEach((issue) => {
                     const card = document.createElement('div');
                     card.style.padding = '10px';
                     card.style.border = '1px solid #cfd8dc';
                     card.style.borderRadius = '6px';
                     card.style.backgroundColor = '#fafafa';

                     const title = document.createElement('div');
                     title.style.fontWeight = 'bold';
                     title.style.marginBottom = '4px';
                     title.textContent = issue.label;

                     const status = document.createElement('div');
                     status.textContent = issue.status || '';

                     card.appendChild(title);
                     card.appendChild(status);
                     issueList.appendChild(card);
                 });

                 if (issuePanel) {
                     issuePanel.style.display = 'block';
                 }
             }

             const issueStatusSource = response.issue_statuses || Qualtrics.SurveyEngine.getEmbeddedData('LUCIDIssueStatusesJSON');
             let issueStatuses = issueStatusDefaults;

             if (typeof issueStatusSource === 'string' && issueStatusSource.trim() !== '') {
                 try {
                     issueStatuses = normalizeIssueStatuses(JSON.parse(issueStatusSource));
                 } catch (e) {
                     issueStatuses = issueStatusDefaults;
                 }
             } else if (issueStatusSource) {
                 issueStatuses = normalizeIssueStatuses(issueStatusSource);
             }

             renderIssueStatuses(issueStatuses);
             // === END NEGOTIATION ISSUE STATUS DISPLAY ==='''
        
        # Find and insert after saveHistoriesToEmbeddedData();
        insertion_point = "             // Save the updated logs to Qualtrics Embedded Data\n             saveHistoriesToEmbeddedData();"
        if insertion_point in old_js:
            new_js = old_js.replace(
                insertion_point,
                insertion_point + issues_js
            )
            payload['QuestionJS'] = new_js
            print("✓ Successfully added issue status display JavaScript")
        else:
            print("✗ Could not find insertion point in JavaScript")
            sys.exit(1)
        
        print("✓ Successfully updated QuestionText with issue status panel HTML")
        break

# Save the updated template
with open(TEMPLATE_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=0)

print(f"\n✅ Template updated successfully: {TEMPLATE_PATH}")
print("The template now includes:")
print("  • Issue status panel in the HTML")
print("  • JavaScript to parse and display issue statuses from embedded data or backend response")
