import json
import sys

# Update QID106's QuestionText with the new HTML (including the reference sheet dropdown)
qsf_path = 'LUCID_Negotiation_Template.qsf'
html_path = 'tmp_qhtml.html'

with open(html_path, 'r', encoding='utf-8') as f:
    new_html = f.read().strip()

with open(qsf_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

updated = False
for el in data.get('SurveyElements', []):
    if el.get('Element') == 'SQ' and el.get('PrimaryAttribute') == 'QID106':
        payload = el['Payload']
        payload['QuestionText'] = new_html
        payload['QuestionText_Unsafe'] = new_html
        updated = True
        print('Updated QID106 QuestionText with new HTML (reference sheet dropdown included)')

if not updated:
    raise SystemExit('QID106 not found; no changes made.')

with open(qsf_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print('Saved', qsf_path)
