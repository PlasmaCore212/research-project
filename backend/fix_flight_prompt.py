import re

with open('agents/flight_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the return format in refine_proposal
old_pattern = r'Return: \{\{"selected_flights": \["FLIGHT_ID"\], "reasoning": "How this addresses feedback"\}\}'
new_text = '''Return: {{"selected_flights": ["SINGLE_FLIGHT_ID"], "reasoning": "How this ONE flight addresses feedback"}}

IMPORTANT: Return exactly ONE flight ID - the single best option!'''

content = re.sub(old_pattern, new_text, content)

with open('agents/flight_agent.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated flight_agent.py")
