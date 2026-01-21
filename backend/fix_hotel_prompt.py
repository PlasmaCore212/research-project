import re

# Update HotelAgent
with open('agents/hotel_agent.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add to system prompt
content = content.replace(
    'Return format: {"selected_hotels": ["HOTEL_ID"], "reasoning": "Why this hotel"}"""',
    'Return format: {"selected_hotels": ["HOTEL_ID"], "reasoning": "Why this hotel"}\n\nIMPORTANT: Select exactly ONE hotel only - not multiple options!"""'
)

# Update refine_proposal
old_pattern = r'Return: \{\{"selected_hotels": \["HOTEL_ID"\], "reasoning": "How this addresses feedback"\}\}'
new_text = '''Return: {{"selected_hotels": ["SINGLE_HOTEL_ID"], "reasoning": "How this ONE hotel addresses feedback"}}

IMPORTANT: Return exactly ONE hotel ID - the single best option!'''

content = re.sub(old_pattern, new_text, content)

with open('agents/hotel_agent.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated hotel_agent.py")
