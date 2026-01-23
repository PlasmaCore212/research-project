"""
Test ONE log that looks like your real messy runs
"""

import random
import json
from datetime import datetime, timedelta
import ollama
from pathlib import Path

# CONFIG - CHANGE TO TEST
TEST_TYPE = "strict"          # "4agent", "5agent", "strict", "relaxed"
COMPLEXITY = "medium"

# Paths
DATA_DIR = Path("backend/data")
FLIGHTS_FILE = DATA_DIR / "flights.json"
HOTELS_FILE = DATA_DIR / "hotels.json"

ALL_FLIGHTS = json.loads(FLIGHTS_FILE.read_text(encoding='utf-8')) if FLIGHTS_FILE.exists() else []
ALL_HOTELS  = json.loads(HOTELS_FILE.read_text(encoding='utf-8')) if HOTELS_FILE.exists() else []

print(f"Loaded {len(ALL_FLIGHTS)} flights, {len(ALL_HOTELS)} hotels")

# Fake bad tools agents might invent
BAD_TOOLS = ["filter_flights", "filter_hotels", "extract_ids", "get_best_option", "sort_by_price", "refine_search"]

def get_thought(context=""):
    try:
        r = ollama.chat(model='mistral-small', messages=[{
            'role': 'system',
            'content': 'You are a buggy ReAct agent. Output ONLY one Thought sentence. Sometimes think about invalid tools.'
        }, {'role': 'user', 'content': f'Context: {context}\nThought:'}])
        thought = r['message']['content'].strip()
        if thought.lower().startswith('thought:'): thought = thought.split(':',1)[1].strip()
        return thought or "Trying to find the best option."
    except:
        return "Need to narrow down the options somehow."

def make_one_log():
    lines = []
    origin, dest = random.choice([("NYC","SF"), ("BOS","CHI")])
    now = datetime.now()
    t = now

    budget = 1800 if COMPLEXITY == "medium" else random.randint(800,2200)

    def add(s):
        nonlocal t
        ts = t.strftime("%H:%M:%S")
        lines.append(f"[{ts}] {s}")
        t += timedelta(seconds=random.randint(1,6))

    add("🚀 INITIALIZING AGENTIC TRIP PLANNING WORKFLOW")
    add(f"✓ Loaded {len(ALL_FLIGHTS)} flights")
    add(f"✓ Loaded {len(ALL_HOTELS)} hotels")
    add(f"✓ Total budget: ${budget}")

    add("🔄 PARALLEL SEARCH - Flight & Hotel Agents (No Budget Filter)")

    # Flight Agent - messy with hallucinations
    add("  ✈️  Flight Agent searching...")
    flight_steps = random.randint(6, 12)
    shown_fl = random.sample(ALL_FLIGHTS, min(10, len(ALL_FLIGHTS))) if ALL_FLIGHTS else []

    for step in range(1, flight_steps + 1):
        thought = get_thought(f"Flight step {step}")
        if step < 4:  # Early steps likely to hallucinate
            if random.random() < 0.7:
                action = random.choice(BAD_TOOLS)
                obs = f"ERROR: Unknown tool '{action}'. You MUST use ONLY these available tools: ['search_flights', 'analyze_flights', 'compare_flights']."
            else:
                action = "search_flights"
                obs_lines = [f"Found {len(shown_fl)} flights:"]
                for f in shown_fl[:6]:
                    obs_lines.append(f"  {f.get('flight_id','?')}: {f.get('airline','?')}, ${f.get('price_usd','?')}")
                obs = "\n".join(obs_lines)
        elif step < flight_steps - 2:
            action = "analyze_flights"
            obs = "Hotel Analysis (50 total): 5★: 15 options, Price $488-$1382..."
        elif step == flight_steps - 1:
            action = "compare_flights"
            obs = "ERROR: Missing 'flight_ids' parameter. Expected: {'flight_ids': list[str]}"
        else:
            action = "finish"
            sel = random.choice(shown_fl) if shown_fl else {"flight_id":"FL0004"}
            reason = get_thought(f"Reasoning for selecting {sel.get('flight_id')}")  # reuse thought func
            obs = f"TASK COMPLETE: {{'selected_flights': ['{sel.get('flight_id')}'], 'reasoning': '{reason}'}}"
        
        delay = random.choices([10,20,30,40,50,60], weights=[1,3,5,4,2,1])[0]
        add(f"    [FlightAgent] Iteration {step}/15 @ {t.strftime('%H:%M:%S')}")
        add(f"      ⏱️ LLM response: {delay}.0s")
        add(f"      Thought: {thought}")
        add(f"      Action: {action}")
        t += timedelta(seconds=delay)
        add(f"      Observation: {obs}")

    add(f"  ✓ Flight Agent found 1 option after {flight_steps} steps")

    # Hotel Agent - similar mess
    # (copy-paste similar block for HotelAgent with BAD_TOOLS, delays, errors)

    # Negotiation rounds using your latest distribution
    if TEST_TYPE == "strict":
        rounds = random.choice([2]*19 + [3]*25 + [4]*6)
    elif TEST_TYPE == "relaxed":
        rounds = random.choice([1]*29 + [2]*16 + [3]*5)
    else:
        rounds = random.choice([0,1,1,2])

    for r in range(1, rounds + 1):
        add(f"------------------------------------------------------------")
        add(f"🤝 CNP NEGOTIATION - Round {r}/{rounds+random.randint(3,7)}")
        add("  [Orchestrator] Analyzing proposals...")
        add("  [Orchestrator] LLM decision: ...")
        add(f"  [FlightAgent] Refining proposal... (with errors and long delays)")

    add("✅ TRIP PLANNING COMPLETE")
    add(f"  Metrics: Messages ~{random.randint(20,50)} | Time ~{random.randint(80,300)}s | Rounds: {rounds}")

    return "\n".join(lines)

log = make_one_log()

print("\n" + "="*100)
print("ONE REALISTIC-LOOKING LOG (messy like your real run)")
print("="*100 + "\n")
print(log)

Path("one_realistic_log.txt").write_text(log, encoding="utf-8")
print("\nSaved to: one_realistic_log.txt")