
import sys
import os
import asyncio
from datetime import datetime
import json

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator.graph import build_workflow
from orchestrator.state import create_initial_state
from main import calculate_hotel_distances

# Mock data
test_data = [
    {
        "id": 1,
        "origin": "NYC",
        "destination": "SF",
        "departure_date": "2026-02-10",
        "return_date": "2026-02-14",
        "hotel_checkin": "2026-02-10",
        "hotel_checkout": "2026-02-13",
        "meeting_date": "2026-02-11",
        "meeting_time": "14:00",
        "meeting_address": "1 Ferry Building, San Francisco, CA 94111",
        "meeting_coordinates": {"lat": 37.7955, "lon": -122.3937},
        "budget": 2000,
        "trip_type": "business",
        "preferences": "Quiet room, gym access"
    },
    {
        "id": 2,
        "origin": "SF",
        "destination": "NYC",
        "departure_date": "2026-03-05",
        "return_date": "2026-03-08",
        "hotel_checkin": "2026-03-05",
        "hotel_checkout": "2026-03-07",
        "meeting_date": "2026-03-06",
        "meeting_time": "10:00",
        "meeting_address": "350 Fifth Avenue, New York, NY 10118",
        "meeting_coordinates": {"lat": 40.748817, "lon": -73.985428},
        "budget": 3000,
        "trip_type": "business",
        "preferences": "Near Central Park"
    },
    {
        "id": 3,
        "origin": "CHI",
        "destination": "BOS",
        "departure_date": "2026-04-12",
        "return_date": "2026-04-15",
        "hotel_checkin": "2026-04-12",
        "hotel_checkout": "2026-04-14",
        "meeting_date": "2026-04-13",
        "meeting_time": "09:00",
        "meeting_address": "100 Northern Ave, Boston, MA 02210",
        "meeting_coordinates": {"lat": 42.353, "lon": -71.045},
        "budget": 1500,
        "trip_type": "business",
        "preferences": "Seafood dining nearby"
    },
    {
        "id": 4,
        "origin": "BOS",
        "destination": "CHI",
        "departure_date": "2026-05-20",
        "return_date": "2026-05-23",
        "hotel_checkin": "2026-05-20",
        "hotel_checkout": "2026-05-22",
        "meeting_date": "2026-05-21",
        "meeting_time": "13:00",
        "meeting_address": "233 S Wacker Dr, Chicago, IL 60606",
        "meeting_coordinates": {"lat": 41.878876, "lon": -87.635915},
        "budget": 1800,
        "trip_type": "business",
        "preferences": "Architecture tour"
    },
    {
        "id": 5,
        "origin": "NYC",
        "destination": "CHI",
        "departure_date": "2026-06-15",
        "return_date": "2026-06-18",
        "hotel_checkin": "2026-06-15",
        "hotel_checkout": "2026-06-17",
        "meeting_date": "2026-06-16",
        "meeting_time": "15:00",
        "meeting_address": "433 W Van Buren St, Chicago, IL 60607",
        "meeting_coordinates": {"lat": 41.876, "lon": -87.639},
        "budget": 2500,
        "trip_type": "business",
        "preferences": "Modern hotel"
    }
]

async def run_test(data):
    print(f"\n{'='*50}")
    print(f"Running Test #{data['id']}: {data['origin']} -> {data['destination']}")
    print(f"Meeting: {data['meeting_address']}")
    print(f"Budget: ${data['budget']}")
    print(f"{'='*50}")

    # Build workflow
    workflow = build_workflow()
    graph = workflow.compile()

    # Create Initial State
    # Note: preferences dict structure matches what main.py constructs
    initial_state = create_initial_state(
        origin=data["origin"],
        destination=data["destination"],
        departure_date=data["departure_date"],
        return_date=data["return_date"],
        budget=data["budget"],
        preferences={
            "meeting_time": data["meeting_time"],
            "meeting_date": data["meeting_date"],
            "meeting_address": data["meeting_address"],
            "meeting_coordinates": data["meeting_coordinates"],
            "meeting_times": [f"{data['meeting_date']} {data['meeting_time']}"],
            "meeting_location": data["meeting_coordinates"] or data["meeting_address"],
            "hotel_location": data["meeting_address"],  # Use meeting address as preferred hotel location
            "hotel_checkin": data["hotel_checkin"],
            "hotel_checkout": data["hotel_checkout"],
            "trip_type": data["trip_type"],
            "user_preferences": data["preferences"]
        }
    )

    # Run Graph
    try:
        final_state = graph.invoke(initial_state)
        
        # Analyze Results
        print("\n--- Results ---")
        
        # Flight
        flight = final_state.get("selected_flight")
        if flight:
             # Handle object vs dict
            f_data = flight.__dict__ if hasattr(flight, '__dict__') else flight
            print(f"✅ Flight Selected: {f_data.get('airline')} {f_data.get('flight_number')} (${f_data.get('price', 'N/A')})")
        else:
            print("❌ No Flight Selected")

        # Hotel
        hotel = final_state.get("selected_hotel")
        if hotel:
            h_data = hotel.__dict__ if hasattr(hotel, '__dict__') else hotel
            print(f"✅ Hotel Selected: {h_data.get('name')} (${h_data.get('price_per_night', 'N/A')}/night)")
            
            # Enrich with distance
            if data['meeting_coordinates']:
                enriched = await calculate_hotel_distances([h_data], data['meeting_coordinates'])
                if enriched:
                    dist = enriched[0].get('distance_to_meeting_km')
                    time_min = enriched[0].get('travel_time_minutes')
                    print(f"   📍 Distance to meeting: {dist} km ({time_min} min)")
        else:
            print("❌ No Hotel Selected")
            
        # Policy
        policy = final_state.get("compliance_status")
        if policy:
            p_data = policy.__dict__ if hasattr(policy, '__dict__') else policy
            print(f"📋 Policy Check: {'Passed' if p_data.get('compliant') else 'Failed'}")
            if not p_data.get('compliant'):
                print(f"   Reason: {p_data.get('issues')}")

        # Cost Analysis
        total_cost = 0
        if flight and hotel:
            f_price = f_data.get('price_usd', f_data.get('price', 0))
            h_price = h_data.get('price_per_night_usd', h_data.get('price_per_night', 0))
            # Rough night calc
            d1 = datetime.strptime(data['hotel_checkin'], "%Y-%m-%d")
            d2 = datetime.strptime(data['hotel_checkout'], "%Y-%m-%d")
            nights = max(1, (d2 - d1).days)
            total_cost = f_price + (h_price * nights)
            
        print(f"💰 Total Estimated Cost: ${total_cost} (Budget: ${data['budget']})")
        if total_cost > data['budget']:
             print("⚠️ Over Budget!")
        else:
             print("✅ Within Budget")

    except Exception as e:
        print(f"❌ Error running test: {e}")
        import traceback
        traceback.print_exc()

async def main():
    print("Starting integration tests with 5 dummy datasets...")
    for test in test_data:
        await run_test(test)
        print("\n\n")

if __name__ == "__main__":
    asyncio.run(main())
