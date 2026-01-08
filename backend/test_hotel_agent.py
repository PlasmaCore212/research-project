#!/usr/bin/env python3
"""Test script for Hotel Agent"""

import sys
sys.path.insert(0, '/mnt/project')

from agents.hotel_agent import HotelAgent
from agents.models import HotelQuery

def test_basic_search():
    """Test basic hotel search"""
    print("="*60)
    print("TEST 1: Basic hotel search in SF")
    print("="*60)
    
    agent = HotelAgent()
    query = HotelQuery(
        city="SF",
        max_price_per_night=300
    )
    
    result = agent.search_hotels(query)
    
    print(f"\nQuery: {query.city}, Max: ${query.max_price_per_night}/night")
    print(f"\nFound {len(result.hotels)} hotels\n")
    
    print(result.reasoning)
    print("\n" + "="*60)

def test_filtered_search():
    """Test search with multiple filters"""
    print("\nTEST 2: Filtered search - 4+ with WiFi & Gym")
    print("="*60)
    
    agent = HotelAgent()
    query = HotelQuery(
        city="NYC",
        max_price_per_night=400,
        min_stars=4,
        max_distance_to_center_km=2.0,
        required_amenities=["WiFi", "Gym"]
    )
    
    result = agent.search_hotels(query)
    
    print(f"\nQuery: {query.city}, Max: ${query.max_price_per_night}/night, {query.min_stars}+")
    print(f"Max distance: {query.max_distance_to_center_km}km")
    print(f"Required amenities: {', '.join(query.required_amenities)}")
    print(f"\nFound {len(result.hotels)} hotels\n")
    
    print(result.reasoning)
    print("\n" + "="*60)

def test_budget_search():
    """Test budget-conscious search"""
    print("\nTEST 3: Budget search in Chicago")
    print("="*60)
    
    agent = HotelAgent()
    query = HotelQuery(
        city="CHI",
        max_price_per_night=150,
        required_amenities=["WiFi"]
    )
    
    result = agent.search_hotels(query)
    
    print(f"\nQuery: {query.city}, Max: ${query.max_price_per_night}/night")
    print(f"Required: WiFi")
    print(f"\nFound {len(result.hotels)} hotels\n")
    
    print(result.reasoning)
    print("\n" + "="*60)

if __name__ == "__main__":
    test_basic_search()
    test_filtered_search()
    test_budget_search()
    
    print("\nAll hotel agent tests completed!")