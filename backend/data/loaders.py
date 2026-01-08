from typing import List, Dict
import json

from data.data_generator import CITIES

class FlightDataLoader:
    def __init__(self, filepath: str = "data/flights.json"):
        with open(filepath, 'r') as f:
            self.flights = json.load(f)
    
    def _time_to_minutes(self, time_str: str) -> int:
        """Convert HH:MM to minutes since midnight"""
        hour, minute = map(int, time_str.split(':'))
        return hour * 60 + minute
    
    def search(
        self,
        from_city: str,
        to_city: str,
        max_price: int = None,
        departure_after: str = None,
        departure_before: str = None
    ) -> List[Dict]:
        """Filter flights matching criteria"""
        results = []
        
        print(f"[DEBUG] Searching for {from_city}→{to_city}")
        if departure_after or departure_before:
            print(f"[DEBUG] Time window: {departure_after or '00:00'} to {departure_before or '23:59'}")
        
        # Convert time constraints to minutes
        after_minutes = self._time_to_minutes(departure_after) if departure_after else 0
        before_minutes = self._time_to_minutes(departure_before) if departure_before else 1439  # 23:59
        
        for flight in self.flights:
            # Match route
            if flight["from_city"] != from_city or flight["to_city"] != to_city:
                continue
            
            # Price filter
            if max_price and flight["price_usd"] > max_price:
                continue
            
            # Time window filter
            departure_minutes = self._time_to_minutes(flight["departure_time"])
            if not (after_minutes <= departure_minutes <= before_minutes):
                continue
            
            results.append(flight)
        
        print(f"[DEBUG] Found {len(results)} flights")
        return sorted(results, key=lambda x: (x["price_usd"], x["departure_time"]))[:10]
    
class HotelDataLoader:
    def __init__(self, filepath: str = "data/hotels.json"):
        with open(filepath, 'r') as f:
            self.hotels = json.load(f)
    
    def search(
        self,
        city: str,
        max_price_per_night: int = None,
        min_stars: int = None,
        max_distance_to_center_km: float = None,
        required_amenities: List[str] = None
    ) -> List[Dict]:
        """Filter hotels matching criteria"""
        results = []
        
        print(f"[DEBUG] Searching hotels in {city}")
        if max_price_per_night:
            print(f"[DEBUG] Max price: ${max_price_per_night}/night")
        if min_stars:
            print(f"[DEBUG] Min stars: {min_stars}")
        if max_distance_to_center_km:
            print(f"[DEBUG] Max distance: {max_distance_to_center_km}km")
        
        for hotel in self.hotels:
            # Match city
            if hotel["city"] != city:
                continue
            
            # Price filter
            if max_price_per_night and hotel["price_per_night_usd"] > max_price_per_night:
                continue
            
            # Stars filter
            if min_stars and hotel["stars"] < min_stars:
                continue
            
            # Distance filter
            if max_distance_to_center_km and hotel["distance_to_business_center_km"] > max_distance_to_center_km:
                continue
            
            # Amenities filter
            if required_amenities:
                hotel_amenities = set(hotel["amenities"])
                if not all(amenity in hotel_amenities for amenity in required_amenities):
                    continue
            
            results.append(hotel)
        
        print(f"[DEBUG] Found {len(results)} hotels")
        return sorted(results, key=lambda x: (x["distance_to_business_center_km"], x["price_per_night_usd"]))[:10]
    
class PolicyDataLoader:
    def __init__(self, filepath: str = "data/policies.json"):
        with open(filepath, 'r') as f:
            self.policies = json.load(f)
    
    def get_policy(self, policy_name: str = "standard") -> Dict:
        """Get policy rules by name"""
        return self.policies.get(policy_name, {})