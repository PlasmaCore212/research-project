from typing import List, Dict
import json
import os

from data.data_generator import CITIES

# City name normalization map
CITY_ALIASES = {
    "san francisco": "SF", "sf": "SF", "sfo": "SF",
    "new york": "NYC", "new york city": "NYC", "nyc": "NYC",
    "boston": "BOS", "bos": "BOS",
    "chicago": "CHI", "chi": "CHI",
}

def normalize_city(city: str) -> str:
    """Normalize city names to standard codes."""
    city_lower = city.lower().strip()
    return CITY_ALIASES.get(city_lower, city.upper())


class FlightDataLoader:
    def __init__(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), "flights.json")
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
        
        # Normalize city names
        from_city = normalize_city(from_city)
        to_city = normalize_city(to_city)
        
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
        
        # Return diverse results: include all flight classes
        # Sort by price within each class, then combine
        by_class = {'Economy': [], 'Business': [], 'First Class': []}
        for f in results:
            flight_class = f.get('class', 'Economy')
            if flight_class in by_class:
                by_class[flight_class].append(f)
        
        diverse_results = []
        for fc in ['Economy', 'Business', 'First Class']:
            sorted_class = sorted(by_class[fc], key=lambda x: (x["price_usd"], x["departure_time"]))
            diverse_results.extend(sorted_class[:15])  # Up to 15 from each class
        
        return diverse_results
    
class HotelDataLoader:
    def __init__(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), "hotels.json")
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
        
        # Normalize city name
        city = normalize_city(city)
        
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
        
        # Return diverse results: include hotels from each star level
        by_stars = {5: [], 4: [], 3: [], 2: [], 1: []}
        for h in results:
            stars = h.get('stars', 3)
            if stars in by_stars:
                by_stars[stars].append(h)
        
        diverse_results = []
        for stars in [5, 4, 3, 2, 1]:
            sorted_star = sorted(by_stars[stars], key=lambda x: (x["distance_to_business_center_km"], x["price_per_night_usd"]))
            diverse_results.extend(sorted_star[:5])  # Up to 5 from each star level
        
        return diverse_results