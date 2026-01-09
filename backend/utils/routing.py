# backend/utils/routing.py
import requests
from typing import Tuple, Optional, Dict
from functools import lru_cache

# Known airport coordinates
AIRPORT_COORDS = {
    "NYC": {"lat": 40.6413, "lon": -73.7781},  # JFK
    "JFK": {"lat": 40.6413, "lon": -73.7781},
    "LGA": {"lat": 40.7769, "lon": -73.8740},
    "EWR": {"lat": 40.6895, "lon": -74.1745},
    "SF": {"lat": 37.6213, "lon": -122.3790},   # SFO
    "SFO": {"lat": 37.6213, "lon": -122.3790},
    "OAK": {"lat": 37.7213, "lon": -122.2208},
    "BOS": {"lat": 42.3656, "lon": -71.0096},   # Logan
    "CHI": {"lat": 41.9742, "lon": -87.9073},   # O'Hare
    "ORD": {"lat": 41.9742, "lon": -87.9073},
    "MDW": {"lat": 41.7868, "lon": -87.7522},
}

# Known city center / business district coordinates
CITY_CENTER_COORDS = {
    "NYC": {"lat": 40.7580, "lon": -73.9855},   # Midtown Manhattan
    "SF": {"lat": 37.7946, "lon": -122.3999},    # Financial District
    "BOS": {"lat": 42.3555, "lon": -71.0565},    # Financial District
    "CHI": {"lat": 41.8819, "lon": -87.6278},    # The Loop
}

# Known landmark/meeting location coordinates
KNOWN_LOCATIONS = {
    # San Francisco
    "salesforce tower": {"lat": 37.7897, "lon": -122.3972},
    "financial district sf": {"lat": 37.7946, "lon": -122.3999},
    "embarcadero center": {"lat": 37.7952, "lon": -122.3954},
    "union square sf": {"lat": 37.7879, "lon": -122.4074},
    "soma": {"lat": 37.7785, "lon": -122.4056},
    "moscone center": {"lat": 37.7840, "lon": -122.4015},
    "ferry building": {"lat": 37.7956, "lon": -122.3934},
    "market street": {"lat": 37.7879, "lon": -122.4074},
    
    # New York
    "times square": {"lat": 40.7580, "lon": -73.9855},
    "wall street": {"lat": 40.7074, "lon": -74.0113},
    "empire state building": {"lat": 40.7484, "lon": -73.9857},
    "world trade center": {"lat": 40.7127, "lon": -74.0134},
    "grand central": {"lat": 40.7527, "lon": -73.9772},
    "midtown": {"lat": 40.7549, "lon": -73.9840},
    "downtown manhattan": {"lat": 40.7128, "lon": -74.0060},
    "rockefeller center": {"lat": 40.7587, "lon": -73.9787},
    "bryant park": {"lat": 40.7536, "lon": -73.9832},
    
    # Boston
    "financial district boston": {"lat": 42.3555, "lon": -71.0565},
    "back bay": {"lat": 42.3503, "lon": -71.0810},
    "seaport district": {"lat": 42.3519, "lon": -71.0446},
    "cambridge": {"lat": 42.3736, "lon": -71.1097},
    "harvard square": {"lat": 42.3732, "lon": -71.1189},
    "kendall square": {"lat": 42.3629, "lon": -71.0901},
    "prudential center": {"lat": 42.3470, "lon": -71.0818},
    "faneuil hall": {"lat": 42.3600, "lon": -71.0560},
    
    # Chicago
    "the loop": {"lat": 41.8819, "lon": -87.6278},
    "magnificent mile": {"lat": 41.8946, "lon": -87.6249},
    "willis tower": {"lat": 41.8789, "lon": -87.6359},
    "navy pier": {"lat": 41.8917, "lon": -87.6086},
    "millennium park": {"lat": 41.8826, "lon": -87.6226},
    "river north": {"lat": 41.8922, "lon": -87.6324},
    "west loop": {"lat": 41.8827, "lon": -87.6505},
}


def get_airport_coords(city_code: str) -> Dict[str, float]:
    """Get airport coordinates for a city code."""
    return AIRPORT_COORDS.get(city_code.upper(), AIRPORT_COORDS.get("NYC"))


def get_city_center_coords(city_code: str) -> Dict[str, float]:
    """Get city center coordinates for a city code."""
    return CITY_CENTER_COORDS.get(city_code.upper(), CITY_CENTER_COORDS.get("NYC"))


def geocode_address(address: str, city_code: str = None) -> Optional[Dict[str, float]]:
    """
    Convert an address/location string to coordinates.
    Uses known locations lookup, with fallback to city center.
    
    Args:
        address: Location name or address string
        city_code: Optional city code (NYC, SF, BOS, CHI) for context
    
    Returns:
        Dict with 'lat' and 'lon' keys, or None
    """
    if not address:
        return get_city_center_coords(city_code) if city_code else None
    
    addr_lower = address.lower().strip()
    
    # Try exact/partial match in known locations
    for key, coords in KNOWN_LOCATIONS.items():
        if key in addr_lower or addr_lower in key:
            return coords
    
    # Try city center as fallback
    if city_code:
        return get_city_center_coords(city_code)
    
    # Try to extract city from address
    for city in ["nyc", "new york", "sf", "san francisco", "bos", "boston", "chi", "chicago"]:
        if city in addr_lower:
            city_map = {"nyc": "NYC", "new york": "NYC", "sf": "SF", "san francisco": "SF",
                       "bos": "BOS", "boston": "BOS", "chi": "CHI", "chicago": "CHI"}
            return get_city_center_coords(city_map.get(city, "NYC"))
    
    return None


class RoutingService:
    """Calculate actual driving distances and times using OSRM (OpenStreetMap)"""
    
    def __init__(self):
        # Public OSRM demo server (uses OpenStreetMap data)
        self.osrm_base = "http://router.project-osrm.org/route/v1/driving"
    
    @lru_cache(maxsize=1000)
    def get_route(
        self, 
        start_lon: float, 
        start_lat: float,
        end_lon: float, 
        end_lat: float
    ) -> Optional[Tuple[float, float]]:
        """
        Get driving distance (km) and duration (minutes) using OSRM/OpenStreetMap.
        
        Returns: (distance_km, duration_minutes) or None if route fails
        """
        # Validate coordinates
        if not all([
            -180 <= start_lon <= 180, -90 <= start_lat <= 90,
            -180 <= end_lon <= 180, -90 <= end_lat <= 90
        ]):
            print(f"[ROUTING] Invalid coordinates: ({start_lat},{start_lon}) → ({end_lat},{end_lon})")
            return None
        
        # Skip if coordinates are too close (same location)
        if abs(start_lat - end_lat) < 0.001 and abs(start_lon - end_lon) < 0.001:
            return (0.1, 5)  # Minimal distance, 5 min walk
        
        url = f"{self.osrm_base}/{start_lon},{start_lat};{end_lon},{end_lat}"
        params = {
            "overview": "false",
            "geometries": "geojson"
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            if data["code"] != "Ok":
                print(f"[ROUTING] OSRM error: {data.get('code')}")
                return None
            
            route = data["routes"][0]
            distance_km = route["distance"] / 1000  # meters to km
            duration_minutes = route["duration"] / 60  # seconds to minutes
            
            return (distance_km, duration_minutes)
            
        except requests.exceptions.Timeout:
            print(f"[ROUTING] Timeout reaching OSRM server")
            return None
        except requests.exceptions.RequestException as e:
            print(f"[ROUTING ERROR] {e}")
            return None
        except Exception as e:
            print(f"[ROUTING ERROR] Unexpected: {e}")
            return None
    
    def get_transit_time(
        self,
        start_coords: dict,
        end_coords: dict,
        include_buffer: bool = True
    ) -> Optional[float]:
        """
        Calculate total transit time using real street routing.
        
        Args:
            start_coords: {"lat": float, "lon": float}
            end_coords: {"lat": float, "lon": float}
            include_buffer: Add 15min buffer for traffic/parking
        
        Returns: total time in minutes, or None if routing fails
        """
        # Extract coordinates safely
        start_lat = start_coords.get("lat", 0)
        start_lon = start_coords.get("lon", start_coords.get("lng", 0))
        end_lat = end_coords.get("lat", 0)
        end_lon = end_coords.get("lon", end_coords.get("lng", 0))
        
        # Validate
        if not all([start_lat, start_lon, end_lat, end_lon]):
            return None
        
        result = self.get_route(start_lon, start_lat, end_lon, end_lat)
        
        if result:
            distance_km, duration = result
            # Add buffer time for real-world conditions
            if include_buffer:
                duration += 15  # parking, traffic, etc.
            return duration
        
        return None
    
    def get_distance_and_time(
        self,
        start_coords: dict,
        end_coords: dict
    ) -> Optional[Tuple[float, float]]:
        """
        Get both distance (km) and time (minutes) between two points.
        Uses real street routing via OSRM/OpenStreetMap.
        
        Returns: (distance_km, duration_minutes) or None
        """
        start_lat = start_coords.get("lat", 0)
        start_lon = start_coords.get("lon", start_coords.get("lng", 0))
        end_lat = end_coords.get("lat", 0)
        end_lon = end_coords.get("lon", end_coords.get("lng", 0))
        
        return self.get_route(start_lon, start_lat, end_lon, end_lat)