# backend/utils/routing.py
"""
Routing utilities for calculating travel distances and times.

Uses OSRM (OpenStreetMap Routing Machine) for real street-level routing.
Only keeps essential airport coordinates needed for airport→hotel routing.
"""

import requests
from typing import Tuple, Optional, Dict
from functools import lru_cache

# Known airport coordinates - needed for airport→hotel routing
# Flight data doesn't include airport coordinates, so we need these
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


def get_airport_coords(city_code: str) -> Dict[str, float]:
    """Get airport coordinates for a city code.
    
    Args:
        city_code: Airport/city code (e.g., 'NYC', 'SF', 'BOS', 'CHI')
    
    Returns:
        Dict with 'lat' and 'lon' keys
    """
    return AIRPORT_COORDS.get(city_code.upper(), AIRPORT_COORDS.get("NYC"))


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