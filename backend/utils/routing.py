# backend/utils/routing.py
import requests
from typing import Tuple, Optional
from functools import lru_cache

class RoutingService:
    """Calculate actual driving distances and times using OSRM"""
    
    def __init__(self):
        # Public OSRM demo server (consider self-hosting for production)
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
        Get driving distance (km) and duration (minutes)
        
        Returns: (distance_km, duration_minutes) or None if route fails
        """
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
                return None
            
            route = data["routes"][0]
            distance_km = route["distance"] / 1000  # meters to km
            duration_minutes = route["duration"] / 60  # seconds to minutes
            
            return (distance_km, duration_minutes)
            
        except Exception as e:
            print(f"[ROUTING ERROR] {e}")
            return None
    
    def get_transit_time(
        self,
        start_coords: dict,
        end_coords: dict,
        include_buffer: bool = True
    ) -> float:
        """
        Calculate total transit time including buffers
        
        Args:
            start_coords: {"lat": float, "lon": float}
            end_coords: {"lat": float, "lon": float}
            include_buffer: Add 15min buffer for traffic/parking
        
        Returns: total time in minutes
        """
        result = self.get_route(
            start_coords["lon"], start_coords["lat"],
            end_coords["lon"], end_coords["lat"]
        )
        
        if not result:
            # Fallback to haversine with typical speeds
            from data.data_generator import haversine_distance
            distance = haversine_distance(
                start_coords["lat"], start_coords["lon"],
                end_coords["lat"], end_coords["lon"]
            )
            duration = distance / 0.6  # Assume 36 km/h average urban speed
        else:
            distance, duration = result
        
        # Add buffer time for real-world conditions
        if include_buffer:
            duration += 15  # parking, traffic, etc.
        
        return duration