# backend/agents/time_agent.py
from utils.routing import RoutingService
from .models import FlightSearchResult, HotelSearchResult
from typing import List, Tuple
from datetime import datetime, timedelta

class TimeManagementAgent:
    def __init__(self):
        self.routing = RoutingService()
    
    def check_feasibility(
        self,
        flight: FlightSearchResult,
        hotel: HotelSearchResult,
        meetings: List[dict]  # [{"time": "14:00", "location": {"lat": ..., "lon": ...}}]
    ) -> Tuple[bool, str]:
        """
        Check if itinerary is time-feasible
        
        Returns: (is_feasible, explanation)
        """
        from data.loaders import CITIES  # Import city coordinates
        
        flight_obj = flight.flights[0]
        hotel_obj = hotel.hotels[0]
        
        # Get coordinates
        arrival_city = CITIES[flight_obj.to_city]
        airport_coords = arrival_city["coordinates"]
        hotel_coords = hotel_obj.coordinates
        
        # 1. Check airport → hotel transit time
        airport_to_hotel_time = self.routing.get_transit_time(
            airport_coords, 
            {"lat": hotel_coords["lat"], "lon": hotel_coords["lon"]}
        )
        
        # Parse arrival time
        arrival_hour, arrival_min = map(int, flight_obj.arrival_time.split(':'))
        arrival_datetime = datetime.now().replace(
            hour=arrival_hour, 
            minute=arrival_min,
            second=0
        )
        hotel_arrival = arrival_datetime + timedelta(minutes=airport_to_hotel_time)
        
        conflicts = []
        
        # 2. Check each meeting
        for meeting in meetings:
            meeting_hour, meeting_min = map(int, meeting["time"].split(':'))
            meeting_datetime = datetime.now().replace(
                hour=meeting_hour,
                minute=meeting_min,
                second=0
            )
            
            # Check if we can reach meeting from hotel
            hotel_to_meeting_time = self.routing.get_transit_time(
                {"lat": hotel_coords["lat"], "lon": hotel_coords["lon"]},
                meeting["location"]
            )
            
            required_departure = meeting_datetime - timedelta(minutes=hotel_to_meeting_time)
            
            if required_departure < hotel_arrival:
                conflicts.append(
                    f"Meeting at {meeting['time']} unreachable - "
                    f"need to leave hotel at {required_departure.strftime('%H:%M')} "
                    f"but won't arrive until {hotel_arrival.strftime('%H:%M')}"
                )
        
        if conflicts:
            return False, "\n".join(conflicts)
        
        return True, f"All meetings reachable. Hotel arrival: {hotel_arrival.strftime('%H:%M')}"