# backend/agents/time_agent.py
from langchain_community.llms import ollama
from .models import TimeCheckResult, TimeConflict, FlightSearchResult, HotelSearchResult, Meeting
from utils.routing import RoutingService
from typing import List, Dict
from datetime import datetime, timedelta
import json


class TimeManagementAgent:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.llm = ollama.Ollama(
            model=model_name,
            temperature=0.0,
            format="json"
        )
        self.routing = RoutingService()
    
    def check_feasibility(
        self,
        flight_result: FlightSearchResult,
        hotel_result: HotelSearchResult,
        meetings: List[Meeting],
        arrival_city_coords: dict,  # {"lat": float, "lon": float}
        airport_coords: dict,
        departure_date: str
    ) -> TimeCheckResult:
        """
        Check if trip timeline is feasible given flight, hotel, and meetings
        """
        
        if not flight_result.flights or not hotel_result.hotels:
            return TimeCheckResult(
                is_feasible=False,
                conflicts=[],
                reasoning="No flight or hotel options available",
                timeline={}
            )
        
        flight = flight_result.flights[0]
        hotel = hotel_result.hotels[0]
        
        # Build timeline and check for conflicts
        conflicts = []
        timeline = {}
        
        # 1. Calculate airport → hotel transit
        airport_to_hotel_time = self.routing.get_transit_time(
            airport_coords,
            hotel.coordinates,
            include_buffer=True
        )
        
        # Handle overnight flights (arrival time < departure time means next day)
        departure_hour, departure_min = map(int, flight.departure_time.split(':'))
        arrival_hour, arrival_min = map(int, flight.arrival_time.split(':'))

        # Create arrival datetime
        arrival_dt = datetime.fromisoformat(f"{departure_date}T{arrival_hour:02d}:{arrival_min:02d}:00")
        if arrival_hour < departure_hour:
            arrival_dt += timedelta(days=1)

        hotel_arrival_dt = arrival_dt + timedelta(minutes=airport_to_hotel_time)
        timeline["flight_arrival"] = flight.arrival_time
        timeline["hotel_arrival"] = hotel_arrival_dt.strftime('%H:%M')
        
        # 2. Check each meeting
        for i, meeting in enumerate(meetings):
            meeting_hour, meeting_min = map(int, meeting.time.split(':'))
            meeting_dt = datetime.fromisoformat(f"{meeting.date}T{meeting.time}:00")
            
            # Calculate hotel → meeting transit
            hotel_to_meeting_time = self.routing.get_transit_time(
                hotel.coordinates,
                meeting.location,
                include_buffer=True
            )
            
            # When do we need to leave hotel?
            required_departure_dt = meeting_dt - timedelta(minutes=hotel_to_meeting_time)
            timeline[f"meeting_{i+1}_departure"] = required_departure_dt.strftime('%H:%M')
            timeline[f"meeting_{i+1}_start"] = meeting.time
            
            # Check if we can make it
            if required_departure_dt < hotel_arrival_dt:
                conflicts.append(TimeConflict(
                    conflict_type="meeting_unreachable",
                    severity="error",
                    message=f"Meeting at {meeting.time} unreachable from hotel",
                    required_time=required_departure_dt.strftime('%H:%M'),
                    actual_time=hotel_arrival_dt.strftime('%H:%M')
                ))
            
            # Check for tight timing (less than 30min buffer)
            buffer_minutes = (required_departure_dt - hotel_arrival_dt).total_seconds() / 60
            if 0 < buffer_minutes < 30:
                conflicts.append(TimeConflict(
                    conflict_type="insufficient_buffer",
                    severity="warning",
                    message=f"Only {int(buffer_minutes)} minutes between hotel arrival and meeting departure",
                    required_time=required_departure_dt.strftime('%H:%M'),
                    actual_time=hotel_arrival_dt.strftime('%H:%M')
                ))
        
        # 3. Get LLM analysis
        if conflicts:
            analysis = self._analyze_conflicts(flight, hotel, meetings, conflicts, timeline)
        else:
            analysis = "All meetings are reachable with adequate time buffers."
        
        reasoning = self._build_reasoning(
            flight, hotel, meetings, conflicts, timeline, 
            airport_to_hotel_time, analysis
        )
        
        return TimeCheckResult(
            is_feasible=len([c for c in conflicts if c.severity == "error"]) == 0,
            conflicts=conflicts,
            reasoning=reasoning,
            timeline=timeline
        )
    
    def _analyze_conflicts(
        self,
        flight,
        hotel,
        meetings: List[Meeting],
        conflicts: List[TimeConflict],
        timeline: Dict
    ) -> str:
        """Get LLM analysis of timeline conflicts"""
        
        conflicts_list = [
            {
                "type": c.conflict_type,
                "severity": c.severity,
                "message": c.message,
                "required": c.required_time,
                "actual": c.actual_time
            }
            for c in conflicts
        ]
        
        prompt = f"""You are a Travel Timeline Analyst. Analyze these scheduling conflicts.

Flight arrives: {flight.arrival_time}
Hotel arrival: {timeline['hotel_arrival']}
Meetings: {[m.time for m in meetings]}

Timeline conflicts:
{json.dumps(conflicts_list, indent=2)}

Provide a brief analysis of the timeline issues and suggest if an earlier flight or closer hotel would resolve them.

Return ONLY this JSON format (no other text):
{{
  "analysis": "Brief explanation of timeline issues and recommendations"
}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            return result.get("analysis", "Timeline conflicts detected")
        except:
            return "Timeline conflicts detected"
    
    def _build_reasoning(
        self,
        flight,
        hotel,
        meetings: List[Meeting],
        conflicts: List[TimeConflict],
        timeline: Dict,
        transit_time: float,
        llm_analysis: str
    ) -> str:
        """Build ReAct-style reasoning chain"""
        
        conflicts_str = "\n".join([
            f"- [{c.severity.upper()}] {c.message}: Need to leave at {c.required_time}, "
            f"but arrival at {c.actual_time}"
            for c in conflicts
        ]) if conflicts else "No conflicts"
        
        timeline_str = "\n".join([
            f"- {k.replace('_', ' ').title()}: {v}"
            for k, v in timeline.items()
        ])
        
        return f"""**Thought**: Checking if traveler can reach all meetings after flight arrival
- Flight: {flight.flight_id} arrives at {flight.arrival_time}
- Hotel: {hotel.hotel_id} ({hotel.distance_to_business_center_km:.2f}km from center)
- Meetings: {len(meetings)} scheduled

**Action**: Calculate transit times and build timeline

**Observation**: Airport → Hotel takes {transit_time:.0f} minutes
Timeline:
{timeline_str}

**Analysis**: {llm_analysis}

Conflicts found:
{conflicts_str}

**Final Answer**: Trip is {'FEASIBLE' if len([c for c in conflicts if c.severity == 'error']) == 0 else 'NOT FEASIBLE'}"""