# backend/agents/time_agent.py
"""Time Management Agent with ReAct Pattern for timeline validation."""

from .base_agent import BaseReActAgent, AgentAction
from .models import TimeCheckResult, TimeConflict, FlightSearchResult, HotelSearchResult, Meeting
from utils.routing import RoutingService
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from math import radians, cos, sin, asin, sqrt


class TimeManagementAgent(BaseReActAgent):
    """Agentic Time Management Agent for timeline feasibility checks."""
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        super().__init__(
            agent_name="TimeManagementAgent", agent_role="Travel Timeline Analyst",
            model_name=model_name, max_iterations=5, verbose=verbose
        )
        self.routing = RoutingService()
        self.tools = self._register_tools()
    
    def _should_stop_early(self, observation: str) -> bool:
        obs_lower = observation.lower()
        has_timeline = self.state.get_belief("trip_timeline") is not None
        signals = ["is feasible", "is not feasible", "feasible: true", "feasible: false",
                   "buffer analysis", "timeline is", "conflict", "unreachable"]
        return has_timeline and any(s in obs_lower for s in signals)
    
    def _extract_best_result_from_state(self) -> dict:
        return {"is_feasible": True, "timeline": self.state.get_belief("trip_timeline", {}),
                "reasoning": "Timeline analysis completed."}
    
    def _register_tools(self) -> Dict[str, AgentAction]:
        return {
            "calculate_transit_time": AgentAction(
                name="calculate_transit_time",
                description="Calculate driving time between two locations",
                parameters={"from_lat": "float", "from_lon": "float", "to_lat": "float",
                           "to_lon": "float", "include_buffer": "bool"},
                function=self._tool_calculate_transit_time
            ),
            "parse_flight_times": AgentAction(
                name="parse_flight_times",
                description="Parse flight departure and arrival times",
                parameters={"departure_time": "str HH:MM", "arrival_time": "str HH:MM",
                           "departure_date": "str YYYY-MM-DD"},
                function=self._tool_parse_flight_times
            ),
            "check_meeting_reachability": AgentAction(
                name="check_meeting_reachability",
                description="Check if meeting can be reached from hotel",
                parameters={"hotel_arrival_time": "str HH:MM", "meeting_time": "str HH:MM",
                           "transit_minutes": "int"},
                function=self._tool_check_meeting_reachability
            ),
            "build_timeline": AgentAction(
                name="build_timeline",
                description="Build a complete trip timeline",
                parameters={"flight_arrival": "str HH:MM", "airport_to_hotel_mins": "int",
                           "meetings": "list of HH:MM"},
                function=self._tool_build_timeline
            ),
            "analyze_buffer_times": AgentAction(
                name="analyze_buffer_times",
                description="Analyze buffer times between events",
                parameters={"timeline": "dict"},
                function=self._tool_analyze_buffer_times
            )
        }
    
    def _get_system_prompt(self) -> str:
        return """You are a Travel Timeline Analyst. Validate trip schedules.
BUFFERS: Airport→hotel +15min, hotel→meeting +15min, between meetings 30min+ recommended.
CONFLICTS: ERROR if unreachable, WARNING if <30min buffer."""
    
    def _tool_calculate_transit_time(self, from_lat: float, from_lon: float, to_lat: float,
                                     to_lon: float, include_buffer: bool = True) -> str:
        # Validate coordinates are reasonable (not 0 or invalid)
        try:
            from_lat = float(from_lat) if from_lat else 0
            from_lon = float(from_lon) if from_lon else 0
            to_lat = float(to_lat) if to_lat else 0
            to_lon = float(to_lon) if to_lon else 0
        except (ValueError, TypeError):
            return "Estimated: 45 min (default transit time)"
        
        # Check for invalid coordinates
        if abs(from_lat) < 1 or abs(to_lat) < 1 or abs(from_lon) < 1 or abs(to_lon) < 1:
            # Likely invalid coordinates, use default estimate
            return "Estimated: 45 min (default - coordinates not available)"
        
        result = self.routing.get_transit_time({"lat": from_lat, "lon": from_lon},
                                               {"lat": to_lat, "lon": to_lon}, include_buffer=include_buffer)
        if result is None:
            # Haversine fallback
            lat1, lon1, lat2, lon2 = map(radians, [from_lat, from_lon, to_lat, to_lon])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            km = 6371 * 2 * asin(sqrt(a))
            minutes = (km / 30) * 60 + (15 if include_buffer else 0)
            return f"Estimated: {int(minutes)} min ({km:.1f}km)"
        return f"Transit: {int(result)} min"
    
    def _tool_parse_flight_times(self, departure_time: str, arrival_time: str, departure_date: str) -> str:
        try:
            dep_h, dep_m = map(int, departure_time.split(':'))
            arr_h, arr_m = map(int, arrival_time.split(':'))
            is_overnight = arr_h < dep_h
            
            arrival_date = departure_date
            if is_overnight:
                arr_dt = datetime.fromisoformat(departure_date) + timedelta(days=1)
                arrival_date = arr_dt.strftime('%Y-%m-%d')
            
            duration = ((24 - dep_h) * 60 - dep_m + arr_h * 60 + arr_m) if is_overnight else ((arr_h - dep_h) * 60 + (arr_m - dep_m))
            
            self.state.add_belief("flight_arrival_time", arrival_time)
            self.state.add_belief("flight_arrival_date", arrival_date)
            
            return f"Departure: {departure_time} on {departure_date}, Arrival: {arrival_time} on {arrival_date}, Duration: {duration//60}h{duration%60}m"
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_check_meeting_reachability(self, hotel_arrival_time: str, meeting_time: str, transit_minutes: int) -> str:
        try:
            hotel_h, hotel_m = map(int, hotel_arrival_time.split(':'))
            meet_h, meet_m = map(int, meeting_time.split(':'))
            
            hotel_mins = hotel_h * 60 + hotel_m
            meeting_mins = meet_h * 60 + meet_m
            must_leave = meeting_mins - transit_minutes
            buffer = must_leave - hotel_mins
            
            if buffer < 0:
                return f"✗ CONFLICT: Meeting at {meeting_time} UNREACHABLE (need {-buffer}min earlier arrival)"
            elif buffer < 30:
                return f"⚠ WARNING: Only {buffer}min buffer before {meeting_time} meeting (risky)"
            return f"✓ OK: {buffer}min buffer before {meeting_time} meeting"
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_build_timeline(self, flight_arrival: str = None, airport_to_hotel_mins: int = None,
                             meetings: List[str] = None, **kwargs) -> str:
        """Build trip timeline. Accepts extra kwargs to handle LLM variations."""
        # Handle LLM passing alternate parameter names
        if flight_arrival is None:
            flight_arrival = kwargs.get('arrival_time', kwargs.get('arrival', '09:00'))
        if airport_to_hotel_mins is None:
            airport_to_hotel_mins = kwargs.get('transit_time', kwargs.get('transit_mins', 45))
        if meetings is None:
            meetings = kwargs.get('meeting_times', kwargs.get('meeting_list', []))
        
        # Ensure correct types
        if isinstance(airport_to_hotel_mins, str):
            try:
                airport_to_hotel_mins = int(airport_to_hotel_mins)
            except:
                airport_to_hotel_mins = 45
        
        meetings = meetings or []
        try:
            arr_h, arr_m = map(int, str(flight_arrival).split(':'))
            hotel_mins = arr_h * 60 + arr_m + airport_to_hotel_mins
            hotel_h, hotel_m = hotel_mins // 60, hotel_mins % 60
            
            timeline = {"flight_arrival": flight_arrival, "hotel_arrival": f"{hotel_h:02d}:{hotel_m:02d}"}
            result = [f"Timeline:", f"  {flight_arrival} - Flight arrives",
                     f"  {hotel_h:02d}:{hotel_m:02d} - Hotel arrival (+{airport_to_hotel_mins}min transit)"]
            
            for i, meeting in enumerate(meetings, 1):
                timeline[f"meeting_{i}"] = meeting
                result.append(f"  {meeting} - Meeting {i}")
            
            self.state.add_belief("trip_timeline", timeline)
            return "\n".join(result)
        except Exception as e:
            return f"Error: {e}"
    
    def _tool_analyze_buffer_times(self, timeline: Dict) -> str:
        if not timeline:
            return "No timeline."
        
        events = []
        for k, v in timeline.items():
            try:
                h, m = map(int, v.split(':'))
                events.append((k, h * 60 + m))
            except:
                continue
        
        events.sort(key=lambda x: x[1])
        result = ["Buffer Analysis:"]
        for i in range(len(events) - 1):
            buffer = events[i+1][1] - events[i][1]
            status = "✓" if buffer >= 30 else "⚠" if buffer >= 0 else "✗"
            result.append(f"  {status} {events[i][0]} → {events[i+1][0]}: {buffer}min")
        return "\n".join(result)
    
    def check_feasibility(self, flight_result: FlightSearchResult, hotel_result: HotelSearchResult,
                          meetings: List[Meeting], arrival_city_coords: dict, airport_coords: dict,
                          departure_date: str) -> TimeCheckResult:
        """Main entry point for timeline feasibility check."""
        self.reset_state()
        
        if not flight_result.flights or not hotel_result.hotels:
            return TimeCheckResult(is_feasible=False, conflicts=[TimeConflict(
                conflict_type="missing_booking", severity="error", message="Missing flight or hotel"
            )], reasoning="Cannot check without bookings.", timeline={})
        
        flight, hotel = flight_result.flights[0], hotel_result.hotels[0]
        
        meetings_info = "\n".join(f"  - {m.time} on {m.date}: {m.description or 'Meeting'}"
                                  for m in meetings) if meetings else "No meetings."
        
        goal = f"""Check timeline feasibility:
FLIGHT: {flight.flight_id}, {flight.departure_time}→{flight.arrival_time}, Date: {departure_date}
HOTEL: {hotel.hotel_id}, {hotel.distance_to_business_center_km}km from center
MEETINGS: {meetings_info}

1. Parse flight times 2. Calculate transit 3. Build timeline 4. Check meeting reachability 5. Analyze buffers
Return: {{"is_feasible": bool, "conflicts": [], "timeline": {{}}}}"""

        result = self.run(goal)
        
        timeline = self.state.get_belief("trip_timeline", {})
        conflicts = []
        
        if not timeline:
            timeline, conflicts = self._fallback_check(flight, hotel, meetings, airport_coords, departure_date)
        
        # Extract conflicts from reasoning trace
        for step in result.get("reasoning_trace", []):
            if "CONFLICT" in step.observation or "UNREACHABLE" in step.observation:
                conflicts.append(TimeConflict(conflict_type="meeting_unreachable", severity="error",
                                             message=step.observation[:100]))
            elif "WARNING" in step.observation and "buffer" in step.observation.lower():
                conflicts.append(TimeConflict(conflict_type="insufficient_buffer", severity="warning",
                                             message=step.observation[:100]))
        
        is_feasible = not any(c.severity == "error" for c in conflicts)
        
        self.log_message("orchestrator", f"Timeline: {'FEASIBLE' if is_feasible else 'CONFLICTS'}", "result")
        
        return TimeCheckResult(is_feasible=is_feasible, conflicts=conflicts,
                              reasoning=f"Checked {len(meetings)} meetings. {len(conflicts)} issues found.",
                              timeline=timeline)
    
    def _fallback_check(self, flight, hotel, meetings: List[Meeting], airport_coords: dict,
                        departure_date: str) -> tuple:
        """Programmatic fallback if ReAct fails."""
        conflicts, timeline = [], {}
        
        dep_h, dep_m = map(int, flight.departure_time.split(':'))
        arr_h, arr_m = map(int, flight.arrival_time.split(':'))
        
        arrival_dt = datetime.fromisoformat(f"{departure_date}T{arr_h:02d}:{arr_m:02d}:00")
        if arr_h < dep_h:
            arrival_dt += timedelta(days=1)
        
        transit = self.routing.get_transit_time(airport_coords, hotel.coordinates, include_buffer=True) or 45
        hotel_arrival_dt = arrival_dt + timedelta(minutes=transit)
        
        timeline["flight_arrival"] = flight.arrival_time
        timeline["hotel_arrival"] = hotel_arrival_dt.strftime('%H:%M')
        
        for i, meeting in enumerate(meetings):
            meeting_dt = datetime.fromisoformat(f"{meeting.date}T{meeting.time}:00")
            must_leave_dt = meeting_dt - timedelta(minutes=30)
            buffer = (must_leave_dt - hotel_arrival_dt).total_seconds() / 60
            
            timeline[f"meeting_{i+1}"] = meeting.time
            
            if buffer < 0:
                conflicts.append(TimeConflict(conflict_type="meeting_unreachable", severity="error",
                                             message=f"Meeting {meeting.time} unreachable by {-int(buffer)}min"))
            elif buffer < 30:
                conflicts.append(TimeConflict(conflict_type="insufficient_buffer", severity="warning",
                                             message=f"Only {int(buffer)}min buffer before {meeting.time}"))
        
        return timeline, conflicts
