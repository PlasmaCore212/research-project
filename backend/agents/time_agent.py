# backend/agents/time_agent.py
"""
Time Management Agent with ReAct Pattern and Chain-of-Thought Prompting

This agent specializes in validating trip timelines and schedules.
It uses the ReAct pattern (Thought -> Action -> Observation) to:
1. Calculate transit times between locations
2. Build trip timelines
3. Check meeting reachability
4. Identify scheduling conflicts

References:
- ReAct: Synergizing Reasoning and Acting (Yao et al., 2023)
- Chain-of-Thought Prompting (Wei et al., 2022)
"""

from .base_agent import BaseReActAgent, AgentAction
from .models import (
    TimeCheckResult, TimeConflict,
    FlightSearchResult, HotelSearchResult, Meeting
)
from utils.routing import RoutingService
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json


class TimeManagementAgent(BaseReActAgent):
    """
    Agentic Time Management Agent with ReAct reasoning.
    
    This agent autonomously:
    - Calculates transit times between locations
    - Builds comprehensive trip timelines
    - Checks if meetings are reachable
    - Identifies scheduling conflicts and buffers
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        super().__init__(
            agent_name="TimeManagementAgent",
            agent_role="Travel Timeline Analyst",
            model_name=model_name,
            max_iterations=5,
            verbose=verbose
        )
        
        self.routing = RoutingService()
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, AgentAction]:
        """Register tools available to the Time Management Agent"""
        return {
            "calculate_transit_time": AgentAction(
                name="calculate_transit_time",
                description="Calculate driving time between two locations",
                parameters={
                    "from_lat": "float - starting latitude",
                    "from_lon": "float - starting longitude",
                    "to_lat": "float - destination latitude",
                    "to_lon": "float - destination longitude",
                    "include_buffer": "bool - add 15 min buffer for delays"
                },
                function=self._tool_calculate_transit_time
            ),
            "parse_flight_times": AgentAction(
                name="parse_flight_times",
                description="Parse flight departure and arrival times",
                parameters={
                    "departure_time": "str - departure time in HH:MM format",
                    "arrival_time": "str - arrival time in HH:MM format",
                    "departure_date": "str - date in YYYY-MM-DD format"
                },
                function=self._tool_parse_flight_times
            ),
            "check_meeting_reachability": AgentAction(
                name="check_meeting_reachability",
                description="Check if a meeting can be reached from hotel after flight arrival",
                parameters={
                    "hotel_arrival_time": "str - time arriving at hotel HH:MM",
                    "meeting_time": "str - meeting start time HH:MM",
                    "transit_minutes": "int - travel time from hotel to meeting in minutes"
                },
                function=self._tool_check_meeting_reachability
            ),
            "build_timeline": AgentAction(
                name="build_timeline",
                description="Build a complete trip timeline",
                parameters={
                    "flight_arrival": "str - flight arrival time HH:MM",
                    "airport_to_hotel_mins": "int - transit time to hotel",
                    "meetings": "list - list of meeting times HH:MM"
                },
                function=self._tool_build_timeline
            ),
            "analyze_buffer_times": AgentAction(
                name="analyze_buffer_times",
                description="Analyze buffer times between events",
                parameters={
                    "timeline": "dict - timeline with event times"
                },
                function=self._tool_analyze_buffer_times
            )
        }
    
    def _get_system_prompt(self) -> str:
        """Get the domain-specific system prompt for Time Management Agent"""
        return """You are an expert Travel Timeline Analyst AI Agent.

YOUR EXPERTISE:
- Calculating transit times between locations
- Building trip timelines with realistic buffers
- Identifying scheduling conflicts
- Ensuring meetings are reachable after travel

REASONING APPROACH (Chain-of-Thought):
When analyzing timelines, think through:
1. ARRIVAL: When does the flight land?
2. TRANSIT: How long to get from airport to hotel?
3. HOTEL ARRIVAL: When will the traveler actually reach the hotel?
4. MEETINGS: For each meeting, when must they leave the hotel?
5. BUFFERS: Is there enough time between events? (30+ min recommended)
6. CONFLICTS: Are there any overlapping or impossible schedules?

TIME BUFFER GUIDELINES:
- Airport to hotel: Add 15 min buffer for customs/baggage
- Hotel to meeting: Add 15 min buffer for traffic
- Between meetings: Minimum 30 min buffer recommended
- Before important meetings: 1 hour buffer ideal

CONFLICT SEVERITY:
- ERROR: Meeting is physically unreachable given timeline
- WARNING: Less than 30 min buffer (risky but possible)"""
    
    def _tool_calculate_transit_time(
        self,
        from_lat: float,
        from_lon: float,
        to_lat: float,
        to_lon: float,
        include_buffer: bool = True
    ) -> str:
        """Tool: Calculate transit time between two locations"""
        
        result = self.routing.get_transit_time(
            {"lat": from_lat, "lon": from_lon},
            {"lat": to_lat, "lon": to_lon},
            include_buffer=include_buffer
        )
        
        if result is None:
            # Fallback estimate based on distance
            from math import radians, cos, sin, asin, sqrt
            
            # Haversine formula
            lat1, lon1, lat2, lon2 = map(radians, [from_lat, from_lon, to_lat, to_lon])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            km = 6371 * c  # Earth radius in km
            
            # Estimate: 30 km/h average urban speed
            minutes = (km / 30) * 60
            if include_buffer:
                minutes += 15
            
            return f"Estimated transit time: {int(minutes)} minutes (based on {km:.1f}km distance)"
        
        return f"Transit time: {int(result)} minutes (including buffer)" if include_buffer else f"Transit time: {int(result)} minutes"
    
    def _tool_parse_flight_times(
        self,
        departure_time: str,
        arrival_time: str,
        departure_date: str
    ) -> str:
        """Tool: Parse and analyze flight times"""
        
        try:
            dep_hour, dep_min = map(int, departure_time.split(':'))
            arr_hour, arr_min = map(int, arrival_time.split(':'))
            
            # Check for overnight flight
            is_overnight = arr_hour < dep_hour
            
            arrival_date = departure_date
            if is_overnight:
                # Arrival is next day
                dep_dt = datetime.fromisoformat(departure_date)
                arr_dt = dep_dt + timedelta(days=1)
                arrival_date = arr_dt.strftime('%Y-%m-%d')
            
            # Calculate duration
            if is_overnight:
                duration_mins = (24 - dep_hour) * 60 - dep_min + arr_hour * 60 + arr_min
            else:
                duration_mins = (arr_hour - dep_hour) * 60 + (arr_min - dep_min)
            
            result = f"""Flight Time Analysis:
- Departure: {departure_time} on {departure_date}
- Arrival: {arrival_time} on {arrival_date}
- Duration: {duration_mins // 60}h {duration_mins % 60}m
- Overnight: {'Yes' if is_overnight else 'No'}"""
            
            # Store in beliefs
            self.state.add_belief("flight_arrival_time", arrival_time)
            self.state.add_belief("flight_arrival_date", arrival_date)
            self.state.add_belief("is_overnight_flight", is_overnight)
            
            return result
            
        except Exception as e:
            return f"Error parsing times: {e}"
    
    def _tool_check_meeting_reachability(
        self,
        hotel_arrival_time: str,
        meeting_time: str,
        transit_minutes: int
    ) -> str:
        """Tool: Check if meeting is reachable from hotel"""
        
        try:
            hotel_hour, hotel_min = map(int, hotel_arrival_time.split(':'))
            meeting_hour, meeting_min = map(int, meeting_time.split(':'))
            
            hotel_arrival_mins = hotel_hour * 60 + hotel_min
            meeting_start_mins = meeting_hour * 60 + meeting_min
            
            # When must they leave hotel?
            must_leave_mins = meeting_start_mins - transit_minutes
            must_leave_hour = must_leave_mins // 60
            must_leave_min = must_leave_mins % 60
            
            # Buffer available
            buffer_mins = must_leave_mins - hotel_arrival_mins
            
            if buffer_mins < 0:
                return f"""✗ CONFLICT: Meeting at {meeting_time} is UNREACHABLE
- Hotel arrival: {hotel_arrival_time}
- Must leave by: {must_leave_hour:02d}:{must_leave_min:02d}
- Transit time: {transit_minutes} min
- Buffer: {buffer_mins} min (IMPOSSIBLE - need to arrive {-buffer_mins} min earlier)"""
            elif buffer_mins < 30:
                return f"""⚠ WARNING: Tight schedule for meeting at {meeting_time}
- Hotel arrival: {hotel_arrival_time}
- Must leave by: {must_leave_hour:02d}:{must_leave_min:02d}
- Transit time: {transit_minutes} min
- Buffer: {buffer_mins} min (RISKY - less than 30 min)"""
            else:
                return f"""✓ OK: Meeting at {meeting_time} is reachable
- Hotel arrival: {hotel_arrival_time}
- Must leave by: {must_leave_hour:02d}:{must_leave_min:02d}
- Transit time: {transit_minutes} min
- Buffer: {buffer_mins} min (adequate)"""
                
        except Exception as e:
            return f"Error checking reachability: {e}"
    
    def _tool_build_timeline(
        self,
        flight_arrival: str,
        airport_to_hotel_mins: int,
        meetings: List[str]
    ) -> str:
        """Tool: Build a complete trip timeline"""
        
        try:
            arr_hour, arr_min = map(int, flight_arrival.split(':'))
            arr_total_mins = arr_hour * 60 + arr_min
            
            hotel_arrival_mins = arr_total_mins + airport_to_hotel_mins
            hotel_hour = hotel_arrival_mins // 60
            hotel_min = hotel_arrival_mins % 60
            
            timeline = {
                "flight_arrival": flight_arrival,
                "hotel_arrival": f"{hotel_hour:02d}:{hotel_min:02d}"
            }
            
            result_lines = [
                "Trip Timeline:",
                f"  {flight_arrival} - Flight arrives",
                f"  {hotel_hour:02d}:{hotel_min:02d} - Arrive at hotel (after {airport_to_hotel_mins} min transit)"
            ]
            
            for i, meeting in enumerate(meetings, 1):
                timeline[f"meeting_{i}"] = meeting
                result_lines.append(f"  {meeting} - Meeting {i}")
            
            # Store timeline
            self.state.add_belief("trip_timeline", timeline)
            
            return "\n".join(result_lines)
            
        except Exception as e:
            return f"Error building timeline: {e}"
    
    def _tool_analyze_buffer_times(self, timeline: Dict) -> str:
        """Tool: Analyze buffer times between events"""
        
        if not timeline:
            return "No timeline to analyze."
        
        events = []
        for key, value in timeline.items():
            try:
                hour, minute = map(int, value.split(':'))
                events.append((key, hour * 60 + minute))
            except:
                continue
        
        events.sort(key=lambda x: x[1])
        
        result_lines = ["Buffer Analysis:"]
        for i in range(len(events) - 1):
            current = events[i]
            next_event = events[i + 1]
            buffer = next_event[1] - current[1]
            
            status = "✓" if buffer >= 30 else "⚠" if buffer >= 0 else "✗"
            result_lines.append(
                f"  {status} {current[0]} → {next_event[0]}: {buffer} min"
            )
        
        return "\n".join(result_lines)
    
    def check_feasibility(
        self,
        flight_result: FlightSearchResult,
        hotel_result: HotelSearchResult,
        meetings: List[Meeting],
        arrival_city_coords: dict,
        airport_coords: dict,
        departure_date: str
    ) -> TimeCheckResult:
        """
        Main entry point for timeline feasibility check using ReAct reasoning.
        
        This method triggers the agentic ReAct loop to validate the trip timeline.
        """
        
        # Reset state for new check
        self.reset_state()
        
        # Validate inputs
        if not flight_result.flights or not hotel_result.hotels:
            return TimeCheckResult(
                is_feasible=False,
                conflicts=[TimeConflict(
                    conflict_type="missing_booking",
                    severity="error",
                    message="Missing flight or hotel selection"
                )],
                reasoning="Cannot check timeline without flight and hotel.",
                timeline={}
            )
        
        flight = flight_result.flights[0]
        hotel = hotel_result.hotels[0]
        
        # Format meetings info
        meetings_info = ""
        if meetings:
            meetings_info = "\nMEETINGS:\n" + "\n".join([
                f"  - {m.time} on {m.date}: {m.description or 'Meeting'} ({m.duration_minutes} min)"
                for m in meetings
            ])
        else:
            meetings_info = "\nNo meetings scheduled."
        
        # Build the goal
        goal = f"""Check if the following trip timeline is feasible:

FLIGHT:
- ID: {flight.flight_id}
- Departure: {flight.departure_time}
- Arrival: {flight.arrival_time}
- Date: {departure_date}

HOTEL:
- ID: {hotel.hotel_id}
- Location: {hotel.distance_to_business_center_km}km from business center
- Coordinates: {hotel.coordinates}

AIRPORT COORDINATES: {airport_coords}
{meetings_info}

Steps to follow:
1. Parse the flight times to understand arrival
2. Calculate transit time from airport to hotel
3. Build the trip timeline
4. For each meeting, check if it's reachable from the hotel
5. Analyze buffer times between events
6. Identify any conflicts or warnings

Return your final answer as a JSON object with:
- is_feasible: true/false
- conflicts: list of any issues found
- timeline: dict of event times"""

        # Run ReAct loop
        result = self.run(goal)
        
        # Build timeline and check conflicts
        conflicts = []
        timeline = self.state.get_belief("trip_timeline", {})
        
        # Fallback: do programmatic check if ReAct didn't produce results
        if not timeline:
            timeline, conflicts = self._fallback_check(
                flight, hotel, meetings, airport_coords, departure_date
            )
        
        # Check for conflicts in reasoning trace
        for step in result.get("reasoning_trace", []):
            if "CONFLICT" in step.observation or "UNREACHABLE" in step.observation:
                conflicts.append(TimeConflict(
                    conflict_type="meeting_unreachable",
                    severity="error",
                    message=step.observation[:100]
                ))
            elif "WARNING" in step.observation and "Tight" in step.observation:
                conflicts.append(TimeConflict(
                    conflict_type="insufficient_buffer",
                    severity="warning",
                    message=step.observation[:100]
                ))
        
        # Determine feasibility
        has_errors = any(c.severity == "error" for c in conflicts)
        is_feasible = not has_errors
        
        # Build reasoning
        reasoning = self._build_react_reasoning(
            flight, hotel, meetings, departure_date, result, timeline, conflicts
        )
        
        # Log message
        self.log_message(
            to_agent="orchestrator",
            content=f"Timeline: {'FEASIBLE' if is_feasible else 'CONFLICTS FOUND'}",
            msg_type="result"
        )
        
        return TimeCheckResult(
            is_feasible=is_feasible,
            conflicts=conflicts,
            reasoning=reasoning,
            timeline=timeline
        )
    
    def _fallback_check(
        self,
        flight,
        hotel,
        meetings: List[Meeting],
        airport_coords: dict,
        departure_date: str
    ) -> tuple:
        """Fallback programmatic check if ReAct fails"""
        
        conflicts = []
        timeline = {}
        
        # Parse flight times
        dep_hour, dep_min = map(int, flight.departure_time.split(':'))
        arr_hour, arr_min = map(int, flight.arrival_time.split(':'))
        
        # Handle overnight
        arrival_dt = datetime.fromisoformat(f"{departure_date}T{arr_hour:02d}:{arr_min:02d}:00")
        if arr_hour < dep_hour:
            arrival_dt += timedelta(days=1)
        
        # Calculate transit time
        transit_time = self.routing.get_transit_time(
            airport_coords,
            hotel.coordinates,
            include_buffer=True
        ) or 45  # Default 45 min
        
        hotel_arrival_dt = arrival_dt + timedelta(minutes=transit_time)
        
        timeline["flight_arrival"] = flight.arrival_time
        timeline["hotel_arrival"] = hotel_arrival_dt.strftime('%H:%M')
        
        # Check meetings
        for i, meeting in enumerate(meetings):
            meeting_dt = datetime.fromisoformat(f"{meeting.date}T{meeting.time}:00")
            
            # Estimate hotel to meeting transit (30 min default)
            hotel_to_meeting = 30
            
            must_leave_dt = meeting_dt - timedelta(minutes=hotel_to_meeting)
            timeline[f"meeting_{i+1}_departure"] = must_leave_dt.strftime('%H:%M')
            timeline[f"meeting_{i+1}_start"] = meeting.time
            
            buffer = (must_leave_dt - hotel_arrival_dt).total_seconds() / 60
            
            if buffer < 0:
                conflicts.append(TimeConflict(
                    conflict_type="meeting_unreachable",
                    severity="error",
                    message=f"Meeting at {meeting.time} unreachable - need {-int(buffer)} min earlier arrival"
                ))
            elif buffer < 30:
                conflicts.append(TimeConflict(
                    conflict_type="insufficient_buffer",
                    severity="warning",
                    message=f"Only {int(buffer)} min buffer before meeting at {meeting.time}"
                ))
        
        return timeline, conflicts
    
    def _build_react_reasoning(
        self,
        flight,
        hotel,
        meetings: List[Meeting],
        departure_date: str,
        react_result: Dict,
        timeline: Dict,
        conflicts: List[TimeConflict]
    ) -> str:
        """Build the full ReAct reasoning trace"""
        
        reasoning_parts = [
            f"## Timeline Feasibility ReAct Reasoning Trace",
            f"**Agent**: {self.agent_name}",
            f"**Flight**: {flight.flight_id} arriving {flight.arrival_time}",
            f"**Hotel**: {hotel.hotel_id} ({hotel.distance_to_business_center_km}km from center)",
            f"**Meetings**: {len(meetings)}",
            f"**Iterations**: {react_result.get('iterations', 0)}",
            "",
            "### Reasoning Steps:",
        ]
        
        for step in react_result.get("reasoning_trace", []):
            reasoning_parts.append(f"""
**Step {step.step_number}**:
- **Thought**: {step.thought}
- **Action**: `{step.action}({json.dumps(step.action_input) if step.action_input else ''})`
- **Observation**: {step.observation[:200]}{'...' if len(step.observation) > 200 else ''}
""")
        
        # Timeline summary
        if timeline:
            reasoning_parts.append("\n### Timeline:")
            for event, time in sorted(timeline.items(), key=lambda x: x[1]):
                reasoning_parts.append(f"- {time}: {event.replace('_', ' ').title()}")
        
        # Conflicts
        if conflicts:
            reasoning_parts.append("\n### Conflicts Found:")
            for c in conflicts:
                reasoning_parts.append(f"- [{c.severity.upper()}] {c.message}")
        else:
            reasoning_parts.append("\n### Result: ✓ Timeline is feasible")
        
        return "\n".join(reasoning_parts)
