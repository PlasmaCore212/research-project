"""
Helper functions and constants for the Trip Planning workflow.

This module contains shared utilities used across workflow nodes:
- Date/night calculations
- Model conversion helpers
- CNP message creation
- Constants for workflow limits
"""

from typing import Dict, Any, Tuple, Optional
from datetime import datetime
from agents.models import Flight, Hotel

# === CONSTANTS ===
MAX_BACKTRACKING_ITERATIONS = 10
MAX_NEGOTIATION_ROUNDS = 7


# === TIME PARSING UTILITIES ===

def parse_time_to_minutes(time_str: str) -> int:
    """
    Parse time string (HH:MM) to minutes since midnight.

    Args:
        time_str: Time in HH:MM format

    Returns:
        Minutes since midnight

    Raises:
        ValueError: If time format is invalid
    """
    try:
        hour, minute = map(int, time_str.split(':'))
        if not (0 <= hour < 24 and 0 <= minute < 60):
            raise ValueError(f"Invalid time: {time_str}")
        return hour * 60 + minute
    except Exception as e:
        raise ValueError(f"Invalid time format '{time_str}': must be HH:MM") from e


def minutes_to_time(minutes: int) -> str:
    """
    Convert minutes since midnight to time string (HH:MM).

    Args:
        minutes: Minutes since midnight

    Returns:
        Time in HH:MM format
    """
    minutes = max(0, minutes)  # Ensure non-negative
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def parse_meeting_time(meeting_time_str: str, default_date: str = "") -> Tuple[str, str]:
    """
    Parse meeting time string into date and time components.

    Args:
        meeting_time_str: Meeting time as "YYYY-MM-DD HH:MM" or "HH:MM"
        default_date: Default date to use if only time provided

    Returns:
        Tuple of (date_str, time_str)
    """
    if ' ' in str(meeting_time_str):
        date_part, time_part = str(meeting_time_str).split(' ', 1)
        return date_part, time_part
    else:
        return default_date, str(meeting_time_str)


def calculate_required_arrival_time(meeting_time_str: str, buffer_minutes: int = 165) -> Optional[str]:
    """
    Calculate required arrival time before a meeting.

    Args:
        meeting_time_str: Meeting time in HH:MM format
        buffer_minutes: Minutes before meeting to arrive (default 165 = 2h45m for transit + prep)

    Returns:
        Required arrival time in HH:MM format, or None if parsing fails
    """
    try:
        meeting_minutes = parse_time_to_minutes(meeting_time_str)
        required_minutes = meeting_minutes - buffer_minutes
        return minutes_to_time(required_minutes)
    except ValueError:
        return None


def calculate_nights(state_or_prefs: Dict) -> int:
    """Calculate nights from departure/return dates or check-in/check-out."""
    dep = state_or_prefs.get('departure_date')
    ret = state_or_prefs.get('return_date')
    if dep and ret:
        try:
            dep_dt = datetime.strptime(dep, '%Y-%m-%d')
            ret_dt = datetime.strptime(ret, '%Y-%m-%d')
            return max(1, (ret_dt - dep_dt).days)
        except:
            pass
    
    prefs = state_or_prefs.get('preferences', state_or_prefs)
    if prefs.get('hotel_checkin') and prefs.get('hotel_checkout'):
        try:
            checkin = datetime.strptime(prefs['hotel_checkin'], '%Y-%m-%d')
            checkout = datetime.strptime(prefs['hotel_checkout'], '%Y-%m-%d')
            return max(1, (checkout - checkin).days)
        except:
            pass
    return 1


def dict_to_flight(f: Dict, origin: str = '', dest: str = '') -> Flight:
    """Convert flight dict to Flight model."""
    return Flight(
        flight_id=f.get('flight_id', ''),
        airline=f.get('airline', ''),
        from_city=f.get('from_city', origin),
        to_city=f.get('to_city', dest),
        departure_time=f.get('departure_time', '09:00'),
        arrival_time=f.get('arrival_time', '12:00'),
        duration_hours=f.get('duration_hours', 3.0),
        price_usd=f.get('price_usd', 0),
        seats_available=f.get('seats_available', 10),
        **{'class': f.get('class', f.get('flight_class', 'Economy'))}
    )


def dict_to_hotel(h: Dict, dest: str = '') -> Hotel:
    """Convert hotel dict to Hotel model."""
    coords = h.get('coordinates', {'lat': 37.7749, 'lon': -122.4194})
    if 'lng' in coords and 'lon' not in coords:
        coords['lon'] = coords['lng']
    
    return Hotel(
        hotel_id=h.get('hotel_id', ''),
        name=h.get('name', ''),
        city=h.get('city', dest),
        city_name=h.get('city_name', dest),
        business_area=h.get('business_area', ''),
        tier=h.get('tier', 'standard'),
        stars=h.get('stars', 3),
        price_per_night_usd=h.get('price_per_night_usd', 0),
        distance_to_business_center_km=h.get('distance_to_business_center_km', 1.0),
        distance_to_airport_km=h.get('distance_to_airport_km', 10.0),
        amenities=h.get('amenities', []),
        rooms_available=h.get('rooms_available', 5),
        coordinates=coords
    )


def create_cnp_message(
    performative: str,
    sender: str,
    receiver: str,
    content: Dict[str, Any],
    conversation_id: str = None
) -> Dict[str, Any]:
    """
    Create a Contract Net Protocol / FIPA-ACL style message.
    
    Performatives:
    - cfp: Call for Proposals (task announcement)
    - propose: Agent proposal/bid
    - accept: Accept proposal
    - reject: Reject proposal
    - inform: Inform of results
    - request: Request action from another agent
    - failure: Report failure
    """
    return {
        "performative": performative,
        "sender": sender,
        "receiver": receiver,
        "content": content,
        "conversation_id": conversation_id or f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat()
    }
