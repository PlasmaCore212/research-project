from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime, time as time_type

# flight model
class FlightQuery(BaseModel):
    from_city: str
    to_city: str
    max_price: Optional[int] = None
    departure_after: Optional[str] = None
    departure_before: Optional[str] = None
    class_preference: Optional[str] = "Economy"
    
    @validator('departure_after', 'departure_before')
    def validate_time_format(cls, v):
        if v is not None:
            try:
                hour, minute = map(int, v.split(':'))
                if not (0 <= hour < 24 and 0 <= minute < 60):
                    raise ValueError
            except:
                raise ValueError('Time must be in HH:MM format (e.g., "09:00")')
        return v

class Flight(BaseModel):
    flight_id: str
    airline: str
    from_city: str
    to_city: str
    departure_time: str
    arrival_time: str
    duration_hours: float
    price_usd: int
    seats_available: int
    flight_class: str = Field(alias="class")

class FlightSearchResult(BaseModel):
    query: FlightQuery
    flights: List[Flight]
    reasoning: str

class HotelQuery(BaseModel):
    city: str
    max_price_per_night: Optional[int] = None
    min_stars: Optional[int] = None
    max_distance_to_center_km: Optional[float] = None
    required_amenities: Optional[List[str]] = None
    
    @validator('min_stars')
    def validate_stars(cls, v):
        if v is not None and not (1 <= v <= 5):
            raise ValueError('Stars must be between 1 and 5')
        return v

# hotel model
class Hotel(BaseModel):
    hotel_id: str
    name: str
    city: str
    city_name: str
    business_area: str
    tier: str
    stars: int
    price_per_night_usd: int
    distance_to_business_center_km: float
    distance_to_airport_km: float
    amenities: List[str]
    rooms_available: int
    coordinates: Dict
class HotelSearchResult(BaseModel):
    query: HotelQuery
    hotels: List[Hotel]
    reasoning: str

# backend/agents/models.py
class Meeting(BaseModel):
    date: str  # "2025-01-15" format (ISO)
    time: str  # "14:30" format
    location: dict
    duration_minutes: int = 60
    description: Optional[str] = None
    
    @validator('date')
    def validate_date_format(cls, v):
        try:
            datetime.fromisoformat(v)
        except:
            raise ValueError('Date must be in YYYY-MM-DD format')
        return v

class TimeConflict(BaseModel):
    conflict_type: str  # "meeting_unreachable", "insufficient_buffer", "overlap"
    severity: str  # "error" or "warning"
    message: str
    required_time: Optional[str] = None
    actual_time: Optional[str] = None

class TimeCheckResult(BaseModel):
    is_feasible: bool
    conflicts: List[TimeConflict]
    reasoning: str
    timeline: Dict[str, str]  # {"hotel_arrival": "11:30", "meeting_1_departure": "13:45", ...}
