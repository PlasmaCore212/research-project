"""
FastAPI Backend for Agentic Trip Planning System
Structured input approach - safe and reliable
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Any, Dict
from datetime import date, time
from pathlib import Path
import uvicorn

# Import the orchestrator graph
from orchestrator.graph import build_workflow
from orchestrator.state import create_initial_state

app = FastAPI(
    title="Agentic Trip Planner API",
    description="Multi-Agent Trip Planning with ReAct Reasoning",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ==================== Helper Functions ====================

import math
import httpx

# OSRM public routing API (uses OpenStreetMap data)
OSRM_URL = "http://router.project-osrm.org/route/v1/driving"


async def get_route_distance(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> Dict[str, Any]:
    """
    Get actual street distance and travel time using OSRM.
    Returns distance in km and duration in minutes.
    """
    try:
        url = f"{OSRM_URL}/{from_lon},{from_lat};{to_lon},{to_lat}?overview=false"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            data = response.json()
        
        if data.get("code") == "Ok" and data.get("routes"):
            route = data["routes"][0]
            return {
                "distance_km": round(route["distance"] / 1000, 2),  # meters to km
                "duration_minutes": round(route["duration"] / 60, 1),  # seconds to minutes
                "success": True
            }
    except Exception as e:
        print(f"OSRM routing error: {e}")
    
    return {"success": False, "distance_km": None, "duration_minutes": None}


async def calculate_hotel_distances(hotels: List[Dict], meeting_coords: Dict) -> List[Dict]:
    """Add actual street distance and travel time to meeting location for each hotel."""
    if not meeting_coords:
        return hotels
    
    meeting_lat = meeting_coords.get('lat')
    meeting_lon = meeting_coords.get('lon')
    
    if not meeting_lat or not meeting_lon:
        return hotels
    
    for hotel in hotels:
        if 'coordinates' in hotel:
            hotel_lat = hotel['coordinates'].get('lat')
            hotel_lon = hotel['coordinates'].get('lon')
            if hotel_lat and hotel_lon:
                route = await get_route_distance(hotel_lat, hotel_lon, meeting_lat, meeting_lon)
                if route["success"]:
                    hotel['distance_to_meeting_km'] = route["distance_km"]
                    hotel['travel_time_minutes'] = route["duration_minutes"]
                else:
                    # Fallback to simple estimate if routing fails
                    hotel['distance_to_meeting_km'] = None
                    hotel['travel_time_minutes'] = None
    
    return hotels


# ==================== Request Models ====================

class TripRequest(BaseModel):
    """Structured trip planning request - all fields validated"""
    
    # Flight details (required)
    origin: str = Field(..., min_length=2, max_length=10, description="Origin city code (e.g., NYC)")
    destination: str = Field(..., min_length=2, max_length=10, description="Destination city code (e.g., SFO)")
    departure_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(None, description="Return date (YYYY-MM-DD)")
    
    # Hotel details (required)
    hotel_checkin: str = Field(..., description="Hotel check-in date (YYYY-MM-DD)")
    hotel_checkout: str = Field(..., description="Hotel check-out date (YYYY-MM-DD)")
    hotel_location: Optional[str] = Field(None, description="Preferred hotel area")
    
    # Meeting details (required)
    meeting_time: str = Field(..., description="Meeting time (HH:MM)")
    meeting_date: str = Field(..., description="Meeting date (YYYY-MM-DD)")
    meeting_address: str = Field(..., description="Full meeting address")
    meeting_coordinates: Optional[Dict[str, Any]] = Field(None, description="Geocoded coordinates {lat, lon}")
    
    # Budget and preferences (required)
    budget: float = Field(..., gt=0, le=50000, description="Maximum budget in USD")
    trip_type: str = Field(..., description="Trip type: 'business' or 'personal'")
    preferences: Optional[str] = Field(None, description="Additional preferences")
    
    @field_validator('origin', 'destination')
    @classmethod
    def validate_city_code(cls, v: str) -> str:
        """Ensure city codes are uppercase and valid"""
        v = v.upper().strip()
        # Only allow cities that exist in our data
        valid_codes = {'NYC', 'BOS', 'CHI', 'SF'}
        if v not in valid_codes:
            raise ValueError(f"City must be one of: NYC, BOS, CHI, SF")
        return v
    
    @field_validator('trip_type')
    @classmethod
    def validate_trip_type(cls, v: str) -> str:
        """Ensure trip type is valid"""
        v = v.lower().strip()
        if v not in ('business', 'personal'):
            raise ValueError("Trip type must be 'business' or 'personal'")
        return v


class Coordinates(BaseModel):
    """Geocoded coordinates"""
    lat: float
    lon: float
    display_name: Optional[str] = None


class TripResponse(BaseModel):
    """Response from trip planning"""
    success: bool
    selected_flight: Optional[Dict[str, Any]] = None
    selected_hotel: Optional[Dict[str, Any]] = None
    policy_check: Optional[Dict[str, Any]] = None
    time_check: Optional[Dict[str, Any]] = None
    total_cost: Optional[float] = None
    reasoning_trace: Optional[List[Dict[str, Any]]] = None
    agent_traces: Optional[Dict[str, List[Dict[str, Any]]]] = None
    errors: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    agents: List[str]
    version: str


# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Serve the frontend"""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "Agentic Trip Planner API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and agent health"""
    return HealthResponse(
        status="healthy",
        agents=["FlightAgent", "HotelAgent", "PolicyAgent", "TimeAgent", "Orchestrator"],
        version="1.0.0"
    )


@app.post("/api/plan-trip", response_model=TripResponse)
async def plan_trip(request: TripRequest):
    """
    Plan a trip using the multi-agent system.
    
    This endpoint accepts structured input and runs the full
    agentic workflow with ReAct reasoning.
    """
    try:
        # Validate origin != destination
        if request.origin == request.destination:
            raise HTTPException(
                status_code=400, 
                detail="Origin and destination cannot be the same"
            )
        
        # Build the workflow graph
        workflow = build_workflow()
        graph = workflow.compile()
        
        # Create user request string for the agents
        user_request = f"Plan a {request.trip_type} trip from {request.origin} to {request.destination} on {request.departure_date}. Meeting at {request.meeting_time} at {request.meeting_address}. Budget: ${request.budget}."
        if request.preferences:
            user_request += f" Preferences: {request.preferences}"
        
        # Create initial state from request
        initial_state = create_initial_state(
            user_request=user_request,
            origin=request.origin,
            destination=request.destination,
            departure_date=request.departure_date,
            return_date=request.return_date,
            budget=request.budget,
            preferences={
                "meeting_time": request.meeting_time,
                "meeting_date": request.meeting_date,
                "meeting_address": request.meeting_address,
                "meeting_coordinates": request.meeting_coordinates,
                "hotel_location": request.hotel_location,
                "hotel_checkin": request.hotel_checkin,
                "hotel_checkout": request.hotel_checkout,
                "trip_type": request.trip_type
            }
        )
        
        # Run the graph
        final_state = graph.invoke(initial_state)
        
        # Extract results
        selected_flight = None
        selected_hotel = None
        policy_result = None
        time_result = None
        agent_traces = {}
        
        # Get flight result
        if final_state.get("flight_options"):
            flights = final_state["flight_options"]
            if isinstance(flights, list) and len(flights) > 0:
                selected_flight = flights[0] if isinstance(flights[0], dict) else flights[0].__dict__ if hasattr(flights[0], '__dict__') else None
        
        # Get hotel result - calculate distances to meeting  
        if final_state.get("hotel_options"):
            hotels = final_state["hotel_options"]
            # Convert to dicts if needed
            hotel_dicts = []
            for h in hotels:
                if isinstance(h, dict):
                    hotel_dicts.append(h)
                elif hasattr(h, '__dict__'):
                    hotel_dicts.append(h.__dict__)
            
            # Add actual street distance to meeting location using OSRM
            if request.meeting_coordinates:
                hotel_dicts = await calculate_hotel_distances(hotel_dicts, request.meeting_coordinates)
                # Sort by travel time (more useful than distance)
                hotel_dicts.sort(key=lambda x: x.get('travel_time_minutes') or 999)
            
            if hotel_dicts:
                selected_hotel = hotel_dicts[0]
        
        # Get policy check result
        if final_state.get("compliance_status"):
            policy_result = final_state["compliance_status"]
            if hasattr(policy_result, '__dict__'):
                policy_result = policy_result.__dict__
        
        # Get time result
        if final_state.get("time_result"):
            time_result = final_state["time_result"]
            if hasattr(time_result, '__dict__'):
                time_result = time_result.__dict__
        
        # Collect agent reasoning traces
        if final_state.get("reasoning_steps"):
            agent_traces["orchestrator"] = final_state["reasoning_steps"]
        
        # Calculate total cost
        total_cost = None
        if selected_flight and selected_hotel:
            flight_price = selected_flight.get('price', 0) if isinstance(selected_flight, dict) else 0
            hotel_price = selected_hotel.get('price_per_night', 0) if isinstance(selected_hotel, dict) else 0
            total_cost = flight_price + hotel_price
        
        return TripResponse(
            success=True,
            selected_flight=selected_flight,
            selected_hotel=selected_hotel,
            policy_check=policy_result,
            time_check=time_result,
            total_cost=total_cost,
            agent_traces=agent_traces if agent_traces else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TripResponse(
            success=False,
            errors=[str(e)]
        )


@app.get("/api/cities", response_model=List[Dict[str, str]])
async def get_cities():
    """Get list of available cities - limited to proof of concept data"""
    cities = [
        {"code": "NYC", "name": "New York City", "airports": ["JFK", "LGA", "EWR"]},
        {"code": "BOS", "name": "Boston", "airports": ["BOS"]},
        {"code": "CHI", "name": "Chicago", "airports": ["ORD", "MDW"]},
        {"code": "SF", "name": "San Francisco", "airports": ["SFO", "OAK"]},
    ]
    return cities


# ==================== Run Server ====================

if __name__ == "__main__":
    print("🚀 Starting Agentic Trip Planner API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Frontend: http://localhost:8000")
    print("📊 Available cities: NYC, BOS, CHI, SF")
    uvicorn.run(app, host="0.0.0.0", port=8000)
