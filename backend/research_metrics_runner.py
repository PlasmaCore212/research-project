#!/usr/bin/env python3
"""
Research Metrics Runner for Multi-Agent Trip Planning System

Collects metrics to answer research sub-questions:
1. Time Agent impact on success rate and iterations
2. Feedback loops between Policy and booking agents
3. Message exchanges across complexity levels
4. Trip complexity vs. planning time

Author: Research Project - Laureys Indy
"""

import csv
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.policy_agent import PolicyComplianceAgent
from agents.time_agent import TimeManagementAgent
from agents.models import FlightQuery, HotelQuery, Meeting, Flight, Hotel
from data.loaders import FlightDataLoader, HotelDataLoader


@dataclass
class TestScenario:
    """Defines a test scenario for metrics collection."""
    scenario_id: str
    scenario_name: str
    origin: str
    destination: str
    budget: float
    nights: int
    num_meetings: int
    trip_complexity: str  # simple, medium, complex
    with_time_agent: bool
    meetings: List[Dict] = None
    
    def __post_init__(self):
        if self.meetings is None:
            self.meetings = []


@dataclass
class TestMetrics:
    """All metrics collected from a single test run."""
    # Test identification
    test_id: str
    timestamp: str
    scenario_id: str
    scenario_name: str
    
    # Input parameters
    origin: str
    destination: str
    budget: float
    nights: int
    num_meetings: int
    trip_complexity: str
    with_time_agent: bool
    
    # Overall results
    planning_success: bool
    planning_time_seconds: float
    error_message: str = ""
    
    # Flight Agent metrics
    flight_agent_iterations: int = 0
    flight_options_found: int = 0
    flight_search_time_seconds: float = 0
    
    # Hotel Agent metrics
    hotel_agent_iterations: int = 0
    hotel_options_found: int = 0
    hotel_search_time_seconds: float = 0
    
    # Policy Agent metrics
    policy_checks_performed: int = 0
    combinations_evaluated: int = 0
    valid_combinations_found: int = 0
    violations_found: int = 0
    
    # Time Agent metrics
    time_agent_enabled: bool = False
    time_conflicts_found: int = 0
    timeline_feasible: bool = True
    
    # Selection results
    selected_flight_id: str = ""
    selected_flight_price: float = 0
    selected_hotel_id: str = ""
    selected_hotel_price: float = 0
    total_cost: float = 0
    budget_remaining: float = 0
    budget_utilization: float = 0
    quality_score: float = 0
    value_score: float = 0
    
    # Backtracking/iteration metrics (Sub-question 2)
    backtracking_count: int = 0
    total_iterations: int = 0


class ResearchMetricsRunner:
    """Runs test scenarios and collects comprehensive metrics."""
    
    def __init__(self, output_file: str = "research_metrics.csv", verbose: bool = True):
        self.output_file = output_file
        self.verbose = verbose
        self.results: List[TestMetrics] = []
        
        # Initialize data loaders to check available data
        self.flight_loader = FlightDataLoader()
        self.hotel_loader = HotelDataLoader()
        
    def log(self, message: str):
        if self.verbose:
            print(f"[MetricsRunner] {message}")
    
    def run_single_test(self, scenario: TestScenario) -> TestMetrics:
        """Run a single test scenario and collect all metrics."""
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.log(f"\n{'='*60}")
        self.log(f"Running: {scenario.scenario_name}")
        self.log(f"  Route: {scenario.origin} → {scenario.destination}")
        self.log(f"  Budget: ${scenario.budget}, Nights: {scenario.nights}")
        self.log(f"  Complexity: {scenario.trip_complexity}, TimeAgent: {scenario.with_time_agent}")
        self.log(f"{'='*60}")
        
        # Initialize metrics
        metrics = TestMetrics(
            test_id=test_id,
            timestamp=datetime.now().isoformat(),
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.scenario_name,
            origin=scenario.origin,
            destination=scenario.destination,
            budget=scenario.budget,
            nights=scenario.nights,
            num_meetings=scenario.num_meetings,
            trip_complexity=scenario.trip_complexity,
            with_time_agent=scenario.with_time_agent,
            planning_success=False,
            planning_time_seconds=0,
            time_agent_enabled=scenario.with_time_agent
        )
        
        start_time = time.time()
        
        try:
            # Initialize agents (fresh for each test)
            flight_agent = FlightAgent(verbose=self.verbose)
            hotel_agent = HotelAgent(verbose=self.verbose)
            policy_agent = PolicyComplianceAgent(verbose=self.verbose)
            time_agent = TimeManagementAgent(verbose=self.verbose) if scenario.with_time_agent else None
            
            # Step 1: Flight search
            self.log("\n--- PHASE 1: Flight Search ---")
            flight_start = time.time()
            flight_query = FlightQuery(
                from_city=scenario.origin,
                to_city=scenario.destination,
                max_price=int(scenario.budget * 0.6),  # Allow up to 60% for flight
                departure_after="06:00",
                departure_before="21:00"
            )
            flight_result = flight_agent.search_flights(flight_query)
            flight_end = time.time()
            
            metrics.flight_search_time_seconds = round(flight_end - flight_start, 2)
            metrics.flight_options_found = len(flight_result.flights)
            metrics.flight_agent_iterations = len(flight_agent.state.reasoning_trace)
            
            self.log(f"  Found {metrics.flight_options_found} flights in {metrics.flight_search_time_seconds}s")
            
            # Step 2: Hotel search
            self.log("\n--- PHASE 2: Hotel Search ---")
            hotel_start = time.time()
            hotel_query = HotelQuery(
                city=scenario.destination,
                max_price_per_night=int(scenario.budget * 0.5 / scenario.nights),
                min_stars=3
            )
            hotel_result = hotel_agent.search_hotels(hotel_query)
            hotel_end = time.time()
            
            metrics.hotel_search_time_seconds = round(hotel_end - hotel_start, 2)
            metrics.hotel_options_found = len(hotel_result.hotels)
            metrics.hotel_agent_iterations = len(hotel_agent.state.reasoning_trace)
            
            self.log(f"  Found {metrics.hotel_options_found} hotels in {metrics.hotel_search_time_seconds}s")
            
            # Step 3: Policy Agent - Find best combination
            self.log("\n--- PHASE 3: Policy Evaluation ---")
            if flight_result.flights and hotel_result.hotels:
                # Convert Flight/Hotel objects to dicts for policy agent
                flights_data = [
                    {
                        "flight_id": f.flight_id,
                        "airline": f.airline,
                        "price_usd": f.price_usd,
                        "departure_time": f.departure_time,
                        "arrival_time": f.arrival_time,
                        "duration_hours": f.duration_hours,
                        "from_city": f.from_city,
                        "to_city": f.to_city
                    }
                    for f in flight_result.flights
                ]
                
                hotels_data = [
                    {
                        "hotel_id": h.hotel_id,
                        "name": h.name,
                        "stars": h.stars,
                        "price_per_night_usd": h.price_per_night_usd,
                        "distance_to_business_center_km": h.distance_to_business_center_km,
                        "business_area": h.business_area,
                        "coordinates": h.coordinates
                    }
                    for h in hotel_result.hotels
                ]
                
                combination_result = policy_agent.find_best_combination(
                    flights=flights_data,
                    hotels=hotels_data,
                    budget=scenario.budget,
                    nights=scenario.nights
                )
                
                # Collect policy metrics
                policy_metrics = policy_agent.get_metrics()
                metrics.policy_checks_performed = policy_metrics.get("checks_performed", 0)
                metrics.combinations_evaluated = policy_metrics.get("combinations_evaluated", 0)
                metrics.valid_combinations_found = policy_metrics.get("valid_combinations_found", 0)
                
                self.log(f"  Evaluated {metrics.combinations_evaluated} combinations")
                self.log(f"  Valid within budget: {metrics.valid_combinations_found}")
                
                if combination_result.success:
                    metrics.selected_flight_id = combination_result.selected_flight.get("flight_id", "")
                    metrics.selected_flight_price = combination_result.selected_flight.get("price_usd", 0)
                    metrics.selected_hotel_id = combination_result.selected_hotel.get("hotel_id", "")
                    metrics.selected_hotel_price = combination_result.selected_hotel.get("price_per_night_usd", 0)
                    metrics.total_cost = combination_result.total_cost
                    metrics.budget_remaining = combination_result.budget_remaining
                    metrics.budget_utilization = round(combination_result.total_cost / scenario.budget, 3) if scenario.budget > 0 else 0
                    metrics.value_score = combination_result.value_score
                    
                    self.log(f"  Selected: {metrics.selected_flight_id} + {metrics.selected_hotel_id}")
                    self.log(f"  Total: ${metrics.total_cost}, Remaining: ${metrics.budget_remaining}")
                    
                    # Step 4: Time Agent (if enabled)
                    if scenario.with_time_agent and time_agent and scenario.meetings:
                        self.log("\n--- PHASE 4: Timeline Validation ---")
                        
                        # Create Meeting objects
                        meetings = [
                            Meeting(
                                date=m.get("date", "2026-01-15"),
                                time=m.get("time", "14:00"),
                                location=m.get("location", {}),
                                duration_minutes=m.get("duration_minutes", 60),
                                description=m.get("description", "Business meeting")
                            )
                            for m in scenario.meetings
                        ]
                        
                        # Get coordinates (simplified)
                        airport_coords = {"lat": 37.7749, "lon": -122.4194}  # SF default
                        
                        time_result = time_agent.check_feasibility(
                            flight_result=flight_result,
                            hotel_result=hotel_result,
                            meetings=meetings,
                            arrival_city_coords=airport_coords,
                            airport_coords=airport_coords,
                            departure_date="2026-01-15"
                        )
                        
                        metrics.timeline_feasible = time_result.is_feasible
                        metrics.time_conflicts_found = len([c for c in time_result.conflicts if c.severity == "error"])
                        
                        self.log(f"  Timeline feasible: {metrics.timeline_feasible}")
                        self.log(f"  Conflicts found: {metrics.time_conflicts_found}")
                    
                    metrics.planning_success = True
                else:
                    metrics.error_message = "No valid combinations within budget"
                    metrics.violations_found = 1
            else:
                metrics.error_message = "No flights or hotels found"
                
        except Exception as e:
            metrics.error_message = str(e)
            self.log(f"  ERROR: {e}")
        
        # Calculate total time
        end_time = time.time()
        metrics.planning_time_seconds = round(end_time - start_time, 2)
        metrics.total_iterations = metrics.flight_agent_iterations + metrics.hotel_agent_iterations
        
        self.log(f"\n  RESULT: {'SUCCESS' if metrics.planning_success else 'FAILED'}")
        self.log(f"  Total time: {metrics.planning_time_seconds}s")
        
        return metrics
    
    def run_all_scenarios(self, scenarios: List[TestScenario]) -> List[TestMetrics]:
        """Run all test scenarios and collect metrics."""
        self.log(f"\n{'#'*60}")
        self.log(f"# STARTING RESEARCH METRICS COLLECTION")
        self.log(f"# Total scenarios: {len(scenarios)}")
        self.log(f"{'#'*60}")
        
        for i, scenario in enumerate(scenarios, 1):
            self.log(f"\n[{i}/{len(scenarios)}] Running scenario: {scenario.scenario_name}")
            metrics = self.run_single_test(scenario)
            self.results.append(metrics)
            
            # Small delay between tests
            time.sleep(0.5)
        
        return self.results
    
    def save_to_csv(self):
        """Save all collected metrics to CSV file."""
        if not self.results:
            self.log("No results to save!")
            return
        
        filepath = os.path.join(os.path.dirname(__file__), self.output_file)
        
        # Get field names from dataclass
        fieldnames = list(asdict(self.results[0]).keys())
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))
        
        self.log(f"\n✓ Saved {len(self.results)} results to {filepath}")
    
    def print_summary(self):
        """Print summary analysis of results."""
        if not self.results:
            return
        
        print("\n" + "="*70)
        print("RESEARCH METRICS SUMMARY")
        print("="*70)
        
        # Overall stats
        total = len(self.results)
        successes = sum(1 for r in self.results if r.planning_success)
        
        print(f"\n📊 OVERALL:")
        print(f"   Total tests: {total}")
        print(f"   Successful: {successes} ({successes/total*100:.1f}%)")
        print(f"   Failed: {total - successes}")
        
        # By complexity
        print(f"\n📈 BY COMPLEXITY:")
        for complexity in ["simple", "medium", "complex"]:
            subset = [r for r in self.results if r.trip_complexity == complexity]
            if subset:
                success_rate = sum(1 for r in subset if r.planning_success) / len(subset) * 100
                avg_time = sum(r.planning_time_seconds for r in subset) / len(subset)
                avg_iterations = sum(r.total_iterations for r in subset) / len(subset)
                print(f"   {complexity.upper()}: {len(subset)} tests, {success_rate:.0f}% success, "
                      f"avg {avg_time:.1f}s, avg {avg_iterations:.1f} iterations")
        
        # Time Agent impact (Sub-question 1)
        print(f"\n🕐 TIME AGENT IMPACT (Sub-question 1):")
        with_time = [r for r in self.results if r.with_time_agent]
        without_time = [r for r in self.results if not r.with_time_agent]
        
        if with_time:
            success_with = sum(1 for r in with_time if r.planning_success) / len(with_time) * 100
            avg_iter_with = sum(r.total_iterations for r in with_time) / len(with_time)
            print(f"   WITH TimeAgent: {success_with:.0f}% success, avg {avg_iter_with:.1f} iterations")
        
        if without_time:
            success_without = sum(1 for r in without_time if r.planning_success) / len(without_time) * 100
            avg_iter_without = sum(r.total_iterations for r in without_time) / len(without_time)
            print(f"   WITHOUT TimeAgent: {success_without:.0f}% success, avg {avg_iter_without:.1f} iterations")
        
        # Quality and cost analysis
        print(f"\n💎 QUALITY & COST METRICS:")
        successful = [r for r in self.results if r.planning_success]
        if successful:
            avg_cost = sum(r.total_cost for r in successful) / len(successful)
            avg_remaining = sum(r.budget_remaining for r in successful) / len(successful)
            avg_value = sum(r.value_score for r in successful) / len(successful)
            print(f"   Avg total cost: ${avg_cost:.0f}")
            print(f"   Avg budget remaining: ${avg_remaining:.0f}")
            print(f"   Avg value score: {avg_value:.1f}")
            
            # Flight and hotel price breakdown
            avg_flight = sum(r.selected_flight_price for r in successful) / len(successful)
            avg_hotel = sum(r.selected_hotel_price for r in successful) / len(successful)
            print(f"   Avg flight price: ${avg_flight:.0f}")
            print(f"   Avg hotel/night: ${avg_hotel:.0f}")
        
        # Timing analysis (Sub-question 4)
        print(f"\n⏱️ PLANNING TIME (Sub-question 4):")
        if successful:
            times = [r.planning_time_seconds for r in successful]
            print(f"   Min: {min(times):.1f}s")
            print(f"   Max: {max(times):.1f}s")
            print(f"   Avg: {sum(times)/len(times):.1f}s")
        
        # Agent performance
        print(f"\n🤖 AGENT PERFORMANCE:")
        if successful:
            avg_flight_iter = sum(r.flight_agent_iterations for r in successful) / len(successful)
            avg_hotel_iter = sum(r.hotel_agent_iterations for r in successful) / len(successful)
            avg_combos = sum(r.combinations_evaluated for r in successful) / len(successful)
            print(f"   Avg flight agent iterations: {avg_flight_iter:.1f}")
            print(f"   Avg hotel agent iterations: {avg_hotel_iter:.1f}")
            print(f"   Avg combinations evaluated: {avg_combos:.0f}")
        
        print("\n" + "="*70)


def create_test_scenarios() -> List[TestScenario]:
    """Create a comprehensive set of 50 test scenarios for research validation."""
    scenarios = []
    
    # ========== SIMPLE SCENARIOS (15 tests) ==========
    # Generous budgets, no meetings, no time constraints
    
    # Route variations with generous budget
    simple_routes = [
        ("NYC", "SF", 2500), ("NYC", "CHI", 2000), ("NYC", "BOS", 1500),
        ("SF", "NYC", 2500), ("SF", "CHI", 2000), ("SF", "BOS", 2200),
        ("CHI", "NYC", 2000), ("CHI", "SF", 2200),
        ("BOS", "NYC", 1800), ("BOS", "SF", 2500),
    ]
    for i, (origin, dest, budget) in enumerate(simple_routes, 1):
        scenarios.append(TestScenario(
            scenario_id=f"simple_{i}",
            scenario_name=f"Simple: {origin}→{dest} ${budget}",
            origin=origin, destination=dest,
            budget=budget, nights=2, num_meetings=0,
            trip_complexity="simple", with_time_agent=False
        ))
    
    # Night variations (simple)
    for nights in [1, 2, 3, 4, 5]:
        scenarios.append(TestScenario(
            scenario_id=f"simple_nights_{nights}",
            scenario_name=f"Simple: NYC→SF {nights} nights",
            origin="NYC", destination="SF",
            budget=1500 + (nights * 300), nights=nights, num_meetings=0,
            trip_complexity="simple", with_time_agent=False
        ))
    
    # ========== MEDIUM SCENARIOS (20 tests) ==========
    # Moderate budgets, 1-2 meetings, time constraints
    
    # Budget variations with meetings
    for budget in [1200, 1500, 1800, 2000, 2500]:
        scenarios.append(TestScenario(
            scenario_id=f"medium_budget_{budget}",
            scenario_name=f"Medium: NYC→SF ${budget} + 1 meeting",
            origin="NYC", destination="SF",
            budget=budget, nights=2, num_meetings=1,
            trip_complexity="medium", with_time_agent=True,
            meetings=[{"date": "2026-01-15", "time": "14:00", "duration_minutes": 60, 
                      "location": {"lat": 37.79, "lon": -122.40}, "description": "Client meeting"}]
        ))
    
    # Meeting time variations
    for meeting_time in ["09:00", "11:00", "14:00", "16:00", "18:00"]:
        scenarios.append(TestScenario(
            scenario_id=f"medium_meeting_{meeting_time.replace(':', '')}",
            scenario_name=f"Medium: Meeting at {meeting_time}",
            origin="NYC", destination="SF",
            budget=2000, nights=2, num_meetings=1,
            trip_complexity="medium", with_time_agent=True,
            meetings=[{"date": "2026-01-15", "time": meeting_time, "duration_minutes": 60,
                      "location": {"lat": 37.79, "lon": -122.40}, "description": "Business meeting"}]
        ))
    
    # Route variations with meetings
    medium_routes = [("NYC", "SF"), ("SF", "NYC"), ("CHI", "NYC"), ("BOS", "NYC"), ("NYC", "CHI")]
    for origin, dest in medium_routes:
        scenarios.append(TestScenario(
            scenario_id=f"medium_route_{origin}_{dest}",
            scenario_name=f"Medium: {origin}→{dest} with meeting",
            origin=origin, destination=dest,
            budget=2000, nights=2, num_meetings=1,
            trip_complexity="medium", with_time_agent=True,
            meetings=[{"date": "2026-01-15", "time": "15:00", "duration_minutes": 90,
                      "location": {"lat": 40.75, "lon": -73.99}, "description": "Partner meeting"}]
        ))
    
    # Multiple meetings
    for num_meetings in [2, 3]:
        meetings = [
            {"date": "2026-01-15", "time": f"{10 + i*3}:00", "duration_minutes": 60,
             "location": {"lat": 37.79, "lon": -122.40}, "description": f"Meeting {i+1}"}
            for i in range(num_meetings)
        ]
        scenarios.append(TestScenario(
            scenario_id=f"medium_meetings_{num_meetings}",
            scenario_name=f"Medium: {num_meetings} meetings same day",
            origin="NYC", destination="SF",
            budget=2200, nights=2, num_meetings=num_meetings,
            trip_complexity="medium", with_time_agent=True,
            meetings=meetings
        ))
    
    # Time agent comparison pairs
    for with_time in [False, True]:
        scenarios.append(TestScenario(
            scenario_id=f"medium_timeagent_{'on' if with_time else 'off'}",
            scenario_name=f"Medium: TimeAgent {'ON' if with_time else 'OFF'}",
            origin="NYC", destination="SF",
            budget=2000, nights=2, num_meetings=2,
            trip_complexity="medium", with_time_agent=with_time,
            meetings=[
                {"date": "2026-01-15", "time": "14:00", "duration_minutes": 60,
                 "location": {"lat": 37.79, "lon": -122.40}, "description": "Meeting 1"},
                {"date": "2026-01-15", "time": "16:30", "duration_minutes": 60,
                 "location": {"lat": 37.79, "lon": -122.40}, "description": "Meeting 2"}
            ]
        ))
    
    # ========== COMPLEX SCENARIOS (10 tests) ==========
    # Tight budgets, multiple meetings, challenging constraints
    
    # Very tight budgets
    for budget in [700, 800, 900, 1000]:
        scenarios.append(TestScenario(
            scenario_id=f"complex_tight_{budget}",
            scenario_name=f"Complex: Tight ${budget} budget",
            origin="NYC", destination="SF",
            budget=budget, nights=2, num_meetings=2,
            trip_complexity="complex", with_time_agent=True,
            meetings=[
                {"date": "2026-01-15", "time": "10:00", "duration_minutes": 90,
                 "location": {"lat": 37.79, "lon": -122.40}, "description": "Workshop"},
                {"date": "2026-01-15", "time": "14:00", "duration_minutes": 60,
                 "location": {"lat": 37.79, "lon": -122.40}, "description": "Follow-up"}
            ]
        ))
    
    # Early morning meetings (challenging for timeline)
    for meeting_time in ["07:00", "08:00", "09:00"]:
        scenarios.append(TestScenario(
            scenario_id=f"complex_early_{meeting_time.replace(':', '')}",
            scenario_name=f"Complex: Early {meeting_time} meeting",
            origin="NYC", destination="SF",
            budget=2000, nights=2, num_meetings=1,
            trip_complexity="complex", with_time_agent=True,
            meetings=[{"date": "2026-01-15", "time": meeting_time, "duration_minutes": 120,
                      "location": {"lat": 37.79, "lon": -122.40}, "description": "Early meeting"}]
        ))
    
    # Many meetings packed together
    scenarios.append(TestScenario(
        scenario_id="complex_packed",
        scenario_name="Complex: 4 packed meetings",
        origin="NYC", destination="SF",
        budget=2500, nights=2, num_meetings=4,
        trip_complexity="complex", with_time_agent=True,
        meetings=[
            {"date": "2026-01-15", "time": "09:00", "duration_minutes": 60,
             "location": {"lat": 37.79, "lon": -122.40}, "description": "Standup"},
            {"date": "2026-01-15", "time": "10:30", "duration_minutes": 90,
             "location": {"lat": 37.79, "lon": -122.40}, "description": "Deep dive"},
            {"date": "2026-01-15", "time": "13:00", "duration_minutes": 60,
             "location": {"lat": 37.79, "lon": -122.40}, "description": "Lunch meeting"},
            {"date": "2026-01-15", "time": "15:00", "duration_minutes": 120,
             "location": {"lat": 37.79, "lon": -122.40}, "description": "Workshop"}
        ]
    ))
    
    # ========== EDGE CASES (5 tests) ==========
    
    # Impossible budget
    scenarios.append(TestScenario(
        scenario_id="edge_impossible",
        scenario_name="Edge: Impossible budget ($300)",
        origin="NYC", destination="SF",
        budget=300, nights=1, num_meetings=0,
        trip_complexity="complex", with_time_agent=False
    ))
    
    # Luxury budget
    scenarios.append(TestScenario(
        scenario_id="edge_luxury",
        scenario_name="Edge: Luxury budget ($5000)",
        origin="NYC", destination="SF",
        budget=5000, nights=3, num_meetings=0,
        trip_complexity="simple", with_time_agent=False
    ))
    
    # Long stay
    scenarios.append(TestScenario(
        scenario_id="edge_longstay",
        scenario_name="Edge: Week-long stay (7 nights)",
        origin="NYC", destination="SF",
        budget=4000, nights=7, num_meetings=0,
        trip_complexity="medium", with_time_agent=False
    ))
    
    # Same day return
    scenarios.append(TestScenario(
        scenario_id="edge_daytrip",
        scenario_name="Edge: Day trip (0 nights)",
        origin="NYC", destination="SF",
        budget=1500, nights=0, num_meetings=1,
        trip_complexity="medium", with_time_agent=True,
        meetings=[{"date": "2026-01-15", "time": "14:00", "duration_minutes": 60,
                  "location": {"lat": 37.79, "lon": -122.40}, "description": "Quick meeting"}]
    ))
    
    # Multiple day meetings
    scenarios.append(TestScenario(
        scenario_id="edge_multiday",
        scenario_name="Edge: Meetings across 2 days",
        origin="NYC", destination="SF",
        budget=2500, nights=2, num_meetings=3,
        trip_complexity="medium", with_time_agent=True,
        meetings=[
            {"date": "2026-01-15", "time": "14:00", "duration_minutes": 60,
             "location": {"lat": 37.79, "lon": -122.40}, "description": "Day 1 meeting"},
            {"date": "2026-01-16", "time": "09:00", "duration_minutes": 120,
             "location": {"lat": 37.79, "lon": -122.40}, "description": "Day 2 morning"},
            {"date": "2026-01-16", "time": "14:00", "duration_minutes": 90,
             "location": {"lat": 37.79, "lon": -122.40}, "description": "Day 2 afternoon"}
        ]
    ))
    
    return scenarios


if __name__ == "__main__":
    print("="*70)
    print("MULTI-AGENT TRIP PLANNING - RESEARCH METRICS COLLECTION")
    print("="*70)
    
    # Create runner
    runner = ResearchMetricsRunner(
        output_file="research_metrics.csv",
        verbose=True
    )
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    print(f"\nCreated {len(scenarios)} test scenarios")
    
    # Run all tests
    results = runner.run_all_scenarios(scenarios)
    
    # Save to CSV
    runner.save_to_csv()
    
    # Print summary
    runner.print_summary()
    
    print("\n✓ Metrics collection complete!")
    print(f"  Results saved to: research_metrics.csv")
