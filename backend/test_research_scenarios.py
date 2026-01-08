"""
Research Test Scenarios for Multi-Agent Trip Planning System

This module provides test scenarios designed to answer the research sub-questions
from the contract plan:

1. Impact of Time Management Agent on planning success rate and iterations
2. Feedback loops between Policy Compliance Agent and booking agents
3. Message exchanges for different trip complexity levels
4. Trip complexity vs. end-to-end planning time
5. Prompt specificity vs. policy violation rate

Author: Research Project - Laureys Indy
"""

import time
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, field, asdict
import statistics

from orchestrator.graph import (
    plan_trip, 
    build_workflow, 
    create_trip_planning_app,
    MAX_BACKTRACKING_ITERATIONS
)
from orchestrator.state import create_initial_state


@dataclass
class TestScenario:
    """A single test scenario configuration."""
    name: str
    description: str
    complexity: str  # simple, medium, complex
    origin: str
    destination: str
    budget: float
    preferences: Dict[str, Any]
    expected_backtracking: int = 0  # Expected number of backtracks
    

@dataclass
class TestResult:
    """Results from running a test scenario."""
    scenario_name: str
    complexity: str
    success: bool
    total_time_seconds: float
    backtracking_iterations: int
    negotiation_rounds: int
    parallel_searches: int
    total_messages: int
    policy_violations_found: int
    flight_found: bool
    hotel_found: bool
    total_cost: float
    compliance_status: str
    error: str = ""


# ============================================================================
# TEST SCENARIOS ALIGNED WITH RESEARCH SUB-QUESTIONS
# ============================================================================

SIMPLE_SCENARIOS = [
    TestScenario(
        name="simple_nyc_sf_high_budget",
        description="Simple NYC to SF trip with generous budget",
        complexity="simple",
        origin="NYC",
        destination="SF",
        budget=3000,  # High budget, should pass policy easily
        preferences={
            "min_rating": 3,
            "meeting_times": ["2026-01-20 14:00"]
        }
    ),
    TestScenario(
        name="simple_bos_chi_high_budget",
        description="Simple Boston to Chicago with generous budget",
        complexity="simple",
        origin="BOS",
        destination="CHI",
        budget=2500,
        preferences={
            "min_rating": 3,
            "meeting_times": ["2026-01-21 10:00"]
        }
    ),
]

MEDIUM_SCENARIOS = [
    TestScenario(
        name="medium_nyc_sf_moderate_budget",
        description="NYC to SF with moderate budget (likely 1-2 backtracks)",
        complexity="medium",
        origin="NYC",
        destination="SF",
        budget=1200,  # Moderate - may trigger some policy issues
        preferences={
            "min_rating": 4,  # Higher quality requirement
            "meeting_times": ["2026-01-20 14:00", "2026-01-21 09:00"],
            "required_amenities": ["WiFi", "Gym"]
        },
        expected_backtracking=1
    ),
    TestScenario(
        name="medium_chi_bos_constraints",
        description="Chicago to Boston with specific constraints",
        complexity="medium",
        origin="CHI",
        destination="BOS",
        budget=1500,
        preferences={
            "min_rating": 4,
            "departure_after": "07:00",
            "departure_before": "12:00",
            "meeting_times": ["2026-01-20 16:00"]
        },
        expected_backtracking=1
    ),
]

COMPLEX_SCENARIOS = [
    TestScenario(
        name="complex_nyc_sf_tight_budget",
        description="NYC to SF with tight budget (expect multiple backtracks)",
        complexity="complex",
        origin="NYC",
        destination="SF",
        budget=800,  # Very tight - will definitely hit policy limits
        preferences={
            "min_rating": 4,
            "meeting_times": ["2026-01-20 09:00", "2026-01-20 14:00", "2026-01-21 10:00"],
            "required_amenities": ["WiFi", "Conference Room", "Gym"],
            "departure_after": "06:00",
            "departure_before": "10:00"  # Morning flights only
        },
        expected_backtracking=3
    ),
    TestScenario(
        name="complex_sf_bos_very_tight",
        description="SF to Boston with very strict constraints",
        complexity="complex",
        origin="SF",
        destination="BOS",
        budget=700,
        preferences={
            "min_rating": 5,  # Top quality only
            "meeting_times": ["2026-01-20 08:00"],  # Very early meeting
            "required_amenities": ["WiFi", "Conference Room", "Restaurant", "Spa"],
        },
        expected_backtracking=4
    ),
]

ALL_SCENARIOS = SIMPLE_SCENARIOS + MEDIUM_SCENARIOS + COMPLEX_SCENARIOS


def run_single_test(scenario: TestScenario, verbose: bool = True) -> TestResult:
    """Run a single test scenario and collect metrics."""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RUNNING TEST: {scenario.name}")
        print(f"Complexity: {scenario.complexity}")
        print(f"Budget: ${scenario.budget}")
        print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = plan_trip(
            user_request=f"Business trip from {scenario.origin} to {scenario.destination}",
            origin=scenario.origin,
            destination=scenario.destination,
            departure_date="2026-01-20",
            return_date="2026-01-22",
            budget=scenario.budget,
            preferences=scenario.preferences
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Extract metrics from result
        metrics = {}
        final_rec = None
        messages = []
        
        if result:
            for node_name, node_state in result.items():
                if isinstance(node_state, dict):
                    if 'metrics' in node_state:
                        metrics = node_state['metrics']
                    if 'final_recommendation' in node_state:
                        final_rec = node_state['final_recommendation']
                    if 'messages' in node_state:
                        messages.extend(node_state.get('messages', []))
        
        test_result = TestResult(
            scenario_name=scenario.name,
            complexity=scenario.complexity,
            success=final_rec is not None,
            total_time_seconds=elapsed,
            backtracking_iterations=metrics.get('backtracking_count', 0),
            negotiation_rounds=metrics.get('negotiation_rounds', 0),
            parallel_searches=metrics.get('parallel_searches_executed', 0),
            total_messages=len(messages),
            policy_violations_found=metrics.get('policy_violations_found', 0),
            flight_found=final_rec.get('flight') is not None if final_rec else False,
            hotel_found=final_rec.get('hotel') is not None if final_rec else False,
            total_cost=final_rec.get('total_estimated_cost', 0) if final_rec else 0,
            compliance_status=final_rec.get('compliance_status', 'unknown') if final_rec else 'unknown'
        )
        
    except Exception as e:
        end_time = time.time()
        test_result = TestResult(
            scenario_name=scenario.name,
            complexity=scenario.complexity,
            success=False,
            total_time_seconds=end_time - start_time,
            backtracking_iterations=0,
            negotiation_rounds=0,
            parallel_searches=0,
            total_messages=0,
            policy_violations_found=0,
            flight_found=False,
            hotel_found=False,
            total_cost=0,
            compliance_status='error',
            error=str(e)
        )
    
    if verbose:
        print(f"\nRESULT: {'SUCCESS' if test_result.success else 'FAILED'}")
        print(f"Time: {test_result.total_time_seconds:.2f}s")
        print(f"Backtracks: {test_result.backtracking_iterations}")
        print(f"Messages: {test_result.total_messages}")
    
    return test_result


def run_experiment_suite(
    scenarios: List[TestScenario], 
    runs_per_scenario: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a complete experiment suite for statistical analysis.
    
    This supports answering research sub-questions by running multiple
    iterations of each scenario type.
    """
    
    all_results: List[TestResult] = []
    
    for scenario in scenarios:
        scenario_results = []
        
        for run in range(runs_per_scenario):
            if verbose:
                print(f"\n[Run {run + 1}/{runs_per_scenario}] {scenario.name}")
            
            result = run_single_test(scenario, verbose=verbose)
            scenario_results.append(result)
            all_results.append(result)
        
        # Calculate stats for this scenario
        if scenario_results:
            times = [r.total_time_seconds for r in scenario_results]
            backtracks = [r.backtracking_iterations for r in scenario_results]
            messages = [r.total_messages for r in scenario_results]
            
            if verbose:
                print(f"\n--- Stats for {scenario.name} ---")
                print(f"Avg time: {statistics.mean(times):.2f}s")
                print(f"Avg backtracks: {statistics.mean(backtracks):.1f}")
                print(f"Avg messages: {statistics.mean(messages):.1f}")
                print(f"Success rate: {sum(r.success for r in scenario_results) / len(scenario_results) * 100:.0f}%")
    
    # Aggregate by complexity
    complexity_stats = {}
    for complexity in ['simple', 'medium', 'complex']:
        complexity_results = [r for r in all_results if r.complexity == complexity]
        if complexity_results:
            complexity_stats[complexity] = {
                'count': len(complexity_results),
                'success_rate': sum(r.success for r in complexity_results) / len(complexity_results),
                'avg_time': statistics.mean([r.total_time_seconds for r in complexity_results]),
                'avg_backtracks': statistics.mean([r.backtracking_iterations for r in complexity_results]),
                'avg_messages': statistics.mean([r.total_messages for r in complexity_results]),
                'avg_violations': statistics.mean([r.policy_violations_found for r in complexity_results])
            }
    
    return {
        'all_results': [asdict(r) for r in all_results],
        'complexity_stats': complexity_stats,
        'total_runs': len(all_results),
        'timestamp': datetime.now().isoformat()
    }


def run_subquestion_1_test(verbose: bool = True) -> Dict[str, Any]:
    """
    SUB-QUESTION 1: Impact of Time Management Agent on success rate and iterations
    
    Test with and without time constraints to measure the Time Agent's impact.
    """
    print("\n" + "="*60)
    print("SUB-QUESTION 1: Time Management Agent Impact")
    print("="*60)
    
    # Scenario WITHOUT time constraints (no meetings)
    no_time_scenario = TestScenario(
        name="no_time_constraints",
        description="Trip without meeting times",
        complexity="medium",
        origin="NYC",
        destination="SF",
        budget=1500,
        preferences={"min_rating": 3}  # No meeting_times
    )
    
    # Scenario WITH time constraints
    with_time_scenario = TestScenario(
        name="with_time_constraints",
        description="Trip with tight meeting schedule",
        complexity="medium",
        origin="NYC",
        destination="SF",
        budget=1500,
        preferences={
            "min_rating": 3,
            "meeting_times": ["2026-01-20 09:00", "2026-01-20 14:00"]
        }
    )
    
    results = {
        'without_time_agent': run_single_test(no_time_scenario, verbose),
        'with_time_agent': run_single_test(with_time_scenario, verbose)
    }
    
    print("\n--- SUB-QUESTION 1 RESULTS ---")
    print(f"Without time constraints: {results['without_time_agent'].backtracking_iterations} backtracks")
    print(f"With time constraints: {results['with_time_agent'].backtracking_iterations} backtracks")
    
    return {k: asdict(v) for k, v in results.items()}


def run_subquestion_2_test(verbose: bool = True) -> Dict[str, Any]:
    """
    SUB-QUESTION 2: Feedback loops between Policy Agent and booking agents
    
    Test with different policy strictness levels to measure feedback loop variation.
    """
    print("\n" + "="*60)
    print("SUB-QUESTION 2: Policy Feedback Loop Analysis")
    print("="*60)
    
    # Lenient policy (high budget, should pass easily)
    lenient = TestScenario(
        name="lenient_policy",
        description="High budget, few violations expected",
        complexity="simple",
        origin="NYC",
        destination="SF",
        budget=3000,
        preferences={"min_rating": 3}
    )
    
    # Moderate policy
    moderate = TestScenario(
        name="moderate_policy",
        description="Moderate budget, some violations expected",
        complexity="medium",
        origin="NYC",
        destination="SF",
        budget=1200,
        preferences={"min_rating": 4}
    )
    
    # Strict policy (tight budget)
    strict = TestScenario(
        name="strict_policy",
        description="Tight budget, many violations expected",
        complexity="complex",
        origin="NYC",
        destination="SF",
        budget=800,
        preferences={"min_rating": 4, "required_amenities": ["WiFi", "Gym"]}
    )
    
    results = {
        'lenient': run_single_test(lenient, verbose),
        'moderate': run_single_test(moderate, verbose),
        'strict': run_single_test(strict, verbose)
    }
    
    print("\n--- SUB-QUESTION 2 RESULTS ---")
    for policy, result in results.items():
        print(f"{policy}: {result.negotiation_rounds} negotiation rounds, "
              f"{result.policy_violations_found} violations found")
    
    return {k: asdict(v) for k, v in results.items()}


def run_subquestion_3_test(verbose: bool = True) -> Dict[str, Any]:
    """
    SUB-QUESTION 3: Message exchanges for different complexity levels
    
    Measure total messages across simple, medium, and complex trips.
    """
    print("\n" + "="*60)
    print("SUB-QUESTION 3: Message Exchange Analysis")
    print("="*60)
    
    scenarios = [
        SIMPLE_SCENARIOS[0],
        MEDIUM_SCENARIOS[0],
        COMPLEX_SCENARIOS[0]
    ]
    
    results = {}
    for scenario in scenarios:
        result = run_single_test(scenario, verbose)
        results[scenario.complexity] = result
    
    print("\n--- SUB-QUESTION 3 RESULTS ---")
    for complexity, result in results.items():
        print(f"{complexity}: {result.total_messages} messages exchanged")
    
    return {k: asdict(v) for k, v in results.items()}


def run_subquestion_4_test(verbose: bool = True) -> Dict[str, Any]:
    """
    SUB-QUESTION 4: Complexity vs. end-to-end planning time
    
    Measure planning time across different complexity levels.
    """
    print("\n" + "="*60)
    print("SUB-QUESTION 4: Complexity vs. Planning Time")
    print("="*60)
    
    scenarios = [
        SIMPLE_SCENARIOS[0],
        MEDIUM_SCENARIOS[0],
        COMPLEX_SCENARIOS[0]
    ]
    
    results = {}
    for scenario in scenarios:
        result = run_single_test(scenario, verbose)
        results[scenario.complexity] = result
    
    print("\n--- SUB-QUESTION 4 RESULTS ---")
    for complexity, result in results.items():
        print(f"{complexity}: {result.total_time_seconds:.2f}s planning time")
    
    return {k: asdict(v) for k, v in results.items()}


def export_results_to_json(results: Dict[str, Any], filename: str = "test_results.json"):
    """Export test results to JSON for analysis."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults exported to {filename}")


# ============================================================================
# QUICK VALIDATION TEST
# ============================================================================

def quick_validation_test():
    """Run a quick validation test to ensure the system works."""
    print("\n" + "="*60)
    print("QUICK VALIDATION TEST")
    print("="*60)
    
    simple_scenario = TestScenario(
        name="quick_test",
        description="Quick validation",
        complexity="simple",
        origin="NYC",
        destination="SF",
        budget=2000,
        preferences={"min_rating": 3}
    )
    
    result = run_single_test(simple_scenario, verbose=True)
    
    print("\n" + "="*60)
    print("VALIDATION RESULT")
    print("="*60)
    print(f"Success: {result.success}")
    print(f"Time: {result.total_time_seconds:.2f}s")
    print(f"Backtracks: {result.backtracking_iterations}")
    print(f"Messages: {result.total_messages}")
    print(f"Flight found: {result.flight_found}")
    print(f"Hotel found: {result.hotel_found}")
    print(f"Total cost: ${result.total_cost}")
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            quick_validation_test()
        elif sys.argv[1] == "q1":
            run_subquestion_1_test()
        elif sys.argv[1] == "q2":
            run_subquestion_2_test()
        elif sys.argv[1] == "q3":
            run_subquestion_3_test()
        elif sys.argv[1] == "q4":
            run_subquestion_4_test()
        elif sys.argv[1] == "full":
            results = run_experiment_suite(ALL_SCENARIOS, runs_per_scenario=1)
            export_results_to_json(results)
    else:
        print("Usage: python test_research_scenarios.py [quick|q1|q2|q3|q4|full]")
        print("\nRunning quick validation test by default...")
        quick_validation_test()
