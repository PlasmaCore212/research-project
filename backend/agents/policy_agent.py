# backend/agents/policy_agent.py
"""
Policy Compliance Agent for Multi-Agent Trip Planning System

This agent implements a SEARCH-FIRST, ALLOCATE-LATER strategy:
1. Receives ALL flight and hotel options from booking agents
2. Evaluates ALL valid combinations within budget
3. Scores each combination on value (price + quality)
4. Selects the OPTIMAL combination

Key Research Value:
- Measures feedback loop iterations between Policy Agent and booking agents
- Tracks compliance checking overhead
- Shows agent decision-making with Chain-of-Thought reasoning

Author: Research Project - Laureys Indy
"""

from langchain_ollama import OllamaLLM
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel
from dataclasses import dataclass
import json


class PolicyViolation(BaseModel):
    """Represents a single policy/budget violation."""
    rule: str
    severity: str  # "error" or "warning"
    message: str
    actual_value: Optional[str] = None
    expected_value: Optional[str] = None


class PolicyCheckResult(BaseModel):
    """Result of a policy compliance check."""
    is_compliant: bool
    violations: List[PolicyViolation]
    reasoning: str
    recommendations: List[str] = []


@dataclass
class CombinationScore:
    """Represents a scored flight+hotel combination."""
    flight: Dict[str, Any]
    hotel: Dict[str, Any]
    flight_cost: float
    hotel_total_cost: float
    total_cost: float
    nights: int
    value_score: float  # Higher is better
    quality_score: float  # Based on stars, amenities, etc.
    reasoning: str


class CombinationResult(BaseModel):
    """Result of finding the best combination."""
    success: bool
    selected_flight: Optional[Dict[str, Any]] = None
    selected_hotel: Optional[Dict[str, Any]] = None
    total_cost: float = 0
    budget_remaining: float = 0
    value_score: float = 0
    combinations_evaluated: int = 0
    reasoning: str = ""
    all_combinations: List[Dict[str, Any]] = []  # For research analysis


class PolicyComplianceAgent:
    """
    Policy Compliance Agent with SEARCH-FIRST, ALLOCATE-LATER strategy.
    
    This agent:
    1. Receives all flight and hotel options (no budget filter applied)
    2. Generates all possible flight+hotel combinations
    3. Filters to those within the user's total budget
    4. Scores each on value (balancing price and quality)
    5. Selects the optimal combination using Chain-of-Thought reasoning
    
    Research Purpose:
    - Demonstrates intelligent budget allocation (no arbitrary splits)
    - Provides rich reasoning traces for analysis
    - Enables measurement of decision-making quality
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.0,
            format="json"
        )
        
        # Metrics for research
        self.metrics = {
            "checks_performed": 0,
            "combinations_evaluated": 0,
            "valid_combinations_found": 0,
            "optimal_selections": 0,
            "no_valid_options": 0
        }
    
    def _log(self, message: str):
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            print(f"  [PolicyAgent] {message}")
    
    def _calculate_quality_score(self, flight: Dict, hotel: Dict) -> float:
        """
        Calculate a quality score for a flight+hotel combination.
        
        Factors:
        - Hotel stars (1-5)
        - Hotel distance to center
        - Flight duration
        - Flight timing (business hours preferred)
        """
        score = 0.0
        
        # Hotel stars (0-25 points)
        stars = hotel.get("stars", 3)
        score += stars * 5
        
        # Hotel distance (0-15 points, closer is better)
        distance = hotel.get("distance_to_business_center_km", 5)
        score += max(0, 15 - distance * 2)
        
        # Flight duration (0-10 points, shorter is better)
        duration = flight.get("duration_hours", 5)
        score += max(0, 10 - duration)
        
        # Flight timing (0-10 points, 8am-6pm preferred)
        try:
            dep_time = flight.get("departure_time", "12:00")
            hour = int(dep_time.split(":")[0])
            if 8 <= hour <= 18:
                score += 10
            elif 6 <= hour <= 20:
                score += 5
        except:
            score += 5  # Default
        
        return score
    
    def _calculate_value_score(
        self, 
        total_cost: float, 
        quality_score: float, 
        budget: float
    ) -> float:
        """
        Calculate value score balancing cost and quality.
        
        Formula: (quality_score * 2) + ((budget - total_cost) / budget * 50)
        
        This rewards:
        - Higher quality (hotel stars, location, flight timing)
        - More budget remaining (efficiency)
        """
        # Quality component (0-60 points from quality score)
        quality_component = quality_score
        
        # Savings component (0-50 points based on budget efficiency)
        savings_ratio = (budget - total_cost) / budget if budget > 0 else 0
        savings_component = savings_ratio * 50
        
        return quality_component + savings_component
    
    def find_best_combination(
        self,
        flights: List[Dict[str, Any]],
        hotels: List[Dict[str, Any]],
        budget: float,
        nights: int = 1,
        preferences: Optional[Dict[str, Any]] = None
    ) -> CombinationResult:
        """
        Find the optimal flight+hotel combination within budget.
        
        This is the core method implementing SEARCH-FIRST, ALLOCATE-LATER:
        1. Generate all possible combinations
        2. Filter to those within budget
        3. Score each on value (price + quality)
        4. Return the best one with reasoning
        
        Args:
            flights: List of flight options from FlightAgent
            hotels: List of hotel options from HotelAgent
            budget: User's total budget
            nights: Number of nights for hotel
            preferences: Optional user preferences for weighting
            
        Returns:
            CombinationResult with the optimal selection
        """
        self.metrics["checks_performed"] += 1
        
        if not flights or not hotels:
            self._log("❌ No flights or hotels to evaluate")
            return CombinationResult(
                success=False,
                reasoning="No flight or hotel options available to evaluate."
            )
        
        self._log(f"Evaluating {len(flights)} flights × {len(hotels)} hotels = {len(flights) * len(hotels)} combinations")
        
        # Generate and score all combinations
        valid_combinations: List[CombinationScore] = []
        all_combinations_data = []
        
        for flight in flights:
            flight_cost = flight.get("price_usd", 0)
            
            for hotel in hotels:
                hotel_per_night = hotel.get("price_per_night_usd", 0)
                hotel_total = hotel_per_night * nights
                total_cost = flight_cost + hotel_total
                
                self.metrics["combinations_evaluated"] += 1
                
                # Check if within budget
                is_valid = total_cost <= budget
                
                # Calculate scores
                quality_score = self._calculate_quality_score(flight, hotel)
                value_score = self._calculate_value_score(total_cost, quality_score, budget) if is_valid else 0
                
                combo_data = {
                    "flight_id": flight.get("flight_id", ""),
                    "hotel_id": hotel.get("hotel_id", ""),
                    "flight_cost": flight_cost,
                    "hotel_cost": hotel_total,
                    "total_cost": total_cost,
                    "within_budget": is_valid,
                    "quality_score": round(quality_score, 2),
                    "value_score": round(value_score, 2)
                }
                all_combinations_data.append(combo_data)
                
                if is_valid:
                    self.metrics["valid_combinations_found"] += 1
                    
                    combo = CombinationScore(
                        flight=flight,
                        hotel=hotel,
                        flight_cost=flight_cost,
                        hotel_total_cost=hotel_total,
                        total_cost=total_cost,
                        nights=nights,
                        value_score=value_score,
                        quality_score=quality_score,
                        reasoning=f"Flight ${flight_cost} + Hotel ${hotel_total} ({nights}n) = ${total_cost}"
                    )
                    valid_combinations.append(combo)
        
        self._log(f"Found {len(valid_combinations)} valid combinations within ${budget} budget")
        
        if not valid_combinations:
            self.metrics["no_valid_options"] += 1
            
            # Find the closest option for feedback
            all_costs = [c["total_cost"] for c in all_combinations_data]
            min_cost = min(all_costs) if all_costs else 0
            
            return CombinationResult(
                success=False,
                combinations_evaluated=len(all_combinations_data),
                reasoning=f"No valid combinations within ${budget} budget. "
                         f"Cheapest option costs ${min_cost} (${min_cost - budget} over budget).",
                all_combinations=all_combinations_data[:10]  # Top 10 for analysis
            )
        
        # Sort by value score (highest first)
        valid_combinations.sort(key=lambda x: x.value_score, reverse=True)
        
        # Get top 3 for LLM reasoning
        top_combinations = valid_combinations[:3]
        
        # Use LLM for final decision with Chain-of-Thought
        best = self._select_with_reasoning(top_combinations, budget, preferences)
        
        self.metrics["optimal_selections"] += 1
        
        return CombinationResult(
            success=True,
            selected_flight=best.flight,
            selected_hotel=best.hotel,
            total_cost=best.total_cost,
            budget_remaining=budget - best.total_cost,
            value_score=best.value_score,
            combinations_evaluated=len(all_combinations_data),
            reasoning=best.reasoning,
            all_combinations=all_combinations_data[:10]
        )
    
    def _select_with_reasoning(
        self,
        top_combinations: List[CombinationScore],
        budget: float,
        preferences: Optional[Dict[str, Any]]
    ) -> CombinationScore:
        """
        Use LLM to select the best combination with Chain-of-Thought reasoning.
        """
        if len(top_combinations) == 1:
            combo = top_combinations[0]
            combo.reasoning = (
                f"Selected the only valid option: "
                f"Flight ${combo.flight_cost} ({combo.flight.get('airline', 'Unknown')}) + "
                f"Hotel ${combo.hotel_total_cost} ({combo.hotel.get('name', 'Unknown')} - "
                f"{combo.hotel.get('stars', '?')}★) = ${combo.total_cost} "
                f"(${budget - combo.total_cost} under budget)"
            )
            return combo
        
        # Prepare options for LLM
        options_text = ""
        for i, combo in enumerate(top_combinations, 1):
            options_text += f"""
Option {i}:
- Flight: {combo.flight.get('airline', 'Unknown')} - ${combo.flight_cost}
  - Departure: {combo.flight.get('departure_time', 'N/A')}, Duration: {combo.flight.get('duration_hours', 'N/A')}h
- Hotel: {combo.hotel.get('name', 'Unknown')} ({combo.hotel.get('stars', '?')}★) - ${combo.hotel.get('price_per_night_usd', 0)}/night
  - Location: {combo.hotel.get('business_area', 'N/A')}, Distance to center: {combo.hotel.get('distance_to_business_center_km', 'N/A')}km
- Total: ${combo.total_cost} (${budget - combo.total_cost} under budget)
- Value Score: {combo.value_score:.1f}
"""
        
        prompt = f"""You are a Policy Compliance Agent selecting the best flight+hotel combination.

BUDGET: ${budget}
NIGHTS: {top_combinations[0].nights}

TOP OPTIONS:
{options_text}

THINK STEP BY STEP:
1. Compare the total costs and savings
2. Compare hotel quality (stars, location)
3. Compare flight convenience (timing, duration)
4. Balance value (quality vs price)

Select the best option and explain why.

Return JSON:
{{
    "selected_option": 1,
    "reasoning": "Step-by-step explanation of why this is the best choice"
}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            selected_idx = result.get("selected_option", 1) - 1
            reasoning = result.get("reasoning", "Selected based on best value score")
            
            if 0 <= selected_idx < len(top_combinations):
                best = top_combinations[selected_idx]
                best.reasoning = reasoning
                return best
        except Exception as e:
            self._log(f"LLM selection failed: {e}, using highest value score")
        
        # Fallback to highest value score
        best = top_combinations[0]
        best.reasoning = (
            f"Selected highest value option: "
            f"Flight ${best.flight_cost} + Hotel ${best.hotel_total_cost} = ${best.total_cost} "
            f"(Value Score: {best.value_score:.1f})"
        )
        return best
    
    def check_compliance(self, state: Dict[str, Any]) -> PolicyCheckResult:
        """
        Legacy method for backward compatibility.
        Checks if a single flight+hotel combination is within budget.
        """
        total_budget = state.get("total_budget", 2000)
        nights = state.get("nights", 1)
        selected_flight = state.get("selected_flight")
        selected_hotel = state.get("selected_hotel")
        
        violations = []
        flight_cost = selected_flight.get("price_usd", 0) if selected_flight else 0
        hotel_total = (selected_hotel.get("price_per_night_usd", 0) * nights) if selected_hotel else 0
        total_cost = flight_cost + hotel_total
        
        if total_cost > total_budget:
            violations.append(PolicyViolation(
                rule="total_budget",
                severity="error",
                message=f"Total ${total_cost} exceeds budget ${total_budget}",
                actual_value=str(total_cost),
                expected_value=str(total_budget)
            ))
        
        return PolicyCheckResult(
            is_compliant=len(violations) == 0,
            violations=violations,
            reasoning=f"Total: ${total_cost} vs Budget: ${total_budget}"
        )
    
    def get_metrics(self) -> Dict[str, int]:
        """Return metrics for research analysis."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset metrics for new test run."""
        self.metrics = {
            "checks_performed": 0,
            "combinations_evaluated": 0,
            "valid_combinations_found": 0,
            "optimal_selections": 0,
            "no_valid_options": 0
        }
