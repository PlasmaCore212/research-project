# backend/agents/policy_agent.py
"""
Policy Compliance Agent for Multi-Agent Trip Planning System

STRATEGY: SEARCH-FIRST, ALLOCATE-LATER
1. Receives ALL flight and hotel options from booking agents
2. Evaluates ALL valid combinations within budget
3. Scores each combination on value (quality + budget utilization)
4. Selects the OPTIMAL combination that MAXIMIZES budget usage for quality
"""

from langchain_ollama import OllamaLLM
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from dataclasses import dataclass
import json


class PolicyViolation(BaseModel):
    """Represents a policy/budget violation."""
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
    value_score: float
    quality_score: float
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
    all_combinations: List[Dict[str, Any]] = []
    cheaper_alternatives: List[Dict[str, Any]] = []


class PolicyComplianceAgent:
    """
    Policy Compliance Agent - MAXIMIZES budget utilization for quality.
    
    Scoring Strategy:
    - Quality is PRIMARY factor (hotel stars, location, flight timing)
    - Budget utilization is SECONDARY (prefer using more of budget for quality)
    - Checks for violations (over budget)
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.llm = OllamaLLM(model=model_name, temperature=0.0, format="json")
        self.metrics = {"checks_performed": 0, "combinations_evaluated": 0,
                       "valid_combinations_found": 0, "optimal_selections": 0, "no_valid_options": 0}
    
    def _log(self, message: str):
        if self.verbose:
            print(f"  [PolicyAgent] {message}")
    
    def _calculate_quality_score(self, flight: Dict, hotel: Dict) -> float:
        """
        Calculate quality score (0-50 points).
        QUALITY IS KING - Higher stars and better location dominate.
        """
        score = 0.0
        
        # Hotel stars (0-30 points) - MOST IMPORTANT: 5★=30, 4★=24, 3★=18, 2★=12, 1★=6
        score += hotel.get("stars", 3) * 6
        
        # Hotel distance (0-10 points, closer is better)
        distance = hotel.get("distance_to_business_center_km", 5)
        if distance < 0.5: score += 10
        elif distance < 1: score += 8
        elif distance < 2: score += 6
        elif distance < 5: score += 4
        else: score += 2
        
        # Flight duration (0-5 points, shorter is better)
        duration = flight.get("duration_hours", 6)
        if duration < 5.5: score += 5
        elif duration < 6.5: score += 4
        elif duration < 7.5: score += 3
        else: score += 1
        
        # Flight timing (0-5 points, morning preferred)
        try:
            hour = int(flight.get("departure_time", "12:00").split(":")[0])
            if 6 <= hour <= 10: score += 5
            elif 10 < hour <= 14: score += 4
            elif 14 < hour <= 18: score += 3
            else: score += 1
        except:
            score += 2
        
        return score
    
    def _calculate_value_score(self, total_cost: float, quality_score: float, budget: float) -> float:
        """
        Calculate value score: MAXIMIZE QUALITY AND BUDGET UTILIZATION.
        
        Formula: quality_score * 3 + utilization_component (0-50)
        
        This ensures we select the highest quality option that ALSO uses more of the budget.
        A 5★ hotel at $400/night beats a 5★ at $200/night when both fit the budget.
        """
        quality_component = quality_score * 3  # 0-150 points from quality
        
        # Budget utilization is now a MAJOR factor (0-50 points)
        utilization_ratio = total_cost / budget if budget > 0 else 0
        
        # Reward higher utilization significantly
        if utilization_ratio >= 0.95: 
            utilization_component = 50  # Excellent - using 95%+ of budget
        elif utilization_ratio >= 0.90: 
            utilization_component = 45
        elif utilization_ratio >= 0.85: 
            utilization_component = 40
        elif utilization_ratio >= 0.80: 
            utilization_component = 35
        elif utilization_ratio >= 0.70: 
            utilization_component = 25
        elif utilization_ratio >= 0.60: 
            utilization_component = 15
        elif utilization_ratio >= 0.50: 
            utilization_component = 8
        else: 
            utilization_component = 2  # Penalize low utilization
        
        return quality_component + utilization_component
    
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
        MAXIMIZES quality while utilizing as much budget as possible.
        """
        self.metrics["checks_performed"] += 1
        
        if not flights or not hotels:
            self._log("❌ No flights or hotels to evaluate")
            return CombinationResult(success=False, reasoning="No options available.")
        
        self._log(f"Evaluating {len(flights)} flights × {len(hotels)} hotels = {len(flights) * len(hotels)} combinations")
        
        valid_combinations: List[CombinationScore] = []
        all_combinations_data = []
        
        for flight in flights:
            flight_cost = flight.get("price_usd", 0)
            for hotel in hotels:
                hotel_total = hotel.get("price_per_night_usd", 0) * nights
                total_cost = flight_cost + hotel_total
                
                self.metrics["combinations_evaluated"] += 1
                is_valid = total_cost <= budget  # VIOLATION CHECK
                
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
                    valid_combinations.append(CombinationScore(
                        flight=flight, hotel=hotel, flight_cost=flight_cost,
                        hotel_total_cost=hotel_total, total_cost=total_cost, nights=nights,
                        value_score=value_score, quality_score=quality_score,
                        reasoning=f"Flight ${flight_cost} + Hotel ${hotel_total} ({nights}n) = ${total_cost}"
                    ))
        
        self._log(f"Found {len(valid_combinations)} valid combinations within ${budget}")
        
        if not valid_combinations:
            self.metrics["no_valid_options"] += 1
            min_cost = min(c["total_cost"] for c in all_combinations_data) if all_combinations_data else 0
            return CombinationResult(
                success=False, combinations_evaluated=len(all_combinations_data),
                reasoning=f"No valid combinations within ${budget}. Cheapest: ${min_cost} (${min_cost - budget} over).",
                all_combinations=all_combinations_data[:10]
            )
        
        # Sort by value score (highest first) - quality + budget utilization
        valid_combinations.sort(key=lambda x: x.value_score, reverse=True)
        best = valid_combinations[0]
        
        # Find cheaper alternatives
        cheaper_sorted = sorted(valid_combinations, key=lambda x: x.total_cost)
        cheaper_alternatives = []
        for combo in cheaper_sorted:
            if (combo.flight.get("flight_id") != best.flight.get("flight_id") or
                combo.hotel.get("hotel_id") != best.hotel.get("hotel_id")):
                if combo.total_cost < best.total_cost:
                    cheaper_alternatives.append({
                        "flight": {"flight_id": combo.flight.get("flight_id"), 
                                  "airline": combo.flight.get("airline", "Unknown"),
                                  "price_usd": combo.flight_cost},
                        "hotel": {"hotel_id": combo.hotel.get("hotel_id"), 
                                 "name": combo.hotel.get("name", "Unknown"),
                                 "stars": combo.hotel.get("stars"),
                                 "price_per_night_usd": combo.hotel.get("price_per_night_usd")},
                        "total_cost": combo.total_cost,
                        "savings_vs_selected": round(best.total_cost - combo.total_cost, 2),
                        "quality_score": round(combo.quality_score, 1)
                    })
                    if len(cheaper_alternatives) >= 3:
                        break
        
        # LLM reasoning for top 3 options
        best = self._select_with_reasoning(valid_combinations[:3], budget, preferences)
        self.metrics["optimal_selections"] += 1
        
        return CombinationResult(
            success=True, selected_flight=best.flight, selected_hotel=best.hotel,
            total_cost=best.total_cost, budget_remaining=budget - best.total_cost,
            value_score=best.value_score, combinations_evaluated=len(all_combinations_data),
            reasoning=best.reasoning, all_combinations=all_combinations_data[:10],
            cheaper_alternatives=cheaper_alternatives
        )
    
    def _select_with_reasoning(self, top_combinations: List[CombinationScore], budget: float,
                               preferences: Optional[Dict[str, Any]]) -> CombinationScore:
        """Use LLM for final selection with Chain-of-Thought reasoning."""
        if len(top_combinations) == 1:
            combo = top_combinations[0]
            combo.reasoning = (f"Selected only valid option: Flight ${combo.flight_cost} + "
                              f"Hotel ${combo.hotel_total_cost} = ${combo.total_cost}")
            return combo
        
        options_text = "\n".join(
            f"Option {i+1}: {c.flight.get('airline','?')} ${c.flight_cost} + "
            f"{c.hotel.get('name','?')} {c.hotel.get('stars','?')}★ ${c.hotel.get('price_per_night_usd',0)}/night "
            f"= ${c.total_cost} (Value: {c.value_score:.1f})"
            for i, c in enumerate(top_combinations)
        )
        
        prompt = f"""Select best flight+hotel combination. Budget: ${budget}

OPTIONS:
{options_text}

Consider: 1) Hotel quality (stars) 2) Location 3) Flight timing 4) Value score
Return JSON: {{"selected_option": 1, "reasoning": "why this is best"}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            idx = result.get("selected_option", 1) - 1
            reasoning = result.get("reasoning", "Selected based on value score")
            
            if 0 <= idx < len(top_combinations):
                best = top_combinations[idx]
                best.reasoning = reasoning
                return best
        except Exception as e:
            self._log(f"LLM selection failed: {e}")
        
        # Fallback to highest value score
        best = top_combinations[0]
        best.reasoning = f"Highest value: ${best.total_cost} (Score: {best.value_score:.1f})"
        return best
    
    def check_compliance(self, state: Dict[str, Any]) -> PolicyCheckResult:
        """Check if flight+hotel is within budget (violation detection)."""
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
                rule="total_budget", severity="error",
                message=f"Total ${total_cost} exceeds budget ${total_budget}",
                actual_value=str(total_cost), expected_value=str(total_budget)
            ))
        
        return PolicyCheckResult(
            is_compliant=len(violations) == 0, violations=violations,
            reasoning=f"Total: ${total_cost} vs Budget: ${total_budget}"
        )
    
    def get_metrics(self) -> Dict[str, int]:
        return self.metrics.copy()
    
    def reset_metrics(self):
        self.metrics = {"checks_performed": 0, "combinations_evaluated": 0,
                       "valid_combinations_found": 0, "optimal_selections": 0, "no_valid_options": 0}
