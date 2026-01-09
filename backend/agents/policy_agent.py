# backend/agents/policy_agent.py
"""
Policy Compliance Agent for Multi-Agent Trip Planning System

TRULY AGENTIC APPROACH:
1. Receives ALL flight and hotel options from booking agents
2. Filters to valid combinations within budget
3. Uses LLM reasoning to SELECT the optimal combination
4. LLM considers: quality, price, budget utilization, timing, value
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
    Policy Compliance Agent - TRULY AGENTIC SELECTION.
    
    The LLM reasons about and selects the best combination by considering:
    - Budget utilization (use budget wisely, don't leave too much unused)
    - Quality (hotel stars, amenities, location)
    - Flight quality (timing, duration, airline)
    - Overall value (balance of all factors)
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        self.model_name = model_name
        self.verbose = verbose
        self.llm = OllamaLLM(model=model_name, temperature=0.1, format="json")  # Slight temp for reasoning variety
        self.metrics = {"checks_performed": 0, "combinations_evaluated": 0,
                       "valid_combinations_found": 0, "optimal_selections": 0, "no_valid_options": 0}
    
    def _log(self, message: str):
        if self.verbose:
            print(f"  [PolicyAgent] {message}")
    
    def _summarize_option(self, flight: Dict, hotel: Dict, nights: int, budget: float) -> Dict:
        """Create a summary of a flight+hotel combination for LLM reasoning."""
        flight_cost = flight.get("price_usd", 0)
        hotel_per_night = hotel.get("price_per_night_usd", 0)
        hotel_total = hotel_per_night * nights
        total = flight_cost + hotel_total
        remaining = budget - total
        utilization = (total / budget * 100) if budget > 0 else 0
        
        return {
            "flight": {
                "id": flight.get("flight_id"),
                "airline": flight.get("airline"),
                "price": flight_cost,
                "class": flight.get("class", "Economy"),
                "departure": flight.get("departure_time"),
                "duration_hours": flight.get("duration_hours")
            },
            "hotel": {
                "id": hotel.get("hotel_id"),
                "name": hotel.get("name"),
                "stars": hotel.get("stars"),
                "price_per_night": hotel_per_night,
                "total_for_stay": hotel_total,
                "distance_km": hotel.get("distance_to_business_center_km")
            },
            "total_cost": total,
            "budget_remaining": remaining,
            "budget_utilization_pct": round(utilization, 1)
        }
    
    def find_best_combination(
        self,
        flights: List[Dict[str, Any]],
        hotels: List[Dict[str, Any]],
        budget: float,
        nights: int = 1,
        preferences: Optional[Dict[str, Any]] = None
    ) -> CombinationResult:
        """
        Find the optimal flight+hotel combination using LLM reasoning.
        The LLM analyzes options and decides based on quality, value, and budget utilization.
        """
        self.metrics["checks_performed"] += 1
        
        if not flights or not hotels:
            self._log("❌ No flights or hotels to evaluate")
            return CombinationResult(success=False, reasoning="No options available.")
        
        self._log(f"Evaluating {len(flights)} flights × {len(hotels)} hotels = {len(flights) * len(hotels)} combinations")
        
        # Step 1: Filter to valid combinations within budget
        valid_combinations = []
        all_combinations_data = []
        
        for flight in flights:
            flight_cost = flight.get("price_usd", 0)
            for hotel in hotels:
                hotel_total = hotel.get("price_per_night_usd", 0) * nights
                total_cost = flight_cost + hotel_total
                
                self.metrics["combinations_evaluated"] += 1
                is_valid = total_cost <= budget
                
                combo_data = {
                    "flight_id": flight.get("flight_id", ""),
                    "hotel_id": hotel.get("hotel_id", ""),
                    "flight_cost": flight_cost,
                    "hotel_cost": hotel_total,
                    "total_cost": total_cost,
                    "within_budget": is_valid
                }
                all_combinations_data.append(combo_data)
                
                if is_valid:
                    self.metrics["valid_combinations_found"] += 1
                    valid_combinations.append({
                        "flight": flight,
                        "hotel": hotel,
                        "total_cost": total_cost
                    })
        
        self._log(f"Found {len(valid_combinations)} valid combinations within ${budget}")
        
        if not valid_combinations:
            self.metrics["no_valid_options"] += 1
            min_cost = min(c["total_cost"] for c in all_combinations_data) if all_combinations_data else 0
            return CombinationResult(
                success=False, combinations_evaluated=len(all_combinations_data),
                reasoning=f"No valid combinations within ${budget}. Cheapest: ${min_cost} (${min_cost - budget} over).",
                all_combinations=all_combinations_data[:10]
            )
        
        # Step 2: Prepare diverse options for LLM to reason about
        # Group by strategy: budget-conscious, balanced, premium
        sorted_by_cost = sorted(valid_combinations, key=lambda x: x["total_cost"])
        sorted_by_hotel_stars = sorted(valid_combinations, key=lambda x: -x["hotel"].get("stars", 0))
        sorted_by_flight_class = sorted(valid_combinations, 
            key=lambda x: (0 if x["flight"].get("class") == "First Class" else 
                          1 if x["flight"].get("class") == "Business" else 2))
        
        # Select diverse candidates for LLM (up to 8 options)
        candidates = []
        seen = set()
        
        def add_candidate(combo, reason):
            key = (combo["flight"].get("flight_id"), combo["hotel"].get("hotel_id"))
            if key not in seen and len(candidates) < 8:
                seen.add(key)
                candidates.append((combo, reason))
        
        # Add budget option (cheapest)
        if sorted_by_cost:
            add_candidate(sorted_by_cost[0], "Most budget-friendly")
        
        # Add premium hotel options (highest stars)
        for c in sorted_by_hotel_stars[:2]:
            add_candidate(c, f"{c['hotel'].get('stars')}★ hotel")
        
        # Add premium flight options
        for c in sorted_by_flight_class[:2]:
            add_candidate(c, f"{c['flight'].get('class', 'Economy')} flight")
        
        # Add high-utilization options (use most of budget)
        sorted_by_utilization = sorted(valid_combinations, key=lambda x: -x["total_cost"])
        for c in sorted_by_utilization[:2]:
            add_candidate(c, "Maximizes budget usage")
        
        # Add middle-ground option
        if len(sorted_by_cost) > 2:
            mid_idx = len(sorted_by_cost) // 2
            add_candidate(sorted_by_cost[mid_idx], "Balanced option")
        
        # Step 3: Use LLM to reason and select the best option
        selected = self._llm_select_best(candidates, budget, nights, preferences)
        
        if selected:
            self.metrics["optimal_selections"] += 1
            
            # Find cheaper alternatives
            cheaper_alternatives = []
            for combo in sorted_by_cost[:5]:
                if combo["total_cost"] < selected["total_cost"]:
                    cheaper_alternatives.append({
                        "flight": {"flight_id": combo["flight"].get("flight_id"),
                                  "airline": combo["flight"].get("airline", "Unknown"),
                                  "price_usd": combo["flight"].get("price_usd")},
                        "hotel": {"hotel_id": combo["hotel"].get("hotel_id"),
                                 "name": combo["hotel"].get("name", "Unknown"),
                                 "stars": combo["hotel"].get("stars"),
                                 "price_per_night_usd": combo["hotel"].get("price_per_night_usd")},
                        "total_cost": combo["total_cost"],
                        "savings_vs_selected": round(selected["total_cost"] - combo["total_cost"], 2)
                    })
            
            return CombinationResult(
                success=True,
                selected_flight=selected["flight"],
                selected_hotel=selected["hotel"],
                total_cost=selected["total_cost"],
                budget_remaining=budget - selected["total_cost"],
                value_score=selected.get("value_score", 0),
                combinations_evaluated=len(all_combinations_data),
                reasoning=selected.get("reasoning", ""),
                all_combinations=all_combinations_data[:10],
                cheaper_alternatives=cheaper_alternatives[:3]
            )
        
        # Fallback: return highest utilization option
        best = sorted_by_utilization[0]
        return CombinationResult(
            success=True,
            selected_flight=best["flight"],
            selected_hotel=best["hotel"],
            total_cost=best["total_cost"],
            budget_remaining=budget - best["total_cost"],
            reasoning="Fallback: selected option that maximizes budget utilization."
        )
    
    def _llm_select_best(self, candidates: List[tuple], budget: float, nights: int,
                         preferences: Optional[Dict[str, Any]]) -> Optional[Dict]:
        """Use LLM to reason about and select the best combination."""
        if not candidates:
            return None
        
        if len(candidates) == 1:
            combo, reason = candidates[0]
            return {
                **combo,
                "reasoning": f"Only valid option: {reason}",
                "value_score": 100
            }
        
        # Build options text for LLM
        options_text = []
        for i, (combo, label) in enumerate(candidates, 1):
            summary = self._summarize_option(combo["flight"], combo["hotel"], nights, budget)
            options_text.append(
                f"Option {i} ({label}):\n"
                f"  Flight: {summary['flight']['airline']} {summary['flight']['class']} - ${summary['flight']['price']} "
                f"(departs {summary['flight']['departure']}, {summary['flight']['duration_hours']:.1f}h)\n"
                f"  Hotel: {summary['hotel']['name']} {summary['hotel']['stars']}★ - ${summary['hotel']['price_per_night']}/night "
                f"({summary['hotel']['distance_km']:.1f}km from center)\n"
                f"  Total: ${summary['total_cost']} | Remaining: ${summary['budget_remaining']} "
                f"| Budget Used: {summary['budget_utilization_pct']}%"
            )
        
        prompt = f"""You are a business travel policy agent. Select the BEST flight+hotel combination.

BUDGET: ${budget} for {nights} night(s)

AVAILABLE OPTIONS:
{chr(10).join(options_text)}

DECISION CRITERIA (in order of importance):
1. QUALITY: Prefer higher hotel stars (5★ > 4★ > 3★) and better flight class (First > Business > Economy)
2. BUDGET UTILIZATION: Use budget wisely - don't leave too much unused if better options exist
3. CONVENIENCE: Morning flights, shorter durations, hotels close to business center
4. VALUE: Best quality for the money spent

IMPORTANT: 
- If budget allows for a premium option (Business/First class flight OR 5★ hotel), prefer it
- Leaving 30%+ of budget unused when premium options exist is wasteful
- Balance quality with practical value

Analyze each option and select the best one. Return JSON:
{{"selected_option": <number 1-{len(candidates)}>, "reasoning": "<detailed explanation of why this is the best choice>", "value_score": <0-200>}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            idx = result.get("selected_option", 1) - 1
            reasoning = result.get("reasoning", "Selected based on overall value")
            value_score = result.get("value_score", 100)
            
            if 0 <= idx < len(candidates):
                combo, label = candidates[idx]
                return {
                    **combo,
                    "reasoning": reasoning,
                    "value_score": value_score
                }
        except Exception as e:
            self._log(f"LLM selection error: {e}")
        
        # Fallback: pick the one with highest utilization
        combo, label = max(candidates, key=lambda x: x[0]["total_cost"])
        return {
            **combo,
            "reasoning": f"Fallback selection: {label}",
            "value_score": 80
        }
    
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
    
    def generate_feedback(
        self,
        flights: List[Dict[str, Any]],
        hotels: List[Dict[str, Any]],
        budget: float,
        nights: int,
        negotiation_round: int
    ) -> Dict[str, Any]:
        """
        CNP NEGOTIATION: Generate feedback for booking agents when no valid combination found.
        
        This enables the negotiation loop - PolicyAgent analyzes WHY combinations failed
        and provides specific, actionable feedback for agents to refine their proposals.
        
        Returns:
            Dict with feedback for both agents:
            {
                "needs_refinement": True/False,
                "flight_feedback": {...} or None,
                "hotel_feedback": {...} or None,
                "reasoning": "explanation"
            }
        """
        self._log(f"Generating negotiation feedback (round {negotiation_round})")
        
        if not flights or not hotels:
            return {
                "needs_refinement": True,
                "flight_feedback": {"issue": "no_options", "reasoning": "No flights available"} if not flights else None,
                "hotel_feedback": {"issue": "no_options", "reasoning": "No hotels available"} if not hotels else None,
                "reasoning": "Missing flight or hotel options"
            }
        
        # Calculate cheapest possible combination
        min_flight_cost = min(f.get("price_usd", 9999) for f in flights)
        min_hotel_cost = min(h.get("price_per_night_usd", 9999) for h in hotels) * nights
        min_total = min_flight_cost + min_hotel_cost
        
        # If even the cheapest combo exceeds budget, we need cheaper options
        if min_total > budget:
            budget_gap = min_total - budget
            
            # Determine which agent needs to reduce more
            flight_portion = min_flight_cost / min_total if min_total > 0 else 0.5
            hotel_portion = 1 - flight_portion
            
            # Both need to help, but proportionally
            flight_reduction_needed = budget_gap * flight_portion
            hotel_reduction_needed = budget_gap * hotel_portion
            
            flight_feedback = None
            hotel_feedback = None
            
            if flight_reduction_needed > 50:  # Significant reduction needed
                max_flight_price = max(50, min_flight_cost - flight_reduction_needed - 50)
                flight_feedback = {
                    "issue": "budget_exceeded",
                    "max_price": int(max_flight_price),
                    "current_min_price": min_flight_cost,
                    "reduction_needed": flight_reduction_needed,
                    "from_city": flights[0].get("from_city", ""),
                    "to_city": flights[0].get("to_city", ""),
                    "reasoning": f"Cheapest flight ${min_flight_cost} too expensive. Need flights under ${int(max_flight_price)}."
                }
            
            if hotel_reduction_needed > 30:  # Significant reduction needed
                max_hotel_price = max(50, (min_hotel_cost / nights) - (hotel_reduction_needed / nights) - 30)
                hotel_feedback = {
                    "issue": "budget_exceeded",
                    "max_price_per_night": int(max_hotel_price),
                    "current_min_price": min_hotel_cost / nights,
                    "reduction_needed": hotel_reduction_needed,
                    "city": hotels[0].get("city", ""),
                    "reasoning": f"Cheapest hotel ${min_hotel_cost/nights:.0f}/night too expensive. Need hotels under ${int(max_hotel_price)}/night."
                }
            
            return {
                "needs_refinement": True,
                "flight_feedback": flight_feedback,
                "hotel_feedback": hotel_feedback,
                "reasoning": f"Budget gap of ${budget_gap:.0f}. Min combo costs ${min_total:.0f} vs budget ${budget}."
            }
        
        # If we have valid combinations but none are optimal, check for quality issues
        max_flight_cost = max(f.get("price_usd", 0) for f in flights)
        max_hotel_stars = max(h.get("stars", 1) for h in hotels)
        
        budget_remaining = budget - min_total
        
        # If we have significant budget remaining, suggest quality upgrades
        if budget_remaining > budget * 0.4:  # More than 40% unused
            feedback = {"needs_refinement": False, "flight_feedback": None, "hotel_feedback": None}
            
            # Check if we can suggest better flights
            has_premium_flights = any(f.get("class") in ["Business", "First Class"] for f in flights)
            if not has_premium_flights and budget_remaining > 300:
                feedback["flight_feedback"] = {
                    "issue": "quality_insufficient",
                    "min_class": "Business",
                    "budget_available": budget_remaining,
                    "from_city": flights[0].get("from_city", ""),
                    "to_city": flights[0].get("to_city", ""),
                    "reasoning": f"${budget_remaining:.0f} budget remaining allows for Business/First class upgrade."
                }
                feedback["needs_refinement"] = True
            
            # Check if we can suggest better hotels
            if max_hotel_stars < 5 and budget_remaining > 200:
                feedback["hotel_feedback"] = {
                    "issue": "quality_insufficient",
                    "min_stars": 5,
                    "budget_available": budget_remaining,
                    "city": hotels[0].get("city", ""),
                    "reasoning": f"${budget_remaining:.0f} budget remaining allows for 5★ hotel upgrade."
                }
                feedback["needs_refinement"] = True
            
            if feedback["needs_refinement"]:
                feedback["reasoning"] = f"${budget_remaining:.0f} unused budget. Suggesting quality upgrades."
                return feedback
        
        # No refinement needed - we have good options
        return {
            "needs_refinement": False,
            "flight_feedback": None,
            "hotel_feedback": None,
            "reasoning": "Valid combinations available. Proceeding to selection."
        }
    
    def get_metrics(self) -> Dict[str, int]:
        return self.metrics.copy()
    
    def reset_metrics(self):
        self.metrics = {"checks_performed": 0, "combinations_evaluated": 0,
                       "valid_combinations_found": 0, "optimal_selections": 0, "no_valid_options": 0}
