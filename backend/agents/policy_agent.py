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
                "distance_km": hotel.get("distance_to_business_center_km") or hotel.get("distance_km") or 0
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
        
        # BUDGET EXCEEDED HANDLING: If no valid combinations, return the cheapest anyway
        # This is business travel - better to exceed budget slightly than fail completely
        if not valid_combinations:
            self.metrics["no_valid_options"] += 1
            
            # Find the cheapest combination overall
            cheapest_combo = min(all_combinations_data, key=lambda x: x["total_cost"])
            cheapest_flight = next((f for f in flights if f.get("flight_id") == cheapest_combo["flight_id"]), flights[0])
            cheapest_hotel = next((h for h in hotels if h.get("hotel_id") == cheapest_combo["hotel_id"]), hotels[0])
            
            min_cost = cheapest_combo["total_cost"]
            over_budget = min_cost - budget
            
            self._log(f"⚠️ Budget exceeded by ${over_budget:.0f}, returning cheapest available")
            
            return CombinationResult(
                success=True,  # Still return as success with best-effort
                selected_flight=cheapest_flight,
                selected_hotel=cheapest_hotel,
                total_cost=min_cost,
                budget_remaining=budget - min_cost,  # Will be negative
                value_score=50,  # Lower score for budget exceeded
                combinations_evaluated=len(all_combinations_data),
                reasoning=f"Budget ${budget} exceeded. Cheapest option is ${min_cost:.0f} (${over_budget:.0f} over). Returning best available.",
                all_combinations=all_combinations_data[:10],
                cheaper_alternatives=[]  # No cheaper alternatives exist
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
        
        prompt = f"""You are a business travel policy agent. Select the BEST flight+hotel combination for a business trip.

BUDGET: ${budget} for {nights} night(s)

AVAILABLE OPTIONS:
{chr(10).join(options_text)}

DECISION CRITERIA - Balance ALL of these factors:
1. VALUE FOR MONEY: Is the quality worth the price? A 3★ hotel at $100/night may be better value than a 5★ at $500/night
2. CONVENIENCE: Morning flights are better for business. Shorter flights reduce fatigue. Hotels near business center save commute time
3. QUALITY: Higher hotel stars and better flight class provide comfort, but only if justified by the trip purpose
4. BUDGET EFFICIENCY: Use budget wisely - neither waste money on unnecessary luxury NOR leave significant budget unused when upgrades would genuinely help

REASONING GUIDELINES:
- A 4★ hotel with great location may be better than a 5★ far from meetings
- Economy class is fine for short flights; Business class makes sense for 6+ hour flights
- Sometimes the mid-range option offers the best balance
- Don't just pick the most expensive - pick what makes SENSE for business travel
- Consider: Would a reasonable business traveler appreciate this choice?

Analyze each option carefully and select the one with the best overall value proposition. Return JSON:
{{"selected_option": <number 1-{len(candidates)}>, "reasoning": "<explain your reasoning about value, convenience, and quality tradeoffs>", "value_score": <0-200>}}"""

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
        
        # Fallback: pick the middle option (balanced choice)
        mid_idx = len(candidates) // 2
        combo, label = candidates[mid_idx]
        return {
            **combo,
            "reasoning": f"Fallback selection (balanced option): {label}",
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
        negotiation_round: int,
        feedback_history: List[str] = None,
        previous_min_cost: float = None
    ) -> Dict[str, Any]:
        """
        CNP NEGOTIATION: Generate feedback for booking agents using LLM reasoning.
        
        The PolicyAgent reasons about the current proposals and decides what
        feedback to give. It has full context about previous rounds to detect
        when negotiation should converge (no hardcoded stalemate logic).
        
        Returns:
            Dict with feedback for both agents
        """
        self._log(f"Generating negotiation feedback (round {negotiation_round})")
        
        feedback_history = feedback_history or []
        
        if not flights or not hotels:
            return {
                "needs_refinement": True,
                "flight_feedback": {"issue": "no_options", "reasoning": "No flights available"} if not flights else None,
                "hotel_feedback": {"issue": "no_options", "reasoning": "No hotels available"} if not hotels else None,
                "reasoning": "Missing flight or hotel options",
                "current_min_cost": 0
            }
        
        # Calculate cost metrics for LLM reasoning
        min_flight_cost = min(f.get("price_usd", 9999) for f in flights)
        max_flight_cost = max(f.get("price_usd", 0) for f in flights)
        min_hotel_cost = min(h.get("price_per_night_usd", 9999) for h in hotels) * nights
        max_hotel_cost = max(h.get("price_per_night_usd", 0) for h in hotels) * nights
        min_total = min_flight_cost + min_hotel_cost
        max_total = max_flight_cost + max_hotel_cost
        
        # Calculate quality metrics
        max_hotel_stars = max(h.get("stars", 1) for h in hotels)
        min_hotel_stars = min(h.get("stars", 5) for h in hotels)
        flight_classes = set(f.get("class", "Economy") for f in flights)
        
        # Detect if we're making progress (for LLM context)
        cost_improved = False
        if previous_min_cost is not None:
            cost_improved = min_total < previous_min_cost
        
        # Format previous cost for prompt
        prev_cost_str = f"${previous_min_cost:.0f}" if previous_min_cost is not None else "N/A (first round)"
        improvement_str = 'Yes' if cost_improved else ('No' if previous_min_cost is not None else 'N/A (first round)')
        
        # Use LLM to reason about what feedback to provide
        prompt = f"""You are a PolicyAgent negotiating business travel arrangements. You must decide whether to continue negotiating or accept the current options.

CURRENT SITUATION:
- Budget: ${budget} for {nights} night(s)
- Negotiation Round: {negotiation_round + 1} of 5 maximum
- Previous feedback given: {feedback_history if feedback_history else "None (first round)"}
- Previous cheapest cost: {prev_cost_str}
- Cost improvement this round: {improvement_str}

FLIGHT OPTIONS ({len(flights)} available):
- Price range: ${min_flight_cost} - ${max_flight_cost}
- Classes: {', '.join(flight_classes)}

HOTEL OPTIONS ({len(hotels)} available):  
- Price range: ${min_hotel_cost/nights:.0f} - ${max_hotel_cost/nights:.0f} per night
- Star ratings: {min_hotel_stars}★ - {max_hotel_stars}★

COST ANALYSIS:
- Cheapest possible combination: ${min_total:.0f}
- Budget: ${budget}
- Gap: ${max(0, min_total - budget):.0f} {'OVER budget' if min_total > budget else 'within budget'}

CONVERGENCE REASONING:
You should ACCEPT current options (needs_refinement: false) when:
1. A valid combination exists within budget
2. No cost improvement was made since last round (negotiation has converged)
3. You've already asked for the same type of refinement before (avoid loops)
4. Round 3+ and budget cannot be met - accept best effort

You should REQUEST REFINEMENT (needs_refinement: true) when:
1. This is round 1 and there's room for improvement
2. Cost improved last round, suggesting more improvement is possible
3. Budget has significant unused headroom (>40%) and quality could be upgraded

IMPORTANT: If you asked for "budget_exceeded" feedback before and cost didn't improve, you MUST accept current options to avoid infinite loops.

Analyze the situation and decide. Return JSON:
{{
    "needs_refinement": true/false,
    "reasoning": "Your analysis explaining WHY you're accepting or requesting refinement",
    "flight_feedback": {{"issue": "budget_exceeded|quality_insufficient", "reasoning": "specific feedback"}} or null,
    "hotel_feedback": {{"issue": "budget_exceeded|quality_insufficient", "reasoning": "specific feedback"}} or null
}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            needs_refinement = result.get("needs_refinement", False)
            reasoning = result.get("reasoning", "LLM decision")
            
            flight_feedback = result.get("flight_feedback")
            hotel_feedback = result.get("hotel_feedback")
            
            # Enrich feedback with actual data for agents
            if flight_feedback and flight_feedback.get("issue") == "budget_exceeded":
                flight_feedback["current_min_price"] = min_flight_cost
                flight_feedback["from_city"] = flights[0].get("from_city", "")
                flight_feedback["to_city"] = flights[0].get("to_city", "")
                
            if hotel_feedback and hotel_feedback.get("issue") == "budget_exceeded":
                hotel_feedback["current_min_price"] = min_hotel_cost / nights
                hotel_feedback["city"] = hotels[0].get("city", "")
            
            self._log(f"LLM feedback: needs_refinement={needs_refinement}, reasoning={reasoning[:80]}...")
            
            return {
                "needs_refinement": needs_refinement,
                "flight_feedback": flight_feedback,
                "hotel_feedback": hotel_feedback,
                "reasoning": reasoning,
                "current_min_cost": min_total
            }
            
        except Exception as e:
            self._log(f"LLM feedback error: {e}, using fallback")
            
            # Fallback: simple rule-based decision
            if min_total <= budget:
                return {
                    "needs_refinement": False,
                    "reasoning": f"Valid combinations available within ${budget} budget.",
                    "current_min_cost": min_total
                }
            elif previous_min_cost and min_total >= previous_min_cost:
                # No improvement - converge
                return {
                    "needs_refinement": False,
                    "reasoning": f"No cost improvement (${min_total:.0f}). Accepting best available.",
                    "current_min_cost": min_total
                }
            else:
                return {
                    "needs_refinement": True,
                    "flight_feedback": {"issue": "budget_exceeded", "reasoning": f"Need cheaper flights (current min: ${min_flight_cost})"},
                    "hotel_feedback": {"issue": "budget_exceeded", "reasoning": f"Need cheaper hotels (current min: ${min_hotel_cost/nights:.0f}/night)"},
                    "reasoning": f"Budget gap: ${min_total - budget:.0f}",
                    "current_min_cost": min_total
                }
    
    def get_metrics(self) -> Dict[str, int]:
        return self.metrics.copy()
    
    def reset_metrics(self):
        self.metrics = {"checks_performed": 0, "combinations_evaluated": 0,
                       "valid_combinations_found": 0, "optimal_selections": 0, "no_valid_options": 0}
