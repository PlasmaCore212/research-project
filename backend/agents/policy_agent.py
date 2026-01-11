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
        # QUALITY-FOCUSED selection - prioritize quality while considering value
        sorted_by_cost = sorted(valid_combinations, key=lambda x: x["total_cost"])
        sorted_by_hotel_stars = sorted(valid_combinations, key=lambda x: -x["hotel"].get("stars", 0))
        
        # Calculate value scores for each combo - QUALITY-WEIGHTED
        for combo in valid_combinations:
            stars = combo["hotel"].get("stars", 3)
            flight_class_score = 2 if combo["flight"].get("class") == "Business" else 1
            
            # Use meeting distance if available, otherwise business center
            distance = combo["hotel"].get("distance_to_meeting_km", 
                                         combo["hotel"].get("distance_to_business_center_km", 5))
            
            # NEW FORMULA: Quality-focused with reduced price impact
            # Quality score: stars (weighted 3x) + flight class (2x) + proximity (10-distance)
            quality_score = stars * 3 + flight_class_score * 2 + (10 - min(distance, 10))
            
            # Price factor: reduced impact using power of 0.7 instead of 1.0
            # This makes expensive options less penalized
            price_factor = 1000 / (combo["total_cost"] ** 0.7)
            
            combo["value_score"] = quality_score * price_factor
        
        sorted_by_value = sorted(valid_combinations, key=lambda x: -x["value_score"])
        
        # Select diverse candidates for LLM (up to 8 options, more premium choices)
        candidates = []
        seen = set()
        
        def add_candidate(combo, reason):
            key = (combo["flight"].get("flight_id"), combo["hotel"].get("hotel_id"))
            if key not in seen and len(candidates) < 8:
                seen.add(key)
                candidates.append((combo, reason))
        
        # Add BEST VALUE options first (quality per dollar with new formula)
        for c in sorted_by_value[:3]:
            add_candidate(c, "Best value for money")
        
        # Add premium hotel options (highest stars)
        for c in sorted_by_hotel_stars[:2]:
            add_candidate(c, f"{c['hotel'].get('stars')}★ premium hotel")
        
        # Add mid-range options (50-75th percentile by cost)
        if len(sorted_by_cost) > 4:
            mid_start = len(sorted_by_cost) // 2
            mid_end = int(len(sorted_by_cost) * 0.75)
            for c in sorted_by_cost[mid_start:mid_end][:2]:
                add_candidate(c, "Quality mid-range option")
        
        # Add one budget option for comparison
        if sorted_by_cost:
            add_candidate(sorted_by_cost[0], "Budget-conscious option")
        
        # Step 3: Use LLM to reason and select the best option
        selected = self._llm_select_best(candidates, budget, nights, preferences)
        
        if selected:
            self.metrics["optimal_selections"] += 1
            
            # Generate HOTEL-ONLY alternatives (keep selected flight fixed)
            # This avoids needing to re-validate through time agent
            hotel_alternatives = []
            selected_hotel_id = selected["hotel"].get("hotel_id")
            selected_hotel_price = selected["hotel"].get("price_per_night_usd", 0)
            selected_flight = selected["flight"]
            selected_flight_cost = selected_flight.get("price_usd", 0)
            
            # Find alternative hotels with the same selected flight
            alternative_hotels = [h for h in hotels if h.get("hotel_id") != selected_hotel_id]
            
            def create_hotel_alternative(hotel, category, reasoning):
                hotel_total = hotel.get("price_per_night_usd", 0) * nights
                total_cost = selected_flight_cost + hotel_total
                vs_selected = total_cost - selected["total_cost"]
                return {
                    "category": category,
                    "hotel": {
                        "hotel_id": hotel.get("hotel_id"),
                        "name": hotel.get("name", "Unknown"),
                        "stars": hotel.get("stars"),
                        "price_per_night_usd": hotel.get("price_per_night_usd"),
                        "distance_km": hotel.get("distance_to_business_center_km", 0),
                        "amenities": hotel.get("amenities", [])[:5]
                    },
                    "total_cost": total_cost,
                    "hotel_cost": hotel_total,
                    "vs_selected": round(vs_selected, 2),
                    "reasoning": reasoning
                }
            
            # Track used hotel IDs to avoid duplicates
            used_hotel_ids = set()
            
            # 1. PREMIUM HOTEL: Higher stars/quality (CAN exceed budget - shows genuine upgrade option)
            premium_hotels = sorted(
                [h for h in alternative_hotels 
                 if h.get("price_per_night_usd", 0) > selected_hotel_price * 1.15],  # No budget cap - show real upgrades
                key=lambda x: (-x.get("stars", 0), x.get("distance_to_business_center_km", 99))
            )
            if premium_hotels:
                h = premium_hotels[0]
                used_hotel_ids.add(h.get("hotel_id"))
                h_stars = h.get("stars", 3)
                price_diff = (h.get("price_per_night_usd", 0) - selected_hotel_price) * nights
                total_cost = selected_flight_cost + h.get("price_per_night_usd", 0) * nights
                over_budget = total_cost > budget
                budget_note = f" (${total_cost - budget:.0f} over budget)" if over_budget else ""
                reasoning = f"Upgrade: {h_stars}★ hotel (+${price_diff:.0f}){budget_note}"
                hotel_alternatives.append(create_hotel_alternative(h, "🔶 PREMIUM", reasoning))
            
            # 2. SIMILAR HOTEL: Same tier, different option (exclude already used)
            similar_hotels = [
                h for h in alternative_hotels
                if abs(h.get("price_per_night_usd", 0) - selected_hotel_price) < selected_hotel_price * 0.2
                and h.get("hotel_id") not in used_hotel_ids
            ]
            if similar_hotels:
                # Pick one with different characteristics
                for h in similar_hotels:
                    used_hotel_ids.add(h.get("hotel_id"))
                    h_stars = h.get("stars", 3)
                    h_dist = h.get("distance_to_business_center_km", 0)
                    reasoning = f"Alternative: {h_stars}★, {h_dist:.1f}km to center"
                    hotel_alternatives.append(create_hotel_alternative(h, "🔷 SIMILAR", reasoning))
                    break
            
            # 3. BUDGET HOTEL: Cheaper option (exclude already used)
            budget_hotels = sorted(
                [h for h in alternative_hotels 
                 if h.get("price_per_night_usd", 0) < selected_hotel_price * 0.85
                 and h.get("hotel_id") not in used_hotel_ids],
                key=lambda x: (-x.get("stars", 0) / max(x.get("price_per_night_usd", 1), 1), x.get("price_per_night_usd", 0))
            )
            if budget_hotels:
                h = budget_hotels[0]
                h_stars = h.get("stars", 3)
                savings = (selected_hotel_price - h.get("price_per_night_usd", 0)) * nights
                reasoning = f"Save ${savings:.0f} ({h_stars}★ hotel)"
                hotel_alternatives.append(create_hotel_alternative(h, "💚 BUDGET", reasoning))
            
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
                cheaper_alternatives=hotel_alternatives  # Now hotel-only
            )
        
        # Fallback: return best value option
        best = sorted_by_value[0] if sorted_by_value else sorted_by_cost[0]
        return CombinationResult(
            success=True,
            selected_flight=best["flight"],
            selected_hotel=best["hotel"],
            total_cost=best["total_cost"],
            budget_remaining=budget - best["total_cost"],
            reasoning="Fallback: selected best value option."
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

⭐ DECISION PHILOSOPHY FOR BUSINESS TRAVEL:
Quality and convenience are IMPORTANT for business travelers. A well-rested, comfortable traveler is more productive.
Aim to use 60-80% of budget on quality accommodations that genuinely enhance the business trip experience.

DECISION CRITERIA (in priority order):
1. QUALITY & COMFORT: Prioritize 4-5★ hotels and convenient flight times. Business travelers deserve comfort.
2. CONVENIENCE: Morning flights, proximity to meetings, minimal travel time. Time is valuable.
3. VALUE PROPOSITION: Quality should justify the price, but don't be overly budget-conscious. A $400 4★ hotel is often worth it vs a $150 3★.
4. APPROPRIATE SPENDING: Using 60-80% of budget is GOOD for business travel. Don't leave excessive budget unused.

REASONING GUIDELINES:
- For business trips, quality matters: 4★+ hotels provide better work environment, WiFi, and amenities
- Proximity to meeting venue is crucial - saves time and reduces stress
- Morning flights (6-9am) are ideal for business - worth paying slightly more
- It's OKAY to spend more for genuine quality improvements
- Only choose budget options if quality difference is minimal
- Ask: "Would a professional appreciate this choice?" not "Is this the cheapest?"

TARGET: Aim for options using 60-80% of available budget while maximizing quality and convenience.

Analyze each option and select the one that best serves a business traveler. Return JSON:
{{"selected_option": <number 1-{len(candidates)}>, "reasoning": "<explain why this quality/convenience justifies the price>", "value_score": <0-200>}}"""

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
        
        # Fallback: pick the FIRST option (which is now sorted by best value)
        combo, label = candidates[0]
        return {
            **combo,
            "reasoning": f"Fallback selection (best value option): {label}",
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
    
    def _should_request_quality_upgrade(
        self,
        budget: float,
        min_total: float,
        budget_utilization: float,
        flights: List[Dict[str, Any]],
        hotels: List[Dict[str, Any]]
    ) -> tuple[bool, str, Dict[str, Any]]:
        """
        LLM-based decision: Should we request premium/quality upgrade options?
        
        Returns:
            tuple: (should_upgrade, reasoning, feedback_dict)
        """
        # Get option summaries for context
        best_hotel_stars = max((h.get("stars", 0) for h in hotels), default=0)
        best_flight_class = "Business" if any(f.get("class") == "Business" for f in flights) else "Economy"
        
        prompt = f"""You are a PolicyAgent deciding whether to request PREMIUM options for a business trip.

CURRENT SITUATION:
- Budget: ${budget:.0f}
- Current best combination: ${min_total:.0f} ({budget_utilization:.0f}% of budget used)
- Budget remaining unused: ${budget - min_total:.0f}
- Best available hotel: {best_hotel_stars}★
- Best available flight class: {best_flight_class}

DECISION CRITERIA:
For business travel, quality matters. Consider:
1. Is there significant budget remaining that could improve the travel experience?
2. Would upgrading to 4-5★ hotels or Business class flights meaningfully improve comfort?
3. Is the current quality level already appropriate for business travel?

GUIDANCE:
- If budget utilization is very low (e.g., under 60-70%) AND current options are basic, consider upgrading
- If current options are already high quality (4-5★ hotels, Business class), no upgrade needed
- If budget remaining is small relative to upgrade cost, don't force upgrades
- Use your judgment - there's no fixed threshold

Respond with JSON:
{{
  "should_upgrade": true/false,
  "reasoning": "Your analysis of whether quality upgrades are warranted and why"
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            should_upgrade = result.get("should_upgrade", False)
            reasoning = result.get("reasoning", "LLM decision")
            
            self._log(f"Quality upgrade decision: {should_upgrade} - {reasoning[:80]}...")
            
            if should_upgrade:
                target_spend = budget * 0.85  # Suggest target, not enforce
                feedback = {
                    "needs_refinement": True,
                    "flight_feedback": {
                        "issue": "quality_upgrade",
                        "reasoning": f"{reasoning}. Please offer premium flight options.",
                        "from_city": flights[0].get("from_city", "") if flights else "",
                        "to_city": flights[0].get("to_city", "") if flights else ""
                    },
                    "hotel_feedback": {
                        "issue": "quality_upgrade",
                        "reasoning": f"{reasoning}. Please offer 4-5★ hotel options.",
                        "city": hotels[0].get("city", "") if hotels else ""
                    },
                    "reasoning": reasoning,
                    "current_min_cost": min_total
                }
                return True, reasoning, feedback
            
            return False, reasoning, {}
            
        except Exception as e:
            self._log(f"Quality upgrade LLM decision failed: {e}")
            return False, f"LLM decision error: {e}", {}
    
    def _should_terminate_negotiation(
        self,
        budget: float,
        min_total: float,
        negotiation_round: int,
        previous_min_cost: float,
        feedback_history: List[str]
    ) -> tuple[bool, str]:
        """
        LLM-based decision: Should we stop negotiating and accept current options?
        
        Returns:
            tuple: (should_terminate, reasoning)
        """
        cost_improved = previous_min_cost is not None and min_total < previous_min_cost
        improvement_amount = previous_min_cost - min_total if previous_min_cost and cost_improved else 0
        budget_gap = min_total - budget if min_total > budget else 0
        
        prompt = f"""You are a PolicyAgent deciding whether to CONTINUE or STOP negotiating for better prices.

CURRENT SITUATION:
- Budget: ${budget:.0f}
- Current best total: ${min_total:.0f}
- Budget gap (if over): ${budget_gap:.0f}
- Negotiation rounds completed: {negotiation_round}
- Cost improved this round: {cost_improved} (by ${improvement_amount:.0f})

NEGOTIATION HISTORY:
{chr(10).join(feedback_history[-5:]) if feedback_history else "No previous feedback"}

DECISION CRITERIA:
1. Is further negotiation likely to yield meaningful improvement?
2. Are we stuck in a loop with no progress?
3. Have we exhausted reasonable options?
4. Is the current best option acceptable even if slightly over budget?

GUIDANCE:
- If costs are not improving after multiple rounds, it may be time to stop
- If we're very close to budget, one more round might help
- If budget gap is large and no progress, accept best effort
- Consider diminishing returns - small improvements may not be worth more negotiation
- Use your judgment based on the specific situation

Respond with JSON:
{{
  "should_terminate": true/false,
  "reasoning": "Your analysis of whether to stop negotiating and why"
}}"""
        
        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            should_terminate = result.get("should_terminate", False)
            reasoning = result.get("reasoning", "LLM decision")
            
            self._log(f"Termination decision: {should_terminate} - {reasoning[:80]}...")
            return should_terminate, reasoning
            
        except Exception as e:
            self._log(f"Termination LLM decision failed: {e}")
            # Fallback: terminate after many rounds with no improvement
            if negotiation_round >= 5 and not cost_improved:
                return True, "Fallback: many rounds without improvement"
            return False, f"LLM decision error: {e}"
    
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
        CNP NEGOTIATION: Generate TARGETED feedback for specific agent(s).
        
        PolicyAgent analyzes which agent should adjust prices based on:
        1. Which component (flight vs hotel) is taking more of the budget
        2. Which component has more price variance (more room to negotiate)
        3. Quality considerations (don't sacrifice too much quality)
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
        
        # Calculate cost metrics
        min_flight = min(f.get("price_usd", 9999) for f in flights)
        max_flight = max(f.get("price_usd", 0) for f in flights)
        min_hotel = min(h.get("price_per_night_usd", 9999) for h in hotels) * nights
        max_hotel = max(h.get("price_per_night_usd", 0) for h in hotels) * nights
        min_total = min_flight + min_hotel
        
        # Calculate price variance (indicator of negotiation room)
        flight_variance = max_flight - min_flight
        hotel_variance = max_hotel - min_hotel
        
        # Detect if we're making progress
        cost_improved = previous_min_cost is not None and min_total < previous_min_cost
        
        # Calculate budget utilization
        budget_utilization = (min_total / budget * 100) if budget > 0 else 100
        
        # LLM-DRIVEN QUALITY UPGRADE DECISION
        # Ask the LLM to reason about whether we should request premium options
        if min_total <= budget and negotiation_round == 0:
            should_upgrade, upgrade_reasoning, upgrade_feedback = self._should_request_quality_upgrade(
                budget=budget,
                min_total=min_total,
                budget_utilization=budget_utilization,
                flights=flights,
                hotels=hotels
            )
            if should_upgrade:
                return upgrade_feedback
        
        # If we're within budget, no further refinement needed
        if min_total <= budget:
            return {
                "needs_refinement": False,
                "reasoning": f"Valid combination exists: ${min_total:.0f} within ${budget} budget.",
                "current_min_cost": min_total
            }
        
        # LLM-DRIVEN TERMINATION DECISION
        # Ask the LLM to reason about whether to continue negotiating
        if negotiation_round > 0:
            should_terminate, term_reasoning = self._should_terminate_negotiation(
                budget=budget,
                min_total=min_total,
                negotiation_round=negotiation_round,
                previous_min_cost=previous_min_cost,
                feedback_history=feedback_history
            )
            if should_terminate:
                return {
                    "needs_refinement": False,
                    "reasoning": term_reasoning,
                    "current_min_cost": min_total
                }
        
        # STRATEGIC DECISION: Use LLM to decide which agent(s) should reduce prices
        budget_gap = min_total - budget
        flight_share = min_flight / min_total if min_total > 0 else 0.5
        hotel_share = min_hotel / min_total if min_total > 0 else 0.5
        
        # Build context for LLM decision
        flight_info = f"Flight: ${min_flight:.0f}-${max_flight:.0f} ({flight_share*100:.0f}% of total)"
        hotel_info = f"Hotel: ${min_hotel:.0f}-${max_hotel:.0f} for {nights} nights ({hotel_share*100:.0f}% of total)"
        
        prompt = f"""You are a PolicyAgent negotiating with FlightAgent and HotelAgent to reduce costs.

BUDGET SITUATION:
- Budget: ${budget:.0f}
- Current best total: ${min_total:.0f} (${budget_gap:.0f} OVER budget)
- {flight_info}
- {hotel_info}
- Flight price variance: ${flight_variance:.0f} (range: ${min_flight:.0f}-${max_flight:.0f})
- Hotel price variance: ${hotel_variance/nights:.0f}/night (range: ${min_hotel/nights:.0f}-${max_hotel/nights:.0f}/night)

NEGOTIATION HISTORY:
{chr(10).join(feedback_history[-3:]) if feedback_history else "No previous rounds"}

YOUR TASK:
Analyze the situation and decide:
1. Which agent(s) should reduce prices? (FlightAgent only, HotelAgent only, or BOTH)
2. What should the target price be for each?

CONSIDERATIONS:
- If one agent takes >70% of cost, they might need to reduce more
- If costs are roughly equal, asking BOTH to reduce a little is often better than one to reduce a lot
- Consider price variance: agent with more variance has more room to negotiate
- Consider previous history: if one agent already reduced, maybe target the other
- If flight is very expensive but hotel is cheap, target flight (and vice versa)

ALL OPTIONS ARE VALID.

Return JSON with YOUR reasoned decision:
{{
  "target_agent": "flight" | "hotel" | "both",
  "flight_should_reduce": true/false,
  "hotel_should_reduce": true/false,
  "reasoning": "Explain your analysis and why you chose this strategy",
  "flight_target_price": <your reasoned max price for flights>,
  "hotel_target_price_per_night": <your reasoned max price per night>
}}"""

        flight_feedback = None
        hotel_feedback = None
        
        try:
            response = self.llm.invoke(prompt)
            decision = json.loads(response)
            
            target = decision.get("target_agent", "both")
            reasoning = decision.get("reasoning", "Budget exceeded, need price reduction")
            
            self._log(f"LLM decision: target {target} - {reasoning[:80]}...")
            
            if decision.get("flight_should_reduce", target in ["flight", "both"]):
                target_price = decision.get("flight_target_price", min_flight - budget_gap * 0.5)
                flight_feedback = {
                    "issue": "budget_exceeded",
                    "reasoning": f"{reasoning}. Need flights under ${max(50, target_price):.0f}.",
                    "max_price": max(50, int(target_price)),
                    "from_city": flights[0].get("from_city", "") if flights else "",
                    "to_city": flights[0].get("to_city", "") if flights else ""
                }
            
            if decision.get("hotel_should_reduce", target in ["hotel", "both"]):
                target_price = decision.get("hotel_target_price_per_night", (min_hotel - budget_gap * 0.5) / nights)
                hotel_feedback = {
                    "issue": "budget_exceeded",
                    "reasoning": f"{reasoning}. Need hotels under ${max(50, target_price):.0f}/night.",
                    "max_price_per_night": max(50, int(target_price)),
                    "city": hotels[0].get("city", "") if hotels else ""
                }
                
        except Exception as e:
            self._log(f"LLM decision failed: {e}, using balanced fallback")
            # Fallback: target BOTH agents with balanced reduction (proportional to cost share)
            flight_reduction = budget_gap * flight_share
            hotel_reduction = budget_gap * hotel_share
            flight_target = max(50, min_flight - flight_reduction)
            hotel_target = max(50, (min_hotel - hotel_reduction) / nights)
            
            flight_feedback = {
                "issue": "budget_exceeded",
                "reasoning": f"Balanced reduction needed. Flights ({flight_share*100:.0f}% of cost).",
                "max_price": max(50, int(flight_target)),
                "from_city": flights[0].get("from_city", "") if flights else "",
                "to_city": flights[0].get("to_city", "") if flights else ""
            }
            hotel_feedback = {
                "issue": "budget_exceeded",
                "reasoning": f"Balanced reduction needed. Hotels ({hotel_share*100:.0f}% of cost).",
                "max_price_per_night": max(50, int(hotel_target)),
                "city": hotels[0].get("city", "") if hotels else ""
            }
        
        reasoning = f"Budget gap: ${budget_gap:.0f}. Flight: ${min_flight} ({flight_share*100:.0f}%), Hotel: ${min_hotel} ({hotel_share*100:.0f}%)"
        
        return {
            "needs_refinement": True,
            "flight_feedback": flight_feedback,
            "hotel_feedback": hotel_feedback,
            "reasoning": reasoning,
            "current_min_cost": min_total
        }
    
    def get_metrics(self) -> Dict[str, int]:
        return self.metrics.copy()
    
    def reset_metrics(self):
        self.metrics = {"checks_performed": 0, "combinations_evaluated": 0,
                       "valid_combinations_found": 0, "optimal_selections": 0, "no_valid_options": 0}
