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
            
            # Generate DIVERSE alternatives: Premium (more expensive), Similar (same range), Budget (cheaper)
            diverse_alternatives = []
            selected_cost = selected["total_cost"]

            # Find alternatives in different price categories
            premium_combos = [c for c in valid_combinations if c["total_cost"] > selected_cost * 1.15]
            similar_combos = [c for c in valid_combinations
                             if abs(c["total_cost"] - selected_cost) < selected_cost * 0.15
                             and c != selected]
            budget_combos = [c for c in valid_combinations if c["total_cost"] < selected_cost * 0.85]

            def create_alternative(combo, category, reasoning):
                return {
                    "category": category,
                    "flight": {
                        "flight_id": combo["flight"].get("flight_id"),
                        "airline": combo["flight"].get("airline", "Unknown"),
                        "class": combo["flight"].get("class", "Economy"),
                        "price_usd": combo["flight"].get("price_usd"),
                        "departure_time": combo["flight"].get("departure_time")
                    },
                    "hotel": {
                        "hotel_id": combo["hotel"].get("hotel_id"),
                        "name": combo["hotel"].get("name", "Unknown"),
                        "stars": combo["hotel"].get("stars"),
                        "price_per_night_usd": combo["hotel"].get("price_per_night_usd"),
                        "distance_km": combo["hotel"].get("distance_to_business_center_km", 0)
                    },
                    "total_cost": combo["total_cost"],
                    "vs_selected": round(combo["total_cost"] - selected_cost, 2),
                    "reasoning": reasoning
                }

            # 1. PREMIUM: More expensive option with better quality/features
            if premium_combos:
                # Sort by best quality/value among premium options
                premium_sorted = sorted(premium_combos, key=lambda x: (
                    -x["hotel"].get("stars", 0),  # Higher stars first
                    x["hotel"].get("distance_to_business_center_km", 99),  # Closer first
                    x["total_cost"]  # Then by price
                ))

                p = premium_sorted[0]
                h_stars = p["hotel"].get("stars", 3)
                f_class = p["flight"].get("class", "Economy")
                cost_diff = p["total_cost"] - selected_cost

                reasoning = f"Premium: {h_stars}★ hotel"
                if f_class != "Economy":
                    reasoning += f", {f_class} flight"
                reasoning += f" (+${cost_diff:.0f})"

                diverse_alternatives.append(create_alternative(p, "🔶 PREMIUM", reasoning))

            # 2. SIMILAR: Comparable price, different option
            if similar_combos:
                # Find option with different hotel/airline for variety
                for c in similar_combos:
                    if (c["hotel"].get("hotel_id") != selected["hotel"].get("hotel_id") or
                        c["flight"].get("airline") != selected["flight"].get("airline")):
                        h_stars = c["hotel"].get("stars", 3)
                        airline = c["flight"].get("airline", "Unknown")
                        cost_diff = c["total_cost"] - selected_cost

                        reasoning = f"Alternative: {h_stars}★ hotel, {airline}"
                        if abs(cost_diff) > 1:
                            reasoning += f" ({'+'if cost_diff > 0 else ''}{cost_diff:.0f})"

                        diverse_alternatives.append(create_alternative(c, "🔷 SIMILAR", reasoning))
                        break

            # 3. BUDGET: Cheaper option with good value
            if budget_combos:
                # Sort budget options by best value (stars per dollar)
                budget_sorted = sorted(budget_combos, key=lambda x: (
                    -x["hotel"].get("stars", 0) / max(x["total_cost"], 1),  # Value score
                    x["total_cost"]  # Then by price
                ))

                c = budget_sorted[0]
                savings = selected_cost - c["total_cost"]
                h_stars = c["hotel"].get("stars", 3)

                reasoning = f"Budget: Save ${savings:.0f}"
                if h_stars >= 3:
                    reasoning += f" ({h_stars}★ hotel)"
                else:
                    reasoning += " (basic accommodation)"

                diverse_alternatives.append(create_alternative(c, "💚 BUDGET", reasoning))
            
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
                cheaper_alternatives=diverse_alternatives  # Now diverse, not just cheaper
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
        
        # If we're within budget, no refinement needed
        if min_total <= budget:
            return {
                "needs_refinement": False,
                "reasoning": f"Valid combination exists: ${min_total:.0f} within ${budget} budget.",
                "current_min_cost": min_total
            }
        
        # If no improvement and already tried, accept best effort
        if previous_min_cost is not None and min_total >= previous_min_cost and negotiation_round >= 2:
            return {
                "needs_refinement": False,
                "reasoning": f"No cost improvement after {negotiation_round} rounds. Accepting best: ${min_total:.0f}",
                "current_min_cost": min_total
            }
        
        # STRATEGIC DECISION: Which agent(s) should reduce prices?
        budget_gap = min_total - budget
        flight_share = min_flight / min_total if min_total > 0 else 0.5
        hotel_share = min_hotel / min_total if min_total > 0 else 0.5
        
        flight_feedback = None
        hotel_feedback = None
        
        # Strategy: Target the component that takes more budget OR has more variance
        if flight_share > 0.55 or flight_variance > hotel_variance * 1.5:
            # Flight is the bigger cost driver - ask for cheaper flights
            target_flight_max = min_flight - (budget_gap * 0.6)  # Ask for 60% of gap from flights
            flight_feedback = {
                "issue": "budget_exceeded",
                "reasoning": f"Flight costs are {flight_share*100:.0f}% of total. Need flights under ${max(50, target_flight_max):.0f} to fit budget.",
                "max_price": max(50, int(target_flight_max)),
                "from_city": flights[0].get("from_city", ""),
                "to_city": flights[0].get("to_city", "")
            }
            self._log(f"Targeting FlightAgent: need price reduction of ~${budget_gap * 0.6:.0f}")
            
        elif hotel_share > 0.55 or hotel_variance > flight_variance * 1.5:
            # Hotel is the bigger cost driver - ask for cheaper hotels
            target_hotel_max = (min_hotel - (budget_gap * 0.6)) / nights
            hotel_feedback = {
                "issue": "budget_exceeded",
                "reasoning": f"Hotel costs are {hotel_share*100:.0f}% of total. Need hotels under ${max(50, target_hotel_max):.0f}/night to fit budget.",
                "max_price_per_night": max(50, int(target_hotel_max)),
                "city": hotels[0].get("city", "")
            }
            self._log(f"Targeting HotelAgent: need price reduction of ~${budget_gap * 0.6:.0f}")
            
        else:
            # Both are roughly equal - ask both to reduce slightly
            flight_target = min_flight - (budget_gap * 0.4)
            hotel_target = (min_hotel - (budget_gap * 0.6)) / nights
            flight_feedback = {
                "issue": "budget_exceeded",
                "reasoning": f"Need flights under ${max(50, flight_target):.0f}",
                "max_price": max(50, int(flight_target)),
                "from_city": flights[0].get("from_city", ""),
                "to_city": flights[0].get("to_city", "")
            }
            hotel_feedback = {
                "issue": "budget_exceeded", 
                "reasoning": f"Need hotels under ${max(50, hotel_target):.0f}/night",
                "max_price_per_night": max(50, int(hotel_target)),
                "city": hotels[0].get("city", "")
            }
            self._log(f"Targeting BOTH agents for price reduction")
        
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
