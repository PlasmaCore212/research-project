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
    reasoning: str


class CombinationResult(BaseModel):
    """Result of finding the best combination."""
    success: bool
    selected_flight: Optional[Dict[str, Any]] = None
    selected_hotel: Optional[Dict[str, Any]] = None
    total_cost: float = 0
    budget_remaining: float = 0
    combinations_evaluated: int = 0
    reasoning: str = ""
    all_combinations: List[Dict[str, Any]] = []
    cheaper_alternatives: List[Dict[str, Any]] = []


class PolicyComplianceAgent:
    """
    Policy Compliance Agent
    
    The LLM reasons about and selects the best combination by considering:
    - Budget utilization (use budget wisely, don't leave too much unused)
    - Quality (hotel stars, amenities, location)
    - Flight quality (timing, duration, airline)
    - Overall value (balance of all factors)
    """
    
    def __init__(self, model_name: str = "qwen2.5:14b", verbose: bool = True):
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
                combinations_evaluated=len(all_combinations_data),
                reasoning=f"Budget ${budget} exceeded. Cheapest option is ${min_cost:.0f} (${over_budget:.0f} over). Returning best available.",
                all_combinations=all_combinations_data[:10],
                cheaper_alternatives=[]  # No cheaper alternatives exist
            )
        
        # Step 2: Prepare diverse options for LLM to reason about
        # Include options across ALL quality tiers for balanced selection
        sorted_by_cost = sorted(valid_combinations, key=lambda x: x["total_cost"])
        
        # Select diverse candidates for LLM (up to 12 options)
        candidates = []
        seen = set()
        
        def add_candidate(combo, reason):
            key = (combo["flight"].get("flight_id"), combo["hotel"].get("hotel_id"))
            if key not in seen and len(candidates) < 12:
                seen.add(key)
                candidates.append((combo, reason))
        
        # Group options by FLIGHT CLASS to ensure diversity in flight quality
        by_flight_class = {}
        for c in valid_combinations:
            flight_class = c["flight"].get("class", "Economy")
            if flight_class not in by_flight_class:
                by_flight_class[flight_class] = []
            by_flight_class[flight_class].append(c)
        
        # Add best combo from each flight class (prioritize higher classes)
        class_priority = ["First Class", "Business", "Economy"]
        for flight_class in class_priority:
            if flight_class in by_flight_class:
                # Sort by hotel stars (high to low), then by total cost
                class_combos = sorted(
                    by_flight_class[flight_class],
                    key=lambda x: (-x["hotel"].get("stars", 0), x["total_cost"])
                )
                if class_combos:
                    add_candidate(class_combos[0], f"Best {flight_class} flight option")
                    # Add a second option if available (different hotel tier)
                    if len(class_combos) > 1:
                        add_candidate(class_combos[1], f"Alternative {flight_class} option")
        
        # Group options by hotel star rating to also ensure hotel diversity
        by_stars = {}
        for c in valid_combinations:
            stars = c["hotel"].get("stars", 0)
            if stars not in by_stars:
                by_stars[stars] = []
            by_stars[stars].append(c)
        
        # From each star category, add the best flight (prioritize Business/First)
        for stars in sorted(by_stars.keys(), reverse=True):  # Start with highest stars
            tier_combos = sorted(
                by_stars[stars],
                key=lambda x: (0 if x["flight"].get("class") == "First Class" 
                              else 1 if x["flight"].get("class") == "Business" 
                              else 2, x["total_cost"])
            )
            if tier_combos:
                add_candidate(tier_combos[0], f"Best flight for {stars}★ hotel")
        
        # Add options at different budget utilization points (80%, 90%, 95%)
        for target_pct in [0.80, 0.90, 0.95]:
            target_cost = budget * target_pct
            closest = min(valid_combinations, key=lambda x: abs(x["total_cost"] - target_cost))
            add_candidate(closest, f"~{int(target_pct*100)}% budget utilization")
        
        # Add cheapest and most expensive for range
        add_candidate(sorted_by_cost[0], "Cheapest option")
        if len(sorted_by_cost) > 1:
            add_candidate(sorted_by_cost[-1], "Premium option")
        
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
                combinations_evaluated=len(all_combinations_data),
                reasoning=selected.get("reasoning", ""),
                all_combinations=all_combinations_data[:10],
                cheaper_alternatives=hotel_alternatives  # Now hotel-only
            )
        
        # Fallback: return cheapest valid option
        best = sorted_by_cost[0]
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
                "reasoning": f"Only valid option: {reason}"
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
        
        prompt = f"""Select the BEST flight+hotel combination for a BUSINESS trip.
Budget: ${budget} for {nights} night(s).

OPTIONS:
{chr(10).join(options_text)}

SELECTION PRINCIPLES:
1. MAXIMIZE VALUE: Choose the highest quality option that fits within or near budget
2. QUALITY INDICATORS:
   - Hotels: Higher stars (5★ > 4★ > 3★), closer distance to meeting location, better amenities
   - Flights: Better class (First class > Business > Economy), reasonable duration
3. BUDGET UTILIZATION:
   - IDEAL: Use 75-95% of budget to get good quality without waste
   - Using <60% usually means missing an upgrade opportunity
   - Going slightly over (up to 5%) may be acceptable for significant quality gains

BALANCE GUIDANCE:
- If a 4★ hotel costs ${int(budget*0.15)} more than a 3★, it's usually worth it within budget
- Don't pick the absolute cheapest if a notably better option exists at 80-90% budget
- Consider the total experience: a great hotel can offset a standard flight

Return JSON: {{"selected_option": <1-{len(candidates)}>, "reasoning": "<explain the quality-price tradeoff>"}}"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response)
            
            idx = result.get("selected_option", 1) - 1
            reasoning = result.get("reasoning", "Selected based on overall analysis")
            
            if 0 <= idx < len(candidates):
                combo, label = candidates[idx]
                return {
                    **combo,
                    "reasoning": reasoning
                }
        except Exception as e:
            self._log(f"LLM selection error: {e}")
        
        # Fallback: pick the FIRST option (cheapest valid)
        combo, label = candidates[0]
        return {
            **combo,
            "reasoning": f"Fallback selection: {label}"
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
        
        prompt = f"""Should we STOP negotiating? Budget: ${budget:.0f}, Best: ${min_total:.0f}, Gap: ${budget_gap:.0f}, Round: {negotiation_round}, Improved: {cost_improved} (${improvement_amount:.0f}).
History: {'; '.join(feedback_history[-3:]) if feedback_history else 'None'}

Stop if: no progress after multiple rounds, stuck in loop, or gap too large to close.
Continue if: close to budget and improvement likely.

Return JSON: {{"should_terminate": true/false, "reasoning": "<brief explanation>"}}"""
        
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
        previous_min_cost: float = None,
        current_selection: Dict[str, float] = None
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
        current_selection = current_selection or {}
        
        if not flights or not hotels:
            return {
                "needs_refinement": True,
                "flight_feedback": {"issue": "no_options", "reasoning": "No flights available"} if not flights else None,
                "hotel_feedback": {"issue": "no_options", "reasoning": "No hotels available"} if not hotels else None,
                "reasoning": "Missing flight or hotel options",
                "current_min_cost": 0
            }
        
        # Calculate cost metrics from all available options
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
        
        # Use CURRENT SELECTION for budget utilization (not min available)
        # This is the key fix - use what was actually selected, not cheapest possible
        selected_flight_price = current_selection.get("flight_price", min_flight)
        selected_hotel_price = current_selection.get("hotel_price", min_hotel / nights) * nights if current_selection.get("hotel_price") else min_hotel
        current_total = selected_flight_price + selected_hotel_price if selected_flight_price > 0 else min_total
        
        budget_utilization = (current_total / budget * 100) if budget > 0 else 100
        
        # THRESHOLD-BASED NEGOTIATION TRIGGER
        MIN_UTILIZATION = 80  # Trigger upgrade negotiation if < 80%
        MAX_UTILIZATION = 100  # Trigger cost reduction if > 100%
        
        self._log(f"Budget utilization: {budget_utilization:.0f}% (current: ${current_total:.0f}, min: {MIN_UTILIZATION}%, max: {MAX_UTILIZATION}%)")
        
        if min_total <= budget:
            if budget_utilization < MIN_UTILIZATION:
                # Under-utilizing budget → request quality upgrades from BOTH agents
                self._log(f"Budget utilization {budget_utilization:.0f}% < {MIN_UTILIZATION}% → requesting quality upgrades")
                
                # Track this as a quality upgrade negotiation
                self.metrics["negotiation_rounds"] = self.metrics.get("negotiation_rounds", 0) + 1
                
                # Calculate TARGET price ranges based on achieving 85% budget utilization
                # Distribute the additional spending proportionally
                current_min_flight = min(f.get("price_usd", 0) for f in flights) if flights else 0
                current_min_hotel = min(h.get("price_per_night_usd", 0) for h in hotels) if hotels else 0
                current_min_hotel_total = current_min_hotel * nights
                
                target_spend = budget * 0.85  # Target 85% utilization
                extra_to_spend = target_spend - min_total
                
                # Split extra budget: 40% to flight upgrade, 60% to hotel upgrade
                extra_flight = extra_to_spend * 0.4
                extra_hotel = extra_to_spend * 0.6
                
                # Calculate target prices that would achieve ~85% utilization
                target_flight_price = current_min_flight + extra_flight
                target_hotel_price = current_min_hotel + (extra_hotel / nights)
                
                # Set ranges: min is 10% above current (some upgrade), max is calculated target or up to 95% budget
                target_flight_min = int(current_min_flight * 1.1)
                target_flight_max = int(min(target_flight_price * 1.2, budget * 0.6))  # Cap at 60% of budget for flight
                target_hotel_min = int(current_min_hotel * 1.1)
                target_hotel_max = int(min(target_hotel_price * 1.2, (budget * 0.7) / nights))  # Cap at 70% of budget for hotel
                
                return {
                    "needs_refinement": True,
                    "flight_feedback": {
                        "issue": "quality_upgrade",
                        "reasoning": f"Budget utilization is only {budget_utilization:.0f}%. Target flight price: ${target_flight_min}-${target_flight_max}. Look for Business class, better timing, or higher-tier Economy.",
                        "target_price_min": target_flight_min,
                        "target_price_max": target_flight_max,
                        "from_city": flights[0].get("from_city", "") if flights else "",
                        "to_city": flights[0].get("to_city", "") if flights else "",
                        "re_search": True
                    },
                    "hotel_feedback": {
                        "issue": "quality_upgrade",
                        "reasoning": f"Budget utilization is only {budget_utilization:.0f}%. Target hotel price: ${target_hotel_min}-${target_hotel_max}/night. Look for 4-5★ hotels closer to meeting venue.",
                        "target_price_min": target_hotel_min,
                        "target_price_max": target_hotel_max,
                        "city": hotels[0].get("city", "") if hotels else "",
                        "re_search": True
                    },
                    "reasoning": f"Under-budget: only using {budget_utilization:.0f}% of ${budget} budget. Requesting quality upgrades with target prices: Flight ${target_flight_min}-${target_flight_max}, Hotel ${target_hotel_min}-${target_hotel_max}/night.",
                    "current_min_cost": min_total
                }
            else:
                # 80-100% utilization → optimal range, accept proposal
                return {
                    "needs_refinement": False,
                    "reasoning": f"Optimal budget utilization ({budget_utilization:.0f}%): ${min_total:.0f} of ${budget} used effectively.",
                    "current_min_cost": min_total
                }
        
        # If > 100%, continue to cost reduction logic below
        
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
        
        prompt = f"""Budget: ${budget:.0f}, Total: ${min_total:.0f} (${budget_gap:.0f} OVER). {flight_info}. {hotel_info}.
Flight variance: ${flight_variance:.0f}, Hotel variance: ${hotel_variance/nights:.0f}/night.
History: {'; '.join(feedback_history[-3:]) if feedback_history else 'None'}

Decide: which agent should reduce prices? Target the one with higher cost share or more variance.

Return JSON: {{"target_agent": "flight"|"hotel"|"both", "flight_should_reduce": bool, "hotel_should_reduce": bool, "reasoning": "<brief>", "flight_target_price": <int>, "hotel_target_price_per_night": <int>}}"""

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
