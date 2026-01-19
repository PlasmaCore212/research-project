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

    def validate_combination(
        self,
        flight: Dict[str, Any],
        hotel: Dict[str, Any],
        budget: float,
        nights: int = 1
    ) -> Dict[str, Any]:
        """
        Pure validation: check if flight+hotel meets budget.
        NO coordination decisions - just return violations.

        Args:
            flight: Flight dict with price_usd
            hotel: Hotel dict with price_per_night_usd
            budget: Total budget
            nights: Number of nights

        Returns:
            {
                "is_valid": bool,
                "total_cost": float,
                "budget_remaining": float,
                "violations": [
                    {"type": "budget_exceeded", "severity": "error", "amount": 150, "component": "flight"},
                    {"type": "low_quality", "severity": "warning", "message": "..."}
                ]
            }
        """
        # Calculate costs
        flight_cost = flight.get('price_usd', 0)
        hotel_total = hotel.get('price_per_night_usd', 0) * nights
        total_cost = flight_cost + hotel_total
        budget_remaining = budget - total_cost

        # Build violations list
        violations = []

        # Check budget violation
        if budget_remaining < 0:
            violations.append({
                "type": "budget_exceeded",
                "severity": "error",
                "amount": -budget_remaining,
                "component": "total",
                "message": f"Total cost ${total_cost} exceeds budget ${budget} by ${-budget_remaining:.0f}"
            })

        # Check quality warnings (not errors, just warnings)
        hotel_stars = hotel.get('stars', 3)
        if hotel_stars < 3:
            violations.append({
                "type": "low_quality",
                "severity": "warning",
                "component": "hotel",
                "message": f"Hotel has only {hotel_stars} stars (below business travel standard of 3+)"
            })

        # Check flight class quality
        flight_class = flight.get('class', 'Economy')
        flight_duration = flight.get('duration_hours', 0)
        if flight_class == 'Economy' and flight_duration > 6:
            violations.append({
                "type": "low_comfort",
                "severity": "warning",
                "component": "flight",
                "message": f"Long flight ({flight_duration:.1f}h) in Economy class may impact productivity"
            })

        is_valid = budget_remaining >= 0

        return {
            "is_valid": is_valid,
            "total_cost": total_cost,
            "budget_remaining": budget_remaining,
            "violations": violations
        }
    
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
        SELECTION ONLY - NO COORDINATION LOGIC

        Find the optimal flight+hotel combination using LLM reasoning.
        The LLM analyzes options and decides based on quality, value, and budget utilization.

        This method ONLY selects the best combo from available options.
        It does NOT coordinate agents or generate feedback.
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
        
        # Add options at different budget utilization points (85%, 90%, 95%)
        for target_pct in [0.85, 0.90, 0.95]:
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
   - IDEAL: Use 85-95% of budget (aim for ~90%) to maximize value
   - Using <75% usually means missing an upgrade opportunity
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
    
    def get_metrics(self) -> Dict[str, int]:
        return self.metrics.copy()
    
    def reset_metrics(self):
        self.metrics = {"checks_performed": 0, "combinations_evaluated": 0,
                       "valid_combinations_found": 0, "optimal_selections": 0, "no_valid_options": 0}
