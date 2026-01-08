# backend/agents/policy_agent.py
"""
Policy Compliance Agent with ReAct Pattern and Chain-of-Thought Prompting

This agent specializes in validating travel bookings against company policies.
It uses the ReAct pattern (Thought -> Action -> Observation) to:
1. Load and understand company travel policies
2. Validate flight and hotel selections
3. Identify policy violations
4. Provide recommendations for compliance

References:
- ReAct: Synergizing Reasoning and Acting (Yao et al., 2023)
- Chain-of-Thought Prompting (Wei et al., 2022)
- Hybrid LLM-Solver Approaches (TRIP-PAL, 2024)
"""

from .base_agent import BaseReActAgent, AgentAction
from .models import (
    PolicyRules, PolicyCheckResult, PolicyViolation,
    FlightSearchResult, HotelSearchResult
)
from data.loaders import PolicyDataLoader
from typing import List, Dict, Any, Optional
import json


class PolicyAgent(BaseReActAgent):
    """
    Agentic Policy Compliance Agent with ReAct reasoning.
    
    This agent autonomously:
    - Loads company travel policies
    - Validates bookings against policy rules
    - Identifies and categorizes violations
    - Suggests remediation actions
    """
    
    def __init__(self, model_name: str = "llama3.1:8b", verbose: bool = True):
        super().__init__(
            agent_name="PolicyAgent",
            agent_role="Corporate Travel Policy Compliance Officer",
            model_name=model_name,
            max_iterations=5,
            verbose=verbose
        )
        
        self.loader = PolicyDataLoader()
        self.tools = self._register_tools()
        
        # Track validation history
        self.validation_history: List[Dict] = []
    
    def _register_tools(self) -> Dict[str, AgentAction]:
        """Register tools available to the Policy Agent"""
        return {
            "load_policy": AgentAction(
                name="load_policy",
                description="Load company travel policy rules by name",
                parameters={
                    "policy_name": "str - policy name (e.g., 'standard', 'strict', 'executive')"
                },
                function=self._tool_load_policy
            ),
            "check_flight_compliance": AgentAction(
                name="check_flight_compliance",
                description="Check if a flight selection complies with policy",
                parameters={
                    "flight_id": "str - flight ID to check",
                    "flight_price": "int - flight price in USD",
                    "airline": "str - airline name"
                },
                function=self._tool_check_flight_compliance
            ),
            "check_hotel_compliance": AgentAction(
                name="check_hotel_compliance",
                description="Check if a hotel selection complies with policy",
                parameters={
                    "hotel_id": "str - hotel ID to check",
                    "price_per_night": "int - hotel price per night in USD",
                    "stars": "int - hotel star rating",
                    "distance_km": "float - distance to business center in km",
                    "amenities": "list - list of hotel amenities"
                },
                function=self._tool_check_hotel_compliance
            ),
            "check_total_budget": AgentAction(
                name="check_total_budget",
                description="Check if total trip cost is within budget",
                parameters={
                    "flight_cost": "int - flight cost in USD",
                    "hotel_cost_per_night": "int - hotel cost per night",
                    "nights": "int - number of nights"
                },
                function=self._tool_check_total_budget
            ),
            "get_policy_summary": AgentAction(
                name="get_policy_summary",
                description="Get a summary of all policy rules",
                parameters={}
            ,
                function=self._tool_get_policy_summary
            ),
            "suggest_adjustments": AgentAction(
                name="suggest_adjustments",
                description="Suggest adjustments to fix policy violations",
                parameters={
                    "violations": "list - list of violation types to address"
                },
                function=self._tool_suggest_adjustments
            )
        }
    
    def _get_system_prompt(self) -> str:
        """Get the domain-specific system prompt for Policy Agent"""
        return """You are an expert Corporate Travel Policy Compliance Officer AI Agent.

YOUR EXPERTISE:
- Understanding and enforcing company travel policies
- Validating bookings against budget and quality standards
- Identifying policy violations and their severity
- Recommending compliant alternatives

REASONING APPROACH (Chain-of-Thought):
When checking compliance, think through:
1. POLICY RULES: What are the specific limits and requirements?
2. FLIGHT CHECK: Does the flight price/airline meet policy?
3. HOTEL CHECK: Does the hotel meet price/stars/location requirements?
4. BUDGET CHECK: Is the total trip cost within budget?
5. SEVERITY: Which violations are errors vs warnings?
6. REMEDIATION: What changes would make this compliant?

VIOLATION SEVERITY LEVELS:
- ERROR: Must be fixed before booking (e.g., over budget)
- WARNING: Should be reviewed but can be approved with justification

COMPLIANCE PRIORITIES:
- Budget limits are strict (errors if exceeded)
- Preferred vendors are suggestions (warnings only)
- Minimum quality standards protect employee welfare"""
    
    def _tool_load_policy(self, policy_name: str) -> str:
        """Tool: Load a company travel policy"""
        
        policy_dict = self.loader.get_policy(policy_name)
        
        if not policy_dict:
            return f"Policy '{policy_name}' not found. Available: standard, strict, executive"
        
        # Store policy in beliefs
        self.state.add_belief("current_policy_name", policy_name)
        self.state.add_belief("current_policy", policy_dict)
        
        # Format policy summary
        policy = PolicyRules(**policy_dict)
        
        summary_lines = [f"Loaded policy: {policy_name}"]
        if policy.max_flight_price:
            summary_lines.append(f"  - Max flight price: ${policy.max_flight_price}")
        if policy.max_hotel_price_per_night:
            summary_lines.append(f"  - Max hotel price: ${policy.max_hotel_price_per_night}/night")
        if policy.max_total_budget:
            summary_lines.append(f"  - Max total budget: ${policy.max_total_budget}")
        if policy.preferred_airlines:
            summary_lines.append(f"  - Preferred airlines: {', '.join(policy.preferred_airlines)}")
        if policy.min_hotel_stars:
            summary_lines.append(f"  - Min hotel stars: {policy.min_hotel_stars}")
        if policy.max_distance_to_center_km:
            summary_lines.append(f"  - Max distance to center: {policy.max_distance_to_center_km}km")
        if policy.required_hotel_amenities:
            summary_lines.append(f"  - Required amenities: {', '.join(policy.required_hotel_amenities)}")
        
        return "\n".join(summary_lines)
    
    def _tool_check_flight_compliance(
        self,
        flight_id: str,
        flight_price: int,
        airline: str
    ) -> str:
        """Tool: Check flight against policy"""
        
        policy_dict = self.state.get_belief("current_policy", {})
        if not policy_dict:
            return "ERROR: No policy loaded. Use load_policy first."
        
        policy = PolicyRules(**policy_dict)
        violations = []
        
        # Check price
        if policy.max_flight_price and flight_price > policy.max_flight_price:
            violations.append({
                "rule": "max_flight_price",
                "severity": "error",
                "message": f"FLIGHT price ${flight_price} exceeds maximum ${policy.max_flight_price}",
                "excess": flight_price - policy.max_flight_price
            })
        
        # Check airline preference
        if policy.preferred_airlines and airline not in policy.preferred_airlines:
            violations.append({
                "rule": "preferred_airlines",
                "severity": "warning",
                "message": f"Airline '{airline}' not in preferred list: {', '.join(policy.preferred_airlines)}"
            })
        
        # Store violations
        self.state.add_belief("flight_violations", violations)
        
        if not violations:
            return f"✓ Flight {flight_id} is COMPLIANT with policy"
        else:
            result = f"✗ Flight {flight_id} has {len(violations)} violation(s):"
            for v in violations:
                result += f"\n  [{v['severity'].upper()}] {v['message']}"
            return result
    
    def _tool_check_hotel_compliance(
        self,
        hotel_id: str,
        price_per_night: int,
        stars: int,
        distance_km: float,
        amenities: List[str]
    ) -> str:
        """Tool: Check hotel against policy"""
        
        policy_dict = self.state.get_belief("current_policy", {})
        if not policy_dict:
            return "ERROR: No policy loaded. Use load_policy first."
        
        policy = PolicyRules(**policy_dict)
        violations = []
        
        # Check price
        if policy.max_hotel_price_per_night and price_per_night > policy.max_hotel_price_per_night:
            violations.append({
                "rule": "max_hotel_price_per_night",
                "severity": "error",
                "message": f"HOTEL price ${price_per_night}/night exceeds maximum ${policy.max_hotel_price_per_night}",
                "excess": price_per_night - policy.max_hotel_price_per_night
            })
        
        # Check stars
        if policy.min_hotel_stars and stars < policy.min_hotel_stars:
            violations.append({
                "rule": "min_hotel_stars",
                "severity": "error",
                "message": f"Hotel {stars}* rating below minimum {policy.min_hotel_stars}*"
            })
        
        # Check distance
        if policy.max_distance_to_center_km and distance_km > policy.max_distance_to_center_km:
            violations.append({
                "rule": "max_distance_to_center_km",
                "severity": "warning",
                "message": f"Hotel {distance_km:.1f}km from center exceeds recommended {policy.max_distance_to_center_km}km"
            })
        
        # Check amenities
        if policy.required_hotel_amenities:
            missing = set(policy.required_hotel_amenities) - set(amenities)
            if missing:
                violations.append({
                    "rule": "required_hotel_amenities",
                    "severity": "error",
                    "message": f"Hotel missing required amenities: {', '.join(missing)}"
                })
        
        # Store violations
        self.state.add_belief("hotel_violations", violations)
        
        if not violations:
            return f"✓ Hotel {hotel_id} is COMPLIANT with policy"
        else:
            result = f"✗ Hotel {hotel_id} has {len(violations)} violation(s):"
            for v in violations:
                result += f"\n  [{v['severity'].upper()}] {v['message']}"
            return result
    
    def _tool_check_total_budget(
        self,
        flight_cost: int,
        hotel_cost_per_night: int,
        nights: int = 1
    ) -> str:
        """Tool: Check total trip cost against budget"""
        
        policy_dict = self.state.get_belief("current_policy", {})
        if not policy_dict:
            return "ERROR: No policy loaded. Use load_policy first."
        
        policy = PolicyRules(**policy_dict)
        
        total_hotel = hotel_cost_per_night * nights
        total_cost = flight_cost + total_hotel
        
        result = f"Budget Analysis:\n"
        result += f"  Flight: ${flight_cost}\n"
        result += f"  Hotel: ${hotel_cost_per_night} x {nights} nights = ${total_hotel}\n"
        result += f"  Total: ${total_cost}\n"
        
        if policy.max_total_budget:
            if total_cost > policy.max_total_budget:
                excess = total_cost - policy.max_total_budget
                result += f"\n✗ OVER BUDGET by ${excess} (max: ${policy.max_total_budget})"
                
                # Store budget violation
                self.state.add_belief("budget_violation", {
                    "rule": "max_total_budget",
                    "severity": "error",
                    "message": f"Total ${total_cost} exceeds budget ${policy.max_total_budget}",
                    "excess": excess
                })
            else:
                remaining = policy.max_total_budget - total_cost
                result += f"\n✓ WITHIN BUDGET (${remaining} remaining)"
        else:
            result += "\n✓ No total budget limit in policy"
        
        return result
    
    def _tool_get_policy_summary(self) -> str:
        """Tool: Get summary of current policy"""
        
        policy_dict = self.state.get_belief("current_policy", {})
        policy_name = self.state.get_belief("current_policy_name", "unknown")
        
        if not policy_dict:
            return "No policy loaded. Use load_policy first."
        
        return f"Current policy: {policy_name}\nRules: {json.dumps(policy_dict, indent=2)}"
    
    def _tool_suggest_adjustments(self, violations: List[str]) -> str:
        """Tool: Suggest how to fix violations"""
        
        policy_dict = self.state.get_belief("current_policy", {})
        if not policy_dict:
            return "No policy loaded."
        
        policy = PolicyRules(**policy_dict)
        suggestions = []
        
        for v in violations:
            if "flight" in v.lower() or "FLIGHT" in v:
                if policy.max_flight_price:
                    suggestions.append(f"- Search for flights under ${policy.max_flight_price}")
                if policy.preferred_airlines:
                    suggestions.append(f"- Prefer airlines: {', '.join(policy.preferred_airlines)}")
            
            if "hotel" in v.lower() or "HOTEL" in v:
                if policy.max_hotel_price_per_night:
                    suggestions.append(f"- Search for hotels under ${policy.max_hotel_price_per_night}/night")
                if policy.min_hotel_stars:
                    suggestions.append(f"- Ensure hotel is {policy.min_hotel_stars}+ stars")
            
            if "budget" in v.lower():
                suggestions.append("- Reduce flight cost by choosing economy or off-peak times")
                suggestions.append("- Reduce hotel cost by choosing a 3-star option")
        
        if not suggestions:
            suggestions.append("- Review current selections and choose lower-cost alternatives")
        
        return "Suggested adjustments:\n" + "\n".join(list(set(suggestions)))
    
    def check_compliance(
        self,
        flight_result: FlightSearchResult,
        hotel_result: HotelSearchResult,
        policy_name: str = "standard"
    ) -> PolicyCheckResult:
        """
        Main entry point for compliance checking using ReAct reasoning.
        
        This method triggers the agentic ReAct loop to validate bookings.
        """
        
        # Reset state for new check
        self.reset_state()
        
        # Get flight and hotel details
        flight = flight_result.flights[0] if flight_result.flights else None
        hotel = hotel_result.hotels[0] if hotel_result.hotels else None
        
        if not flight or not hotel:
            return PolicyCheckResult(
                is_compliant=False,
                violations=[PolicyViolation(
                    rule="missing_booking",
                    severity="error",
                    message="Missing flight or hotel selection"
                )],
                reasoning="Cannot check compliance without both flight and hotel selections."
            )
        
        # Build the goal description
        goal = f"""Check if the following trip bookings comply with the '{policy_name}' policy:

FLIGHT:
- ID: {flight.flight_id}
- Airline: {flight.airline}
- Price: ${flight.price_usd}

HOTEL:
- ID: {hotel.hotel_id}
- Name: {hotel.name}
- Price: ${hotel.price_per_night_usd}/night
- Stars: {hotel.stars}
- Distance to center: {hotel.distance_to_business_center_km}km
- Amenities: {', '.join(hotel.amenities)}

Steps to follow:
1. Load the '{policy_name}' policy to understand the rules
2. Check if the flight complies with policy
3. Check if the hotel complies with policy
4. Check if total budget is within limits (assume 1 night)
5. Compile all violations found
6. If violations exist, suggest adjustments

Return your final answer as a JSON object with:
- is_compliant: true/false
- violations: list of violation descriptions
- reasoning: detailed explanation of compliance status"""

        # Run ReAct loop
        result = self.run(goal)
        
        # Collect all violations from beliefs
        violations = []
        
        flight_violations = self.state.get_belief("flight_violations", [])
        for v in flight_violations:
            violations.append(PolicyViolation(
                rule=v["rule"],
                severity=v["severity"],
                message=v["message"],
                actual_value=str(v.get("excess", "")),
                expected_value=""
            ))
        
        hotel_violations = self.state.get_belief("hotel_violations", [])
        for v in hotel_violations:
            violations.append(PolicyViolation(
                rule=v["rule"],
                severity=v["severity"],
                message=v["message"],
                actual_value=str(v.get("excess", "")),
                expected_value=""
            ))
        
        budget_violation = self.state.get_belief("budget_violation")
        if budget_violation:
            violations.append(PolicyViolation(
                rule=budget_violation["rule"],
                severity=budget_violation["severity"],
                message=budget_violation["message"],
                actual_value=str(budget_violation.get("excess", "")),
                expected_value=""
            ))
        
        # Fallback: if ReAct didn't find violations but we should check programmatically
        if not violations and not result["success"]:
            violations = self._fallback_check(flight, hotel, policy_name)
        
        # Determine compliance
        has_errors = any(v.severity == "error" for v in violations)
        is_compliant = len(violations) == 0
        
        # Build reasoning
        reasoning = self._build_react_reasoning(
            flight, hotel, policy_name, result, violations
        )
        
        # Log message
        self.log_message(
            to_agent="orchestrator",
            content=f"Compliance: {'PASS' if is_compliant else f'FAIL ({len(violations)} violations)'}",
            msg_type="result"
        )
        
        return PolicyCheckResult(
            is_compliant=is_compliant,
            violations=violations,
            reasoning=reasoning
        )
    
    def _fallback_check(self, flight, hotel, policy_name: str) -> List[PolicyViolation]:
        """Fallback programmatic check if ReAct fails"""
        
        policy_dict = self.loader.get_policy(policy_name)
        if not policy_dict:
            return []
        
        policy = PolicyRules(**policy_dict)
        violations = []
        
        # Check flight
        if policy.max_flight_price and flight.price_usd > policy.max_flight_price:
            violations.append(PolicyViolation(
                rule="max_flight_price",
                severity="error",
                message=f"FLIGHT price ${flight.price_usd} exceeds max ${policy.max_flight_price}",
                actual_value=str(flight.price_usd),
                expected_value=str(policy.max_flight_price)
            ))
        
        # Check hotel
        if policy.max_hotel_price_per_night and hotel.price_per_night_usd > policy.max_hotel_price_per_night:
            violations.append(PolicyViolation(
                rule="max_hotel_price_per_night",
                severity="error",
                message=f"HOTEL price ${hotel.price_per_night_usd} exceeds max ${policy.max_hotel_price_per_night}",
                actual_value=str(hotel.price_per_night_usd),
                expected_value=str(policy.max_hotel_price_per_night)
            ))
        
        if policy.min_hotel_stars and hotel.stars < policy.min_hotel_stars:
            violations.append(PolicyViolation(
                rule="min_hotel_stars",
                severity="error",
                message=f"Hotel {hotel.stars}* below minimum {policy.min_hotel_stars}*",
                actual_value=str(hotel.stars),
                expected_value=str(policy.min_hotel_stars)
            ))
        
        return violations
    
    def _build_react_reasoning(
        self,
        flight,
        hotel,
        policy_name: str,
        react_result: Dict,
        violations: List[PolicyViolation]
    ) -> str:
        """Build the full ReAct reasoning trace"""
        
        reasoning_parts = [
            f"## Policy Compliance ReAct Reasoning Trace",
            f"**Agent**: {self.agent_name}",
            f"**Policy**: {policy_name}",
            f"**Flight**: {flight.flight_id} ({flight.airline}) - ${flight.price_usd}",
            f"**Hotel**: {hotel.hotel_id} ({hotel.name}) - ${hotel.price_per_night_usd}/night",
            f"**Iterations**: {react_result.get('iterations', 0)}",
            "",
            "### Reasoning Steps:",
        ]
        
        for step in react_result.get("reasoning_trace", []):
            reasoning_parts.append(f"""
**Step {step.step_number}**:
- **Thought**: {step.thought}
- **Action**: `{step.action}({json.dumps(step.action_input)})`
- **Observation**: {step.observation[:200]}{'...' if len(step.observation) > 200 else ''}
""")
        
        # Violations summary
        if violations:
            reasoning_parts.append("\n### Violations Found:")
            for v in violations:
                reasoning_parts.append(f"- [{v.severity.upper()}] {v.message}")
        else:
            reasoning_parts.append("\n### Result: ✓ All checks passed - Trip is compliant")
        
        return "\n".join(reasoning_parts)
