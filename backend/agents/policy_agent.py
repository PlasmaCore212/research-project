from langchain_community.llms import ollama
from .models import PolicyRules, PolicyCheckResult, PolicyViolation, FlightSearchResult, HotelSearchResult
from data.loaders import PolicyDataLoader
from typing import List
import json


class PolicyAgent:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.llm = ollama.Ollama(
            model=model_name,
            temperature=0.0,
            format="json"
        )
        self.loader = PolicyDataLoader()

    def check_compliance(
        self, 
        flight_result: FlightSearchResult,
        hotel_result: HotelSearchResult,
        policy_name: str = "standard"
    ) -> PolicyCheckResult:
        """Check if trip complies with company policy"""
        
        # Load policy rules
        policy_dict = self.loader.get_policy(policy_name)
        policy = PolicyRules(**policy_dict)
        
        # Step 1: Programmatic validation
        violations = self._check_violations(flight_result, hotel_result, policy)
        
        if not violations:
            return PolicyCheckResult(
                is_compliant=True,
                violations=[],
                reasoning="All bookings comply with company policy."
            )
        
        # Step 2: Get LLM analysis of violations
        analysis_prompt = self._create_analysis_prompt(
            flight_result, hotel_result, policy, violations
        )
        
        try:
            response = self.llm.invoke(analysis_prompt)
            analysis = json.loads(response)
            
            reasoning = self._build_reasoning(
                flight_result, hotel_result, policy, violations, 
                analysis.get("summary", "")
            )
            
        except (json.JSONDecodeError, KeyError):
            reasoning = self._build_reasoning(
                flight_result, hotel_result, policy, violations, 
                "Analysis complete"
            )
        
        return PolicyCheckResult(
            is_compliant=False,
            violations=violations,
            reasoning=reasoning
        )

    def _check_violations(
        self, 
        flight_result: FlightSearchResult,
        hotel_result: HotelSearchResult,
        policy: PolicyRules
    ) -> List[PolicyViolation]:
        """Check for policy violations programmatically"""
        violations = []
        
        # Check flight violations
        if flight_result.flights:
            flight = flight_result.flights[0]  # Check top recommendation
            
            if policy.max_flight_price and flight.price_usd > policy.max_flight_price:
                violations.append(PolicyViolation(
                    rule="max_flight_price",
                    severity="error",
                    message=f"FLIGHT price exceeds maximum allowed",
                    actual_value=f"${flight.price_usd}",
                    expected_value=f"${policy.max_flight_price}"
                ))
            
            if policy.preferred_airlines and flight.airline not in policy.preferred_airlines:
                violations.append(PolicyViolation(
                    rule="preferred_airlines",
                    severity="warning",
                    message=f"Airline not in preferred list",
                    actual_value=flight.airline,
                    expected_value=", ".join(policy.preferred_airlines)
                ))
        
        # Check hotel violations
        if hotel_result.hotels:
            hotel = hotel_result.hotels[0]  # Check top recommendation
            
            if policy.max_hotel_price_per_night and hotel.price_per_night_usd > policy.max_hotel_price_per_night:
                violations.append(PolicyViolation(
                    rule="max_hotel_price_per_night",
                    severity="error",
                    message=f"HOTEL price exceeds maximum allowed per night",  # Added HOTEL
                    actual_value=f"${hotel.price_per_night_usd}",
                    expected_value=f"${policy.max_hotel_price_per_night}"
                ))
            
            if policy.min_hotel_stars and hotel.stars < policy.min_hotel_stars:
                violations.append(PolicyViolation(
                    rule="min_hotel_stars",
                    severity="error",
                    message=f"Hotel rating below minimum requirement",
                    actual_value=f"{hotel.stars} stars",
                    expected_value=f"{policy.min_hotel_stars}+ stars"
                ))
            
            if policy.max_distance_to_center_km and hotel.distance_to_business_center_km > policy.max_distance_to_center_km:
                violations.append(PolicyViolation(
                    rule="max_distance_to_center_km",
                    severity="warning",
                    message=f"Hotel too far from business center",
                    actual_value=f"{hotel.distance_to_business_center_km:.2f}km",
                    expected_value=f"<{policy.max_distance_to_center_km}km"
                ))
            
            if policy.required_hotel_amenities:
                missing = set(policy.required_hotel_amenities) - set(hotel.amenities)
                if missing:
                    violations.append(PolicyViolation(
                        rule="required_hotel_amenities",
                        severity="error",
                        message=f"Hotel missing required amenities",
                        actual_value=f"Missing: {', '.join(missing)}",
                        expected_value=", ".join(policy.required_hotel_amenities)
                    ))
        
        # Check total budget
        if policy.max_total_budget and flight_result.flights and hotel_result.hotels:
            flight_cost = flight_result.flights[0].price_usd
            hotel_cost = hotel_result.hotels[0].price_per_night_usd
            total = flight_cost + hotel_cost
            
            if total > policy.max_total_budget:
                violations.append(PolicyViolation(
                    rule="max_total_budget",
                    severity="error",
                    message=f"Total trip cost exceeds budget",
                    actual_value=f"${total}",
                    expected_value=f"${policy.max_total_budget}"
                ))
        
        return violations

    def _create_analysis_prompt(
        self,
        flight_result: FlightSearchResult,
        hotel_result: HotelSearchResult,
        policy: PolicyRules,
        violations: List[PolicyViolation]
    ) -> str:
        """Create prompt for LLM to analyze violations"""
        
        violations_list = [
            {
                "rule": v.rule,
                "severity": v.severity,
                "message": v.message,
                "actual": v.actual_value,
                "expected": v.expected_value
            }
            for v in violations
        ]
        
        return f"""You are a Corporate Travel Policy Compliance Officer. Analyze these policy violations.

Policy Violations Found:
{json.dumps(violations_list, indent=2)}

Provide a brief summary of the compliance issues and their business impact.

Return ONLY this JSON format (no other text):
{{
  "summary": "Brief explanation of violations and recommendations"
}}"""

    def _build_reasoning(
        self,
        flight_result: FlightSearchResult,
        hotel_result: HotelSearchResult,
        policy: PolicyRules,
        violations: List[PolicyViolation],
        llm_summary: str
    ) -> str:
        """Build ReAct-style reasoning chain"""
        
        # Format violations
        violations_str = "\n".join([
            f"- [{v.severity.upper()}] {v.message}: {v.actual_value} (expected: {v.expected_value})"
            for v in violations
        ])
        
        # Get flight and hotel details
        flight = flight_result.flights[0] if flight_result.flights else None
        hotel = hotel_result.hotels[0] if hotel_result.hotels else None
        
        flight_str = f"{flight.flight_id} - {flight.airline}: ${flight.price_usd}" if flight else "None"
        hotel_str = f"{hotel.hotel_id} - {hotel.name}: ${hotel.price_per_night_usd}/night, {hotel.stars}*" if hotel else "None"
        
        return f"""**Thought**: Checking trip compliance against {policy.__class__.__name__} policy
- Flight: {flight_str}
- Hotel: {hotel_str}

**Action**: Validated bookings against company policy rules

**Observation**: Found {len(violations)} violation(s):
{violations_str}

**Analysis**: {llm_summary}

**Final Answer**: Trip is NOT compliant. Booking agents must revise selections."""