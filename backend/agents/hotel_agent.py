from langchain_community.llms import ollama
from .models import HotelQuery, HotelSearchResult, Hotel
from data.loaders import HotelDataLoader
from typing import List, Dict
import json


class HotelAgent:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.llm = ollama.Ollama(
            model=model_name,
            temperature=0.0,
            format="json"
        )
        self.loader = HotelDataLoader()

    def search_hotels(self, query: HotelQuery) -> HotelSearchResult:
        """Execute search and get LLM recommendation based on actual results"""
        
        # Step 1: Search database first
        hotels = self.loader.search(
            city=query.city,
            max_price_per_night=query.max_price_per_night,
            min_stars=query.min_stars,
            max_distance_to_center_km=query.max_distance_to_center_km,
            required_amenities=query.required_amenities
        )
        
        if not hotels:
            return HotelSearchResult(
                query=query,
                hotels=[],
                reasoning="No hotels found matching the specified criteria."
            )
        
        # Step 2: Get LLM to rank the actual hotels
        candidates = hotels[:10]
        ranking_prompt = self._create_ranking_prompt(query, candidates)
        
        try:
            response = self.llm.invoke(ranking_prompt)
            rankings = json.loads(response)
            
            # Extract recommended hotel IDs
            top_ids = rankings.get("top_3_hotel_ids", [])[:3]
            
            # Map IDs to actual hotel objects
            hotel_dict = {h['hotel_id']: h for h in hotels}
            top_hotels = [
                Hotel(**hotel_dict[hid]) 
                for hid in top_ids 
                if hid in hotel_dict
            ]
            
            # Fallback if LLM didn't provide valid IDs
            if len(top_hotels) < 3:
                top_hotels = [Hotel(**h) for h in candidates[:3]]
            
            reasoning = self._build_reasoning(query, candidates, rankings.get("reasoning", ""), top_hotels)
            
        except (json.JSONDecodeError, KeyError):
            # Fallback: return top 3 by distance/price
            top_hotels = [Hotel(**h) for h in candidates[:3]]
            reasoning = self._build_reasoning(query, candidates, "Analysis failed, sorted by proximity and price", top_hotels)
        
        return HotelSearchResult(
            query=query,
            hotels=top_hotels,
            reasoning=reasoning
        )

    def _create_ranking_prompt(self, query: HotelQuery, hotels: List[Dict]) -> str:
        """Create prompt asking LLM to rank existing hotels"""
        
        hotels_list = [
            {
                "hotel_id": h['hotel_id'],
                "name": h['name'],
                "business_area": h['business_area'],
                "tier": h['tier'],
                "stars": h['stars'],
                "price_per_night_usd": h['price_per_night_usd'],
                "distance_to_business_center_km": round(h['distance_to_business_center_km'], 2),
                "amenities": h['amenities']
            }
            for h in hotels
        ]
        
        # Build constraint description
        constraints = []
        if query.max_price_per_night:
            constraints.append(f"Max price: ${query.max_price_per_night}/night")
        if query.min_stars:
            constraints.append(f"Min {query.min_stars} stars")
        if query.max_distance_to_center_km:
            constraints.append(f"Within {query.max_distance_to_center_km}km of business center")
        if query.required_amenities:
            constraints.append(f"Required amenities: {', '.join(query.required_amenities)}")
        
        constraints_str = "\n".join(constraints) if constraints else "No specific constraints"
        
        return f"""You are a Hotel Booking Specialist. Rank these hotels for a business traveler.

City: {query.city}
Constraints:
{constraints_str}

Available hotels:
{json.dumps(hotels_list, indent=2)}

Select the best 3 hotels considering:
- Proximity to business centers (closer is better)
- Price-to-value ratio
- Star rating and amenities quality
- Business traveler needs (WiFi, Conference Room, etc.)

Return ONLY this JSON format (no other text):
{{
  "top_3_hotel_ids": ["HT0001", "HT0002", "HT0003"],
  "reasoning": "Brief explanation of why these 3 are best"
}}

The hotel_id values MUST be from the list above. Do not invent new IDs."""

    def _build_reasoning(
        self, 
        query: HotelQuery,
        candidates: List[Dict],
        llm_reasoning: str,
        selected: List[Hotel]
    ) -> str:
        """Build the ReAct-style reasoning chain"""
        
        # Build constraint description
        constraints = []
        if query.max_price_per_night:
            constraints.append(f"Max ${query.max_price_per_night}/night")
        if query.min_stars:
            constraints.append(f"Min {query.min_stars} stars")
        if query.max_distance_to_center_km:
            constraints.append(f"Max {query.max_distance_to_center_km}km from center")
        if query.required_amenities:
            constraints.append(f"Required: {', '.join(query.required_amenities)}")
        
        constraints_str = ", ".join(constraints) if constraints else "No specific constraints"
        
        # Format candidate list
        candidates_str = "\n".join([
            f"- {h['hotel_id']} ({h['name']}): ${h['price_per_night_usd']}/night, "
            f"{h['stars']}*, {h['distance_to_business_center_km']:.2f}km from {h['business_area']}"
            for h in candidates
        ])
        
        # Format selected hotels
        selected_str = "\n".join([
            f"{i+1}. {h.hotel_id} - {h.name}: ${h.price_per_night_usd}/night, "
            f"{h.stars}*, {h.distance_to_business_center_km:.2f}km from {h.business_area}"
            for i, h in enumerate(selected)
        ])
        
        return f"""**Thought**: User needs hotels in {query.city}
- Constraints: {constraints_str}

**Action**: Searched hotel database

**Observation**: Found {len(candidates)} matching hotels:
{candidates_str}

**Analysis**: {llm_reasoning}

**Final Answer**: Top 3 recommendations:
{selected_str}"""