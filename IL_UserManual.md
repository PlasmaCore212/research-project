# User Manual
## Agentic Trip Planning System

---

## Overview

The Agentic Trip Planning System is an AI-powered multi-agent application that helps plan business or personal trips by automatically finding optimal flight and hotel combinations based on your requirements.

**Key Features:**
- Multi-agent AI system with ReAct reasoning
- Automatic flight and hotel selection
- Budget optimization
- Meeting time conflict detection
- Distance-based hotel recommendations
- Policy compliance checking

---

## Getting Started

### Accessing the Application

1. Ensure the backend server is running (see Installation Manual)
2. Open your web browser
3. Navigate to: **http://localhost:8000**

You should see the Trip Planner interface with a form.

---

## Using the Trip Planner

### 1. Flight Details

**Origin City** (Required)
- Select your departure city from the dropdown
- Available options: NYC, BOS, CHI, SF

**Destination City** (Required)
- Select your arrival city from the dropdown
- Available options: NYC, BOS, CHI, SF
- Must be different from origin

**Departure Date** (Required)
- Click the date field and select your departure date
- Format: YYYY-MM-DD

**Return Date** (Optional)
- Select your return date if booking a round trip
- Must be after departure date

### 2. Hotel Details

**Check-in Date** (Required)
- Select hotel check-in date
- Typically matches or follows departure date

**Check-out Date** (Required)
- Select hotel check-out date
- Must be after check-in date

**Preferred Hotel Area** (Optional)
- Specify a neighborhood or area (e.g., "Downtown", "Financial District")

### 3. Meeting Details

**Meeting Address** (Required)
- Enter the full address of your meeting location
- Example: "123 Main St, San Francisco, CA 94102"
- Used to calculate hotel proximity

**Meeting Date** (Required)
- Select the date of your meeting

**Meeting Time** (Required)
- Enter meeting time in HH:MM format (24-hour)
- Example: "09:00" for 9:00 AM

### 4. Budget & Preferences

**Budget** (Required)
- Enter total budget in USD
- Range: $1 - $50,000
- The system will try to maximize budget usage for best value

**Trip Type** (Required)
- Select either "Business" or "Personal"
- Affects policy compliance rules

**Additional Preferences** (Optional)
- Free text field for special requests
- Examples: "Window seat preferred", "Pet-friendly hotel"

### 5. Submit Request

Click the **"Plan My Trip"** button to start the AI planning process.

---

## Understanding the Results

### Processing Time

The system uses multiple AI agents to analyze your request. Processing typically takes 30-90 seconds depending on complexity.

You'll see a loading indicator while the agents work.

### Results Display

#### Selected Flight
- **Flight ID**: Unique identifier
- **Airline**: Carrier name
- **Departure Time**: Local departure time
- **Arrival Time**: Local arrival time
- **Class**: Economy, Business, or First Class
- **Price**: Flight cost in USD

#### Selected Hotel
- **Hotel ID**: Unique identifier
- **Name**: Hotel name
- **Stars**: Star rating (1-5)
- **Price per Night**: Nightly rate in USD
- **Amenities**: Available facilities (WiFi, Parking, Gym, etc.)
- **Distance to Meeting**: Driving distance from hotel to meeting location
- **Travel Time**: Estimated driving time to meeting

#### Total Cost
Sum of flight price + hotel cost (price per night × number of nights)

#### Policy Check
- **Compliant**: Whether booking meets budget and policy requirements
- **Issues**: Any policy violations detected
- **Recommendations**: Suggestions for compliance

#### Time Feasibility
- **Feasible**: Whether timeline allows for all activities
- **Conflicts**: Any scheduling conflicts detected
- **Arrival Buffer**: Time between flight arrival and meeting

#### Cheaper Alternatives
If budget allows, the system may suggest more economical options that still meet your requirements.

---

## Features Explained

### Multi-Agent System

The application uses five specialized AI agents:

1. **Flight Agent**
   - Searches available flights
   - Analyzes options by price, time, and class
   - Selects optimal flights

2. **Hotel Agent**
   - Searches available hotels
   - Considers proximity to meeting location
   - Analyzes amenities and star ratings

3. **Policy Agent**
   - Checks budget compliance
   - Validates trip type requirements
   - Suggests cost optimizations

4. **Time Agent**
   - Analyzes timeline feasibility
   - Detects scheduling conflicts
   - Ensures adequate travel buffers

5. **Orchestrator Agent**
   - Coordinates all other agents
   - Makes final decisions
   - Handles agent negotiations

### Budget Optimization

The system aims to maximize your budget usage by:
- Finding the best value options (not just cheapest)
- Balancing flight and hotel quality
- Suggesting premium options when budget allows
- Providing cheaper alternatives for flexibility

### Meeting Proximity

Hotels are ranked by actual driving distance to your meeting location using real routing data from OpenStreetMap, not straight-line distance.

### ReAct Reasoning

Each agent uses ReAct (Reasoning + Acting) pattern:
- **Thought**: Agent analyzes the situation
- **Action**: Agent takes a specific action (search, compare, select)
- **Observation**: Agent evaluates results
- Repeats until optimal solution found

---

## Tips for Best Results

### 1. Be Specific with Meeting Address
- Use full street addresses, not just city names
- Include ZIP/postal codes when possible
- This ensures accurate distance calculations

### 2. Set Realistic Budgets
- Budget should cover both flight and hotel costs
- Minimum recommended: $300-500 for short trips
- System will notify if budget is insufficient

### 3. Allow Adequate Time
- Book departing flights before meeting time
- Allow at least 2-3 hours between landing and meeting
- Consider time zones

### 4. Use Preferences Field
- Specify important requirements
- Examples: "Accessible room needed", "Early morning flights preferred"

### 5. Check Policy Results
- Review policy check section for budget issues
- Consider cheaper alternatives if suggested
- Adjust budget if needed

---

## Common Scenarios

### Business Trip with Tight Schedule
```
Origin: NYC
Destination: SF
Departure: 2026-02-15
Return: 2026-02-16
Meeting: 123 Market St, San Francisco, CA 94105
Meeting Date: 2026-02-15
Meeting Time: 14:00
Budget: $1500
Trip Type: Business
```

### Personal Trip with Flexibility
```
Origin: BOS
Destination: CHI
Departure: 2026-03-10
Return: 2026-03-13
Hotel Check-in: 2026-03-10
Hotel Check-out: 2026-03-13
Budget: $800
Trip Type: Personal
Preferences: "Prefer morning flights, hotel with free breakfast"
```

---

## Limitations

### Available Cities
The proof-of-concept currently supports only:
- NYC (New York City)
- BOS (Boston)
- CHI (Chicago)
- SF (San Francisco)

### Data Currency
Flight and hotel data is generated for demonstration purposes. For production use, integrate with real booking APIs.

### Processing Time
AI reasoning takes time. Complex requests may take up to 2 minutes.

---

## Troubleshooting

### "Origin and destination cannot be the same"
- Select different cities for origin and destination

### "City must be one of: NYC, BOS, CHI, SF"
- Use only supported cities in current version

### "Budget too low" or No results
- Increase budget amount
- Check that dates are valid and in correct order

### Processing takes very long
- Ensure Ollama service is running
- Check system resources (CPU, RAM)
- Try simplifying requirements

### Results seem incorrect
- Verify all input fields are correctly filled
- Check meeting address is accurate
- Review budget amount

---

## API Access (Advanced)

For developers, the system provides a REST API:

**API Documentation:**
http://localhost:8000/docs

**Plan Trip Endpoint:**
POST http://localhost:8000/api/plan-trip

**Get Available Cities:**
GET http://localhost:8000/api/cities

**Health Check:**
GET http://localhost:8000/health

---

## Support & Feedback

For issues or questions:
1. Check troubleshooting section above
2. Review Installation Manual for setup issues
3. Check API documentation at /docs endpoint
4. Review backend console logs for error messages

---

## Privacy & Data

- All processing happens locally on your machine
- No data is sent to external AI services
- Meeting addresses are geocoded via public OpenStreetMap API only
- No personal data is stored or transmitted to third parties

---

**Version:** 1.0.0
**Last Updated:** 2026-01-24
