// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const NOMINATIM_URL = 'https://nominatim.openstreetmap.org';

// Available cities with their airports
const CITY_DATA = {
    'NYC': { name: 'New York City', airports: ['JFK', 'LGA', 'EWR'] },
    'BOS': { name: 'Boston', airports: ['BOS'] },
    'CHI': { name: 'Chicago', airports: ['ORD', 'MDW'] },
    'SF': { name: 'San Francisco', airports: ['SFO', 'OAK'] }
};

// Agent definitions for status display
const AGENTS = [
    { id: 'flight', name: 'Flight Agent', icon: '✈️' },
    { id: 'hotel', name: 'Hotel Agent', icon: '🏨' },
    { id: 'policy', name: 'Policy Agent', icon: '📋' },
    { id: 'time', name: 'Time Agent', icon: '⏰' },
    { id: 'orchestrator', name: 'Orchestrator', icon: '🎯' }
];

// Store geocoded location
let meetingLocation = null;

// Initialize the form
document.addEventListener('DOMContentLoaded', () => {
    // Set minimum date to today
    const today = new Date().toISOString().split('T')[0];
    document.getElementById('departure_date').min = today;
    document.getElementById('return_date').min = today;
    document.getElementById('hotel_checkin').min = today;
    document.getElementById('hotel_checkout').min = today;
    document.getElementById('meeting_date').min = today;

    // Sync dates automatically
    document.getElementById('departure_date').addEventListener('change', syncDates);
    document.getElementById('hotel_checkout').addEventListener('change', syncReturnDate);

    // Address geocoding on blur
    document.getElementById('meeting_address').addEventListener('blur', geocodeAddress);

    // Form submission
    document.getElementById('tripForm').addEventListener('submit', handleSubmit);
});

// Geocode address using OpenStreetMap Nominatim
async function geocodeAddress() {
    const addressInput = document.getElementById('meeting_address');
    const preview = document.getElementById('addressPreview');
    const address = addressInput.value.trim();

    if (!address) {
        preview.style.display = 'none';
        meetingLocation = null;
        return;
    }

    // Show loading state
    preview.style.display = 'flex';
    preview.className = 'address-preview loading';
    preview.querySelector('.preview-text').textContent = 'Looking up address...';

    try {
        const response = await fetch(
            `${NOMINATIM_URL}/search?format=json&q=${encodeURIComponent(address)}&limit=1`,
            {
                headers: {
                    'User-Agent': 'TripPlannerResearchProject/1.0'
                }
            }
        );

        const results = await response.json();

        if (results.length > 0) {
            const result = results[0];
            meetingLocation = {
                lat: parseFloat(result.lat),
                lon: parseFloat(result.lon),
                display_name: result.display_name
            };

            preview.className = 'address-preview';
            preview.querySelector('.preview-text').textContent = 
                `✓ Found: ${result.display_name.substring(0, 80)}...`;
        } else {
            meetingLocation = null;
            preview.className = 'address-preview error';
            preview.querySelector('.preview-text').textContent = 
                '⚠ Address not found. Please check and try again.';
        }
    } catch (error) {
        console.error('Geocoding error:', error);
        meetingLocation = null;
        preview.className = 'address-preview error';
        preview.querySelector('.preview-text').textContent = 
            '⚠ Could not look up address. Check your connection.';
    }
}

// Sync hotel check-in with departure date
function syncDates(e) {
    const departureDate = e.target.value;
    const checkinInput = document.getElementById('hotel_checkin');
    
    if (!checkinInput.value || checkinInput.value < departureDate) {
        checkinInput.value = departureDate;
    }
    
    // Update minimum for other dates
    document.getElementById('return_date').min = departureDate;
    document.getElementById('hotel_checkin').min = departureDate;
}

// Sync return date with checkout
function syncReturnDate(e) {
    const checkoutDate = e.target.value;
    const returnInput = document.getElementById('return_date');
    
    if (!returnInput.value) {
        returnInput.value = checkoutDate;
    }
}

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();

    // Validate origin != destination
    const origin = document.getElementById('origin').value;
    const destination = document.getElementById('destination').value;
    
    if (origin === destination) {
        showError('Origin and destination cannot be the same!');
        return;
    }

    // Validate dates
    const departureDate = document.getElementById('departure_date').value;
    const hotelCheckin = document.getElementById('hotel_checkin').value;
    const hotelCheckout = document.getElementById('hotel_checkout').value;

    if (hotelCheckout <= hotelCheckin) {
        showError('Hotel check-out must be after check-in!');
        return;
    }

    // Validate meeting address was geocoded
    if (!meetingLocation) {
        showError('Please enter a valid meeting address and wait for it to be verified.');
        return;
    }

    // Collect form data
    const formData = {
        origin: origin,
        destination: destination,
        departure_date: departureDate,
        return_date: document.getElementById('return_date').value || null,
        hotel_checkin: hotelCheckin,
        hotel_checkout: hotelCheckout,
        hotel_location: document.getElementById('hotel_location').value || null,
        meeting_time: document.getElementById('meeting_time').value,
        meeting_date: document.getElementById('meeting_date').value,
        meeting_address: document.getElementById('meeting_address').value,
        meeting_coordinates: meetingLocation,
        budget: parseFloat(document.getElementById('budget').value),
        trip_type: document.getElementById('trip_type').value,
        preferences: document.getElementById('preferences').value || null
    };

    // Show loading state
    setLoadingState(true);
    hideError();

    try {
        // Initialize agent status display
        showResults();
        initAgentStatus();

        // Call the API
        const response = await fetch(`${API_BASE_URL}/api/plan-trip`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to plan trip');
        }

        const result = await response.json();
        
        // Display results
        displayResults(result);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while planning your trip. Make sure the backend is running.');
        hideResults();
    } finally {
        setLoadingState(false);
    }
}

// Set loading state on button
function setLoadingState(loading) {
    const btn = document.getElementById('submitBtn');
    const btnText = btn.querySelector('.btn-text');
    const btnLoading = btn.querySelector('.btn-loading');

    btn.disabled = loading;
    btnText.style.display = loading ? 'none' : 'inline';
    btnLoading.style.display = loading ? 'inline-flex' : 'none';
}

// Show results section
function showResults() {
    document.getElementById('results').style.display = 'block';
    document.getElementById('tripForm').style.display = 'none';
}

// Hide results section
function hideResults() {
    document.getElementById('results').style.display = 'none';
    document.getElementById('tripForm').style.display = 'block';
}

// Initialize agent status display
function initAgentStatus() {
    const statusList = document.getElementById('statusList');
    statusList.innerHTML = AGENTS.map(agent => `
        <div class="status-item" id="status-${agent.id}">
            <span class="status-icon">${agent.icon}</span>
            <span class="status-name">${agent.name}</span>
            <span class="status-state">Waiting...</span>
        </div>
    `).join('');
}

// Update agent status
function updateAgentStatus(agentId, state) {
    const statusItem = document.getElementById(`status-${agentId}`);
    if (statusItem) {
        statusItem.className = `status-item ${state}`;
        const stateText = statusItem.querySelector('.status-state');
        stateText.textContent = state === 'active' ? 'Working...' : 
                                state === 'complete' ? 'Complete ✓' : 'Waiting...';
    }
}

// Display results from API
function displayResults(result) {
    const resultCards = document.getElementById('resultCards');
    
    // Mark all agents as complete
    AGENTS.forEach(agent => updateAgentStatus(agent.id, 'complete'));

    // Build result cards
    let cardsHTML = '';

    // Flight results
    if (result.selected_flight) {
        const flight = result.selected_flight;
        cardsHTML += `
            <div class="result-card flight-card">
                <h4>✈️ Selected Flight</h4>
                <div class="detail-row">
                    <span class="detail-label">Route</span>
                    <span class="detail-value">${flight.origin || 'N/A'} → ${flight.destination || 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Airline</span>
                    <span class="detail-value">${flight.airline || 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Departure</span>
                    <span class="detail-value">${flight.departure_time || 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Arrival</span>
                    <span class="detail-value">${flight.arrival_time || 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Price</span>
                    <span class="detail-value">$${flight.price || 'N/A'}</span>
                </div>
            </div>
        `;
    }

    // Hotel results
    if (result.selected_hotel) {
        const hotel = result.selected_hotel;
        cardsHTML += `
            <div class="result-card hotel-card">
                <h4>🏨 Selected Hotel</h4>
                <div class="detail-row">
                    <span class="detail-label">Name</span>
                    <span class="detail-value">${hotel.name || 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Location</span>
                    <span class="detail-value">${hotel.location || hotel.city || 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Rating</span>
                    <span class="detail-value">${hotel.rating ? '⭐'.repeat(Math.round(hotel.rating)) : 'N/A'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Price/Night</span>
                    <span class="detail-value">$${hotel.price_per_night || hotel.price || 'N/A'}</span>
                </div>
                ${hotel.amenities ? `
                <div class="detail-row">
                    <span class="detail-label">Amenities</span>
                    <span class="detail-value">${Array.isArray(hotel.amenities) ? hotel.amenities.join(', ') : hotel.amenities}</span>
                </div>
                ` : ''}
            </div>
        `;
    }

    // Policy compliance
    if (result.policy_check) {
        const policy = result.policy_check;
        const isCompliant = policy.compliant || policy.is_compliant;
        cardsHTML += `
            <div class="result-card policy-card">
                <h4>📋 Policy Compliance</h4>
                <div class="detail-row">
                    <span class="detail-label">Status</span>
                    <span class="detail-value">
                        <span class="compliance-badge ${isCompliant ? 'compliant' : 'non-compliant'}">
                            ${isCompliant ? 'Compliant' : 'Review Needed'}
                        </span>
                    </span>
                </div>
                ${policy.issues && policy.issues.length > 0 ? `
                <div class="detail-row">
                    <span class="detail-label">Issues</span>
                    <span class="detail-value">${policy.issues.join(', ')}</span>
                </div>
                ` : ''}
                ${policy.suggestions && policy.suggestions.length > 0 ? `
                <div class="detail-row">
                    <span class="detail-label">Suggestions</span>
                    <span class="detail-value">${policy.suggestions.join(', ')}</span>
                </div>
                ` : ''}
            </div>
        `;
    }

    // Timeline feasibility
    if (result.time_check) {
        const time = result.time_check;
        const isFeasible = time.feasible || time.is_feasible;
        cardsHTML += `
            <div class="result-card time-card">
                <h4>⏰ Timeline Analysis</h4>
                <div class="detail-row">
                    <span class="detail-label">Feasibility</span>
                    <span class="detail-value">
                        <span class="compliance-badge ${isFeasible ? 'compliant' : 'non-compliant'}">
                            ${isFeasible ? 'Feasible' : 'Needs Adjustment'}
                        </span>
                    </span>
                </div>
                ${time.buffer_time ? `
                <div class="detail-row">
                    <span class="detail-label">Buffer Time</span>
                    <span class="detail-value">${time.buffer_time}</span>
                </div>
                ` : ''}
                ${time.warnings && time.warnings.length > 0 ? `
                <div class="detail-row">
                    <span class="detail-label">Warnings</span>
                    <span class="detail-value">${time.warnings.join(', ')}</span>
                </div>
                ` : ''}
            </div>
        `;
    }

    // Total cost summary
    if (result.total_cost || (result.selected_flight && result.selected_hotel)) {
        const flightCost = result.selected_flight?.price || 0;
        const hotelCost = result.selected_hotel?.price_per_night || result.selected_hotel?.price || 0;
        const totalCost = result.total_cost || (flightCost + hotelCost);
        
        cardsHTML += `
            <div class="result-card" style="border-left-color: #9b59b6;">
                <h4>💰 Cost Summary</h4>
                <div class="detail-row">
                    <span class="detail-label">Flight</span>
                    <span class="detail-value">$${flightCost}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Hotel</span>
                    <span class="detail-value">$${hotelCost}</span>
                </div>
                <div class="detail-row" style="border-top: 2px solid rgba(255,255,255,0.1); padding-top: 10px; margin-top: 5px;">
                    <span class="detail-label" style="font-weight: bold;">Total</span>
                    <span class="detail-value" style="font-size: 1.2rem; color: #00d9ff;">$${totalCost}</span>
                </div>
            </div>
        `;
    }

    resultCards.innerHTML = cardsHTML;

    // Show reasoning trace if available
    if (result.reasoning_trace || result.agent_traces) {
        displayReasoningTrace(result.reasoning_trace || result.agent_traces);
    }
}

// Display reasoning trace
function displayReasoningTrace(traces) {
    const reasoningSection = document.getElementById('reasoningSection');
    const reasoningContent = document.getElementById('reasoningContent');
    
    if (!traces || (Array.isArray(traces) && traces.length === 0)) {
        reasoningSection.style.display = 'none';
        return;
    }

    reasoningSection.style.display = 'block';

    let traceHTML = '';
    
    if (Array.isArray(traces)) {
        traces.forEach((trace, index) => {
            traceHTML += `
                <div class="reasoning-step">
                    <strong>Step ${index + 1}:</strong><br>
                    ${trace.thought ? `<span class="thought">💭 Thought: ${trace.thought}</span><br>` : ''}
                    ${trace.action ? `<span class="action">⚡ Action: ${trace.action}</span><br>` : ''}
                    ${trace.observation ? `<span class="observation">👁️ Observation: ${trace.observation}</span>` : ''}
                </div>
            `;
        });
    } else if (typeof traces === 'object') {
        for (const [agent, agentTraces] of Object.entries(traces)) {
            traceHTML += `<h4 style="color: #00d9ff; margin: 15px 0 10px;">${agent}</h4>`;
            if (Array.isArray(agentTraces)) {
                agentTraces.forEach((trace, index) => {
                    traceHTML += `
                        <div class="reasoning-step">
                            <strong>Step ${index + 1}:</strong><br>
                            ${trace.thought ? `<span class="thought">💭 ${trace.thought}</span><br>` : ''}
                            ${trace.action ? `<span class="action">⚡ ${trace.action}</span><br>` : ''}
                            ${trace.observation ? `<span class="observation">👁️ ${trace.observation}</span>` : ''}
                        </div>
                    `;
                });
            }
        }
    }

    reasoningContent.innerHTML = traceHTML || '<p>No reasoning trace available.</p>';
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

// Hide error message
function hideError() {
    document.getElementById('errorMessage').style.display = 'none';
}

// Reset form
function resetForm() {
    document.getElementById('tripForm').reset();
    document.getElementById('tripForm').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    hideError();
}
