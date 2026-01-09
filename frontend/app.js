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
    { id: 'flight', name: 'Flight Agent', icon: '✈️', description: 'Finding best flights' },
    { id: 'hotel', name: 'Hotel Agent', icon: '🏨', description: 'Searching hotels' },
    { id: 'policy', name: 'Policy Agent', icon: '📋', description: 'Checking budget & compliance' },
    { id: 'time', name: 'Time Agent', icon: '⏰', description: 'Validating timeline' },
    { id: 'orchestrator', name: 'Orchestrator', icon: '🎯', description: 'Coordinating agents' }
];

// Store geocoded location and form data
let meetingLocation = null;
let currentRequest = null;

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

            preview.className = 'address-preview success';
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
    const meetingInput = document.getElementById('meeting_date');
    
    if (!checkinInput.value || checkinInput.value < departureDate) {
        checkinInput.value = departureDate;
    }
    
    // Set meeting date to day after departure by default
    if (!meetingInput.value) {
        const nextDay = new Date(departureDate);
        nextDay.setDate(nextDay.getDate() + 1);
        meetingInput.value = nextDay.toISOString().split('T')[0];
    }
    
    // Update minimum for other dates
    document.getElementById('return_date').min = departureDate;
    document.getElementById('hotel_checkin').min = departureDate;
    document.getElementById('meeting_date').min = departureDate;
}

// Sync return date with checkout
function syncReturnDate(e) {
    const checkoutDate = e.target.value;
    const returnInput = document.getElementById('return_date');
    
    if (!returnInput.value) {
        returnInput.value = checkoutDate;
    }
}

// Calculate nights between two dates
function calculateNights(checkin, checkout) {
    const checkinDate = new Date(checkin);
    const checkoutDate = new Date(checkout);
    const diffTime = Math.abs(checkoutDate - checkinDate);
    return Math.ceil(diffTime / (1000 * 60 * 60 * 24));
}

// Format date for display
function formatDate(dateStr) {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
}

// Format currency
function formatCurrency(amount) {
    if (amount === null || amount === undefined || isNaN(amount)) return 'N/A';
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount);
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

    // Store for later use in results
    currentRequest = formData;
    currentRequest.nights = calculateNights(hotelCheckin, hotelCheckout);

    // Show loading state
    setLoadingState(true);
    hideError();

    try {
        // Initialize agent status display
        showResults();
        initAgentStatus();

        // Animate through agent statuses
        simulateAgentProgress();

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
        
        // Mark all agents complete
        AGENTS.forEach(agent => updateAgentStatus(agent.id, 'complete'));
        
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

// Simulate agent progress for better UX
function simulateAgentProgress() {
    const delays = [100, 500, 1500, 2500, 3500];
    AGENTS.forEach((agent, index) => {
        setTimeout(() => {
            updateAgentStatus(agent.id, 'active');
        }, delays[index] || 100);
    });
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
            <div class="status-info">
                <span class="status-name">${agent.name}</span>
                <span class="status-desc">${agent.description}</span>
            </div>
            <span class="status-state">
                <span class="status-dot"></span>
                <span class="status-text">Waiting</span>
            </span>
        </div>
    `).join('');
}

// Update agent status
function updateAgentStatus(agentId, state) {
    const statusItem = document.getElementById(`status-${agentId}`);
    if (statusItem) {
        statusItem.className = `status-item ${state}`;
        const stateText = statusItem.querySelector('.status-text');
        stateText.textContent = state === 'active' ? 'Working...' : 
                                state === 'complete' ? 'Done' : 'Waiting';
    }
}

// Display results from API
function displayResults(result) {
    const resultCards = document.getElementById('resultCards');
    
    // Calculate values
    const flight = result.selected_flight || {};
    const hotel = result.selected_hotel || {};
    const policy = result.policy_check || {};
    const nights = currentRequest?.nights || 2;
    
    const flightPrice = flight.price_usd || flight.price || 0;
    const hotelPricePerNight = hotel.price_per_night_usd || hotel.price_per_night || 0;
    const hotelTotal = hotelPricePerNight * nights;
    const totalCost = result.total_cost || (flightPrice + hotelTotal);
    const budget = currentRequest?.budget || policy.budget || 0;
    const remaining = budget - totalCost;

    let cardsHTML = '';

    // ============ TRIP SUMMARY HERO ============
    cardsHTML += `
        <div class="trip-summary-hero">
            <div class="trip-route">
                <span class="city-code">${currentRequest?.origin || 'N/A'}</span>
                <span class="route-arrow">✈️ →</span>
                <span class="city-code">${currentRequest?.destination || 'N/A'}</span>
            </div>
            <div class="trip-dates">
                ${formatDate(currentRequest?.departure_date)} - ${formatDate(currentRequest?.hotel_checkout)}
                <span class="nights-badge">${nights} night${nights > 1 ? 's' : ''}</span>
            </div>
            <div class="trip-total">
                <span class="total-label">Total Cost</span>
                <span class="total-amount">${formatCurrency(totalCost)}</span>
                <span class="budget-status ${remaining >= 0 ? 'under' : 'over'}">
                    ${remaining >= 0 ? `${formatCurrency(remaining)} under budget` : `${formatCurrency(Math.abs(remaining))} over budget`}
                </span>
            </div>
        </div>
    `;

    // ============ FLIGHT CARD ============
    if (result.selected_flight) {
        cardsHTML += `
            <div class="result-card flight-card">
                <div class="card-header">
                    <h4>✈️ Flight</h4>
                    <span class="card-price">${formatCurrency(flightPrice)}</span>
                </div>
                <div class="card-body">
                    <div class="flight-info">
                        <div class="flight-airline">
                            <span class="airline-name">${flight.airline || 'Unknown Airline'}</span>
                            <span class="flight-id">${flight.flight_id || ''}</span>
                        </div>
                        <div class="flight-times">
                            <div class="time-block departure">
                                <span class="time">${flight.departure_time || 'N/A'}</span>
                                <span class="city">${flight.from_city || currentRequest?.origin || ''}</span>
                            </div>
                            <div class="flight-duration">
                                <span class="duration-line"></span>
                                <span class="duration-text">${flight.duration_hours ? flight.duration_hours.toFixed(1) + 'h' : 'Direct'}</span>
                            </div>
                            <div class="time-block arrival">
                                <span class="time">${flight.arrival_time || 'N/A'}</span>
                                <span class="city">${flight.to_city || currentRequest?.destination || ''}</span>
                            </div>
                        </div>
                    </div>
                    <div class="flight-details">
                        <span class="detail-chip">🪑 ${flight.class || flight.flight_class || 'Economy'}</span>
                        ${flight.seats_available ? `<span class="detail-chip">🎫 ${flight.seats_available} seats left</span>` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    // ============ HOTEL CARD ============
    if (result.selected_hotel) {
        const stars = hotel.stars || hotel.rating || 0;
        cardsHTML += `
            <div class="result-card hotel-card">
                <div class="card-header">
                    <h4>🏨 Hotel</h4>
                    <span class="card-price">${formatCurrency(hotelPricePerNight)}<small>/night</small></span>
                </div>
                <div class="card-body">
                    <div class="hotel-info">
                        <div class="hotel-name-row">
                            <span class="hotel-name">${hotel.name || 'Unknown Hotel'}</span>
                            <span class="hotel-stars">${'★'.repeat(stars)}${'☆'.repeat(5-stars)}</span>
                        </div>
                        <div class="hotel-location">
                            📍 ${hotel.business_area || hotel.city_name || hotel.city || currentRequest?.destination || 'N/A'}
                            ${hotel.distance_to_business_center_km ? ` • ${hotel.distance_to_business_center_km.toFixed(1)}km to center` : ''}
                        </div>
                    </div>
                    <div class="hotel-stay">
                        <div class="stay-dates">
                            <span>Check-in: ${formatDate(currentRequest?.hotel_checkin)}</span>
                            <span>Check-out: ${formatDate(currentRequest?.hotel_checkout)}</span>
                        </div>
                        <div class="stay-total">
                            ${nights} night${nights > 1 ? 's' : ''} × ${formatCurrency(hotelPricePerNight)} = <strong>${formatCurrency(hotelTotal)}</strong>
                        </div>
                    </div>
                    ${hotel.amenities && hotel.amenities.length > 0 ? `
                        <div class="hotel-amenities">
                            ${hotel.amenities.slice(0, 5).map(a => `<span class="amenity-chip">${a}</span>`).join('')}
                            ${hotel.amenities.length > 5 ? `<span class="amenity-chip more">+${hotel.amenities.length - 5} more</span>` : ''}
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    // ============ CHEAPER ALTERNATIVES ============
    if (result.cheaper_alternatives && result.cheaper_alternatives.length > 0) {
        cardsHTML += `
            <div class="result-card alternatives-card">
                <div class="card-header">
                    <h4>💰 Budget-Friendly Alternatives</h4>
                </div>
                <div class="card-body">
                    <p class="alternatives-intro">Want to save money? Here are some cheaper options:</p>
                    <div class="alternatives-list">
                        ${result.cheaper_alternatives.map((alt, i) => `
                            <div class="alternative-item">
                                <div class="alt-rank">${i + 1}</div>
                                <div class="alt-details">
                                    <div class="alt-combo">
                                        <span class="alt-flight">${alt.flight?.airline || 'Flight'}</span>
                                        <span class="alt-plus">+</span>
                                        <span class="alt-hotel">${alt.hotel?.name || 'Hotel'} (${alt.hotel?.stars || '?'}★)</span>
                                    </div>
                                    <div class="alt-meta">
                                        Quality Score: ${alt.quality_score || 'N/A'}
                                    </div>
                                </div>
                                <div class="alt-price">
                                    <span class="alt-total">${formatCurrency(alt.total_cost)}</span>
                                    <span class="alt-savings">Save ${formatCurrency(alt.savings_vs_selected)}</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    }

    // ============ STATUS CARDS ROW ============
    cardsHTML += '<div class="status-cards-row">';
    
    // Policy Status
    const isCompliant = policy.is_compliant || policy.overall_status === 'compliant';
    cardsHTML += `
        <div class="status-card ${isCompliant ? 'success' : 'warning'}">
            <span class="status-card-icon">📋</span>
            <span class="status-card-label">Budget</span>
            <span class="status-card-value">${isCompliant ? 'Compliant' : 'Review Needed'}</span>
        </div>
    `;

    // Timeline Status
    const timeCheck = result.time_check || {};
    const isFeasible = timeCheck.feasible !== false && timeCheck.is_feasible !== false;
    cardsHTML += `
        <div class="status-card ${isFeasible ? 'success' : 'warning'}">
            <span class="status-card-icon">⏰</span>
            <span class="status-card-label">Timeline</span>
            <span class="status-card-value">${isFeasible ? 'Feasible' : 'Tight Schedule'}</span>
        </div>
    `;

    // Meeting Info
    cardsHTML += `
        <div class="status-card info">
            <span class="status-card-icon">📅</span>
            <span class="status-card-label">Meeting</span>
            <span class="status-card-value">${currentRequest?.meeting_time || 'N/A'}</span>
        </div>
    `;

    cardsHTML += '</div>';

    // ============ COST BREAKDOWN ============
    cardsHTML += `
        <div class="result-card cost-breakdown-card">
            <div class="card-header">
                <h4>🧾 Cost Breakdown</h4>
            </div>
            <div class="card-body">
                <div class="cost-row">
                    <span class="cost-label">Flight (${flight.airline || 'Selected'})</span>
                    <span class="cost-value">${formatCurrency(flightPrice)}</span>
                </div>
                <div class="cost-row">
                    <span class="cost-label">Hotel (${nights} night${nights > 1 ? 's' : ''} × ${formatCurrency(hotelPricePerNight)})</span>
                    <span class="cost-value">${formatCurrency(hotelTotal)}</span>
                </div>
                <div class="cost-row total">
                    <span class="cost-label">Total</span>
                    <span class="cost-value">${formatCurrency(totalCost)}</span>
                </div>
                <div class="cost-row budget">
                    <span class="cost-label">Your Budget</span>
                    <span class="cost-value">${formatCurrency(budget)}</span>
                </div>
                <div class="cost-row remaining ${remaining >= 0 ? 'positive' : 'negative'}">
                    <span class="cost-label">${remaining >= 0 ? 'Remaining' : 'Over Budget'}</span>
                    <span class="cost-value">${formatCurrency(Math.abs(remaining))}</span>
                </div>
            </div>
        </div>
    `;

    // ============ POLICY REASONING ============
    if (policy.reasoning) {
        cardsHTML += `
            <div class="result-card reasoning-card">
                <div class="card-header">
                    <h4>🤖 Agent Reasoning</h4>
                </div>
                <div class="card-body">
                    <p class="reasoning-text">${policy.reasoning}</p>
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
    
    if (!traces || (Array.isArray(traces) && traces.length === 0) || 
        (typeof traces === 'object' && Object.keys(traces).length === 0)) {
        reasoningSection.style.display = 'none';
        return;
    }

    reasoningSection.style.display = 'block';

    let traceHTML = '';
    
    if (Array.isArray(traces)) {
        traces.forEach((trace, index) => {
            traceHTML += `
                <div class="reasoning-step">
                    <div class="step-number">${index + 1}</div>
                    <div class="step-content">
                        ${trace.thought ? `<div class="thought"><strong>Thought:</strong> ${trace.thought}</div>` : ''}
                        ${trace.action ? `<div class="action"><strong>Action:</strong> ${trace.action}</div>` : ''}
                        ${trace.observation ? `<div class="observation"><strong>Observation:</strong> ${trace.observation}</div>` : ''}
                    </div>
                </div>
            `;
        });
    } else if (typeof traces === 'object') {
        for (const [agent, agentTraces] of Object.entries(traces)) {
            if (!agentTraces || agentTraces.length === 0) continue;
            
            const agentInfo = AGENTS.find(a => a.id === agent.replace('_agent', '')) || { icon: '🤖', name: agent };
            traceHTML += `
                <div class="agent-trace">
                    <h5>${agentInfo.icon} ${agentInfo.name}</h5>
                    <div class="trace-steps">
                        ${agentTraces.map((trace, index) => `
                            <div class="reasoning-step">
                                <div class="step-number">${index + 1}</div>
                                <div class="step-content">
                                    ${trace.thought ? `<div class="thought">${trace.thought}</div>` : ''}
                                    ${trace.action ? `<div class="action">⚡ ${trace.action}</div>` : ''}
                                    ${trace.observation ? `<div class="observation">→ ${trace.observation.substring(0, 200)}${trace.observation.length > 200 ? '...' : ''}</div>` : ''}
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
    }

    reasoningContent.innerHTML = traceHTML || '<p>No detailed reasoning trace available.</p>';
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.innerHTML = `<span class="error-icon">⚠️</span> ${message}`;
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
    document.getElementById('addressPreview').style.display = 'none';
    meetingLocation = null;
    currentRequest = null;
    hideError();
}
