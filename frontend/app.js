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
    {
        id: 'flight',
        name: 'Flight Agent',
        icon: '✈️',
        steps: [
            'Searching available flights...',
            'Found 8 flight options',
            'Comparing prices and schedules...',
            'Selected optimal flight'
        ]
    },
    {
        id: 'hotel',
        name: 'Hotel Agent',
        icon: '🏨',
        steps: [
            'Searching hotels in destination...',
            'Found 12 hotels matching criteria',
            'Analyzing amenities and ratings...',
            'Selected best-value hotel'
        ]
    },
    {
        id: 'policy',
        name: 'Policy Agent',
        icon: '📋',
        steps: [
            'Reviewing budget constraints...',
            'Checking policy compliance...',
            'Validating cost breakdown...',
            'Budget analysis complete'
        ]
    },
    {
        id: 'time',
        name: 'Time Agent',
        icon: '⏰',
        steps: [
            'Calculating travel times...',
            'Checking meeting feasibility...',
            'Analyzing schedule conflicts...',
            'Timeline validated'
        ]
    },
    {
        id: 'orchestrator',
        name: 'Orchestrator',
        icon: '🎯',
        steps: [
            'Coordinating agent tasks...',
            'Negotiating optimal combinations...',
            'Resolving conflicts...',
            'Finalizing recommendations'
        ]
    }
];

// Store geocoded location and form data
let meetingLocation = null;
let currentRequest = null;
let selectedAmenities = [];
const MAX_AMENITIES = 5;

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

    // Initialize amenity selector
    initAmenitySelector();

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

// Initialize amenity selector with click handlers
function initAmenitySelector() {
    const selector = document.getElementById('amenitySelector');
    const countDisplay = document.getElementById('amenityCount');

    if (!selector) return;

    const buttons = selector.querySelectorAll('.amenity-btn');

    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const amenity = btn.dataset.amenity;

            if (btn.classList.contains('selected')) {
                // Deselect
                btn.classList.remove('selected');
                selectedAmenities = selectedAmenities.filter(a => a !== amenity);
            } else if (selectedAmenities.length < MAX_AMENITIES) {
                // Select
                btn.classList.add('selected');
                selectedAmenities.push(amenity);
            }

            // Update count display
            updateAmenityCount();

            // Enable/disable unselected buttons based on limit
            updateAmenityButtonStates();
        });
    });
}

// Update the amenity count display
function updateAmenityCount() {
    const countDisplay = document.getElementById('amenityCount');
    if (countDisplay) {
        countDisplay.textContent = `${selectedAmenities.length} of ${MAX_AMENITIES} selected`;
        if (selectedAmenities.length >= MAX_AMENITIES) {
            countDisplay.classList.add('max-reached');
        } else {
            countDisplay.classList.remove('max-reached');
        }
    }
}

// Enable/disable amenity buttons based on selection limit
function updateAmenityButtonStates() {
    const selector = document.getElementById('amenitySelector');
    if (!selector) return;

    const buttons = selector.querySelectorAll('.amenity-btn');
    const atLimit = selectedAmenities.length >= MAX_AMENITIES;

    buttons.forEach(btn => {
        if (!btn.classList.contains('selected')) {
            btn.disabled = atLimit;
        }
    });
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

    // Get origin and destination
    const origin = document.getElementById('origin').value;
    const destination = document.getElementById('destination').value;

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
        hotel_location: null,  // Optional field - removed from UI but backend expects it
        meeting_time: document.getElementById('meeting_time').value,
        meeting_date: document.getElementById('meeting_date').value,
        meeting_address: document.getElementById('meeting_address').value,
        meeting_coordinates: meetingLocation,
        budget: parseFloat(document.getElementById('budget').value),
        trip_type: 'personal',  // Default value - removed from UI but backend requires it
        required_amenities: selectedAmenities.length > 0 ? selectedAmenities : null
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

            // Handle validation errors (422) which return an array of error details
            if (response.status === 422 && Array.isArray(errorData.detail)) {
                const errorMessages = errorData.detail.map(err => {
                    const field = err.loc ? err.loc.join(' -> ') : 'Unknown field';
                    return `${field}: ${err.msg}`;
                }).join('\n');
                throw new Error('Validation Error:\n' + errorMessages);
            }

            // Handle other error formats
            const errorMessage = typeof errorData.detail === 'string'
                ? errorData.detail
                : (errorData.message || JSON.stringify(errorData.detail || errorData));
            throw new Error(errorMessage || 'Failed to plan trip');
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

// Simulate agent progress for better UX with detailed steps
function simulateAgentProgress() {
    const agentStartDelays = [200, 800, 1800, 3000, 4200];

    AGENTS.forEach((agent, agentIndex) => {
        // Start the agent
        setTimeout(() => {
            updateAgentStatus(agent.id, 'active', 0);

            // Progress through steps
            agent.steps.forEach((step, stepIndex) => {
                setTimeout(() => {
                    updateAgentStatus(agent.id, 'active', stepIndex);
                }, (stepIndex + 1) * 800);
            });
        }, agentStartDelays[agentIndex] || 200);
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

// Initialize agent status display with enhanced UI
function initAgentStatus() {
    const statusList = document.getElementById('statusList');

    statusList.innerHTML = AGENTS.map(agent => `
        <div class="status-item waiting" id="status-${agent.id}" data-agent-id="${agent.id}">
            <span class="status-icon">${agent.icon}</span>
            <div class="status-info">
                <span class="status-name">${agent.name}</span>
                <div class="agent-progress-bar" id="progress-${agent.id}">
                    <div class="agent-progress-fill"></div>
                </div>
            </div>
            <span class="status-state">
                <span class="status-spinner" style="display: none;"></span>
                <span class="status-dot"></span>
                <span class="status-text">Waiting</span>
            </span>
        </div>
    `).join('');
}

// Update agent status with progress tracking
function updateAgentStatus(agentId, state, stepIndex = 0) {
    const statusItem = document.getElementById(`status-${agentId}`);
    if (!statusItem) return;

    const agent = AGENTS.find(a => a.id === agentId);
    if (!agent) return;

    const statusText = statusItem.querySelector('.status-text');
    const statusDot = statusItem.querySelector('.status-dot');
    const statusSpinner = statusItem.querySelector('.status-spinner');
    const progressBar = document.querySelector(`#progress-${agentId} .agent-progress-fill`);

    // Update status item class
    statusItem.className = `status-item ${state}`;

    if (state === 'active') {
        // Show spinner, hide dot
        statusSpinner.style.display = 'inline-block';
        statusDot.style.display = 'none';
        statusText.textContent = 'Working...';

        // Update progress bar
        if (stepIndex < agent.steps.length) {
            const progress = ((stepIndex + 1) / agent.steps.length) * 100;
            if (progressBar) {
                progressBar.style.width = `${progress}%`;
            }
        }
    } else if (state === 'complete') {
        // Hide spinner, show dot
        statusSpinner.style.display = 'none';
        statusDot.style.display = 'inline-block';
        statusText.textContent = 'Done';

        // Complete progress bar
        if (progressBar) {
            progressBar.style.width = '100%';
        }
    } else {
        // Waiting state
        statusSpinner.style.display = 'none';
        statusDot.style.display = 'inline-block';
        statusText.textContent = 'Waiting';

        if (progressBar) {
            progressBar.style.width = '0%';
        }
    }
}

// Display results from API
function displayResults(result) {
    const resultCards = document.getElementById('resultCards');

    // Calculate values
    const flight = result.selected_flight || {};
    const hotel = result.selected_hotel || {};
    const policy = result.policy_check || {};
    const timeCheck = result.time_check || {};
    const nights = currentRequest?.nights || 2;

    const flightPrice = flight.price_usd || flight.price || 0;
    const hotelPricePerNight = hotel.price_per_night_usd || hotel.price_per_night || 0;
    const hotelTotal = hotelPricePerNight * nights;
    const totalCost = result.total_cost || (flightPrice + hotelTotal);
    const budget = currentRequest?.budget || policy.budget || 0;
    const remaining = budget - totalCost;
    const isCompliant = policy.is_compliant || policy.overall_status === 'compliant';
    const isFeasible = timeCheck.feasible !== false && timeCheck.is_feasible !== false;

    let cardsHTML = '';

    // ============ KEY HIGHLIGHTS ============
    cardsHTML += `
        <div class="key-highlights">
            <h3 class="highlights-title">✨ Your Trip at a Glance</h3>
            <div class="highlights-stack">
                <div class="highlight-rect">
                    <div class="highlight-icon">💰</div>
                    <div class="highlight-content">
                        <div class="highlight-label">Total Cost</div>
                        <div class="highlight-value">${formatCurrency(totalCost)}</div>
                        <div class="highlight-subtext ${remaining >= 0 ? 'positive' : 'negative'}">
                            ${remaining >= 0 ? `${formatCurrency(remaining)} under budget` : `${formatCurrency(Math.abs(remaining))} over budget`}
                        </div>
                    </div>
                </div>
                <div class="highlight-rect">
                    <div class="highlight-icon">✈️</div>
                    <div class="highlight-content">
                        <div class="highlight-label">Flight</div>
                        <div class="highlight-value">${flight.airline || 'N/A'}</div>
                        <div class="highlight-subtext">${flight.departure_time || ''} - ${flight.arrival_time || ''}</div>
                    </div>
                </div>
                <div class="highlight-rect">
                    <div class="highlight-icon">🏨</div>
                    <div class="highlight-content">
                        <div class="highlight-label">Hotel</div>
                        <div class="highlight-value">${hotel.name ? hotel.name.substring(0, 30) : 'N/A'}${hotel.name?.length > 30 ? '...' : ''}</div>
                        <div class="highlight-subtext">${'★'.repeat(hotel.stars || 0)} • ${nights} night${nights > 1 ? 's' : ''}</div>
                    </div>
                </div>
                <div class="highlight-rect ${isCompliant ? 'success' : 'warning'}">
                    <div class="highlight-icon">${isCompliant ? '✅' : '⚠️'}</div>
                    <div class="highlight-content">
                        <div class="highlight-label">Budget Status</div>
                        <div class="highlight-value">${isCompliant ? 'Compliant' : 'Review Needed'}</div>
                        <div class="highlight-subtext">Budget: ${formatCurrency(budget)}</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TRIP TIMELINE -->
        <div class="trip-timeline">
            <h3 class="timeline-title">📅 Your Trip Timeline</h3>
            <div class="timeline-container">
                <div class="timeline-item">
                    <div class="timeline-marker departure"></div>
                    <div class="timeline-content">
                        <div class="timeline-date">${formatDate(currentRequest?.departure_date)}</div>
                        <div class="timeline-event">
                            <strong>Departure from ${CITY_DATA[currentRequest?.origin]?.name || currentRequest?.origin}</strong>
                            <div class="timeline-detail">Flight ${flight.flight_id || ''} at ${flight.departure_time || 'TBD'}</div>
                        </div>
                    </div>
                </div>
                <div class="timeline-connector"></div>
                <div class="timeline-item">
                    <div class="timeline-marker arrival"></div>
                    <div class="timeline-content">
                        <div class="timeline-date">${formatDate(currentRequest?.departure_date)}</div>
                        <div class="timeline-event">
                            <strong>Arrival in ${CITY_DATA[currentRequest?.destination]?.name || currentRequest?.destination}</strong>
                            <div class="timeline-detail">Arrives ${flight.arrival_time || 'TBD'} • Check-in at ${hotel.name || 'hotel'}</div>
                        </div>
                    </div>
                </div>
                <div class="timeline-connector"></div>
                <div class="timeline-item">
                    <div class="timeline-marker meeting"></div>
                    <div class="timeline-content">
                        <div class="timeline-date">${formatDate(currentRequest?.meeting_date)}</div>
                        <div class="timeline-event">
                            <strong>Meeting</strong>
                            <div class="timeline-detail">At ${currentRequest?.meeting_time || 'TBD'}${hotel.travel_time_minutes ? ` • ${Math.round(hotel.travel_time_minutes)} min from hotel` : ''}</div>
                        </div>
                    </div>
                </div>
                <div class="timeline-connector"></div>
                <div class="timeline-item">
                    <div class="timeline-marker return"></div>
                    <div class="timeline-content">
                        <div class="timeline-date">${formatDate(currentRequest?.hotel_checkout)}</div>
                        <div class="timeline-event">
                            <strong>Check-out & Departure</strong>
                            <div class="timeline-detail">Return to ${CITY_DATA[currentRequest?.origin]?.name || currentRequest?.origin}</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    // ============ FLIGHT DETAILS CARD ============
    if (result.selected_flight) {
        cardsHTML += `
            <div class="result-card flight-card-enhanced">
                <div class="card-header-enhanced">
                    <div class="card-title-group">
                        <div class="card-icon-badge">✈️</div>
                        <div>
                            <h4>Flight Details</h4>
                            <p class="card-subtitle">Your selected flight option</p>
                        </div>
                    </div>
                    <div class="price-tag">
                        <div class="price-amount">${formatCurrency(flightPrice)}</div>
                        <div class="price-label">Per person</div>
                    </div>
                </div>
                <div class="card-body-enhanced">
                    <div class="flight-route-visual">
                        <div class="route-point">
                            <div class="route-time">${flight.departure_time || 'N/A'}</div>
                            <div class="route-city">${CITY_DATA[currentRequest?.origin]?.name || currentRequest?.origin || 'N/A'}</div>
                            <div class="route-code">${currentRequest?.origin || ''}</div>
                        </div>
                        <div class="route-middle">
                            <div class="route-line"></div>
                            <div class="route-duration">${flight.duration_hours ? flight.duration_hours.toFixed(1) + ' hours' : 'Direct Flight'}</div>
                            <div class="route-airline">${flight.airline || 'Unknown Airline'} ${flight.flight_id || ''}</div>
                        </div>
                        <div class="route-point">
                            <div class="route-time">${flight.arrival_time || 'N/A'}</div>
                            <div class="route-city">${CITY_DATA[currentRequest?.destination]?.name || currentRequest?.destination || 'N/A'}</div>
                            <div class="route-code">${currentRequest?.destination || ''}</div>
                        </div>
                    </div>
                    <div class="info-badges">
                        <div class="info-badge">
                            <span class="badge-icon">🪑</span>
                            <span class="badge-text">${flight.class || flight.flight_class || 'Economy'} Class</span>
                        </div>
                        ${flight.seats_available ? `
                        <div class="info-badge ${flight.seats_available < 5 ? 'warning' : ''}">
                            <span class="badge-icon">🎫</span>
                            <span class="badge-text">${flight.seats_available} seats available</span>
                        </div>` : ''}
                        ${flight.amenities ? `
                        <div class="info-badge">
                            <span class="badge-icon">✨</span>
                            <span class="badge-text">${flight.amenities}</span>
                        </div>` : ''}
                    </div>
                </div>
            </div>
        `;
    }

    // ============ HOTEL DETAILS CARD ============
    if (result.selected_hotel) {
        const stars = hotel.stars || hotel.rating || 0;
        cardsHTML += `
            <div class="result-card hotel-card-enhanced">
                <div class="card-header-enhanced">
                    <div class="card-title-group">
                        <div class="card-icon-badge hotel">🏨</div>
                        <div>
                            <h4>Hotel Accommodation</h4>
                            <p class="card-subtitle">Recommended based on your preferences</p>
                        </div>
                    </div>
                    <div class="price-tag">
                        <div class="price-amount">${formatCurrency(hotelPricePerNight)}</div>
                        <div class="price-label">Per night</div>
                    </div>
                </div>
                <div class="card-body-enhanced">
                    <div class="hotel-header-info">
                        <div class="hotel-main-info">
                            <h3 class="hotel-name-large">${hotel.name || 'Unknown Hotel'}</h3>
                            <div class="hotel-rating-row">
                                <div class="hotel-stars-large">${'★'.repeat(stars)}${'☆'.repeat(5 - stars)}</div>
                                <div class="hotel-rating-text">${stars}-Star Hotel</div>
                            </div>
                        </div>
                    </div>

                    <div class="hotel-quick-facts-stack">
                        <div class="quick-fact-rect">
                            <span class="fact-icon">📍</span>
                            <div class="fact-content">
                                <div class="fact-label">Location</div>
                                <div class="fact-value">${hotel.business_area || hotel.city_name || hotel.city || currentRequest?.destination || 'N/A'}</div>
                            </div>
                        </div>
                        ${hotel.distance_to_meeting_km ? `
                        <div class="quick-fact-rect highlight">
                            <span class="fact-icon">🚗</span>
                            <div class="fact-content">
                                <div class="fact-label">To Meeting</div>
                                <div class="fact-value">${hotel.distance_to_meeting_km.toFixed(1)}km • ${Math.round(hotel.travel_time_minutes || 0)} min drive</div>
                            </div>
                        </div>` : ''}
                        ${hotel.distance_to_business_center_km ? `
                        <div class="quick-fact-rect">
                            <span class="fact-icon">🏙️</span>
                            <div class="fact-content">
                                <div class="fact-label">To City Center</div>
                                <div class="fact-value">${hotel.distance_to_business_center_km.toFixed(1)}km</div>
                            </div>
                        </div>` : ''}
                        <div class="quick-fact-rect">
                            <span class="fact-icon">🛏️</span>
                            <div class="fact-content">
                                <div class="fact-label">Stay Duration</div>
                                <div class="fact-value">${nights} night${nights > 1 ? 's' : ''}</div>
                            </div>
                        </div>
                    </div>

                    <div class="stay-summary">
                        <div class="stay-summary-row">
                            <span>Check-in</span>
                            <strong>${formatDate(currentRequest?.hotel_checkin)}</strong>
                        </div>
                        <div class="stay-summary-row">
                            <span>Check-out</span>
                            <strong>${formatDate(currentRequest?.hotel_checkout)}</strong>
                        </div>
                        <div class="stay-summary-row total">
                            <span>Total Stay</span>
                            <strong>${formatCurrency(hotelTotal)}</strong>
                        </div>
                    </div>

                    ${hotel.amenities && hotel.amenities.length > 0 ? `
                        <div class="amenities-section">
                            <div class="amenities-title">Amenities Included</div>
                            <div class="amenities-compact">
                                ${hotel.amenities.map(a => {
                                    const amenityIcons = {
                                        'WiFi': '📶', 'Pool': '🏊', 'Gym': '🏋️', 'Restaurant': '🍽️',
                                        'Parking': '🅿️', 'Spa': '🧖', 'Bar': '🍸', 'Room Service': '🛎️',
                                        'Breakfast': '🍳', 'Business Center': '💼'
                                    };
                                    const icon = amenityIcons[a] || '✓';
                                    return `<span class="amenity-chip-compact">${icon} ${a}</span>`;
                                }).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }

    // ============ WHY THIS CHOICE ============
    if (policy.reasoning) {
        cardsHTML += `
            <div class="why-choice-card">
                <div class="why-choice-header">
                    <div class="why-choice-icon">🤖</div>
                    <div>
                        <h3>Why We Recommend This Option</h3>
                        <p>Our AI agents analyzed multiple factors to find the best match for you</p>
                    </div>
                </div>
                <div class="why-choice-content">
                    <div class="reasoning-item">
                        <div class="reasoning-badge">💰 Budget Analysis</div>
                        <p>${policy.reasoning}</p>
                    </div>
                </div>
            </div>
        `;
    }

    // ============ HOTEL ALTERNATIVES ============
    if (result.cheaper_alternatives && result.cheaper_alternatives.length > 0) {
        cardsHTML += `
            <div class="alternatives-card-enhanced">
                <div class="alternatives-header">
                    <div class="alternatives-icon">💡</div>
                    <div>
                        <h3>Alternative Options</h3>
                        <p>Consider these options if you want to adjust your budget or preferences</p>
                    </div>
                </div>
                <div class="alternatives-list">
                    ${result.cheaper_alternatives.map((alt, index) => {
            const altHotel = alt.hotel || {};
            const vsDiff = alt.vs_selected || 0;
            const isUpgrade = vsDiff > 0;
            const altTotalCost = alt.total_cost || 0;
            const savingsPercent = Math.abs((vsDiff / altTotalCost) * 100);

            return `
                        <div class="alternative-option ${isUpgrade ? 'upgrade' : 'budget'}">
                            <div class="alt-rank-badge">#${index + 1}</div>
                            <div class="alt-content">
                                <div class="alt-header-row">
                                    <div class="alt-title">
                                        <div class="alt-category-badge ${isUpgrade ? 'premium' : 'value'}">${alt.category || (isUpgrade ? '⭐ Upgrade' : '💵 Budget')}</div>
                                        <h4 class="alt-hotel-name">${altHotel.name || 'Alternative Hotel'}</h4>
                                    </div>
                                    <div class="alt-pricing">
                                        <div class="alt-total-cost">${formatCurrency(altTotalCost)}</div>
                                        <div class="alt-difference ${isUpgrade ? 'higher' : 'lower'}">
                                            ${isUpgrade ? '+' : ''}${formatCurrency(vsDiff)} ${isUpgrade ? 'more' : 'savings'}
                                        </div>
                                    </div>
                                </div>
                                <div class="alt-details-row">
                                    <div class="alt-detail">
                                        <span class="alt-detail-icon">${'★'.repeat(altHotel.stars || 0)}${'☆'.repeat(5 - (altHotel.stars || 0))}</span>
                                    </div>
                                    ${altHotel.distance_km ? `
                                    <div class="alt-detail">
                                        <span class="alt-detail-icon">📍</span>
                                        <span>${altHotel.distance_km.toFixed(1)}km to center</span>
                                    </div>` : ''}
                                    ${!isUpgrade && savingsPercent > 0 ? `
                                    <div class="alt-detail highlight">
                                        <span class="alt-detail-icon">💰</span>
                                        <span>Save ${savingsPercent.toFixed(0)}%</span>
                                    </div>` : ''}
                                </div>
                                ${alt.reasoning ? `
                                    <div class="alt-reasoning">
                                        <span class="reasoning-icon">💭</span>
                                        ${alt.reasoning}
                                    </div>
                                ` : ''}
                            </div>
                        </div>
                    `;
        }).join('')}
                </div>
            </div>
        `;
    }

    // ============ COST BREAKDOWN ============
    const costPercentage = budget > 0 ? Math.min((totalCost / budget) * 100, 100) : 0;
    cardsHTML += `
        <div class="result-card cost-breakdown-enhanced">
            <div class="card-header-enhanced">
                <div class="card-title-group">
                    <div class="card-icon-badge cost">💰</div>
                    <div>
                        <h4>Cost Summary</h4>
                        <p class="card-subtitle">Complete breakdown of your trip expenses</p>
                    </div>
                </div>
            </div>
            <div class="card-body-enhanced">
                <div class="cost-visual">
                    <div class="cost-total-display">
                        <div class="cost-total-label">Total Trip Cost</div>
                        <div class="cost-total-amount">${formatCurrency(totalCost)}</div>
                        <div class="cost-budget-bar">
                            <div class="cost-budget-fill ${remaining >= 0 ? 'under' : 'over'}" style="width: ${costPercentage}%"></div>
                        </div>
                        <div class="cost-budget-labels">
                            <span>$0</span>
                            <span class="budget-label">Budget: ${formatCurrency(budget)}</span>
                        </div>
                    </div>
                </div>

                <div class="cost-breakdown-list">
                    <div class="cost-item">
                        <div class="cost-item-left">
                            <span class="cost-item-icon">✈️</span>
                            <div class="cost-item-info">
                                <div class="cost-item-name">Flight</div>
                                <div class="cost-item-desc">${flight.airline || 'Selected flight'} ${flight.flight_id || ''}</div>
                            </div>
                        </div>
                        <div class="cost-item-amount">${formatCurrency(flightPrice)}</div>
                    </div>
                    <div class="cost-item">
                        <div class="cost-item-left">
                            <span class="cost-item-icon">🏨</span>
                            <div class="cost-item-info">
                                <div class="cost-item-name">Hotel Accommodation</div>
                                <div class="cost-item-desc">${nights} night${nights > 1 ? 's' : ''} × ${formatCurrency(hotelPricePerNight)}</div>
                            </div>
                        </div>
                        <div class="cost-item-amount">${formatCurrency(hotelTotal)}</div>
                    </div>
                </div>

                <div class="cost-summary-box">
                    <div class="cost-summary-row">
                        <span>Subtotal</span>
                        <strong>${formatCurrency(totalCost)}</strong>
                    </div>
                    <div class="cost-summary-row">
                        <span>Your Budget</span>
                        <strong>${formatCurrency(budget)}</strong>
                    </div>
                    <div class="cost-summary-row final ${remaining >= 0 ? 'positive' : 'negative'}">
                        <span>${remaining >= 0 ? '💰 Remaining Budget' : '⚠️ Over Budget'}</span>
                        <strong class="${remaining >= 0 ? 'positive' : 'negative'}">${formatCurrency(Math.abs(remaining))}</strong>
                    </div>
                </div>

                ${isCompliant ? `
                    <div class="compliance-badge success">
                        <span class="compliance-icon">✅</span>
                        <span class="compliance-text">This trip is within your budget!</span>
                    </div>
                ` : `
                    <div class="compliance-badge warning">
                        <span class="compliance-icon">⚠️</span>
                        <span class="compliance-text">This trip exceeds your budget. Consider the alternatives below.</span>
                    </div>
                `}
            </div>
        </div>
    `;

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
    // Replace newlines with <br> for multi-line errors
    const formattedMessage = message.replace(/\n/g, '<br>');
    errorDiv.innerHTML = `<span class="error-icon">⚠️</span> <div class="error-text">${formattedMessage}</div>`;
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

    // Reset amenity selection
    selectedAmenities = [];
    const selector = document.getElementById('amenitySelector');
    if (selector) {
        selector.querySelectorAll('.amenity-btn').forEach(btn => {
            btn.classList.remove('selected');
            btn.disabled = false;
        });
    }
    updateAmenityCount();

    hideError();
}
