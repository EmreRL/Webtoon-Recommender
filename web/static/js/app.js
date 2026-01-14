/**
 * Webtoon RAG Recommendation System - Frontend JavaScript
 * Enhanced with cover image support and popularity-based borders
 */

// API Configuration
const API_BASE_URL = window.location.origin + '/api';

// DOM Elements
let searchInput, searchButton, resultsSection;
let loadingState, errorState, successState;
let errorMessage, retryButton;
let queryDisplay, resultCount, queryType;
let recommendationsGrid, noResultsMessage, rejectionMessage;
let availableGenres;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    attachEventListeners();
    loadDatabaseStats();
});

/**
 * Initialize DOM element references
 */
function initializeElements() {
    searchInput = document.getElementById('searchInput');
    searchButton = document.getElementById('searchButton');
    resultsSection = document.getElementById('resultsSection');
    
    loadingState = document.getElementById('loadingState');
    errorState = document.getElementById('errorState');
    successState = document.getElementById('successState');
    
    errorMessage = document.getElementById('errorMessage');
    retryButton = document.getElementById('retryButton');
    
    queryDisplay = document.getElementById('queryDisplay');
    resultCount = document.getElementById('resultCount');
    queryType = document.getElementById('queryType');
    
    recommendationsGrid = document.getElementById('recommendationsGrid');
    noResultsMessage = document.getElementById('noResultsMessage');
    rejectionMessage = document.getElementById('rejectionMessage');
    availableGenres = document.getElementById('availableGenres');
}

/**
 * Attach event listeners to interactive elements
 */
function attachEventListeners() {
    // Search button click
    searchButton.addEventListener('click', handleSearch);
    
    // Enter key in search input
    searchInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // Example query chips
    document.querySelectorAll('.example-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            searchInput.value = query;
            handleSearch();
        });
    });
    
    // Retry button
    retryButton.addEventListener('click', handleSearch);
}

/**
 * Handle search button click
 */
async function handleSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showError('Please enter a search query');
        return;
    }
    
    // Show results section and loading state
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Failed to get recommendations');
        }
        
        displayResults(data);
        
    } catch (error) {
        console.error('Search error:', error);
        showError(error.message || 'Failed to connect to server');
    }
}

/**
 * Display search results
 */
function displayResults(data) {
    console.log('Received data:', data); // Debug log
    
    // Update query display
    queryDisplay.textContent = data.query;
    
    // Normalize recommendations to always be an array
    let recommendations = [];
    
    if (Array.isArray(data.recommendations)) {
        recommendations = data.recommendations;
    } else if (data.recommendations && typeof data.recommendations === 'object') {
        // If it's an object, try to extract an array from it
        recommendations = Object.values(data.recommendations);
    } else {
        console.warn('Unexpected recommendations format:', data.recommendations);
    }
    
    // Update metadata
    const total = recommendations.length;
    resultCount.textContent = `${total} ${total === 1 ? 'Result' : 'Results'}`;
    
    if (data.metadata && data.metadata.query_type) {
        queryType.textContent = data.metadata.query_type.replace('_', ' ').toUpperCase();
    }
    
    // Check if no results
    if (total === 0 && data.message) {
        displayNoResults(data);
    } else if (total === 0) {
        displayNoResults({
            message: 'No recommendations found for your query.',
            database_stats: data.database_stats
        });
    } else {
        displayRecommendations(recommendations);
    }
    
    showSuccess();
}

/**
 * Display recommendations grid
 */
function displayRecommendations(recommendations) {
    noResultsMessage.classList.add('hidden');
    recommendationsGrid.classList.remove('hidden');
    
    // Ensure recommendations is an array
    if (!Array.isArray(recommendations)) {
        console.error('displayRecommendations received non-array:', recommendations);
        displayNoResults({
            message: 'Error displaying recommendations. Invalid data format.'
        });
        return;
    }
    
    recommendationsGrid.innerHTML = recommendations.map((rec, index) => {
        // Safely access properties with fallbacks
        const title = escapeHtml(rec.title || 'Untitled');
        const description = escapeHtml(rec.description || 'No description available.');
        const genre = rec.genre || '';
        const popularity = rec.popularity || 'Unpopular';
        const explanation = rec.explanation || '';
        const similarityScore = rec.similarity_score || null;
        const imageUrl = rec.image_url || null;
        
        // Create cover image HTML
        const coverHtml = imageUrl 
            ? `<img src="${escapeHtml(imageUrl)}" alt="${title}" class="card-cover-image" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'card-cover-placeholder\\'>🎨</div>'">` 
            : '<div class="card-cover-placeholder">🎨</div>';
        
        return `
            <div class="webtoon-card" data-popularity="${popularity}" style="animation-delay: ${index * 0.1}s">
                <div class="webtoon-card-inner">
                    <div class="card-cover">
                        ${coverHtml}
                    </div>
                    
                    <div class="card-content">
                        <div class="card-header">
                            <div class="card-rank">#${index + 1}</div>
                            <div class="card-similarity">
                                ${similarityScore !== null ? `
                                    <div class="similarity-bar">
                                        <div class="similarity-fill" style="width: ${similarityScore * 100}%"></div>
                                    </div>
                                    <span class="similarity-text">${(similarityScore * 100).toFixed(0)}% match</span>
                                ` : ''}
                            </div>
                        </div>
                        
                        <h3 class="card-title">${title}</h3>
                        
                        <div class="card-badges">
                            ${genre ? `<span class="badge badge-genre">${escapeHtml(genre)}</span>` : ''}
                            ${popularity ? `<span class="badge badge-popularity">${escapeHtml(popularity)}</span>` : ''}
                        </div>
                        
                        <p class="card-description">${description}</p>
                        
                        ${explanation ? `
                            <div class="card-recommendation">
                                <div class="recommendation-label">💡 Why this?</div>
                                <p class="recommendation-text">${escapeHtml(explanation)}</p>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

/**
 * Display no results message
 */
function displayNoResults(data) {
    recommendationsGrid.classList.add('hidden');
    noResultsMessage.classList.remove('hidden');
    
    rejectionMessage.textContent = data.message || 'No results found.';
    
    // Show available genres if provided
    if (data.database_stats && data.database_stats.genres) {
        availableGenres.innerHTML = `
            <p class="available-label">Available genres:</p>
            <div class="genre-chips">
                ${data.database_stats.genres.map(genre => `
                    <span class="genre-chip">${escapeHtml(genre)}</span>
                `).join('')}
            </div>
        `;
    } else {
        availableGenres.innerHTML = '';
    }
}

/**
 * Show loading state
 */
function showLoading() {
    resultsSection.classList.remove('hidden');
    loadingState.classList.remove('hidden');
    errorState.classList.add('hidden');
    successState.classList.add('hidden');
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Show error state
 */
function showError(message) {
    resultsSection.classList.remove('hidden');
    loadingState.classList.add('hidden');
    errorState.classList.remove('hidden');
    successState.classList.add('hidden');
    
    errorMessage.textContent = message;
}

/**
 * Show success state
 */
function showSuccess() {
    loadingState.classList.add('hidden');
    errorState.classList.add('hidden');
    successState.classList.remove('hidden');
}

/**
 * Load and display database statistics
 */
async function loadDatabaseStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();
        
        if (data.success && data.stats) {
            displayStats(data.stats);
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

/**
 * Display database statistics
 */
function displayStats(stats) {
    const totalElement = document.getElementById('totalWebtoons');
    const genreCountElement = document.getElementById('genreCount');
    const genresListElement = document.getElementById('genresList');
    
    if (totalElement) {
        totalElement.textContent = stats.total_webtoons || 0;
    }
    
    if (genreCountElement) {
        genreCountElement.textContent = stats.genres?.length || 0;
    }
    
    if (genresListElement && stats.genres && stats.genres.length > 0) {
        genresListElement.innerHTML = `
            <div class="genres-grid">
                ${stats.genres.map(genre => `
                    <span class="genre-tag">${escapeHtml(genre)}</span>
                `).join('')}
            </div>
        `;
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    if (text === null || text === undefined) {
        return '';
    }
    const div = document.createElement('div');
    div.textContent = String(text);
    return div.innerHTML;
}