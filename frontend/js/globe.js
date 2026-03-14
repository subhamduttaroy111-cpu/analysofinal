/**
 * globe.js — 3D News Sentiment Globe Visualization
 * ═══════════════════════════════════════════════════
 * ADDITIVE: Brand-new file. Uses Globe.gl (CDN) to render
 * a 3D interactive globe with live Indian market sentiment.
 */

// ── Config ──────────────────────────────────────────────────
const BACKEND_URL = window.GLOBE_BACKEND_URL || "";
const REFRESH_INTERVAL = 300000; // 5 minutes
const INDIA_CENTER = { lat: 22.5, lng: 78.9, altitude: 2.0 };

// ── State ───────────────────────────────────────────────────
let globe = null;
let globeData = null;

// ── Initialize Globe ────────────────────────────────────────
function initGlobe() {
    const container = document.getElementById("globeViz");

    globe = Globe()(container)
        // Earth appearance
        .globeImageUrl("https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg")
        .bumpImageUrl("https://unpkg.com/three-globe/example/img/earth-topology.png")
        .backgroundImageUrl("https://unpkg.com/three-globe/example/img/night-sky.png")
        .showAtmosphere(true)
        .atmosphereColor("rgba(99, 102, 241, 0.3)")
        .atmosphereAltitude(0.22)

        // Points config (financial hubs)
        .pointsData([])
        .pointLat("lat")
        .pointLng("lng")
        .pointColor("color")
        .pointAltitude(0.02)
        .pointRadius("radius")
        .pointLabel(d => `
            <div class="globe-tooltip">
                <strong>${d.label || d.name}</strong><br/>
                <span style="color:${d.color}">● ${d.news_count || 0} headlines</span><br/>
                <span>Sentiment: ${d.avg_sentiment > 0 ? '+' : ''}${d.avg_sentiment}</span>
            </div>
        `)

        // Arcs config (news flow lines)
        .arcsData([])
        .arcStartLat("startLat")
        .arcStartLng("startLng")
        .arcEndLat("endLat")
        .arcEndLng("endLng")
        .arcColor("color")
        .arcAltitude(0.15)
        .arcStroke(0.5)
        .arcDashLength(0.6)
        .arcDashGap(0.3)
        .arcDashAnimateTime(2000)

        // Labels config (city names)
        .labelsData([])
        .labelLat("lat")
        .labelLng("lng")
        .labelText("name")
        .labelSize(1.2)
        .labelDotRadius(0.4)
        .labelColor(() => "rgba(165, 180, 252, 0.85)")
        .labelResolution(2);

    // Auto-rotate slowly
    globe.controls().autoRotate = true;
    globe.controls().autoRotateSpeed = 0.4;
    globe.controls().enableDamping = true;
    globe.controls().dampingFactor = 0.1;

    // Point camera at India
    setTimeout(() => {
        globe.pointOfView(INDIA_CENTER, 1500);
    }, 500);

    // Handle window resize
    window.addEventListener("resize", () => {
        globe.width(window.innerWidth);
        globe.height(window.innerHeight);
    });
}

// ── Fetch Sentiment Data ────────────────────────────────────
async function fetchSentimentData() {
    try {
        const apiBase = BACKEND_URL || 
            (window.location.protocol === "file:" ? "http://127.0.0.1:5001" : "");
        
        const configScript = document.querySelector('script[src="js/config.js"]');
        const backendUrl = (typeof API_BASE !== "undefined" && API_BASE) ? API_BASE : apiBase;

        const response = await fetch(`${backendUrl}/api/sentiment-globe`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        globeData = await response.json();
        updateGlobe(globeData);
        updatePanels(globeData);

        // Hide loading screen
        const loader = document.getElementById("globeLoading");
        if (loader) loader.classList.add("hidden");

    } catch (error) {
        console.error("❌ Sentiment fetch error:", error);
        showFallbackData();
    }
}

// ── Update Globe with Data ──────────────────────────────────
function updateGlobe(data) {
    if (!globe || !data) return;

    // Hub points (big glowing markers)
    const hubPoints = data.hubs.map(hub => ({
        ...hub,
        radius: Math.max(0.3, hub.news_count * 0.08),
    }));

    // News arcs (from a central "source" point to each hub)
    const sourcePoint = { lat: 5.0, lng: 55.0 }; // Indian Ocean — looks cool
    const arcs = data.news_points.map(np => ({
        startLat: sourcePoint.lat + (Math.random() - 0.5) * 8,
        startLng: sourcePoint.lng + (Math.random() - 0.5) * 8,
        endLat: np.lat,
        endLng: np.lng,
        color: [np.sentiment.color, np.sentiment.color],
    }));

    // Labels for city names
    const labels = data.hubs.map(hub => ({
        lat: hub.lat,
        lng: hub.lng,
        name: hub.name,
    }));

    globe.pointsData(hubPoints);
    globe.arcsData(arcs);
    globe.labelsData(labels);
}

// ── Update UI Panels ────────────────────────────────────────
function updatePanels(data) {
    if (!data) return;

    // Stats panel
    const totalEl = document.getElementById("statTotal");
    const bullishEl = document.getElementById("statBullish");
    const bearishEl = document.getElementById("statBearish");
    const neutralEl = document.getElementById("statNeutral");
    const timeEl = document.getElementById("statTime");

    if (totalEl) totalEl.textContent = data.total_headlines || 0;
    if (bullishEl) bullishEl.textContent = data.summary.bullish || 0;
    if (bearishEl) bearishEl.textContent = data.summary.bearish || 0;
    if (neutralEl) neutralEl.textContent = data.summary.neutral || 0;
    if (timeEl) {
        const dt = new Date(data.timestamp);
        timeEl.textContent = dt.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
    }

    // Legend counts
    const legendBullish = document.getElementById("legendBullish");
    const legendBearish = document.getElementById("legendBearish");
    const legendNeutral = document.getElementById("legendNeutral");
    if (legendBullish) legendBullish.textContent = data.summary.bullish || 0;
    if (legendBearish) legendBearish.textContent = data.summary.bearish || 0;
    if (legendNeutral) legendNeutral.textContent = data.summary.neutral || 0;

    // News feed
    const feedContainer = document.getElementById("newsFeedList");
    if (feedContainer && data.news_points) {
        feedContainer.innerHTML = data.news_points.slice(0, 12).map(np => `
            <a href="${np.link}" target="_blank" rel="noopener" class="news-feed-item">
                <div class="news-item-header">
                    <span class="sentiment-badge ${np.sentiment.label.toLowerCase()}">${np.sentiment.label}</span>
                    <span style="font-size:0.65rem; color:#64748b;">${np.city}</span>
                </div>
                <div class="news-item-title">${np.title}</div>
                <div class="news-item-meta">${np.source}</div>
            </a>
        `).join("");
    }
}

// ── Fallback Data (if API is unavailable) ───────────────────
function showFallbackData() {
    const loader = document.getElementById("globeLoading");
    if (loader) loader.classList.add("hidden");

    // Show hubs with neutral colors
    const fallbackHubs = [
        { name: "Mumbai", label: "Mumbai — BSE/NSE HQ", lat: 19.076, lng: 72.878, color: "#64748b", radius: 0.4, news_count: 0, avg_sentiment: 0 },
        { name: "Delhi", label: "Delhi — Policy Hub", lat: 28.614, lng: 77.209, color: "#64748b", radius: 0.3, news_count: 0, avg_sentiment: 0 },
        { name: "Bengaluru", label: "Bengaluru — Tech/IT", lat: 12.972, lng: 77.595, color: "#64748b", radius: 0.3, news_count: 0, avg_sentiment: 0 },
        { name: "Chennai", label: "Chennai — Industrial", lat: 13.083, lng: 80.271, color: "#64748b", radius: 0.3, news_count: 0, avg_sentiment: 0 },
        { name: "Hyderabad", label: "Hyderabad — Pharma/IT", lat: 17.385, lng: 78.487, color: "#64748b", radius: 0.3, news_count: 0, avg_sentiment: 0 },
        { name: "Kolkata", label: "Kolkata — Eastern Trade", lat: 22.573, lng: 88.364, color: "#64748b", radius: 0.3, news_count: 0, avg_sentiment: 0 },
    ];

    if (globe) {
        globe.pointsData(fallbackHubs);
        globe.labelsData(fallbackHubs);
    }

    // Show error in stats
    const totalEl = document.getElementById("statTotal");
    if (totalEl) totalEl.textContent = "—";
}

// ── Boot ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initGlobe();
    fetchSentimentData();

    // Auto-refresh every 5 minutes
    setInterval(fetchSentimentData, REFRESH_INTERVAL);
});
