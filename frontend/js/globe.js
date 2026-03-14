/**
 * globe.js — 3D News Sentiment Globe (REDESIGNED)
 * ═══════════════════════════════════════════════════
 * Clearer, more intuitive UI showing exactly where
 * good/bad news is coming from in Indian markets.
 */

// ── Config ──────────────────────────────────────────────────
const GLOBE_REFRESH_INTERVAL = 300000; // 5 minutes
const INDIA_CENTER = { lat: 22.5, lng: 78.9, altitude: 2.0 };

// ── State ───────────────────────────────────────────────────
let globe = null;
let globeData = null;
let currentFilter = "all";

// ── Initialize Globe ────────────────────────────────────────
function initGlobe() {
    const container = document.getElementById("globeViz");
    if (!container) return;

    try {
        globe = Globe()(container)
            .globeImageUrl("https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg")
            .bumpImageUrl("https://unpkg.com/three-globe/example/img/earth-topology.png")
            .backgroundImageUrl("https://unpkg.com/three-globe/example/img/night-sky.png")
            .showAtmosphere(true)
            .atmosphereColor("rgba(99, 102, 241, 0.25)")
            .atmosphereAltitude(0.2)

            // Hub Points (larger, clearer)
            .pointsData([])
            .pointLat("lat")
            .pointLng("lng")
            .pointColor("color")
            .pointAltitude(d => 0.01 + d.intensity * 0.03)
            .pointRadius(d => 0.25 + d.intensity * 0.15)
            .pointLabel(d => {
                const mood = d.avg_sentiment >= 0.1 ? "📈 Bullish" : d.avg_sentiment <= -0.1 ? "📉 Bearish" : "➡️ Neutral";
                return `
                    <div style="background:rgba(15,23,42,0.95);border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:14px 18px;color:#e2e8f0;font-family:Inter,sans-serif;font-size:0.82rem;max-width:260px;line-height:1.6;box-shadow:0 8px 32px rgba(0,0,0,0.4);">
                        <div style="font-weight:800;font-size:0.95rem;margin-bottom:6px;">${d.name}</div>
                        <div style="color:${d.color};font-weight:700;margin-bottom:4px;">${mood}</div>
                        <div style="color:#94a3b8;font-size:0.75rem;">${d.news_count} headlines • Score: ${d.avg_sentiment > 0 ? '+' : ''}${d.avg_sentiment}</div>
                    </div>
                `;
            })

            // Arcs (news flow)
            .arcsData([])
            .arcStartLat("startLat")
            .arcStartLng("startLng")
            .arcEndLat("endLat")
            .arcEndLng("endLng")
            .arcColor("colors")
            .arcAltitude(0.12)
            .arcStroke(0.6)
            .arcDashLength(0.5)
            .arcDashGap(0.25)
            .arcDashAnimateTime(1800)

            // City Labels
            .labelsData([])
            .labelLat("lat")
            .labelLng("lng")
            .labelText("labelText")
            .labelSize(1.6)
            .labelDotRadius(0.5)
            .labelColor(d => d.color || "rgba(165, 180, 252, 0.85)")
            .labelResolution(2);

        globe.controls().autoRotate = true;
        globe.controls().autoRotateSpeed = 0.35;
        globe.controls().enableDamping = true;
        globe.controls().dampingFactor = 0.1;

        setTimeout(() => {
            globe.pointOfView(INDIA_CENTER, 1500);
        }, 500);

        window.addEventListener("resize", () => {
            globe.width(window.innerWidth);
            globe.height(window.innerHeight);
        });

        console.log("✅ Globe initialized");
    } catch (err) {
        console.error("❌ Globe init error:", err);
        hideLoader();
    }
}

// ── Hide Loader ─────────────────────────────────────────────
function hideLoader() {
    const el = document.getElementById("globeLoading");
    if (el) el.classList.add("hidden");
}

// ── Fetch Sentiment Data ────────────────────────────────────
async function fetchSentimentData() {
    try {
        const backendUrl = (typeof API_BASE !== "undefined" && API_BASE)
            ? API_BASE
            : "https://analysofinal-backend.onrender.com";

        console.log("🌍 Fetching:", backendUrl + "/api/sentiment-globe");

        const res = await fetch(`${backendUrl}/api/sentiment-globe`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        globeData = await res.json();
        if (globeData.error && typeof globeData.error === "string") throw new Error(globeData.error);

        updateGlobe(globeData);
        updateSentimentBar(globeData);
        updateCityPanel(globeData);
        updateNewsFeed(globeData);
        hideLoader();

        console.log("✅ Loaded", globeData.total_headlines, "headlines");
    } catch (err) {
        console.error("❌ Fetch error:", err);
        showFallback();
    }
}

// ── Update Globe ────────────────────────────────────────────
function updateGlobe(data) {
    if (!globe || !data) return;

    // Hub city points
    const points = data.hubs.map(h => ({
        ...h,
        intensity: Math.min(h.news_count, 10),
    }));

    // Create arcs from each news point to its assigned city
    // Use different source coordinates based on news source for visual variety
    const sourceCoords = {
        "Economic Times": { lat: 19.076, lng: 65.0 },  // West of Mumbai
        "MoneyControl":   { lat: 25.0, lng: 68.0 },     // NW India
        "Livemint":       { lat: 15.0, lng: 66.0 },     // SW coast
    };

    const arcs = data.news_points.map(np => {
        const src = sourceCoords[np.source] || { lat: 10.0, lng: 60.0 };
        return {
            startLat: src.lat + (Math.random() - 0.5) * 3,
            startLng: src.lng + (Math.random() - 0.5) * 3,
            endLat: np.lat,
            endLng: np.lng,
            colors: [np.sentiment.color + "88", np.sentiment.color],
        };
    });

    // Labels with emoji indicators
    const labels = data.hubs.map(h => ({
        lat: h.lat,
        lng: h.lng,
        labelText: h.name,
        color: h.color,
    }));

    globe.pointsData(points);
    globe.arcsData(arcs);
    globe.labelsData(labels);
}

// ── Update Sentiment Bar ────────────────────────────────────
function updateSentimentBar(data) {
    setText("barBullish", data.summary.bullish || 0);
    setText("barBearish", data.summary.bearish || 0);
    setText("barNeutral", data.summary.neutral || 0);
    setText("barTotal", data.total_headlines || 0);
}

// ── Update City Panel ───────────────────────────────────────
function updateCityPanel(data) {
    const container = document.getElementById("cityList");
    if (!container) return;

    // Sort: most bearish cities first so user sees "where bad news comes from" at top
    const sorted = [...data.hubs].sort((a, b) => a.avg_sentiment - b.avg_sentiment);

    container.innerHTML = sorted.map(hub => {
        let sentLabel, sentClass;
        if (hub.avg_sentiment >= 0.1) { sentLabel = "Bullish"; sentClass = "bullish"; }
        else if (hub.avg_sentiment <= -0.1) { sentLabel = "Bearish"; sentClass = "bearish"; }
        else if (hub.news_count === 0) { sentLabel = "No Data"; sentClass = "inactive"; }
        else { sentLabel = "Neutral"; sentClass = "neutral"; }

        return `
            <div class="city-row">
                <span class="city-indicator" style="background:${hub.color}; box-shadow: 0 0 6px ${hub.color}50;"></span>
                <span class="city-name">${hub.name}</span>
                <span class="city-sentiment-tag ${sentClass}">${sentLabel}</span>
                <span class="city-count">${hub.news_count}</span>
            </div>
        `;
    }).join("");
}

// ── Update News Feed ────────────────────────────────────────
function updateNewsFeed(data) {
    if (!data || !data.news_points) return;
    renderNewsItems(data.news_points, currentFilter);
}

function renderNewsItems(newsPoints, filter) {
    const container = document.getElementById("newsFeedList");
    if (!container) return;

    const filtered = filter === "all"
        ? newsPoints
        : newsPoints.filter(np => np.sentiment.label.toLowerCase() === filter);

    if (filtered.length === 0) {
        container.innerHTML = `<div style="text-align:center; padding:24px; color:#64748b; font-size:0.82rem;">
            No ${filter} news at the moment.
        </div>`;
        return;
    }

    container.innerHTML = filtered.slice(0, 15).map(np => {
        const sentClass = np.sentiment.label.toLowerCase();
        const scoreClass = np.sentiment.score > 0 ? "positive" : np.sentiment.score < 0 ? "negative" : "neutral-score";
        const scoreText = np.sentiment.score > 0 ? `+${np.sentiment.score}` : `${np.sentiment.score}`;

        return `
            <a href="${np.link}" target="_blank" rel="noopener" class="news-feed-item ${sentClass}-item">
                <div class="news-item-top">
                    <span class="sentiment-badge ${sentClass}">${np.sentiment.label}</span>
                    <span class="news-city-tag">📍 ${np.city}</span>
                </div>
                <div class="news-item-title">${np.title}</div>
                <div class="news-item-footer">
                    <span class="news-source">📡 ${np.source}</span>
                    <span class="news-score ${scoreClass}">${scoreText}</span>
                </div>
            </a>
        `;
    }).join("");
}

// ── Tab Filter ──────────────────────────────────────────────
function filterNews(filter, tabEl) {
    currentFilter = filter;

    // Update active tab
    document.querySelectorAll(".feed-tab").forEach(t => t.classList.remove("active"));
    if (tabEl) tabEl.classList.add("active");

    // Re-render with filter
    if (globeData && globeData.news_points) {
        renderNewsItems(globeData.news_points, filter);
    }
}

// ── Fallback ────────────────────────────────────────────────
function showFallback() {
    hideLoader();

    const fallbackHubs = [
        { name: "Mumbai", lat: 19.076, lng: 72.878, color: "#64748b", intensity: 1, news_count: 0, avg_sentiment: 0 },
        { name: "Delhi", lat: 28.614, lng: 77.209, color: "#64748b", intensity: 1, news_count: 0, avg_sentiment: 0 },
        { name: "Bengaluru", lat: 12.972, lng: 77.595, color: "#64748b", intensity: 1, news_count: 0, avg_sentiment: 0 },
        { name: "Chennai", lat: 13.083, lng: 80.271, color: "#64748b", intensity: 1, news_count: 0, avg_sentiment: 0 },
        { name: "Hyderabad", lat: 17.385, lng: 78.487, color: "#64748b", intensity: 1, news_count: 0, avg_sentiment: 0 },
        { name: "Kolkata", lat: 22.573, lng: 88.364, color: "#64748b", intensity: 1, news_count: 0, avg_sentiment: 0 },
    ];

    if (globe) {
        globe.pointsData(fallbackHubs);
        globe.labelsData(fallbackHubs.map(h => ({ lat: h.lat, lng: h.lng, labelText: h.name, color: h.color })));
    }

    const feed = document.getElementById("newsFeedList");
    if (feed) {
        feed.innerHTML = `
            <div style="text-align:center; padding:24px; color:#94a3b8; font-size: 0.82rem; line-height:1.6;">
                <div style="font-size:1.5rem; margin-bottom:10px;">⏳</div>
                <strong>Waking up the server...</strong><br>
                <span style="color:#64748b; font-size:0.75rem;">First load may take 1-2 minutes on Render free tier.<br>The page will auto-refresh.</span>
            </div>
        `;
    }
}

// ── Utility ─────────────────────────────────────────────────
function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ── Boot ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    console.log("🌍 Globe page loaded");
    initGlobe();
    fetchSentimentData();
    setInterval(fetchSentimentData, GLOBE_REFRESH_INTERVAL);
});
