/**
 * globe.js — Terminal V2 Logic
 * Aesthetics: Dense, Monospace, High-Contrast Red/Green
 */

// ── Config ──────────────────────────────────────────────────
const GLOBE_REFRESH_INTERVAL = 300000; // 5 minutes
const INDIA_CENTER = { lat: 21.0, lng: 78.9, altitude: 2.2 };
const RETRY_INTERVAL = 15000; 

// ── State ───────────────────────────────────────────────────
let globe = null;
let globeData = null;
let currentFilter = "all";
let dataLoaded = false;
let retryTimer = null;

// ── Initialize Globe ────────────────────────────────────────
function initGlobe() {
    const container = document.getElementById("globeViz");
    if (!container) return;

    try {
        globe = Globe()(container)
            // High contrast visible earth instead of pure darkness
            .globeImageUrl("//unpkg.com/three-globe/example/img/earth-blue-marble.jpg")
            .bumpImageUrl("//unpkg.com/three-globe/example/img/earth-topology.png")
            .backgroundImageUrl("//unpkg.com/three-globe/example/img/night-sky.png")
            .showAtmosphere(true)
            .atmosphereColor("#1F2937") // Hard terminal gray ring
            .atmosphereAltitude(0.15)
            .backgroundColor("#02040A")

            // Hub Points (sharp terminal dots)
            .pointsData([])
            .pointLat("lat")
            .pointLng("lng")
            .pointColor(d => getTerminalColorCode(d.avg_sentiment, d.news_count))
            .pointAltitude(0.01)
            .pointRadius(d => 0.4 + (d.news_count * 0.1))
            .pointLabel(d => `
                <div style="background:#090E17;border:1px solid #1F2937;padding:8px;font-family:'Roboto Mono',monospace;font-size:0.7rem;color:#E2E8F0;">
                    <div style="color:#FFF;font-weight:700;margin-bottom:4px;border-bottom:1px solid #1F2937;padding-bottom:2px;">[ ${d.name.toUpperCase()} ]</div>
                    <div>VOL: ${d.news_count} SCANS</div>
                    <div>SCR: <span style="color:${getTerminalColorCode(d.avg_sentiment, d.news_count)}">${d.avg_sentiment}</span></div>
                </div>
            `)

            // Arcs (news flow) - High contrast laser pulses
            .arcsData([])
            .arcStartLat("startLat")
            .arcStartLng("startLng")
            .arcEndLat("endLat")
            .arcEndLng("endLng")
            .arcColor("colors")
            .arcAltitude(0.15)
            .arcStroke(0.8)
            .arcDashLength(0.15) // Short laser pulse
            .arcDashGap(1.5)     // Large gap so it's a single pulse flying
            .arcDashInitialGap(() => Math.random()) // Random start timing
            .arcDashAnimateTime(2000) // Speed of the pulse

            // Impact Rings (ripples when news hits a city)
            .ringsData([])
            .ringLat("lat")
            .ringLng("lng")
            .ringColor("color")
            .ringMaxRadius(3)
            .ringPropagationSpeed(1.5)
            .ringRepeatPeriod(1000)

            // City Labels (Monospaced)
            .labelsData([])
            .labelLat("lat")
            .labelLng("lng")
            .labelText("labelText")
            .labelSize(1.5)
            .labelDotRadius(0.3)
            .labelColor(() => "#6B7280")
            .labelResolution(3);

        globe.controls().autoRotate = true;
        globe.controls().autoRotateSpeed = 0.5;
        globe.controls().enableDamping = true;
        globe.controls().dampingFactor = 0.1;

        // Custom lighting for terminal look (less diffuse)
        if (globe.scene && globe.scene()) {
            const ambientLight = globe.scene().children.find(o => o.type === 'AmbientLight');
            if (ambientLight) ambientLight.intensity = 0.4;
        }

        setTimeout(() => {
            globe.pointOfView(INDIA_CENTER, 1500);
        }, 300);

        window.addEventListener("resize", () => {
            globe.width(window.innerWidth);
            globe.height(window.innerHeight);
        });

    } catch (err) {
        console.error("SYS ERR: GLOBE INIT", err);
        hideLoader();
    }
}

// Map sentiment float to hard terminal color hex
function getTerminalColorCode(score, count) {
    if (count === 0) return "#374151"; // Inactive
    if (score <= -0.1) return "#FF2A2A"; // Bearish (Red)
    if (score >= 0.1) return "#00FF55"; // Bullish (Green)
    return "#AAAAAA"; // Neutral (Gray)
}

function getTerminalLabel(score, count) {
    if (count === 0) return "N/A    ";
    if (score <= -0.1) return "BEARISH";
    if (score >= 0.1) return "BULLISH";
    return "NEUTRAL";
}

// ── Loader / Fallback ───────────────────────────────────────
function hideLoader() {
    const el = document.getElementById("globeLoading");
    if (el) el.classList.add("hidden");
}

function showLoaderStatus() {
    const feed = document.getElementById("newsFeedList");
    if (feed) {
        feed.innerHTML = `
            <div style="padding:15px; font-family:'Roboto Mono',monospace; font-size:0.75rem; color:#6B7280; line-height:1.6;">
                <div style="color:#3B82F6">> SYS.CONNECTING [API_BASE]</div>
                <div>> AWAITING HANDSHAKE...</div>
                <div style="color:#FF2A2A; margin-top:8px;">> WARN: RENDER COLD START DETECTED</div>
                <div>> ESTIMATED WAKE TTL: 60s</div>
                <div>> RETRY LOOP ENGAGED [15000ms] <span class="loader-cursor" style="display:inline-block;width:6px;height:10px;background:#3B82F6;"></span></div>
            </div>
        `;
    }
}

function showFallback() {
    hideLoader();

    const hubs = [
        { name: "Mumbai", lat: 19.076, lng: 72.878, color: "#374151", news_count: 0, avg_sentiment: 0 },
        { name: "Delhi", lat: 28.614, lng: 77.209, color: "#374151", news_count: 0, avg_sentiment: 0 },
        { name: "Bengaluru", lat: 12.972, lng: 77.595, color: "#374151", news_count: 0, avg_sentiment: 0 },
        { name: "Chennai", lat: 13.083, lng: 80.271, color: "#374151", news_count: 0, avg_sentiment: 0 },
        { name: "Hyderabad", lat: 17.385, lng: 78.487, color: "#374151", news_count: 0, avg_sentiment: 0 },
        { name: "Kolkata", lat: 22.573, lng: 88.364, color: "#374151", news_count: 0, avg_sentiment: 0 },
    ];

    if (globe) {
        globe.pointsData(hubs);
        globe.labelsData(hubs.map(h => ({ lat: h.lat, lng: h.lng, labelText: h.name })));
    }
}

// ── Fetch Sentiment Data ────────────────────────────────────
async function fetchSentimentData() {
    try {
        const backendUrl = (typeof API_BASE !== "undefined" && API_BASE)
            ? API_BASE
            : "https://analysofinal-backend.onrender.com";

        if (!dataLoaded) showLoaderStatus();

        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 45000);

        const res = await fetch(`${backendUrl}/api/sentiment-globe`, { signal: controller.signal });
        clearTimeout(timeout);
        if (!res.ok) throw new Error(`HTTP_ERR_${res.status}`);

        globeData = await res.json();
        if (globeData.error) throw new Error(globeData.error);

        dataLoaded = true;
        if (retryTimer) { clearInterval(retryTimer); retryTimer = null; }

        updateUIComponents(globeData);
        hideLoader();

    } catch (err) {
        console.error("SYS ERR: FETCH_FAIL", err.message);
        if (!dataLoaded) {
            showLoaderStatus();
            if (!retryTimer) {
                retryTimer = setInterval(fetchSentimentData, RETRY_INTERVAL);
            }
        }
    }
}

// ── Core UI Updates ─────────────────────────────────────────
function updateUIComponents(data) {
    if (!globe || !data) return;

    // 1. Globe Layer
    updateGlobeLayer(data);

    // 2. Status Bar Layer
    setText("statTotal", data.total_headlines || 0);
    setText("statBullish", data.summary.bullish || 0);
    setText("statBearish", data.summary.bearish || 0);
    setText("statNeutral", data.summary.neutral || 0);
    
    const dt = new Date(data.timestamp);
    setText("statTime", `TIME: ${dt.toLocaleTimeString("en-IN", { hour12: false })} IST`);

    // 3. Risk Panel Layer
    updateRiskPanel(data);
    updateSectorPanel(data);

    // 4. Intelligence Feed Layer
    updateNewsFeed(data);
}

function updateGlobeLayer(data) {
    const points = data.hubs.map(h => ({ ...h }));

    // Sources positioned globally to show lines flying into India
    const sources = {
        "Economic Times": { lat: 15.0, lng: 55.0 }, // Arabian Sea
        "MoneyControl":   { lat: 10.0, lng: 90.0 }, // Bay of Bengal
        "Livemint":       { lat: 35.0, lng: 70.0 }, // North West Land
    };

    const arcs = data.news_points.map(np => {
        const src = sources[np.source] || { lat: 0.0, lng: 75.0 };
        const hColor = getTerminalColorCode(np.sentiment.score, 1);
        
        return {
            startLat: src.lat + (Math.random() - 0.5) * 5,
            startLng: src.lng + (Math.random() - 0.5) * 5,
            endLat: np.lat,
            endLng: np.lng,
            colors: [hColor + "00", hColor], // Transparent trail fading into solid head
        };
    });

    const labels = data.hubs.map(h => ({
        lat: h.lat,
        lng: h.lng,
        labelText: h.name
    }));

    // Generate ripples for the most active cities
    const rings = data.hubs.filter(h => h.news_count > 0).map(h => ({
        lat: h.lat,
        lng: h.lng,
        color: () => getTerminalColorCode(h.avg_sentiment, 1) + "AA"
    }));

    globe.pointsData(points);
    globe.arcsData(arcs);
    globe.labelsData(labels);
    globe.ringsData(rings);
}

// ── Risk Component ──────────────────────────────────────────
function updateRiskPanel(data) {
    const list = document.getElementById("cityList");
    if (!list) return;

    // Hard sort: High risk (Bearish/Lowest score) at the very top
    const sorted = [...data.hubs].sort((a, b) => a.avg_sentiment - b.avg_sentiment);

    list.innerHTML = sorted.map(hub => {
        const hex = getTerminalColorCode(hub.avg_sentiment, hub.news_count);
        const lbl = getTerminalLabel(hub.avg_sentiment, hub.news_count);
        
        // Intensity meter calculation
        const intensity = Math.min((hub.news_count / 15) * 100, 100);

        return `
            <div class="risk-row">
                <div class="risk-cell risk-col-city">${hub.name.toUpperCase()}</div>
                <div class="risk-cell risk-col-score" style="color:${hex}">${lbl}</div>
                <div class="risk-cell risk-col-meter">
                    <div class="meter-bg">
                        <div class="meter-fill" style="width:${intensity}%; background:${hex};"></div>
                    </div>
                </div>
            </div>
        `;
    }).join("");
}

function updateSectorPanel(data) {
    const list = document.getElementById("sectorList");
    if (!list || !data.sectors) return;

    // Sort: High risk sectors at the top
    const sorted = [...data.sectors].sort((a, b) => a.avg_sentiment - b.avg_sentiment);

    list.innerHTML = sorted.map(sec => {
        const hex = getTerminalColorCode(sec.avg_sentiment, sec.news_count);
        const lbl = getTerminalLabel(sec.avg_sentiment, sec.news_count);
        
        // Intensity meter calculation (sectors might have more news)
        const intensity = Math.min((sec.news_count / 20) * 100, 100);

        return `
            <div class="risk-row">
                <div class="risk-cell risk-col-city" style="font-size:0.65rem;">${sec.name.toUpperCase()}</div>
                <div class="risk-cell risk-col-score" style="color:${hex}">${lbl}</div>
                <div class="risk-cell risk-col-meter">
                    <div class="meter-bg">
                        <div class="meter-fill" style="width:${intensity}%; background:${hex};"></div>
                    </div>
                </div>
            </div>
        `;
    }).join("");
}


// ── Intelligence Component ──────────────────────────────────
function updateNewsFeed(data) {
    if (!data || !data.news_points) return;
    renderFeedItems(data.news_points, currentFilter);
}

function renderFeedItems(news, filter) {
    const list = document.getElementById("newsFeedList");
    if (!list) return;

    const filtered = filter === "all" ? news : news.filter(n => n.sentiment.label.toLowerCase() === filter);

    if (filtered.length === 0) {
        list.innerHTML = `<div style="padding:15px;font-family:'Roboto Mono',monospace;color:#6B7280;font-size:0.75rem;">> NO MATCHING SIGNALS IN BUFFER.</div>`;
        return;
    }

    list.innerHTML = filtered.slice(0, 50).map(n => {
        const tag = n.sentiment.label.toUpperCase();
        return `
            <a href="${n.link}" target="_blank" rel="noopener" class="ticker-item">
                <div class="ticker-meta">
                    <span class="ticker-time">[NEW]</span>
                    <span class="ticker-sentiment ${tag}">${tag}</span>
                    <span class="ticker-source">${n.source.toUpperCase().substr(0,10)}</span>
                    <span class="ticker-city">&lt;${n.city.substring(0,3).toUpperCase()}&gt;</span>
                    <span class="ticker-city" style="color:#A78BFA;">[${n.sector.toUpperCase()}]</span>
                </div>
                <div class="ticker-headline">${n.title}</div>
            </a>
        `;
    }).join("");
}

// Filter Action
function filterNews(filter, btn) {
    currentFilter = filter;
    document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    if (btn) btn.classList.add("active");
    if (globeData && globeData.news_points) {
        renderFeedItems(globeData.news_points, filter);
    }
}

// ── Utility ─────────────────────────────────────────────────
function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

// ── Boot ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    initGlobe();
    showFallback();
    
    // Quick blink loader off, switch to panel loader
    setTimeout(hideLoader, 500);
    
    fetchSentimentData();
    setInterval(() => { if (dataLoaded) fetchSentimentData(); }, GLOBE_REFRESH_INTERVAL);
});
