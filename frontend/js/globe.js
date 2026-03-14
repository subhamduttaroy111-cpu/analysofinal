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
            // White & Blue Theme visible earth
            .globeImageUrl("//unpkg.com/three-globe/example/img/earth-day.jpg")
            .bumpImageUrl("//unpkg.com/three-globe/example/img/earth-topology.png")
            .showAtmosphere(true)
            .atmosphereColor("#3b82f6") // Light blue
            .atmosphereAltitude(0.2)
            .backgroundColor("#f8fafc")

            // Hub Points (sharp terminal dots)
            .pointsData([])
            .pointLat("lat")
            .pointLng("lng")
            .pointColor(d => getTerminalColorCode(d.avg_sentiment, d.news_count))
            .pointAltitude(0.01)
            .pointRadius(d => 0.4 + (d.news_count * 0.1))
            .pointLabel(d => `
                <div style="background:#ffffff;border:1px solid #e2e8f0;padding:8px;font-family:'Inter',sans-serif;font-size:0.75rem;color:#0f172a;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);border-radius:8px;">
                    <div style="color:#0f172a;font-weight:800;margin-bottom:4px;border-bottom:1px solid #e2e8f0;padding-bottom:4px;">${d.name.toUpperCase()}</div>
                    <div style="color:#64748b;font-weight:600;margin-top:4px;">VOL: <span style="color:#0f172a">${d.news_count}</span></div>
                    <div style="color:#64748b;font-weight:600;">SCR: <span style="color:${getTerminalColorCode(d.avg_sentiment, d.news_count)}">${d.avg_sentiment}</span></div>
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
            .labelColor(() => "#64748b")
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
    if (count === 0) return "#94a3b8"; // Inactive
    if (score <= -0.1) return "#ef4444"; // Bearish
    if (score >= 0.1) return "#10b981"; // Bullish
    return "#64748b"; // Neutral
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
            <div style="padding:15px; font-family:'Inter',sans-serif; font-size:0.8rem; color:#64748b; line-height:1.6;">
                <div style="color:#3b82f6; font-weight:600;">> SYS.CONNECTING [API_BASE]</div>
                <div>> AWAITING HANDSHAKE...</div>
                <div style="color:#ef4444; margin-top:8px; font-weight:600;">> WARN: RENDER COLD START DETECTED</div>
                <div>> ESTIMATED WAKE TTL: 60s</div>
                <div>> RETRY LOOP ENGAGED [15000ms] <span class="loader-cursor" style="display:inline-block;width:6px;height:10px;background:#3b82f6;"></span></div>
            </div>
        `;
    }
}

function showFallback() {
    hideLoader();

    if (globe) {
        // Initial empty state until fetch completes
        globe.pointsData([]);
        globe.labelsData([]);
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

    // 3. Sector Panel
    updateSectorPanel(data);
    updateSectorPanel(data);

    // 4. Intelligence Feed Layer
    updateNewsFeed(data);
}

function updateGlobeLayer(data) {
    const points = data.hubs.map(h => ({ ...h }));

    // Sources positioned globally to show lines flying into India and other hubs
    const sources = {
        "Economic Times": { lat: 15.0, lng: 55.0 }, // Arabian Sea
        "MoneyControl":   { lat: 10.0, lng: 90.0 }, // Bay of Bengal
        "Livemint":       { lat: 35.0, lng: 70.0 }, // North West Land
        "Yahoo Fin US":   { lat: 38.0, lng: -97.0 }, // North America
        "CNBC Fin":       { lat: 45.0, lng: -10.0 }, // North Atlantic
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

// ── Market Indices Component ──────────────────────────────────
async function fetchAndDisplayMarketIndices() {
    const list = document.getElementById("marketIndicesList");
    if (!list) return;

    try {
        const backendUrl = typeof API_BASE !== "undefined" ? API_BASE : "https://analysofinal-backend.onrender.com";

        const res = await fetch(`${backendUrl}/api/market-indices`, {
           headers: { 'Accept': 'application/json' }
        });
        
        let json;
        try {
            json = await res.json();
        } catch (je) {
            throw new Error(`Parse failed: ${res.status} ${res.statusText}`);
        }
        
        if (json.status === "success" && json.data && json.data.length > 0) {
            list.innerHTML = json.data.map(idx => {
                const isBullish = idx.change >= 0;
                const hex = isBullish ? "#10b981" : "#ef4444"; // match new theme css var manually
                const sign = isBullish ? "+" : "";
                
                return `
                    <div class="risk-row" style="justify-content: space-between; padding-right: 16px;">
                        <div class="risk-cell risk-col-city">${idx.name}</div>
                        <div class="risk-cell" style="font-weight: 700; font-family: var(--font-mono);">${idx.price.toFixed(2)}</div>
                        <div class="risk-cell" style="color:${hex}; font-weight: 700; font-size: 0.75rem; text-align: right; width: 80px;">
                            ${sign}${idx.change.toFixed(2)}<br>
                            (${sign}${idx.change_pct.toFixed(2)}%)
                        </div>
                    </div>
                `;
            }).join("");
        } else {
            list.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--signal-bearish); font-size: 0.8rem;">DATA UNAVAILABLE</div>`;
        }
    } catch (err) {
        console.error("Market Indices Error:", err);
        list.innerHTML = `<div style="padding: 16px; text-align: center; color: var(--signal-bearish); font-size: 0.8rem;">CONNECTION ERROR</div>`;
    }
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
                    <span class="ticker-city" style="color:#8b5cf6;">[${n.sector.toUpperCase()}]</span>
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
    fetchAndDisplayMarketIndices();
    
    setInterval(() => { 
        if (dataLoaded) fetchSentimentData(); 
        fetchAndDisplayMarketIndices();
    }, GLOBE_REFRESH_INTERVAL);
});
