// ⚙️ API URL is configured in js/config.js (load that file first in HTML)
//    BACKEND_URL="" → same-origin (local Flask)
//    BACKEND_URL="https://..." → Render backend for Vercel deploy
//    API_BASE is set by config.js; fall back for safety:
const API_URL = (typeof API_BASE !== 'undefined') ? API_BASE :
    (window.location.protocol === 'file:' ? 'http://127.0.0.1:5001' : '');


let STOCKS = [];
let WIN_RATE = null;

async function runScan() {
    const loader = document.getElementById("loader");
    const grid = document.getElementById("grid");

    loader.style.display = "flex";
    grid.innerHTML = "";

    // Start cycling loading messages if available
    if (typeof startLoadingAnimation === 'function') {
        startLoadingAnimation();
    }

    const mode = document.getElementById("mode").value;

    try {
        const res = await fetch(`${API_URL}/scan`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode })
        });
        const json = await res.json();

        if (typeof stopLoadingAnimation === 'function') {
            stopLoadingAnimation();
        }
        loader.style.display = "none";

        if (json.status === "success") {
            STOCKS = json.data;
            WIN_RATE = json.win_rate;
            renderCards(STOCKS);
        }
    } catch (error) {
        if (typeof stopLoadingAnimation === 'function') {
            stopLoadingAnimation();
        }
        loader.style.display = "none";
        grid.innerHTML = `
            <div style="grid-column: 1/-1; text-align:center; padding:60px;">
                <h3 style="color:var(--danger); margin-bottom:12px;">❌ Cannot connect to backend</h3>
                <p style="color:var(--muted);">Make sure Python backend is running:</p>
                <code style="background:var(--bg); padding:12px 20px; border-radius:8px; display:inline-block; margin-top:12px;">python server.py</code>
                <br><br>
                <small style="color:var(--muted); font-size: 0.8em;">(Open 'run_app.bat' to start)</small>
            </div>
        `;
    }
}

function renderCards(data) {
    const grid = document.getElementById("grid");

    if (data.length === 0) {
        grid.innerHTML = "<p style='text-align:center; padding:60px; color:var(--muted);'>No setups found. Try different timeframe.</p>";
        return;
    }

    grid.innerHTML = "";

    data.forEach((s, i) => {
        const cls = s.bias.includes("BULL") ? "bull" : s.bias.includes("BEAR") ? "bear" : "neutral";

        const range = s.execution.target1 - s.execution.sl;
        const currentPos = s.ltp - s.execution.sl;
        const percentage = Math.max(0, Math.min(100, (currentPos / range) * 100));

        const card = document.createElement('div');
        card.className = 'card';
        card.onclick = () => openModal(i);

        card.innerHTML = `
            <div>
                <h3>${s.symbol}</h3>
                <span class="badge ${cls}">${s.bias}</span>
                ${s.score >= 85 ? '<span class="badge high-conviction">🔥 HIGH CONVICTION</span>' : ''}
            </div>
            
            <div class="price">₹${s.ltp.toLocaleString()}</div>
            
            <div class="indicators">
                <span class="indicator-badge rsi">RSI ${s.indicators.rsi}</span>
                <span class="indicator-badge macd-${s.indicators.macd.toLowerCase()}">${s.indicators.macd}</span>
                <span class="indicator-badge volume">${s.indicators.volume} VOL</span>
            </div>
            
            <div class="price-range">
                <div class="range-labels">
                    <span class="sl">SL</span>
                    <span class="entry">ENTRY</span>
                    <span class="target">TARGET</span>
                </div>
                <div class="range-bar">
                    <div class="current-price-marker" style="left: ${percentage}%"></div>
                </div>
                <div class="range-values">
                    <div><span>SL</span><span style="color:var(--danger)">₹${s.execution.sl}</span></div>
                    <div><span>ENTRY</span><span style="color:var(--warning)">₹${s.execution.entry}</span></div>
                    <div><span>TGT</span><span style="color:var(--success)">₹${s.execution.target1}</span></div>
                </div>
            </div>
            
            <div class="risk-metrics">
                <div class="risk-row">
                    <span class="risk-label">Risk per trade</span>
                    <span class="risk-value danger">-₹${(s.ltp - s.execution.sl).toFixed(2)} (${((s.ltp - s.execution.sl) / s.ltp * 100).toFixed(1)}%)</span>
                </div>
                <div class="risk-row">
                    <span class="risk-label">Potential profit</span>
                    <span class="risk-value success">+₹${(s.execution.target1 - s.ltp).toFixed(2)} (${((s.execution.target1 - s.ltp) / s.ltp * 100).toFixed(1)}%)</span>
                </div>
                <div class="risk-row">
                    <span class="risk-label">Distance to Target</span>
                    <span class="risk-value">₹${(s.execution.target1 - s.ltp).toFixed(2)}</span>
                </div>
                <div class="risk-row">
                    <span class="risk-label">Distance to SL</span>
                    <span class="risk-value">₹${(s.ltp - s.execution.sl).toFixed(2)}</span>
                </div>
            </div>
            
            <div class="card-footer">
                <div class="score-display">
                    <span>SCORE</span>
                    <div class="score-bar"><div class="score-fill" style="width:${s.score}%"></div></div>
                    <span>${s.score}%</span>
                </div>
                ${WIN_RATE && WIN_RATE.available ? `<div class="win-rate-display">📊 ${WIN_RATE.win_rate}% Win</div>` : ''}
                <div class="rr-display">R:R ${s.execution.rr_ratio}:1</div>
            </div>
        `;

        grid.appendChild(card);
    });
}

async function openModal(i) {
    const s = STOCKS[i];
    const modal = document.getElementById("modal");
    const mTitle = document.getElementById("mTitle");
    const mReasons = document.getElementById("mReasons");
    const mSector = document.getElementById("mSector");
    const mHigh = document.getElementById("mHigh");
    const mLow = document.getElementById("mLow");
    const mCap = document.getElementById("mCap");

    // Store stock symbol for news
    currentStockSymbol = s.symbol;

    // Reset news section
    document.getElementById("newsContent").style.display = "none";
    document.getElementById("newsLoader").style.display = "none";
    document.getElementById("newsError").style.display = "none";
    document.getElementById("viewNewsBtn").textContent = "View Top 5 News";
    document.getElementById("viewNewsBtn").disabled = false;

    modal.style.display = "flex";
    mTitle.innerText = s.symbol;
    loadChart(s.symbol);
    mReasons.innerHTML = s.reason.map(r => `<li>${r}</li>`).join("");

    try {
        const res = await fetch(`${API_URL}/get_stock_details`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol: s.symbol })
        });
        const json = await res.json();

        if (json.status === "success") {
            // Render Fundamentals
            mSector.innerText = json.fundamentals.sector;
            mHigh.innerText = json.fundamentals.high52 !== 'N/A' ? `₹${json.fundamentals.high52}` : 'N/A';
            mLow.innerText = json.fundamentals.low52 !== 'N/A' ? `₹${json.fundamentals.low52}` : 'N/A';
            mCap.innerText = json.fundamentals.marketCap !== 'N/A' ? formatCap(json.fundamentals.marketCap) : 'N/A';

            // Render AI Probability Matrix
            if (json.ai_analysis) {
                renderAIMatrix(json.ai_analysis);
            }
        }
    } catch (e) {
        console.error("Details fetch error:", e);
        document.getElementById("aiMatrix").innerHTML = `<p style="color:var(--danger)">⚠️ Failed to align AI models</p>`;
    }
}

function renderAIMatrix(analysis) {
    const container = document.getElementById("aiMatrix");
    container.innerHTML = "";

    const modes = [
        { key: "INTRADAY", label: "🚀 Intraday (15m)" },
        { key: "SWING", label: "🌊 Swing (Daily)" },
        { key: "LONG_TERM", label: "💎 Long Term" }
    ];

    modes.forEach(m => {
        const data = analysis[m.key];
        if (!data) return;

        if (!data.available) {
            container.innerHTML += `
                <div class="ai-row">
                    <div class="ai-header">
                        <span>${m.label}</span>
                        <span style="color:var(--muted)">Insufficient Data</span>
                    </div>
                </div>`;
            return;
        }

        const score = data.score;
        const signal = data.signal;
        const isBullish = signal === "BULLISH";
        const isBearish = signal === "BEARISH";

        const color = isBullish ? "var(--success)" : isBearish ? "var(--danger)" : "var(--warning)";
        const width = score; // Score is 0-100

        // Parse reasons to bullets
        const reasonHTML = data.reasons && data.reasons.length > 0
            ? `<div class="ai-reasons"><ul>${data.reasons.map(r => `<li>${r}</li>`).join('')}</ul></div>`
            : '';

        const html = `
            <div class="ai-row">
                <div class="ai-header">
                    <span>${m.label}</span>
                    <span style="color:${color}">${score.toFixed(0)}% ${signal}</span>
                </div>
                <div class="ai-progress-bg">
                    <div class="ai-progress-fill" style="width: ${width}%; background: ${color}"></div>
                </div>
                ${reasonHTML}
            </div>
        `;
        container.innerHTML += html;
    });
}

function formatCap(cap) {
    const cr = cap / 10000000;
    return cr >= 100000 ? `₹${(cr / 100000).toFixed(2)}L Cr` : `₹${cr.toFixed(0)} Cr`;
}

function loadChart(symbol) {
    const mode = document.getElementById("mode").value;
    const tv_iframe = document.getElementById("tv_iframe");
    let interval = mode === "SWING" ? "D" : mode === "LONG_TERM" ? "W" : "15";

    tv_iframe.src = `https://www.tradingview.com/widgetembed/?symbol=${symbol}&exchange=NSE&interval=${interval}&theme=light&timezone=Asia%2FKolkata&studies=RSI@tv-basicstudies%1FMACD@tv-basicstudies`;
}

function closeModal(e) {
    const modal = document.getElementById("modal");
    if (!e || e.target.id === "modal" || e.target.classList.contains("close-btn")) modal.style.display = "none";
}

// ============== NEWS FUNCTIONALITY ==============
let currentStockSymbol = null;

async function viewStockNews() {
    if (!currentStockSymbol) return;

    const loader = document.getElementById("newsLoader");
    const content = document.getElementById("newsContent");
    const error = document.getElementById("newsError");
    const btn = document.getElementById("viewNewsBtn");

    // Show loading state
    loader.style.display = "flex";
    content.style.display = "none";
    error.style.display = "none";
    btn.disabled = true;
    btn.textContent = "Loading...";

    try {
        const res = await fetch(`${API_URL}/get_news`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ symbol: currentStockSymbol })
        });

        const data = await res.json();

        loader.style.display = "none";
        btn.disabled = false;
        btn.textContent = "Refresh News";

        if (data.error || !data.news || data.news.length === 0) {
            error.style.display = "block";
            error.innerHTML = `
                <strong>📰 No News Found</strong>
                <p>Unable to fetch recent news for this stock. Try again later.</p>
            `;
            return;
        }

        // Render news
        const newsList = document.getElementById("newsList");
        newsList.innerHTML = data.news.map((item, index) => `
            <div class="news-item">
                <div class="news-number">${index + 1}</div>
                <div class="news-content">
                    <a href="${item.link}" target="_blank" class="news-title">${item.title}</a>
                    <div class="news-meta">
                        <span class="news-publisher">${item.publisher}</span>
                        <span class="news-time">${item.published}</span>
                    </div>
                </div>
            </div>
        `).join("");

        content.style.display = "block";

    } catch (e) {
        loader.style.display = "none";
        error.style.display = "block";
        error.innerHTML = `<strong>❌ Error</strong><p>Failed to load news. Please try again.</p>`;
        btn.disabled = false;
        btn.textContent = "Retry";
    }
}

