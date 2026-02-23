/**
 * config.js — Backend API Configuration
 * ──────────────────────────────────────
 * On Vercel: front-end + back-end live on the SAME domain.
 * API calls are relative (/scan, /get_stock_details, /get_news).
 * No separate URL needed — leave BACKEND_URL = "".
 *
 * Only set BACKEND_URL if you host backend elsewhere.
 */

const BACKEND_URL = ""; // Same-origin on Vercel — leave empty

const API_BASE = BACKEND_URL ||
    (window.location.protocol === 'file:' ? 'http://127.0.0.1:5001' : '');

console.log(`🔗 API_BASE: ${API_BASE || '(relative / same-origin)'}`);
