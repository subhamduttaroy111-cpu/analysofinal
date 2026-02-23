/**
 * config.js — Backend API Configuration
 * ──────────────────────────────────────
 * BACKEND_URL: Your Render backend URL.
 * Set this to your Render service URL or custom api subdomain.
 * Example: "https://api.yourdomain.com"
 *          "https://analysofinal-backend.onrender.com"
 */

// ✅ SET THIS to your Render backend URL after deploying:
const BACKEND_URL = "https://analysofinal-backend.onrender.com";

const API_BASE = BACKEND_URL ||
    (window.location.protocol === 'file:' ? 'http://127.0.0.1:5001' : '');

console.log(`🔗 API_BASE: ${API_BASE || '(relative / same-origin)'}`);
