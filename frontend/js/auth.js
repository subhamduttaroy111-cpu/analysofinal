/**
 * auth.js — Session-based auth (uses localStorage, set by firebase-config.js)
 * ─────────────────────────────────────────────────────────────────────────────
 * This module checks the login state on index.html and updates the UI.
 * The actual login/save logic lives in firebase-config.js.
 *
 * NEW: No longer redirects to login.html. Instead, sets guest mode
 * and lets firebase-config.js handle showing the login modal.
 */

// DOM Elements
const logoutBtn = document.getElementById("logoutBtn");
const userProfile = document.getElementById("userProfile");
const userName = document.getElementById("userName");
const userInitial = document.getElementById("userInitial");

// Logout Function
if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
        if (typeof window.analysoLogout === 'function') {
            window.analysoLogout();
        } else {
            localStorage.removeItem("analyso_user");
            window.isGuestMode = true;
            // Show login modal instead of redirecting
            if (typeof window.showLoginModal === 'function') {
                window.showLoginModal();
            }
        }
    });
}

// Auth State Check (lightweight — firebase-config.js handles the heavy lifting)
(function checkAuth() {
    const isLoginPage = window.location.pathname.includes("login.html");
    const isIndexPage = window.location.pathname === "/" || window.location.pathname.endsWith("/index.html") || window.location.pathname === "";
    const userData = localStorage.getItem("analyso_user");

    if (userData) {
        const user = JSON.parse(userData);

        if (isLoginPage) {
            window.location.href = "/";
        }

        // Update User Profile UI
        if (userName) userName.textContent = user.name;
        if (userInitial) userInitial.textContent = user.name.charAt(0).toUpperCase();
        if (userProfile) userProfile.style.display = "flex";
        if (logoutBtn) logoutBtn.style.display = "block";

        window.isGuestMode = false;

    } else {
        // User is not logged in — enter Guest Mode (no redirect)
        window.isGuestMode = true;
        // firebase-config.js will show the login modal via onAuthStateChanged
    }
})();
