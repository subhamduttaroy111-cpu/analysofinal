/**
 * auth.js — Session-based auth (uses localStorage, set by firebase-config.js)
 * ─────────────────────────────────────────────────────────────────────────────
 * This module checks the login state on index.html and updates the UI.
 * The actual login/save logic lives in firebase-config.js.
 */

// DOM Elements
const logoutBtn = document.getElementById("logoutBtn");
const userProfile = document.getElementById("userProfile");
const userName = document.getElementById("userName");
const userInitial = document.getElementById("userInitial");

// Logout Function
if (logoutBtn) {
    logoutBtn.addEventListener("click", () => {
        localStorage.removeItem("analyso_user");
        window.location.href = "login.html";
    });
}

// Auth State Check
(function checkAuth() {
    const isLoginPage = window.location.pathname.includes("login.html");
    const isIndexPage = window.location.pathname.includes("index.html") || window.location.pathname.endsWith("/");
    const userData = localStorage.getItem("analyso_user");

    if (userData) {
        const user = JSON.parse(userData);

        if (isLoginPage) {
            window.location.href = "index.html";
        }

        // Update User Profile UI
        if (userName) userName.textContent = user.name;
        if (userInitial) userInitial.textContent = user.name.charAt(0).toUpperCase();
        if (userProfile) userProfile.style.display = "flex";
        if (logoutBtn) logoutBtn.style.display = "block";

    } else {
        // User is not logged in
        if (isIndexPage) {
            window.location.href = "login.html";
        }
    }
})();
