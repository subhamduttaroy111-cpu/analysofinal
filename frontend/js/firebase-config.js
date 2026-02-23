/**
 * firebase-config.js — Simple Name + Phone Login (localStorage only)
 * ───────────────────────────────────────────────────────────────────
 * No external database needed. User data is stored in localStorage
 * to gate access to the app. Simple, fast, and free.
 */

// ── Login Handler ────────────────────────────────────────────
window.handleLogin = function (event) {
    event.preventDefault();

    const nameInput = document.getElementById("nameInput");
    const phoneInput = document.getElementById("phoneInput");
    const errorDiv = document.getElementById("loginError");
    const loginBtn = document.getElementById("loginBtn");

    const name = nameInput.value.trim();
    const phone = phoneInput.value.trim();

    // Validate
    if (!name || name.length < 2) {
        errorDiv.textContent = "Please enter a valid name (at least 2 characters).";
        errorDiv.style.display = "block";
        return;
    }

    if (!/^[0-9]{10}$/.test(phone)) {
        errorDiv.textContent = "Please enter a valid 10-digit phone number.";
        errorDiv.style.display = "block";
        return;
    }

    errorDiv.style.display = "none";
    loginBtn.disabled = true;
    loginBtn.textContent = "⏳ Signing in...";

    // Save session to localStorage
    localStorage.setItem("analyso_user", JSON.stringify({
        name: name,
        phone: phone,
        loggedInAt: new Date().toISOString()
    }));

    console.log("✅ User logged in:", name);

    // Redirect to main app
    window.location.href = "index.html";
};

// ── Logout ───────────────────────────────────────────────────
window.analysoLogout = function () {
    localStorage.removeItem("analyso_user");
    window.location.href = "login.html";
};

// ── Auth State Check (on page load) ─────────────────────────
(function checkAuthState() {
    const path = window.location.pathname;
    const isLoginPage = path.includes("login.html") || path.includes("/login");
    const user = localStorage.getItem("analyso_user");

    if (user) {
        const userData = JSON.parse(user);

        if (isLoginPage) {
            // Already logged in, redirect to app
            window.location.href = "index.html";
            return;
        }

        // Populate header profile on main page
        const userName = document.getElementById("userName");
        const userProfile = document.getElementById("userProfile");
        const userInitial = document.getElementById("userInitial");

        if (userName) userName.textContent = userData.name;
        if (userInitial) userInitial.textContent = userData.name.charAt(0).toUpperCase();
        if (userProfile) userProfile.style.display = "flex";

        // Show educational disclaimer once per session
        if (typeof window.showDisclaimer === "function") window.showDisclaimer();

    } else {
        // Not logged in
        if (!isLoginPage) window.location.href = "login.html";
    }
})();
