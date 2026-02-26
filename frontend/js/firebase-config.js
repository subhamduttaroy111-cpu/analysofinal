/**
 * firebase-config.js — Name + Phone Login
 * ─────────────────────────────────────────
 * Saves user data to Firebase Realtime Database (free, no billing needed).
 * Uses localStorage for session tracking on the client side.
 */

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import {
    getDatabase, ref, set, update, get, query, orderByChild, equalTo
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-database.js";

// ── Firebase project config ──────────────────────────────────
const firebaseConfig = {
    apiKey: "AIzaSyBilZw28YWuTXnMtNRmDYyTITzznOSZABs",
    authDomain: "analyso-7ee72.firebaseapp.com",
    projectId: "analyso-7ee72",
    storageBucket: "analyso-7ee72.firebasestorage.app",
    messagingSenderId: "732917431870",
    appId: "1:732917431870:web:12b7e7846d86db48eb1b6f",
    databaseURL: "https://analyso-7ee72-default-rtdb.firebaseio.com"
};

const app = initializeApp(firebaseConfig);
const db = getDatabase(app);

// ── Login Handler ────────────────────────────────────────────
window.handleLogin = async function (event) {
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

    try {
        console.log("🔍 Checking user in Firebase...");

        // Check if phone already exists
        const usersRef = ref(db, "users");
        const phoneQuery = query(usersRef, orderByChild("phone"), equalTo(phone));
        const snapshot = await get(phoneQuery);

        const timestamp = new Date().toISOString();
        const userId = "user_" + phone; // Use phone as unique identifier

        if (!snapshot.exists()) {
            // New user — save to database with set()
            console.log("📝 Saving new user...");
            await set(ref(db, `users/${userId}`), {
                name: name,
                phone: phone,
                createdAt: timestamp,
                lastLogin: timestamp
            }).catch((err) => {
                console.error("❌ Error saving new user:", err.code, err.message);
                throw err;
            });
            console.log("✅ New user saved to Firebase:", name, "- UID:", userId);
        } else {
            // Existing user — update lastLogin
            console.log("👋 Existing user found - updating lastLogin...");
            await update(ref(db, `users/${userId}`), {
                lastLogin: timestamp
            }).catch((err) => {
                console.error("❌ Error updating lastLogin:", err.code, err.message);
                throw err;
            });
            console.log("✅ Updated lastLogin for:", name);
        }

        console.log("✨ Firebase database write completed successfully");

    } catch (err) {
        console.error("🔴 Firebase database error:", {
            code: err.code,
            message: err.message,
            fullError: err
        });
        errorDiv.textContent = "Database error: " + err.message + ". Try again.";
        errorDiv.style.display = "block";
        loginBtn.disabled = false;
        loginBtn.textContent = "🚀 Enter Analyso — It's Free";
        return;
    }

    // Save session to localStorage (always, even if Firebase fails)
    localStorage.setItem("analyso_user", JSON.stringify({
        name: name,
        phone: phone,
        loggedInAt: new Date().toISOString()
    }));

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
        if (!isLoginPage) window.location.href = "login.html";
    }
})();
