/**
 * firebase-config.js — Name + Phone Login
 * ─────────────────────────────────────────
 * Saves user data to Firebase Realtime Database (free, no billing needed).
 * Uses localStorage for session tracking on the client side.
 */

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import {
    getFirestore, collection, addDoc, query, where, getDocs
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";

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
const db = getFirestore(app);

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
        // Save user to Firebase Firestore
        const usersRef = collection(db, "users");

        // Check if phone already exists
        const phoneQuery = query(usersRef, where("phone", "==", phone));
        const snapshot = await getDocs(phoneQuery);

        if (snapshot.empty) {
            // New user — save to database
            await addDoc(usersRef, {
                name: name,
                phone: phone,
                createdAt: new Date().toISOString(),
                lastLogin: new Date().toISOString()
            });
            console.log("✅ New user saved to Firestore:", name);
        } else {
            console.log("👋 Existing user logged in:", name);
        }

    } catch (err) {
        // If Firebase fails, log the error but still let user in (localStorage will handle session)
        console.error("⚠️ Firebase save failed (Check Firestore rules):", err);
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
