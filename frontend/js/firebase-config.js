/**
 * firebase-config.js — Name + Phone Authentication with Firestore
 * ─────────────────────────────────────────────────────────────────
 * Uses Firestore to save user data (name + phone).
 * Uses localStorage to track login session (no Google Auth needed).
 */

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import {
    getFirestore, collection, addDoc, query, where, getDocs, serverTimestamp
} from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";

// ── Firebase project config ──────────────────────────────────
const firebaseConfig = {
    apiKey: "AIzaSyBilZw28YWuTXnMtNRmDYyTITzznOSZABs",
    authDomain: "analyso-7ee72.firebaseapp.com",
    projectId: "analyso-7ee72",
    storageBucket: "analyso-7ee72.firebasestorage.app",
    messagingSenderId: "732917431870",
    appId: "1:732917431870:web:12b7e7846d86db48eb1b6f"
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
        // Check if user already exists in Firestore
        const usersRef = collection(db, "users");
        const q = query(usersRef, where("phone", "==", phone));
        const querySnapshot = await getDocs(q);

        if (querySnapshot.empty) {
            // New user — save to Firestore
            await addDoc(usersRef, {
                name: name,
                phone: phone,
                createdAt: serverTimestamp(),
                lastLogin: serverTimestamp()
            });
            console.log("✅ New user saved to Firestore:", name);
        } else {
            console.log("👋 Existing user logged in:", name);
        }

        // Save session to localStorage
        localStorage.setItem("analyso_user", JSON.stringify({
            name: name,
            phone: phone,
            loggedInAt: new Date().toISOString()
        }));

        // Redirect to main app
        window.location.href = "index.html";

    } catch (err) {
        console.error("Login error:", err);
        errorDiv.textContent = "Something went wrong. Please try again.";
        errorDiv.style.display = "block";
        loginBtn.disabled = false;
        loginBtn.textContent = "🚀 Enter Analyso — It's Free";
    }
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
