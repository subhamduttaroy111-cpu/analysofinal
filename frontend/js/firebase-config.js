/**
 * firebase-config.js — Authentication
 * ─────────────────────────────────────
 * Firebase Web SDK (CDN modules, no build step needed).
 * Config values are intentionally public — Firebase security
 * is enforced via Authentication rules and Authorized Domains,
 * not by keeping these keys secret.
 */

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import {
    getAuth, GoogleAuthProvider, signInWithPopup,
    signOut, onAuthStateChanged
}
    from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

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
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// ── Google Login ─────────────────────────────────────────────
window.googleLogin = async function () {
    try {
        await signInWithPopup(auth, provider);
        window.location.href = "index.html";
    } catch (err) {
        console.error("Login failed:", err.code);
        alert("Login failed: " + err.message);
    }
};

// ── Logout ───────────────────────────────────────────────────
window.googleLogout = async function () {
    try {
        await signOut(auth);
        window.location.href = "login.html";
    } catch (err) {
        console.error("Logout failed:", err.message);
    }
};

// ── Auth State Observer ──────────────────────────────────────
onAuthStateChanged(auth, (user) => {
    const path = window.location.pathname;
    const isLoginPage = path.includes("login.html") || path.includes("/login");
    const isIndexPage = !isLoginPage;

    if (user) {
        if (isLoginPage) {
            window.location.href = "index.html";
            return;
        }

        // Populate header profile
        const userImg = document.getElementById("userImg");
        const userName = document.getElementById("userName");
        const userProfile = document.getElementById("userProfile");

        if (userImg) userImg.src = user.photoURL || "";
        if (userName) userName.textContent = user.displayName || user.email;
        if (userProfile) userProfile.style.display = "flex";

        // Show educational disclaimer once per session
        if (typeof window.showDisclaimer === "function") window.showDisclaimer();

    } else {
        if (isIndexPage) window.location.href = "login.html";
    }
});
