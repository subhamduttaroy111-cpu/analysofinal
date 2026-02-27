/**
 * firebase-config.js — Email + Password Login
 * ─────────────────────────────────────────
 * Saves user data to Firebase Realtime Database.
 * Uses Firebase Auth for secure session tracking.
 */

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth, createUserWithEmailAndPassword, signInWithEmailAndPassword, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";
import { getFirestore, doc, setDoc, updateDoc } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";

// ── Firebase project config ──────────────────────────────────
const firebaseConfig = {
    apiKey: "AIzaSyBf6lC9Vu8JH4J_6zZsgoKhHMSTkn1RzGw",
    authDomain: "analysodb.firebaseapp.com",
    projectId: "analysodb",
    storageBucket: "analysodb.firebasestorage.app",
    messagingSenderId: "1000715298283",
    appId: "1:1000715298283:web:50326294e98d1a10e225ae",
    measurementId: "G-KP5Q2G5QX9"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

// ── UI Toggle Logic (Segmented Tabs) ─────────────────────────
let isSignupMode = false;
const loginBtn = document.getElementById("loginBtn");
const authTabs = document.getElementById("authTabs");
const tabLogin = document.getElementById("tabLogin");
const tabSignup = document.getElementById("tabSignup");
const errorDiv = document.getElementById("loginError");

window.setAuthMode = function (signupActive) {
    isSignupMode = signupActive;

    // Clear any previous errors when switching tabs
    if (errorDiv) {
        errorDiv.style.display = "none";
        errorDiv.textContent = "";
    }

    if (isSignupMode) {
        // Switch to Create Account mode
        if (authTabs) authTabs.setAttribute("data-mode", "signup");
        if (tabSignup) tabSignup.classList.add("active");
        if (tabLogin) tabLogin.classList.remove("active");
        if (loginBtn) loginBtn.innerHTML = "🚀 Create Account";
    } else {
        // Switch to Sign In mode
        if (authTabs) authTabs.setAttribute("data-mode", "login");
        if (tabLogin) tabLogin.classList.add("active");
        if (tabSignup) tabSignup.classList.remove("active");
        if (loginBtn) loginBtn.innerHTML = "🚀 Sign In";
    }
};

// ── Login / Signup Handler ───────────────────────────────────
const loginForm = document.getElementById("loginForm");
if (loginForm) {
    loginForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const emailInput = document.getElementById("emailInput");
        const passwordInput = document.getElementById("passwordInput");
        const errorDiv = document.getElementById("loginError");

        const email = emailInput.value.trim();
        const password = passwordInput.value.trim();

        if (errorDiv) errorDiv.style.display = "none";
        if (loginBtn) {
            loginBtn.disabled = true;
            loginBtn.textContent = "⏳ Processing...";
        }

        try {
            if (isSignupMode) {
                // SIGNUP
                const userCredential = await createUserWithEmailAndPassword(auth, email, password);
                const user = userCredential.user;

                // Save to Firestore using setDoc()
                const timestamp = new Date().toISOString();
                try {
                    await setDoc(doc(db, "users", user.uid), {
                        email: user.email,
                        uid: user.uid,
                        createdAt: timestamp,
                        lastLogin: timestamp
                    });
                } catch (dbErr) {
                    console.error("Database save error:", dbErr);
                    throw new Error("Account created, but Database write failed: " + dbErr.message);
                }

            } else {
                // SIGNIN
                const userCredential = await signInWithEmailAndPassword(auth, email, password);
                const user = userCredential.user;

                // Save last login to Firestore using updateDoc()
                const timestamp = new Date().toISOString();
                try {
                    await updateDoc(doc(db, "users", user.uid), {
                        lastLogin: timestamp
                    });
                } catch (dbErr) {
                    console.error("Database update error:", dbErr);
                    throw new Error("Logged in, but Database update failed: " + dbErr.message);
                }
            }

            // Redirect to main app ONLY IF DATABASE WROTE SUCCESSFULLY
            window.location.href = "index.html";

        } catch (err) {
            console.error("Auth error:", err);
            let errMsg = err.message;
            if (err.code === "auth/email-already-in-use") errMsg = "Email is already in use.";
            if (err.code === "auth/wrong-password" || err.code === "auth/invalid-credential") errMsg = "Incorrect email or password.";
            if (err.code === "auth/user-not-found") errMsg = "No account found with this email.";

            if (errorDiv) {
                errorDiv.textContent = errMsg;
                errorDiv.style.display = "block";
            }
            if (loginBtn) {
                loginBtn.disabled = false;
                loginBtn.innerHTML = isSignupMode ? "🚀 Create Account" : "🚀 Sign In";
            }
        }
    });
}

// ── Auth State Check (on page load) ─────────────────────────
onAuthStateChanged(auth, (user) => {
    const path = window.location.pathname;
    const isLoginPage = path.includes("login.html") || path.includes("/login");

    if (user) {
        // User IS logged in
        if (isLoginPage) {
            window.location.href = "index.html";
            return;
        }

        // Set local storage for backward compatibility with auth.js
        const displayName = user.email.split('@')[0];
        localStorage.setItem("analyso_user", JSON.stringify({
            name: displayName,
            email: user.email,
            uid: user.uid
        }));

        // Populate header profile on main page
        const userName = document.getElementById("userName");
        const userProfile = document.getElementById("userProfile");
        const userInitial = document.getElementById("userInitial");

        if (userName) userName.textContent = displayName;
        if (userInitial) userInitial.textContent = displayName.charAt(0).toUpperCase();
        if (userProfile) userProfile.style.display = "flex";

        // Show educational disclaimer once per session
        if (typeof window.showDisclaimer === "function") window.showDisclaimer();

    } else {
        // User is NOT logged in
        localStorage.removeItem("analyso_user");
        if (!isLoginPage) window.location.href = "login.html";
    }
});

// ── Logout ───────────────────────────────────────────────────
window.analysoLogout = function () {
    signOut(auth).then(() => {
        localStorage.removeItem("analyso_user");
        window.location.href = "login.html";
    }).catch((error) => {
        console.error("Logout error", error);
    });
};
