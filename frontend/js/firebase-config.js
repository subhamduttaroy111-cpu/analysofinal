/**
 * firebase-config.js — Email + Password Login (Modal-Based)
 * ─────────────────────────────────────────────────────────
 * Saves user data to Firebase Realtime Database.
 * Uses Firebase Auth for secure session tracking.
 * 
 * NEW: Uses inline login modal on index.html instead of
 * redirecting to login.html. Supports Guest Mode.
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

// ── Guest Mode Flag ──────────────────────────────────────────
window.isGuestMode = true;

// ── Login Modal: Show / Hide / Tab Toggle ────────────────────
window.showLoginModal = function () {
    const overlay = document.getElementById("loginModalOverlay");
    if (overlay) overlay.style.display = "flex";
};

window.closeLoginModal = function () {
    const overlay = document.getElementById("loginModalOverlay");
    if (overlay) overlay.style.display = "none";

    // Enter Guest Mode
    window.isGuestMode = true;
    const guestBanner = document.getElementById("guestBanner");
    if (guestBanner) guestBanner.style.display = "flex";

    // Apply blur/lock to any already-rendered cards
    if (typeof window.applyGuestLock === "function") window.applyGuestLock();
};

let isModalSignup = false;

window.setLoginModalMode = function (signupActive) {
    isModalSignup = signupActive;

    const tabs = document.getElementById("loginModalTabs");
    const tabLogin = document.getElementById("lmTabLogin");
    const tabSignup = document.getElementById("lmTabSignup");
    const submitBtn = document.getElementById("lmSubmitBtn");
    const phoneGroup = document.getElementById("lmPhoneGroup");
    const errorDiv = document.getElementById("lmError");

    if (errorDiv) { errorDiv.style.display = "none"; errorDiv.textContent = ""; }

    if (signupActive) {
        if (tabs) tabs.setAttribute("data-mode", "signup");
        if (tabSignup) tabSignup.classList.add("active");
        if (tabLogin) tabLogin.classList.remove("active");
        if (submitBtn) submitBtn.innerHTML = "🚀 Create Account";
        if (phoneGroup) phoneGroup.style.display = "block";
    } else {
        if (tabs) tabs.setAttribute("data-mode", "login");
        if (tabLogin) tabLogin.classList.add("active");
        if (tabSignup) tabSignup.classList.remove("active");
        if (submitBtn) submitBtn.innerHTML = "🚀 Sign In";
        if (phoneGroup) phoneGroup.style.display = "none";
    }
};

// ── ALSO keep the old login.html page working ────────────────
// (setAuthMode is used on login.html — keep it functional)
window.setAuthMode = window.setLoginModalMode;

// ── Login Modal Form Handler ─────────────────────────────────
const loginModalForm = document.getElementById("loginModalForm");
if (loginModalForm) {
    loginModalForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const email = document.getElementById("lmEmail").value.trim();
        const password = document.getElementById("lmPassword").value.trim();
        const phoneInput = document.getElementById("lmPhone");
        const phone = phoneInput ? phoneInput.value.trim() : "";
        const errorDiv = document.getElementById("lmError");
        const submitBtn = document.getElementById("lmSubmitBtn");

        if (errorDiv) errorDiv.style.display = "none";
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = "⏳ Processing...";
        }

        try {
            if (isModalSignup) {
                // SIGNUP
                const userCredential = await createUserWithEmailAndPassword(auth, email, password);
                const user = userCredential.user;
                const timestamp = new Date().toISOString();
                try {
                    await setDoc(doc(db, "users", user.uid), {
                        email: user.email,
                        phone: phone,
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

            // Success — onAuthStateChanged will handle UI update

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
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = isModalSignup ? "🚀 Create Account" : "🚀 Sign In";
            }
        }
    });
}

// ── Also handle the OLD login.html form (if on that page) ────
const oldLoginForm = document.getElementById("loginForm");
if (oldLoginForm && !document.getElementById("loginModalForm")) {
    oldLoginForm.addEventListener("submit", async function (event) {
        event.preventDefault();

        const email = document.getElementById("emailInput").value.trim();
        const password = document.getElementById("passwordInput").value.trim();
        const phoneInput = document.getElementById("phoneInput");
        const phone = phoneInput ? phoneInput.value.trim() : "";
        const errorDiv = document.getElementById("loginError");
        const loginBtn = document.getElementById("loginBtn");

        if (errorDiv) errorDiv.style.display = "none";
        if (loginBtn) {
            loginBtn.disabled = true;
            loginBtn.textContent = "⏳ Processing...";
        }

        try {
            if (isModalSignup) {
                const userCredential = await createUserWithEmailAndPassword(auth, email, password);
                const user = userCredential.user;
                const timestamp = new Date().toISOString();
                try {
                    await setDoc(doc(db, "users", user.uid), {
                        email: user.email,
                        phone: phone,
                        uid: user.uid,
                        createdAt: timestamp,
                        lastLogin: timestamp
                    });
                } catch (dbErr) {
                    console.error("Database save error:", dbErr);
                    throw new Error("Account created, but Database write failed: " + dbErr.message);
                }
            } else {
                const userCredential = await signInWithEmailAndPassword(auth, email, password);
                const user = userCredential.user;
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

            // Redirect to main app from login.html
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
                loginBtn.innerHTML = isModalSignup ? "🚀 Create Account" : "🚀 Sign In";
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
        window.isGuestMode = false;

        if (isLoginPage) {
            window.location.href = "index.html";
            return;
        }

        // Hide login modal & guest banner
        const loginModal = document.getElementById("loginModalOverlay");
        const guestBanner = document.getElementById("guestBanner");
        if (loginModal) loginModal.style.display = "none";
        if (guestBanner) guestBanner.style.display = "none";

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

        // Remove locks from any cards
        if (typeof window.removeGuestLock === "function") window.removeGuestLock();

        // Show educational disclaimer once per session
        if (typeof window.showDisclaimer === "function") window.showDisclaimer();

    } else {
        // User is NOT logged in
        window.isGuestMode = true;
        localStorage.removeItem("analyso_user");

        if (isLoginPage) {
            // On login.html page — do nothing, let them use it
            return;
        }

        // On index.html — show the login modal (guest can close it)
        showLoginModal();
    }
});

// ── Logout ───────────────────────────────────────────────────
window.analysoLogout = function () {
    signOut(auth).then(() => {
        localStorage.removeItem("analyso_user");
        window.isGuestMode = true;

        // Hide profile, show modal
        const userProfile = document.getElementById("userProfile");
        if (userProfile) userProfile.style.display = "none";

        showLoginModal();

        // Reapply locks
        if (typeof window.applyGuestLock === "function") window.applyGuestLock();

        const guestBanner = document.getElementById("guestBanner");
        if (guestBanner) guestBanner.style.display = "flex";
    }).catch((error) => {
        console.error("Logout error", error);
    });
};
