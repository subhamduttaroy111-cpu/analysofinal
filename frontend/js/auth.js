import { auth, provider, signInWithPopup, signOut, onAuthStateChanged } from "./firebase-config.js";

// DOM Elements
const loginBtn = document.getElementById("loginBtn");
const logoutBtn = document.getElementById("logoutBtn");
const userProfile = document.getElementById("userProfile");
const userImg = document.getElementById("userImg");
const userName = document.getElementById("userName");

// Login Function
if (loginBtn) {
    loginBtn.addEventListener("click", async () => {
        try {
            const result = await signInWithPopup(auth, provider);
            const user = result.user;
            console.log("Logged in:", user.displayName);
            window.location.href = "index.html";
        } catch (error) {
            console.error("Login Failed:", error.message);
            alert("Login Failed: " + error.message);
        }
    });
}

// Logout Function
if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
        try {
            await signOut(auth);
            console.log("Logged out");
            window.location.href = "login.html";
        } catch (error) {
            console.error("Logout Failed:", error.message);
        }
    });
}

// Auth State Observer
onAuthStateChanged(auth, (user) => {
    const isLoginPage = window.location.pathname.includes("login.html");
    const isIndexPage = window.location.pathname.includes("index.html") || window.location.pathname.endsWith("/");

    if (user) {
        // User is signed in
        if (isLoginPage) {
            window.location.href = "index.html";
        }

        // Update User Profile UI
        if (userImg) userImg.src = user.photoURL;
        if (userName) userName.textContent = user.displayName;
        if (userProfile) userProfile.style.display = "flex";
        if (logoutBtn) logoutBtn.style.display = "block";

    } else {
        // User is signed out
        if (isIndexPage) {
            window.location.href = "login.html";
        }
    }
});
