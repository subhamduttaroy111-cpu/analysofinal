/**
 * disclaimer.js — Educational Warning Popup
 * ─────────────────────────────────────────
 * Shows a mandatory educational disclaimer popup ONCE per browser session
 * after the user logs in. The user MUST click "I Understand & Proceed"
 * to access the dashboard. The popup cannot be dismissed otherwise.
 *
 * Uses sessionStorage so it resets every time the browser tab is closed.
 */

(function () {
    'use strict';

    /* ─── Constants ──────────────────────────────────────────────── */
    const SESSION_KEY = 'analyso_disclaimerShown';

    /* ─── Check whether disclaimer has already been shown this session ─── */
    function hasShownThisSession() {
        return sessionStorage.getItem(SESSION_KEY) === 'true';
    }

    /* ─── Mark disclaimer as shown for this session ─────────────────── */
    function markShown() {
        sessionStorage.setItem(SESSION_KEY, 'true');
    }

    /* ─── Accept & close the popup ───────────────────────────────────── */
    function acceptDisclaimer() {
        const overlay = document.getElementById('disclaimerOverlay');
        if (!overlay) return;

        // Smooth fade-out
        overlay.classList.add('disclaimer-hide');
        overlay.addEventListener('animationend', function handler() {
            overlay.style.display = 'none';
            overlay.removeEventListener('animationend', handler);
        });

        markShown();
    }

    /* ─── Main: show the popup ───────────────────────────────────────── */
    window.showDisclaimer = function () {
        // Only show if not already accepted in this session
        if (hasShownThisSession()) return;

        const overlay = document.getElementById('disclaimerOverlay');
        if (!overlay) return;

        overlay.style.display = 'flex';
        // Trigger entrance animation
        overlay.classList.remove('disclaimer-hide');
        overlay.classList.add('disclaimer-show');
    };

    /* ─── Expose accept handler globally for button onclick ─────────── */
    window.acceptDisclaimer = acceptDisclaimer;

    /* ─── Prevent closing by clicking outside (swallow backdrop clicks) ─ */
    document.addEventListener('DOMContentLoaded', function () {
        const overlay = document.getElementById('disclaimerOverlay');
        if (!overlay) return;

        overlay.addEventListener('click', function (e) {
            // Only the button inside can close this — stop propagation from backdrop
            e.stopPropagation();
        });
    });
})();
