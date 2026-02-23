// Loading status messages that cycle during scan
const loadingMessages = [
    "🔍 Analyzing stocks from NSE...",
    "📊 Checking MACD indicators...",
    "📈 Analyzing RSI levels...",
    "💹 Evaluating volume patterns...",
    "🎯 Checking price action...",
    "📉 Identifying chart patterns...",
    "🔬 Analyzing SMC setups...",
    "⚡ Detecting momentum shifts...",
    "🎢 Checking support & resistance...",
    "💎 Finding best opportunities..."
];

let messageIndex = 0;
let messageInterval = null;

function startLoadingAnimation() {
    const loaderStatus = document.getElementById('loaderStatus');
    if (!loaderStatus) return;

    messageIndex = 0;
    loaderStatus.textContent = loadingMessages[0];

    // Cycle through messages every 1.5 seconds
    messageInterval = setInterval(() => {
        messageIndex = (messageIndex + 1) % loadingMessages.length;
        loaderStatus.textContent = loadingMessages[messageIndex];
    }, 1500);
}

function stopLoadingAnimation() {
    if (messageInterval) {
        clearInterval(messageInterval);
        messageInterval = null;
    }
}
