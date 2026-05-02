// AI Phishing Detector - Robust Content Script for Gmail
console.log("%c AI Phishing Detector: ACTIVATED ", "background: #1a1a1a; color: #ffd700; font-weight: bold; padding: 5px;");

let lastAnalyzedId = null;

// Multi-selector lists for Gmail's ever-changing DOM
const BODY_SELECTORS = ['.a3s.aiL', '.ii.gt', '.adn.ads', '[role="main"] .adn'];
const CONTAINER_SELECTORS = ['.nH.V837eb', '.ha', '.h7', '.gE.iv.gt'];

function findElement(selectors) {
    for (const selector of selectors) {
        const el = document.querySelector(selector);
        if (el) return { el, selector };
    }
    return null;
}

function extractEmailContent() {
    const bodyObj = findElement(BODY_SELECTORS);
    const subject = document.querySelector('h2.hP')?.innerText || "No Subject";
    
    if (!bodyObj) return null;

    console.log(`[AI Phish] Found body using: ${bodyObj.selector}`);
    return `Subject: ${subject}\n\n${bodyObj.el.innerText}`;
}

function injectVerdict(data) {
    const existing = document.getElementById('phish-detector-banner');
    if (existing) existing.remove();

    const containerObj = findElement(CONTAINER_SELECTORS);
    const bodyObj = findElement(BODY_SELECTORS);

    // Prefer the top container, fallback to the body itself
    const target = containerObj?.el || bodyObj?.el;
    if (!target) {
        console.error("[AI Phish] Could not find a place to inject the banner.");
        return;
    }

    const banner = document.createElement('div');
    banner.id = 'phish-detector-banner';
    banner.className = data.verdict === 'PHISH' ? 'verdict-phish' : 'verdict-ham';

    const scorePct = (data.threat_score * 100).toFixed(1);
    const icon = data.verdict === 'PHISH' ? '⚠️' : '✅';
    const title = data.verdict === 'PHISH' ? 'PHISHING DETECTED' : 'SECURE EMAIL';

    banner.innerHTML = `
        <div class="banner-header">
            <span class="banner-icon">${icon}</span>
            <span class="banner-title">${title}</span>
            <span class="banner-score">Threat Score: ${scorePct}%</span>
        </div>
        <div class="banner-details">
            ${data.verdict === 'PHISH' ? 'This email exhibits strong phishing characteristics. Do not click any links.' : 'Our AI model considers this email to be safe.'}
        </div>
    `;

    target.prepend(banner);
    console.log(`[AI Phish] Banner injected into: ${containerObj ? containerObj.selector : bodyObj.selector}`);
}

async function analyzeEmail(text) {
    try {
        const response = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        if (!response.ok) throw new Error(`Server returned ${response.status}`);
        
        const data = await response.json();
        injectVerdict(data);
    } catch (err) {
        console.error("[AI Phish] API Error. Is app.py running?", err);
    }
}

function checkEmail() {
    const bodyObj = findElement(BODY_SELECTORS);
    if (bodyObj) {
        const text = bodyObj.el.innerText;
        // Use text length and content snippet as a unique ID for the current email
        const currentId = text.substring(0, 100) + text.length;

        if (currentId !== lastAnalyzedId) {
            console.log("[AI Phish] New email content detected. Scanning...");
            lastAnalyzedId = currentId;
            const fullContent = extractEmailContent();
            if (fullContent) {
                analyzeEmail(fullContent);
            }
        }
    }
}

// Watch for DOM changes (navigation in Gmail)
const observer = new MutationObserver(() => {
    checkEmail();
});

observer.observe(document.body, { childList: true, subtree: true });

// Initial check in case page is already loaded
setTimeout(checkEmail, 2000);
