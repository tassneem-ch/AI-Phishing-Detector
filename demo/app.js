/**
 * PhishShield - Frontend Analysis Logic
 * 
 * This script simulates the backend detection engine using:
 * 1. Keyword analysis (Urgency, Reward, Threat)
 * 2. Structural analysis (Links, Headers)
 * 3. Domain Spoofing heuristics
 */

// --- Constants ---
const URGENCY_WORDS = ['urgent', 'immediately', 'suspended', 'terminated', '24 hours', 'verify', 'action required'];
const REWARD_WORDS  = ['won', 'winner', 'claim', 'prize', 'gift card', 'congratulations', 'exclusive'];
const THREAT_WORDS  = ['security', 'unauthorized', 'legal action', 'police', 'fbi', 'unusual activity'];

const EXAMPLES = [
    {
        title: "🚨 PayPal Spoof (Critical)",
        text: `From: "PayPal Security Team" <security@paypa1-alert.ru>
Reply-To: collect@phishnet.tk
Subject: URGENT — Account Suspended Due to Unauthorized Access

Dear Valued Customer,

We have detected suspicious activity on your PayPal account. Your account has been temporarily SUSPENDED. You must verify your identity immediately or your account will be permanently terminated within 24 hours.

Click here to verify: http://paypa1-alert.ru/verify?token=abc123

Failure to act will result in legal action.

PayPal Security`
    },
    {
        title: "🎁 Gift Card Trap (High)",
        text: `From: "Prize Department" <winners@free-rewards.xyz>
Subject: Congratulations! You Have Been Selected!

Dear Lucky Winner,

Congratulations! You have won a $1,000 Amazon gift card! This is a limited time exclusive offer. Act now to claim your FREE prize. Click: http://192.168.1.254/claim-prize

Expires in 24 hours!`
    },
    {
        title: "💼 Q3 Report (Safe)",
        text: `From: "Alice Johnson" <alice.johnson@mycompany.com>
Subject: Q3 Report — Review Required

Hi team,

Please find the Q3 financial report attached for your review. Kindly send any feedback by end of week. Let me know if you have questions.

Best regards,
Alice`
    },
    {
        title: "🔑 GitHub Alert (Safe)",
        text: `From: "GitHub Notifications" <noreply@github.com>
Subject: [GitHub] A new SSH key was added to your account

Hi, A new public key was added to your GitHub account.

If you did not perform this action, please remove the key from your account settings and review your account security.

Thanks,
The GitHub Team`
    }
];

// --- State Management ---
const state = {
    analyzing: false,
    result: null
};

// --- DOM Elements ---
const emailInput = document.getElementById('email-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn   = document.getElementById('clear-btn');
const resetBtn   = document.getElementById('reset-btn');
const inputSection  = document.getElementById('input-section');
const resultsSection = document.getElementById('results-section');

// Modals
const showExamplesBtn = document.getElementById('show-examples');
const showEduBtn      = document.getElementById('show-educational');
const examplesModal   = document.getElementById('examples-modal');
const eduModal        = document.getElementById('educational-modal');
const closeButtons    = document.querySelectorAll('.close-modal');

// --- Analysis Logic ---

function countHits(text, keywords) {
    let count = 0;
    keywords.forEach(word => {
        const regex = new RegExp(`\\b${word}\\b`, 'gi');
        const matches = text.match(regex);
        if (matches) count += matches.length;
    });
    return count;
}

function analyzeEmail(text) {
    const cleanText = text.toLowerCase();
    
    // 1. Keyword Scores
    const urgencyScore = countHits(cleanText, URGENCY_WORDS);
    const rewardScore  = countHits(cleanText, REWARD_WORDS);
    const threatScoreLang = countHits(cleanText, THREAT_WORDS);

    // 2. Structural Signals
    const links = text.match(/https?:\/\/[^\s]+/g) || [];
    const linkCount = links.length;
    const suspiciousLink = links.some(link => {
        return link.includes('.ru') || link.includes('.tk') || link.includes('.xyz') || link.includes('.gq') || /\d{1,3}\.\d{1,3}/.test(link);
    });

    // 3. Sender Analysis (Simulated)
    const senderLine = text.split('\n').find(l => l.toLowerCase().startsWith('from:')) || "";
    const replyToLine = text.split('\n').find(l => l.toLowerCase().startsWith('reply-to:')) || "";
    
    let senderSpoof = false;
    if (senderLine.includes('<') && senderLine.includes('>')) {
        const namePart = senderLine.split('<')[0].toLowerCase();
        const emailPart = senderLine.split('<')[1].split('>')[0].toLowerCase();
        // Simple heuristic: name mentions big brand but email domain is different
        if ((namePart.includes('paypal') || namePart.includes('github') || namePart.includes('microsoft')) && !emailPart.includes(namePart.trim().split(' ')[0])) {
            senderSpoof = true;
        }
    }

    const replyToMismatch = replyToLine !== "" && !senderLine.toLowerCase().includes(replyToLine.split(':')[1]?.trim()?.toLowerCase());

    // 4. Calculate Final Threat Probability (0-1)
    let totalScore = 0;
    totalScore += Math.min(urgencyScore * 0.15, 0.3);
    totalScore += Math.min(rewardScore * 0.2, 0.4);
    totalScore += Math.min(threatScoreLang * 0.1, 0.2);
    if (suspiciousLink) totalScore += 0.4;
    if (senderSpoof) totalScore += 0.3;
    if (replyToMismatch) totalScore += 0.3;
    if (linkCount > 3) totalScore += 0.1;

    const finalProb = Math.min(totalScore, 0.99);
    
    return {
        score: finalProb,
        indicators: {
            urgency: urgencyScore,
            reward: rewardScore,
            threat: threatScoreLang,
            links: linkCount,
            suspiciousLink: suspiciousLink,
            senderSpoof: senderSpoof,
            replyToMismatch: replyToMismatch
        }
    };
}

// --- UI Updates ---

function updateResults(result) {
    const { score, indicators } = result;
    const isPhish = score >= 0.5;

    // Verdict Header
    const icon = document.getElementById('verdict-icon');
    const title = document.getElementById('verdict-title');
    const risk = document.getElementById('risk-level');
    
    if (isPhish) {
        icon.textContent = '🚨';
        title.textContent = 'Malicious Email Detected';
        title.style.color = 'var(--phish)';
        risk.textContent = score >= 0.85 ? 'CRITICAL RISK' : 'HIGH RISK';
        risk.style.color = 'var(--phish)';
    } else {
        icon.textContent = '✅';
        title.textContent = 'Email Appears Legitimate';
        title.style.color = 'var(--ham)';
        risk.textContent = score < 0.2 ? 'LOW / SAFE' : 'CAUTION';
        risk.style.color = score < 0.2 ? 'var(--ham)' : 'var(--caution)';
    }

    // Score
    const scoreVal = document.getElementById('threat-score-value');
    const scoreBar = document.getElementById('score-bar');
    scoreVal.textContent = `${(score * 100).toFixed(1)}%`;
    scoreBar.style.width = `${score * 100}%`;
    scoreBar.style.backgroundColor = isPhish ? 'var(--phish)' : (score > 0.3 ? 'var(--caution)' : 'var(--ham)');

    // Indicators
    const list = document.getElementById('indicators-list');
    list.innerHTML = '';
    
    const addIndicator = (label, value, isFail) => {
        const li = document.createElement('li');
        const status = document.createElement('span');
        status.className = `indicator-status ${isFail ? 'status-fail' : 'status-ok'}`;
        status.textContent = isFail ? '[!]' : '[OK]';
        
        const text = document.createElement('span');
        text.textContent = `${label}: ${value}`;
        
        li.appendChild(status);
        li.appendChild(text);
        list.appendChild(li);
    };

    addIndicator('Urgency Keywords', `${indicators.urgency} hit(s)`, indicators.urgency > 0);
    addIndicator('Reward/Greed Language', `${indicators.reward} hit(s)`, indicators.reward > 0);
    addIndicator('URLs Found', indicators.links, indicators.links > 3);
    addIndicator('Suspicious Links', indicators.suspiciousLink ? 'YES' : 'NO', indicators.suspiciousLink);
    addIndicator('Sender Spoofing', indicators.senderSpoof ? 'YES' : 'NO', indicators.senderSpoof);

    // Educational Insight
    const insight = document.getElementById('educational-insight');
    if (isPhish) {
        let text = "<p><strong>Why is this flagged?</strong> This email contains several classic phishing techniques:</p><ul>";
        if (indicators.urgency > 0) text += "<li><strong>Artificial Urgency:</strong> The sender is trying to rush you into a decision.</li>";
        if (indicators.reward > 0) text += "<li><strong>False Rewards:</strong> Scammers often promise 'prizes' to steal credentials.</li>";
        if (indicators.suspiciousLink) text += "<li><strong>Suspicious Links:</strong> The links point to unusual domains (.ru, .tk) or IP addresses.</li>";
        if (indicators.senderSpoof) text += "<li><strong>Domain Spoofing:</strong> The sender's name says 'PayPal', but the actual email address is suspicious.</li>";
        text += "</ul>";
        insight.innerHTML = text;
    } else {
        insight.innerHTML = "<p>This email doesn't show common attack patterns. It uses standard professional language and doesn't contain suspicious links or urgent threats.</p>";
    }

    // Recommendation
    const action = document.getElementById('action-text');
    if (isPhish) {
        action.textContent = "Do NOT click any links or download attachments. Report this email to your security team or delete it immediately.";
    } else {
        action.textContent = "No immediate threat detected. You can interact with this email safely, but always remain vigilant.";
    }
}

// --- Event Listeners ---

analyzeBtn.addEventListener('click', () => {
    const text = emailInput.value.trim();
    if (!text) return alert('Please paste an email first.');
    
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';
    
    // Simulate processing delay
    setTimeout(() => {
        const result = analyzeEmail(text);
        updateResults(result);
        
        inputSection.classList.add('hidden');
        resultsSection.classList.remove('hidden');
        
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Email';
        
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 800);
});

clearBtn.addEventListener('click', () => {
    emailInput.value = '';
});

resetBtn.addEventListener('click', () => {
    inputSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    emailInput.value = '';
});

// Examples
showExamplesBtn.addEventListener('click', () => {
    const list = document.querySelector('.examples-list');
    list.innerHTML = '';
    
    EXAMPLES.forEach(ex => {
        const div = document.createElement('div');
        div.className = 'example-item';
        div.innerHTML = `
            <div class="example-title">${ex.title}</div>
            <div class="example-snippet">${ex.text.split('\n').slice(0, 2).join(' ')}...</div>
        `;
        div.onclick = () => {
            emailInput.value = ex.text;
            examplesModal.classList.add('hidden');
        };
        list.appendChild(div);
    });
    
    examplesModal.classList.remove('hidden');
});

showEduBtn.addEventListener('click', () => {
    eduModal.classList.remove('hidden');
});

// Close modals
closeButtons.forEach(btn => {
    btn.onclick = () => {
        examplesModal.classList.add('hidden');
        eduModal.classList.add('hidden');
    };
});

window.onclick = (e) => {
    if (e.target === examplesModal) examplesModal.classList.add('hidden');
    if (e.target === eduModal) eduModal.classList.add('hidden');
};
