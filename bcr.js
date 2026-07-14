const express = require("express");
const app = express();

const API = "https://bcf-ayt4.onrender.com/sexy/all";

//==============================================================================
// 🚀 BCR PROMAX AI ENGINE V7 - FULL CODE
// - Markov Backoff (Order 5 → 4 → 3 → 2)
// - Cầu (Bridge) với trọng số đồng đều 28
// - Bridge Trust tự học từ lịch sử
// - Cập nhật Win/Lose thực tế
// - Pattern Confidence giảm điểm khi cầu không chắc
// - Tất cả trọng số được cân bằng để AI tự học tối ưu
//==============================================================================

const AI_MEMORY = {
    markov: {}, // markov[order][context] = {B, P}
    bayesian: { conditional: {} },
    bridge_history: {},
    lastPrediction: {}, // { ban: "B" hoặc "P" }
    stats: { predict_total: 0, win: 0, lose: 0, skip: 0 }
};

// ------------------- CẦU (BRIDGE) WEIGHTS ĐỒNG ĐỀU -------------------
const BRIDGE_BASE_WEIGHTS = {
    "BET_STRONG": 28,
    "BET_MEDIUM": 28,
    "PATTERN_1_1": 28,
    "PATTERN_2_2": 28,
    "PATTERN_2_1_2": 28,
    "PATTERN_3_2_1": 28,
    "PATTERN_DAO": 28,
    "PATTERN_XIEN": 28,
    "PATTERN_KEP": 28
};

// Trust mặc định đều bằng 1.0 - AI sẽ tự học từ dữ liệu thực tế
const DEFAULT_TRUST = {
    "BET_STRONG": 1.0,
    "BET_MEDIUM": 1.0,
    "PATTERN_1_1": 1.0,
    "PATTERN_2_2": 1.0,
    "PATTERN_2_1_2": 1.0,
    "PATTERN_3_2_1": 1.0,
    "PATTERN_DAO": 1.0,
    "PATTERN_XIEN": 1.0,
    "PATTERN_KEP": 1.0
};

const MAX_BRIDGE_POINTS = 35;

function convert(v) {
    return v === "B" ? "Banker" : "Player";
}
function opposite(v) { return v === "B" ? "P" : "B"; }
function clamp(n, min, max) { return Math.max(min, Math.min(max, n)); }
function normalizeRaw(raw) {
    return (raw || "").toUpperCase().replace(/T/g, "").replace(/[^BP]/g, "");
}
function recentSlice(s, n) { return s.slice(Math.max(0, s.length - n)); }

function buildGroups(s) {
    if (!s.length) return [];
    const groups = [];
    let current = s[0], count = 1;
    for (let i = 1; i < s.length; i++) {
        if (s[i] === current) count++;
        else {
            groups.push({ side: current, len: count });
            current = s[i];
            count = 1;
        }
    }
    groups.push({ side: current, len: count });
    return groups;
}

// ==================== GET BRIDGE TRUST TỰ HỌC ====================
function getBridgeTrust(name) {
    const h = AI_MEMORY.bridge_history[name];
    if (!h || h.total < 15) {
        return DEFAULT_TRUST[name] || 1.0;
    }
    const rate = h.correct / h.total;
    // Trust = rate + bonus 0.2, nhưng giới hạn trong [0.6, 1.5]
    return clamp(rate + 0.2, 0.6, 1.5);
}

// ==================== PATTERN CONFIDENCE ====================
function getPatternConfidence(s, patternName) {
    if (s.length < 10) return 0.7;
    const groups = buildGroups(s);
    const recent = groups.slice(-10);
    let matchCount = 0;
    let totalCheck = 0;

    for (let i = 0; i < recent.length - 1; i++) {
        const sub = recent.slice(i);
        if (patternName.includes("BET") && sub[0] && sub[0].len >= 3) matchCount++;
        else if (patternName.includes("1-1") && sub.length >= 2 && sub[0].len === 1 && sub[1].len === 1) matchCount++;
        else if (patternName.includes("2-2") && sub.length >= 2 && sub[0].len === 2 && sub[1].len === 2) matchCount++;
        else if (patternName.includes("2-1-2") && sub.length >= 3 && sub[0].len === 2 && sub[1].len === 1 && sub[2].len === 2) matchCount++;
        else if (patternName.includes("3-2-1") && sub.length >= 3 && sub[0].len === 3 && sub[1].len === 2 && sub[2].len === 1) matchCount++;
        totalCheck++;
    }

    const conf = totalCheck > 0 ? clamp(matchCount / totalCheck + 0.3, 0.5, 0.95) : 0.7;
    return conf;
}

// ==================== PHÁT HIỆN CẦU (BRIDGE DETECTOR) ====================
function detectBridge(s) {
    if (s.length < 4) return { side: "NONE", weight: 0, name: "NONE", trust: 1.0, confidence: 0.7 };
    const groups = buildGroups(s);
    const lastGroups = groups.slice(-8);
    if (lastGroups.length < 2) return { side: "NONE", weight: 0, name: "NONE", trust: 1.0, confidence: 0.7 };

    const last = lastGroups[lastGroups.length - 1];
    const prev = lastGroups[lastGroups.length - 2];
    const prev2 = lastGroups.length >= 3 ? lastGroups[lastGroups.length - 3] : null;
    const prev3 = lastGroups.length >= 4 ? lastGroups[lastGroups.length - 4] : null;

    let detected = null;

    // === CẦU BỆT ===
    if (last.len >= 5) {
        detected = { side: last.side, weight: BRIDGE_BASE_WEIGHTS.BET_STRONG, name: "BET_STRONG" };
    } else if (last.len === 4) {
        detected = { side: last.side, weight: BRIDGE_BASE_WEIGHTS.BET_STRONG, name: "BET_STRONG" };
    } else if (last.len === 3) {
        detected = { side: last.side, weight: BRIDGE_BASE_WEIGHTS.BET_MEDIUM, name: "BET_MEDIUM" };
    }
    // === CẦU 1-1 ===
    else if (lastGroups.length >= 4) {
        const last4 = lastGroups.slice(-4);
        if (last4.every(g => g.len === 1)) {
            const sides = last4.map(g => g.side);
            if (sides[0] !== sides[1] && sides[1] !== sides[2] && sides[2] !== sides[3]) {
                detected = { side: opposite(last.side), weight: BRIDGE_BASE_WEIGHTS.PATTERN_1_1, name: "PATTERN_1_1" };
            }
        }
    }
    // === CẦU 2-2 ===
    else if (lastGroups.length >= 2 && prev.len === 2 && last.len === 2) {
        detected = { side: last.side, weight: BRIDGE_BASE_WEIGHTS.PATTERN_2_2, name: "PATTERN_2_2" };
    }
    // === CẦU 2-1-2 ===
    else if (lastGroups.length >= 3 && prev2) {
        if (prev2.len === 2 && prev.len === 1 && last.len === 2 && prev2.side === last.side) {
            detected = { side: prev2.side, weight: BRIDGE_BASE_WEIGHTS.PATTERN_2_1_2, name: "PATTERN_2_1_2" };
        }
    }
    // === CẦU 3-2-1 ===
    else if (lastGroups.length >= 3 && prev2) {
        if (prev2.len === 3 && prev.len === 2 && last.len === 1) {
            detected = { side: last.side, weight: BRIDGE_BASE_WEIGHTS.PATTERN_3_2_1, name: "PATTERN_3_2_1" };
        }
    }
    // === CẦU XIÊN ===
    else if (lastGroups.length >= 3 && prev2) {
        if (prev2.side !== prev.side && prev.side !== last.side && prev2.side === last.side) {
            if (prev2.len >= 2 && prev.len === 1 && last.len >= 2) {
                detected = { side: prev2.side, weight: BRIDGE_BASE_WEIGHTS.PATTERN_XIEN, name: "PATTERN_XIEN" };
            }
        }
    }
    // === CẦU KÉP ===
    else if (lastGroups.length >= 4 && prev2 && prev3) {
        if (prev3.len === prev2.len && prev2.len === prev.len && prev.len === last.len) {
            const sides = [prev3.side, prev2.side, prev.side, last.side];
            if (sides[0] === sides[2] && sides[1] === sides[3] && sides[0] !== sides[1]) {
                detected = { side: last.side, weight: BRIDGE_BASE_WEIGHTS.PATTERN_KEP, name: "PATTERN_KEP" };
            }
        }
    }
    // === CẦU ĐẢO ===
    else if (lastGroups.length >= 3) {
        const last3 = lastGroups.slice(-3);
        if (last3.every(g => g.len === 1)) {
            const sides = last3.map(g => g.side);
            if (sides[0] !== sides[1] && sides[1] !== sides[2]) {
                detected = { side: opposite(last.side), weight: BRIDGE_BASE_WEIGHTS.PATTERN_DAO, name: "PATTERN_DAO" };
            }
        }
    }

    if (!detected) {
        return { side: "NONE", weight: 0, name: "NONE", trust: 1.0, confidence: 0.7 };
    }

    const trust = getBridgeTrust(detected.name);
    const confidence = getPatternConfidence(s, detected.name);

    let adjustedWeight = Math.round(detected.weight * trust * confidence);
    adjustedWeight = Math.min(adjustedWeight, MAX_BRIDGE_POINTS);

    return {
        side: detected.side,
        weight: adjustedWeight,
        name: detected.name,
        trust: trust,
        confidence: confidence
    };
}

// ==================== MARKOV BACKOFF (Order 5 → 4 → 3 → 2) ====================
function markovBackoff(s) {
    if (s.length < 2) return { B: 0, P: 0 };

    const orders = [5, 4, 3, 2];
    for (const order of orders) {
        if (s.length < order) continue;
        const ctx = s.slice(-order);
        const data = AI_MEMORY.markov[order] && AI_MEMORY.markov[order][ctx];
        if (data && (data.B + data.P) >= 3) {
            const total = data.B + data.P;
            return {
                B: Math.round((data.B / total) * 100),
                P: Math.round((data.P / total) * 100)
            };
        }
    }
    return { B: 0, P: 0 };
}

// ==================== TREND ANALYZER ====================
function analyzeTrend(s) {
    if (s.length < 6) return { side: "NONE", weight: 0 };
    const block = recentSlice(s, 20);
    const bCount = (block.match(/B/g) || []).length;
    const total = block.length;
    if (total < 4) return { side: "NONE", weight: 0 };
    const pB = bCount / total;
    const diff = Math.abs(pB - 0.5);
    if (diff < 0.05) return { side: "NONE", weight: 0 };
    const weight = Math.round(diff * 45);
    return {
        side: pB > 0.5 ? "B" : "P",
        weight: clamp(weight, 5, 25)
    };
}

// ==================== BAYESIAN UPDATE ====================
function bayesianUpdate(s) {
    if (s.length < 2) return { B: 0, P: 0 };
    const last = s[s.length - 1];
    const data = AI_MEMORY.bayesian.conditional[last];
    if (!data || data.count < 3) return { B: 0, P: 0 };
    const pB = data.next_B / data.count;
    const pP = data.next_P / data.count;
    const diff = Math.abs(pB - pP);
    if (diff < 0.05) return { B: 0, P: 0 };
    return {
        B: Math.round(pB * 100),
        P: Math.round(pP * 100)
    };
}

// ==================== SCORE FUSION ENGINE ====================
function predict(raw, ban) {
    const s = normalizeRaw(raw);
    AI_MEMORY.stats.predict_total++;

    if (s.length < 4) {
        AI_MEMORY.stats.skip++;
        return {
            result: "SKIP",
            confidence: 0,
            reason: "Chuỗi quá ngắn (<4 ván)",
            details: {}
        };
    }

    // 1. MARKOV BACKOFF
    const markov = markovBackoff(s);
    let scoreB = markov.B || 0;
    let scoreP = markov.P || 0;

    // 2. CẦU (BRIDGE) - đã giới hạn tối đa 35 điểm
    const bridge = detectBridge(s);
    if (bridge.side === "B") scoreB += bridge.weight;
    else if (bridge.side === "P") scoreP += bridge.weight;

    // 3. TREND
    const trend = analyzeTrend(s);
    if (trend.side === "B") scoreB += trend.weight;
    else if (trend.side === "P") scoreP += trend.weight;

    // 4. BAYESIAN
    const bayes = bayesianUpdate(s);
    if (bayes.B > 0 && bayes.P > 0) {
        const total = bayes.B + bayes.P;
        const bWeight = Math.round((bayes.B / total) * 35);
        const pWeight = Math.round((bayes.P / total) * 35);
        scoreB += bWeight;
        scoreP += pWeight;
    }

    // Fallback
    if (scoreB === 0 && scoreP === 0) {
        if (markov.B > markov.P) { scoreB = 50; scoreP = 50; }
        else { scoreP = 50; scoreB = 50; }
    }

    const totalScore = scoreB + scoreP;
    const pB = totalScore > 0 ? (scoreB / totalScore) * 100 : 50;
    const pP = totalScore > 0 ? (scoreP / totalScore) * 100 : 50;

    const predictSide = pB >= pP ? "B" : "P";
    const confidence = Math.round(Math.max(pB, pP));

    // Lưu dự đoán để cập nhật Win/Lose sau
    if (ban) {
        AI_MEMORY.lastPrediction[ban] = predictSide;
    }

    // Xây dựng lý do
    const reasons = [];
    if (markov.B > 0 || markov.P > 0) reasons.push(`✔ Markov (B:${markov.B}% P:${markov.P}%)`);
    if (bridge.name !== "NONE") {
        reasons.push(`✔ ${bridge.name} (+${bridge.weight}) [trust:${bridge.trust.toFixed(2)} conf:${bridge.confidence.toFixed(2)}]`);
    }
    if (trend.side !== "NONE") reasons.push(`✔ Xu hướng ${trend.side} (+${trend.weight})`);
    if (bayes.B > 0 || bayes.P > 0) reasons.push(`✔ Bayesian (B:${bayes.B}% P:${bayes.P}%)`);

    if (reasons.length === 0) reasons.push("⚠ Dùng Markov fallback");

    return {
        result: convert(predictSide),
        confidence: clamp(confidence, 50, 99),
        reason: reasons.join(" | "),
        details: {
            markov: markov,
            bridge: { name: bridge.name, weight: bridge.weight, trust: bridge.trust, confidence: bridge.confidence },
            trend: trend,
            bayesian: bayes,
            raw_score: { B: scoreB, P: scoreP },
            probability: { B: Math.round(pB), P: Math.round(pP) }
        }
    };
}

// ==================== LEARN ====================
function learn(raw, ban) {
    const s = normalizeRaw(raw);
    if (s.length < 3) return;
    const before = s.slice(0, -1);
    const real = s[s.length - 1];
    if (!before.length || !real) return;

    // === CẬP NHẬT MARKOV (tất cả các order) ===
    const orders = [5, 4, 3, 2];
    for (const order of orders) {
        if (before.length >= order) {
            const ctx = before.slice(-order);
            if (!AI_MEMORY.markov[order]) AI_MEMORY.markov[order] = {};
            if (!AI_MEMORY.markov[order][ctx]) AI_MEMORY.markov[order][ctx] = { B: 0, P: 0 };
            AI_MEMORY.markov[order][ctx][real] += 1;
        }
    }

    // === BAYESIAN ===
    const lastBefore = before[before.length - 1];
    if (!AI_MEMORY.bayesian.conditional[lastBefore]) {
        AI_MEMORY.bayesian.conditional[lastBefore] = { count: 0, next_B: 0, next_P: 0 };
    }
    AI_MEMORY.bayesian.conditional[lastBefore].count++;
    if (real === "B") AI_MEMORY.bayesian.conditional[lastBefore].next_B++;
    else AI_MEMORY.bayesian.conditional[lastBefore].next_P++;

    // === CẬP NHẬT BRIDGE HISTORY ===
    const bridge = detectBridge(before);
    if (bridge.name !== "NONE") {
        if (!AI_MEMORY.bridge_history[bridge.name]) {
            AI_MEMORY.bridge_history[bridge.name] = { correct: 0, total: 0 };
        }
        AI_MEMORY.bridge_history[bridge.name].total++;
        if (bridge.side === real) AI_MEMORY.bridge_history[bridge.name].correct++;
    }

    // === CẬP NHẬT WIN/LOSE ===
    if (ban && AI_MEMORY.lastPrediction[ban]) {
        const pred = AI_MEMORY.lastPrediction[ban];
        if (pred === real) AI_MEMORY.stats.win++;
        else AI_MEMORY.stats.lose++;
        delete AI_MEMORY.lastPrediction[ban];
    }
}

// ==================== API ====================
app.get("/dudoan/sexy/all", async (req, res) => {
    try {
        const r = await fetch(API);
        const data = await r.json();

        const result = data.map(item => {
            const raw = normalizeRaw(item.ket_qua || "");
            const ban = item.ban || `ban_${item.phien}`;

            learn(raw, ban);

            const ai = predict(raw, ban);
            const lastRaw = raw.length ? raw[raw.length - 1] : "";

            return {
                ban: item.ban,
                phien: Number(item.phien),
                ket_qua_van_truoc: lastRaw,
                ket_qua: raw,
                phien_hien_tai: Number(item.phien) + 1,
                du_doan: ai.result,
                do_tin_cay: `${ai.confidence}%`,
                ly_do: ai.reason,
                chi_tiet: ai.details
            };
        });

        const betTotal = AI_MEMORY.stats.win + AI_MEMORY.stats.lose;
        res.json({
            success: true,
            engine: "BCR PROMAX AI v7 - Full Code",
            ai_stats: {
                predict_total: AI_MEMORY.stats.predict_total,
                bet_total: betTotal,
                win: AI_MEMORY.stats.win,
                lose: AI_MEMORY.stats.lose,
                skip: AI_MEMORY.stats.skip,
                win_rate: betTotal > 0 ? `${((AI_MEMORY.stats.win / betTotal) * 100).toFixed(2)}%` : "0%"
            },
            bridge_history: AI_MEMORY.bridge_history,
            total_room: result.length,
            data: result
        });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`BCR PROMAX AI v7 đang chạy tại cổng ${PORT}`));
