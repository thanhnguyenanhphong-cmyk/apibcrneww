const express = require("express");
const app = express();

const API = "https://bcf-ayt4.onrender.com/sexy/all";

//==============================================================================
// 🚀 BCR PROMAX AI ENGINE V3
// - Fix pattern follow/reverse
// - Giảm SKIP
// - Tách predict_total / win / lose / skip cho chuẩn
// - Không học lẫn nhịp với chính chuỗi đang dự đoán
//==============================================================================

const AI_MEMORY = {
    patterns: {},
    markov_o2: {},
    markov_o3: {},
    markov_o4: {},
    bayesian: { conditional: {} },
    ngram: {},
    ensemble: {
        weights: {
            markov_o2: 1.25,
            markov_o3: 1.45,
            markov_o4: 1.65,
            bayesian_cond: 1.15,
            pattern_match: 1.55,
            ngram_pred: 1.45,
            lstm_sliding: 1.80,
            shannon_entropy: 1.10,
            trend_detector: 1.90
        },
        meta_history: {
            markov_o2: { win: 0, total: 0, streak: 0 },
            markov_o3: { win: 0, total: 0, streak: 0 },
            markov_o4: { win: 0, total: 0, streak: 0 },
            bayesian_cond: { win: 0, total: 0, streak: 0 },
            pattern_match: { win: 0, total: 0, streak: 0 },
            ngram_pred: { win: 0, total: 0, streak: 0 },
            lstm_sliding: { win: 0, total: 0, streak: 0 },
            shannon_entropy: { win: 0, total: 0, streak: 0 },
            trend_detector: { win: 0, total: 0, streak: 0 }
        }
    },
    stats: { predict_total: 0, win: 0, lose: 0, skip: 0 }
};

function convert(v) {
    if (v === "SKIP") return "Bỏ qua (Chờ thêm dữ liệu)";
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
function getPatternKey(groups) {
    return groups.map(v => `${v.side}${v.len}`).join("-");
}

function algoMarkovOrder2(s) {
    if (s.length < 2) return { side: "NONE", score: 0 };
    const data = AI_MEMORY.markov_o2[s.slice(-2)];
    if (!data) return { side: "NONE", score: 0 };
    const total = (data.B || 0) + (data.P || 0);
    if (total < 3 || data.B === data.P) return { side: "NONE", score: 0 };
    const diff = Math.abs(data.B - data.P) / total;
    return data.B > data.P ? { side: "B", score: 1.6 + diff * 2 } : { side: "P", score: 1.6 + diff * 2 };
}
function algoMarkovOrder3(s) {
    if (s.length < 3) return { side: "NONE", score: 0 };
    const data = AI_MEMORY.markov_o3[s.slice(-3)];
    if (!data) return { side: "NONE", score: 0 };
    const total = (data.B || 0) + (data.P || 0);
    if (total < 3 || data.B === data.P) return { side: "NONE", score: 0 };
    const diff = Math.abs(data.B - data.P) / total;
    return data.B > data.P ? { side: "B", score: 2 + diff * 2.5 } : { side: "P", score: 2 + diff * 2.5 };
}
function algoMarkovOrder4(s) {
    if (s.length < 4) return { side: "NONE", score: 0 };
    const data = AI_MEMORY.markov_o4[s.slice(-4)];
    if (!data) return { side: "NONE", score: 0 };
    const total = (data.B || 0) + (data.P || 0);
    if (total < 4 || data.B === data.P) return { side: "NONE", score: 0 };
    const diff = Math.abs(data.B - data.P) / total;
    return data.B > data.P ? { side: "B", score: 2.4 + diff * 3 } : { side: "P", score: 2.4 + diff * 3 };
}
function algoBayesianConditional(s) {
    if (s.length < 2) return { side: "NONE", score: 0 };
    const last = s[s.length - 1];
    const data = AI_MEMORY.bayesian.conditional[last];
    if (!data || data.count < 4) return { side: "NONE", score: 0 };
    const pB = data.next_B / data.count;
    const pP = data.next_P / data.count;
    const diff = Math.abs(pB - pP);
    if (diff < 0.08) return { side: "NONE", score: 0 };
    return pB > pP ? { side: "B", score: 1.8 + diff * 4 } : { side: "P", score: 1.8 + diff * 4 };
}
function algoPatternMatching(memory, lastSide) {
    if (!memory || !lastSide) return { side: "NONE", score: 0 };
    const f = memory.follow || 0, r = memory.reverse || 0;
    const total = f + r;
    if (total < 4 || f === r) return { side: "NONE", score: 0 };
    const diff = Math.abs(f - r) / total;
    return f > r
        ? { side: lastSide, score: 2.1 + diff * 3.2 }
        : { side: opposite(lastSide), score: 2.1 + diff * 3.2 };
}
function algoNGramPredictor(s) {
    if (s.length < 5) return { side: "NONE", score: 0 };
    const data = AI_MEMORY.ngram[s.slice(-4)];
    if (!data) return { side: "NONE", score: 0 };
    const total = (data.B || 0) + (data.P || 0);
    if (total < 4 || data.B === data.P) return { side: "NONE", score: 0 };
    const diff = Math.abs(data.B - data.P) / total;
    return data.B > data.P ? { side: "B", score: 2 + diff * 2.6 } : { side: "P", score: 2 + diff * 2.6 };
}
function algoDeepLSTMSlidingWindow(s) {
    if (s.length < 10) return { side: "NONE", score: 0 };
    const targetWindow = s.slice(-4);
    let scoreB = 0, scoreP = 0;
    for (let i = 0; i < s.length - 4; i++) {
        if (s.slice(i, i + 4) === targetWindow) {
            const ageFactor = (i + 4) / s.length;
            const decay = 0.6 + ageFactor * 1.8;
            if (s[i + 4] === "B") scoreB += decay;
            if (s[i + 4] === "P") scoreP += decay;
        }
    }
    if (scoreB === 0 && scoreP === 0) return { side: "NONE", score: 0 };
    const total = scoreB + scoreP;
    const diff = Math.abs(scoreB - scoreP) / total;
    if (diff < 0.10) return { side: "NONE", score: 0 };
    return scoreB > scoreP ? { side: "B", score: 2.4 + diff * 4 } : { side: "P", score: 2.4 + diff * 4 };
}
function algoShannonEntropy(s) {
    if (s.length < 8) return { side: "NONE", score: 0 };
    const block = recentSlice(s, 10);
    const bCount = (block.match(/B/g) || []).length;
    const total = block.length;
    const pB = bCount / total;
    const pP = 1 - pB;
    const entropy = -((pB > 0 ? pB * Math.log2(pB) : 0) + (pP > 0 ? pP * Math.log2(pP) : 0));
    const last = s[s.length - 1];
    if (entropy < 0.55) return { side: last, score: 2.6 };
    if (entropy > 0.96) return { side: opposite(last), score: 2.2 };
    return { side: "NONE", score: 0 };
}
function algoTrendDetector(s) {
    if (s.length < 6) return { side: "NONE", score: 0 };
    const groups = buildGroups(s);
    const lastGroups = groups.slice(-5);
    const last = lastGroups[lastGroups.length - 1];
    if (!last) return { side: "NONE", score: 0 };
    if (last.len >= 4) return { side: last.side, score: 4.2 };
    if (lastGroups.length >= 4) {
        const lens = lastGroups.slice(-4).map(x => x.len);
        if (lens.every(v => v === 1)) return { side: opposite(last.side), score: 3.5 };
    }
    if (lastGroups.length >= 3) {
        const a = lastGroups[lastGroups.length - 3];
        const b = lastGroups[lastGroups.length - 2];
        const c = lastGroups[lastGroups.length - 1];
        if (a.side === c.side && a.len >= 2 && b.len === 1 && c.len === 1) {
            return { side: a.side, score: 3.0 };
        }
    }
    if (last.len === 2 || last.len === 3) return { side: last.side, score: 2.2 };
    return { side: "NONE", score: 0 };
}
function algoAntiNoise(s) {
    if (s.length < 8) return { risk: 0.94 };
    const block = recentSlice(s, 12);
    const flips = buildGroups(block).length - 1;
    const last4 = block.slice(-4);
    let risk = 1.0;
    if (flips >= 8) risk -= 0.22;
    else if (flips >= 6) risk -= 0.12;
    if (new Set(last4.split("")).size === 2 && buildGroups(last4).every(g => g.len === 1)) risk -= 0.08;
    const lastGroup = buildGroups(block).slice(-1)[0];
    if (lastGroup && lastGroup.len >= 3) risk += 0.08;
    return { risk: clamp(risk, 0.72, 1.12) };
}
function getMetaMultiplier(key) {
    const meta = AI_MEMORY.ensemble.meta_history[key];
    if (!meta || meta.total < 5) return 1.0;
    const winRate = meta.win / meta.total;
    let m = 1.0;
    if (winRate >= 0.62) m += 0.30;
    else if (winRate >= 0.55) m += 0.15;
    else if (winRate <= 0.38) m -= 0.28;
    else if (winRate <= 0.45) m -= 0.12;
    if (meta.streak >= 3) m += 0.12;
    if (meta.streak <= -3) m -= 0.12;
    return clamp(m, 0.55, 1.55);
}

function predict(raw) {
    const s = normalizeRaw(raw);
    AI_MEMORY.stats.predict_total++;

    if (s.length < 4) {
        AI_MEMORY.stats.skip++;
        return { result: "SKIP", confidence: 0, pattern: "SHORT_DATA", reason: "Chuỗi quá ngắn (<4 ván)" };
    }

    const groups = buildGroups(s);
    const recent = groups.slice(-6);
    const pattern = getPatternKey(recent);
    if (!AI_MEMORY.patterns[pattern]) AI_MEMORY.patterns[pattern] = { follow: 1, reverse: 1, total: 0, win: 0 };

    const antiNoise = algoAntiNoise(s);
    const lastSide = s[s.length - 1];

    const models = [
        { key: "markov_o2", res: algoMarkovOrder2(s) },
        { key: "markov_o3", res: algoMarkovOrder3(s) },
        { key: "markov_o4", res: algoMarkovOrder4(s) },
        { key: "bayesian_cond", res: algoBayesianConditional(s) },
        { key: "pattern_match", res: algoPatternMatching(AI_MEMORY.patterns[pattern], lastSide) },
        { key: "ngram_pred", res: algoNGramPredictor(s) },
        { key: "lstm_sliding", res: algoDeepLSTMSlidingWindow(s) },
        { key: "shannon_entropy", res: algoShannonEntropy(s) },
        { key: "trend_detector", res: algoTrendDetector(s) }
    ];

    let totalScoreB = 0, totalScoreP = 0, agreeB = 0, agreeP = 0;
    const weights = AI_MEMORY.ensemble.weights;

    for (const model of models) {
        if (!model.res || model.res.side === "NONE" || model.res.score <= 0) continue;
        const impact = model.res.score * (weights[model.key] || 1) * getMetaMultiplier(model.key) * antiNoise.risk;
        if (model.res.side === "B") { totalScoreB += impact; agreeB++; }
        if (model.res.side === "P") { totalScoreP += impact; agreeP++; }
    }

    if ((totalScoreB + totalScoreP) === 0) {
        AI_MEMORY.stats.skip++;
        return { result: "SKIP", confidence: 0, pattern, reason: "Không đủ tín hiệu hội tụ" };
    }

    const predictSide = totalScoreB > totalScoreP ? "B" : "P";
    const winScore = Math.max(totalScoreB, totalScoreP);
    const loseScore = Math.min(totalScoreB, totalScoreP);
    const total = totalScoreB + totalScoreP;
    const voteMajor = predictSide === "B" ? agreeB : agreeP;
    const voteMinor = predictSide === "B" ? agreeP : agreeB;

    const scoreEdge = total > 0 ? (winScore - loseScore) / total : 0;
    const voteEdge = (voteMajor + voteMinor) > 0 ? (voteMajor - voteMinor) / (voteMajor + voteMinor) : 0;

    let confidence = Math.round(50 + scoreEdge * 32 + voteEdge * 10 + (antiNoise.risk - 1) * 30);
    const trendRes = models.find(x => x.key === "trend_detector")?.res;
    if (trendRes && trendRes.side === predictSide) confidence += 4;
    if (scoreEdge < 0.08) confidence -= 8;
    else if (scoreEdge < 0.15) confidence -= 4;
    confidence = clamp(confidence, 35, 98);

    if (confidence < 52 || voteMajor < 1) {
        AI_MEMORY.stats.skip++;
        return { result: "SKIP", confidence, pattern, reason: `Tín hiệu yếu (${confidence}%)` };
    }

    AI_MEMORY.patterns[pattern].lastPredict = predictSide;
    AI_MEMORY.patterns[pattern].snapshots = {};
    for (const model of models) AI_MEMORY.patterns[pattern].snapshots[model.key] = model.res.side;

    return {
        result: convert(predictSide),
        confidence,
        pattern,
        reason: `Promax AI v3 hội tụ ${voteMajor}/${voteMajor + voteMinor} mô hình`
    };
}

function learn(raw) {
    const s = normalizeRaw(raw);
    if (s.length < 3) return;
    const before = s.slice(0, -1);
    const real = s[s.length - 1];
    if (!before.length || !real) return;

    const lastBefore = before[before.length - 1];
    if (!AI_MEMORY.bayesian.conditional[lastBefore]) AI_MEMORY.bayesian.conditional[lastBefore] = { count: 0, next_B: 0, next_P: 0 };
    AI_MEMORY.bayesian.conditional[lastBefore].count++;
    if (real === "B") AI_MEMORY.bayesian.conditional[lastBefore].next_B++;
    else AI_MEMORY.bayesian.conditional[lastBefore].next_P++;

    if (before.length >= 2) {
        const ctx = before.slice(-2);
        if (!AI_MEMORY.markov_o2[ctx]) AI_MEMORY.markov_o2[ctx] = { B: 1, P: 1 };
        AI_MEMORY.markov_o2[ctx][real] += 1;
    }
    if (before.length >= 3) {
        const ctx = before.slice(-3);
        if (!AI_MEMORY.markov_o3[ctx]) AI_MEMORY.markov_o3[ctx] = { B: 1, P: 1 };
        AI_MEMORY.markov_o3[ctx][real] += 1;
    }
    if (before.length >= 4) {
        const ctx = before.slice(-4);
        if (!AI_MEMORY.markov_o4[ctx]) AI_MEMORY.markov_o4[ctx] = { B: 1, P: 1 };
        AI_MEMORY.markov_o4[ctx][real] += 1;
        if (!AI_MEMORY.ngram[ctx]) AI_MEMORY.ngram[ctx] = { B: 1, P: 1 };
        AI_MEMORY.ngram[ctx][real] += 1;
    }

    const groups = buildGroups(before);
    const recent = groups.slice(-6);
    const pattern = getPatternKey(recent);
    if (!AI_MEMORY.patterns[pattern]) AI_MEMORY.patterns[pattern] = { follow: 1, reverse: 1, total: 0, win: 0 };

    const lastGroup = recent[recent.length - 1];
    if (lastGroup) {
        if (real === lastGroup.side) AI_MEMORY.patterns[pattern].follow++;
        else AI_MEMORY.patterns[pattern].reverse++;
    }

    const memory = AI_MEMORY.patterns[pattern];
    if (!memory.snapshots || !memory.lastPredict) return;

    const weights = AI_MEMORY.ensemble.weights;
    const meta = AI_MEMORY.ensemble.meta_history;

    for (const key of Object.keys(memory.snapshots)) {
        const predicted = memory.snapshots[key];
        const isCorrect = predicted === real;
        if (!meta[key]) continue;

        meta[key].total++;
        if (isCorrect) {
            meta[key].win++;
            meta[key].streak = meta[key].streak >= 0 ? meta[key].streak + 1 : 1;
        } else {
            meta[key].streak = meta[key].streak <= 0 ? meta[key].streak - 1 : -1;
        }

        if (weights[key] !== undefined) {
            let delta = isCorrect ? 0.04 : -0.05;
            if (meta[key].streak >= 3) delta += 0.02;
            if (meta[key].streak <= -3) delta -= 0.02;
            weights[key] = clamp(weights[key] + delta, 0.35, 4.8);
        }
    }

    if (memory.lastPredict === real) AI_MEMORY.stats.win++;
    else AI_MEMORY.stats.lose++;
    memory.total++;
}

app.get("/dudoan/sexy/all", async (req, res) => {
    try {
        const r = await fetch(API);
        const data = await r.json();

        const result = data.map(item => {
            const raw = normalizeRaw(item.ket_qua || "");

            // học trên dữ liệu quá khứ của chính chuỗi
            learn(raw);

            // dự đoán phiên tiếp theo từ toàn bộ chuỗi hiện có
            const ai = predict(raw);
            const lastRaw = raw.length ? raw[raw.length - 1] : "";

            return {
                ban: item.ban,
                phien: Number(item.phien),
                ket_qua_van_truoc: lastRaw,
                ket_qua: raw,
                phien_hien_tai: Number(item.phien) + 1,
                du_doan: ai.result,
                do_tin_cay: `${ai.confidence || 0}%`,
                ly_do: ai.reason,
                pattern_hien_tai: ai.pattern
            };
        });

        const betTotal = AI_MEMORY.stats.win + AI_MEMORY.stats.lose;
        res.json({
            success: true,
            engine: "BCR PROMAX AI v3",
            ai_stats: {
                predict_total: AI_MEMORY.stats.predict_total,
                bet_total: betTotal,
                win: AI_MEMORY.stats.win,
                lose: AI_MEMORY.stats.lose,
                skip: AI_MEMORY.stats.skip,
                win_rate: betTotal > 0 ? `${((AI_MEMORY.stats.win / betTotal) * 100).toFixed(2)}%` : "0%"
            },
            total_room: result.length,
            data: result
        });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`BCR PROMAX AI v3 đang chạy tại cổng ${PORT}`));
