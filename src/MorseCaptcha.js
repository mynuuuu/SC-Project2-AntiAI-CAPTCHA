import React, { useState, useEffect, useRef } from "react";
import { evaluateFinalAccess } from "./finalHumanGate";

const DOT_TIME = 500;
const DASH_TIME = 1200;
const HOLD_THRESHOLD = 400;

const generatePattern = () => {
    const length = Math.floor(Math.random() * 2) + 3; // 3 - 5
    const symbols = [".", "-"];
    let p = "";
    for (let i = 0; i < length; i++) {
        p += symbols[Math.floor(Math.random() * 2)];
    }
    return p;
};

const getRandomColor = () => {
    const colors = ["yellow", "cyan", "lime", "orange", "magenta"];
    return colors[Math.floor(Math.random() * colors.length)];
};

export default function MorseCaptcha() {
    const [pattern, setPattern] = useState(generatePattern());
    const [flashColor, setFlashColor] = useState(getRandomColor());

    const [stage, setStage] = useState("intro");
    // "intro" → "countdown" → "watch" → "repeat" → "result"
    const [countdown, setCountdown] = useState(3);

    const [flashOn, setFlashOn] = useState(false);
    const [userInput, setUserInput] = useState("");
    const [result, setResult] = useState(null);

    const pressStart = useRef(null);

    // Countdown logic
    useEffect(() => {
        if (stage !== "countdown") return;

        if (countdown > 0) {
            const timer = setTimeout(() => setCountdown(c => c - 1), 1000);
            return () => clearTimeout(timer);
        } else {
            setStage("watch");
        }
    }, [stage, countdown]);

    // Play pattern when stage becomes "watch"
    useEffect(() => {
        if (stage !== "watch") return;

        let i = 0;
        const play = () => {
            if (i >= pattern.length) {
                setStage("repeat");
                return;
            }

            const symbol = pattern[i];
            setFlashOn(true);

            setTimeout(() => {
                setFlashOn(false);
                i++;
                setTimeout(play, 600);
            }, symbol === "." ? DOT_TIME : DASH_TIME);
        };

        play();
    }, [stage, pattern]);

    const handleStart = () => {
        setCountdown(3);
        setStage("countdown");
    };

    const handlePressStart = () => {
        pressStart.current = Date.now();
    };

    const handlePressEnd = async () => {
        const duration = Date.now() - pressStart.current;
        const symbol = duration < HOLD_THRESHOLD ? "." : "-";

        const updated = userInput + symbol;
        setUserInput(updated);

        if (updated.length === pattern.length) {
            const solved = updated === pattern;

            // Ask the defensive logic what to do next
            const decision = await evaluateFinalAccess({
                isSolved: solved,
                sessionId: `morse_${Date.now()}`
            });

            if (decision === "allow" && solved) {
                // Human + solved → redirect to entry form (Next button will be enabled)
                window.location.href = "/entry";
                return;
            }

            if (decision === "block") {
                // Classified as bot → redirect to bot detected page
                window.location.href = "/bot-detected";
                return;
            }

            // decision === "retry" (or fallback) → human but failed, give another chance
            setStage("result");
            setResult(solved ? "✅ Correct! (try again to proceed)" : "❌ Incorrect, try again.");
        }
    };

    const clearInput = () => {
        setUserInput("");
    };

    const reset = () => {
        setPattern(generatePattern());
        setFlashColor(getRandomColor());
        setUserInput("");
        setResult(null);
        setStage("intro");
    };

    return (
        <div style={{ textAlign: "center", marginTop: "40px" }}>
            <h2>Morse Pattern CAPTCHA</h2>

            {/* ✅ INTRO SCREEN */}
            {stage === "intro" && (
                <>
                    <p style={{ maxWidth: 380, margin: "10px auto" }}>
                        This is a simple human check.
                        <br /><br />
                        ✅ You will watch a box flash 3–5 times.
                        <br />Short flash = DOT (•)
                        <br />Long flash = DASH (—)
                        <br /><br />
                        ✅ Then you'll repeat the same pattern:
                        <br /><b>Click</b> for DOT (•)
                        <br /><b>Press & Hold</b> for DASH (—)
                        <br /><br />
                        Click START when you're ready.
                    </p>

                    <button
                        style={{ padding: "18px 40px", fontSize: "18px", cursor: "pointer" }}
                        onClick={handleStart}
                    >
                        START
                    </button>
                </>
            )}

            {/* ✅ COUNTDOWN SCREEN */}
            {(stage === "countdown" || stage === "watch") && (
                <div style={{ fontSize: "48px", fontWeight: "bold", margin: "40px" }}>
                    {countdown}
                </div>
            )}

            {/* ✅ FLASH DISPLAY */}
            {(stage === "watch" || stage === "repeat" || stage === "result" || stage === "countdown") && (
                <div
                    style={{
                        width: 130,
                        height: 130,
                        margin: "20px auto",
                        background: flashOn ? flashColor : "#222",
                        transition: "background 0.2s",
                        borderRadius: 10,
                    }}
                />
            )}

            {/* ✅ WATCHING INSTRUCTIONS */}
            {stage === "watch" && (
                <p>
                    ✅ Step 1: Watch carefully...
                    <br />Short flash = DOT (•)
                    <br />Long flash = DASH (—)
                </p>
            )}

            {/* ✅ REPEAT STAGE */}
            {stage === "repeat" && (
                <>
                    <p>
                        ✅ Step 2: Repeat the pattern
                        <br /><b>Click</b> = DOT (•)
                        <br /><b>Hold</b> = DASH (—)
                    </p>

                    <button
                        style={{ padding: "18px 40px", fontSize: "18px" }}
                        onMouseDown={handlePressStart}
                        onMouseUp={handlePressEnd}
                        onTouchStart={handlePressStart}
                        onTouchEnd={handlePressEnd}
                    >
                        Click / Hold
                    </button>

                    <p style={{ marginTop: 15 }}>
                        Your Input: {userInput || ""}
                    </p>

                    <button onClick={clearInput}>Clear</button>
                </>
            )}

            {/* ✅ RESULT */}
            {stage === "result" && (
                <>
                    <p style={{ fontSize: "20px" }}>{result}</p>
                    <button onClick={reset}>New Pattern</button>
                </>
            )}
        </div>
    );
}