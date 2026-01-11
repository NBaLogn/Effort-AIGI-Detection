"use client";

import React, { useState } from 'react';
import styles from './ResultCard.module.css';

interface ResultCardProps {
    filename: string;
    originalImage: string; // URL or Base64
    gradCamImage?: string; // Base64
    label: string;
    score: number;
    reasoning?: string;
    kind?: "image" | "video";
    videoMeta?: {
        sampledFrames: number;
        worstFrameIndex: number;
        worstFrameScore: number;
    };
}

export default function ResultCard({ filename, originalImage, gradCamImage, label, score, reasoning, kind, videoMeta }: ResultCardProps) {
    const [showGradCam, setShowGradCam] = useState(false);

    const isFake = label.toUpperCase() === 'FAKE';
    const scorePercent = (score * 100).toFixed(1);

    // Format score based on label. 
    // If FAKE, strict score. If REAL, it's 1-score or similar.
    // The backend returns Prob(Fake).
    // So Score displayed should probably be "Probability of being FAKE"

    return (
        <div className={styles.card}>
            <div className={styles.imageContainer} onMouseEnter={() => setShowGradCam(true)} onMouseLeave={() => setShowGradCam(false)}>
                {kind === 'video' ? (
                    <video
                        src={originalImage}
                        className={styles.image}
                        muted
                        controls
                    />
                ) : (
                    <img
                        src={originalImage}
                        alt={filename}
                        className={styles.image}
                    />
                )}
                {gradCamImage && (
                    <img
                        src={gradCamImage}
                        alt="Grad-CAM"
                        className={`${styles.overlay} ${showGradCam ? styles.visible : ''}`}
                    />
                )}
                <div className={styles.hoverHint}>{kind === 'video' ? 'Hover to see worst frame Grad-CAM' : 'Hover to see Grad-CAM'}</div>
            </div>

            <div className={styles.info}>
                <div className={styles.header}>
                    <h3 className={styles.filename} title={filename}>{filename}</h3>
                    <span className={`${styles.badge} ${isFake ? styles.fake : styles.real}`}>
                        {label}
                    </span>
                </div>

                <div className={styles.meterContainer}>
                    <div className={styles.meterInfo}>
                        <span className={styles.meterLabel}>Confidence (Fake)</span>
                        <span className={styles.meterValue}>{scorePercent}%</span>
                    </div>
                    <div className={styles.meterTrack}>
                        <div
                            className={`${styles.meterFill} ${isFake ? styles.fillFake : styles.fillReal}`}
                            style={{ width: `${scorePercent}%` }}
                        />
                    </div>
                </div>

                {reasoning && (
                    <div className={styles.reasoning}>
                        <span className={styles.reasoningLabel}>Analysis Explanation:</span>
                        <p className={styles.reasoningText}>{reasoning}</p>
                    </div>
                )}

                {kind === 'video' && videoMeta && (
                    <div className={styles.videoMeta}>
                        <small>Sampled frames: {videoMeta.sampledFrames}</small>
                        <br />
                        <small>Worst frame index: {videoMeta.worstFrameIndex}</small>
                        <br />
                        <small>Worst frame score: {(videoMeta.worstFrameScore * 100).toFixed(1)}%</small>
                    </div>
                )
                }
            </div>
        </div>
    );
}
