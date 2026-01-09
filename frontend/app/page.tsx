"use client";

import { useState } from "react";
import Dropzone from "./components/Dropzone";
import ResultSummary from "./components/ResultSummary";
import ResultsGrid from "./components/ResultsGrid";
import ProgressBar from "./components/ProgressBar";

interface AnalysisResult {
  filename: string;
  originalImage: string;
  gradCamImage?: string;
  label: string;
  score: number;
  reasoning?: string;
}

interface Batch {
  id: string;
  timestamp: number;
  isFinalized: boolean;
  results: AnalysisResult[];
}

function calculateSummary(results: AnalysisResult[]) {
  const total = results.length;
  if (total === 0) {
    return null;
  }

  let realCount = 0;
  let fakeCount = 0;

  for (const result of results) {
    const normalizedLabel = result.label.trim().toUpperCase();
    if (normalizedLabel === "REAL") {
      realCount += 1;
    } else if (normalizedLabel === "FAKE") {
      fakeCount += 1;
    }
  }

  const roundToOneDecimal = (value: number) => Math.round(value * 10) / 10;

  const realPercent = roundToOneDecimal((realCount / total) * 100);
  const fakePercent = roundToOneDecimal((fakeCount / total) * 100);

  return { total, realCount, fakeCount, realPercent, fakePercent };
}

export default function Home() {
  const [batches, setBatches] = useState<Batch[]>([
    {
      id: crypto.randomUUID(),
      timestamp: Date.now(),
      isFinalized: false,
      results: [],
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);

  // The active batch is always the last one in the list (or the one !isFinalized)
  // We'll assume the last one is always the active one based on our logic.
  const activeBatchIndex = batches.findIndex((b) => !b.isFinalized);
  const activeBatch = batches[activeBatchIndex];
  const finalizedBatches = batches.filter((b) => b.isFinalized);

  const handleFilesSelected = async (files: File[]) => {
    setLoading(true);
    setError(null);
    setProgress({ current: 0, total: files.length });

    // Helper to add a single result to the active batch
    const addResult = (result: AnalysisResult) => {
      setBatches((prev) => {
        const next = [...prev];
        const current = next[activeBatchIndex];
        next[activeBatchIndex] = {
          ...current,
          results: [result, ...current.results],
        };
        return next;
      });
    };

    try {
      // Create an array of potential promises, but manage them so we can update progress
      // We still run them concurrently, but we intercept each completion.
      const promises = files.map(async (file) => {
        const formData = new FormData();
        formData.append("file", file);

        try {
          const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            body: formData,
          });

          if (!response.ok) {
            throw new Error(`Failed to process ${file.name}`);
          }

          const data = await response.json();
          const objectUrl = URL.createObjectURL(file);

          const result: AnalysisResult = {
            filename: file.name,
            originalImage: objectUrl,
            gradCamImage: data.grad_cam_image,
            label: data.label,
            score: data.score,
            reasoning: data.reasoning,
          };

          addResult(result);

        } catch (err) {
          console.error(err);
          // For now, we just log errors but don't stop the whole batch.
          // We could optionally add an "ErrorResult" type to the UI.
        } finally {
          setProgress((prev) => {
            if (!prev) return null;
            return { ...prev, current: prev.current + 1 };
          });
        }
      });

      await Promise.all(promises);

    } catch (err) {
      setError("An error occurred while processing images.");
      console.error(err);
    } finally {
      setLoading(false);
      setProgress(null);
    }
  };

  const handleFinalizeBatch = () => {
    if (!activeBatch || activeBatch.results.length === 0) return;

    setBatches((prev) => {
      const next = [...prev];
      // Mark current as finalized
      next[activeBatchIndex] = { ...next[activeBatchIndex], isFinalized: true };
      // Add new active batch
      next.push({
        id: crypto.randomUUID(),
        timestamp: Date.now(),
        isFinalized: false,
        results: [],
      });
      return next;
    });
  };

  return (
    <main className="container">
      <h1 className="title">Deepfake Verification</h1>
      <p className="subtitle">
        Upload images to detect AI manipulation using our advanced Effort-ViT model.
        Hover over results to reveal Grad-CAM heatmaps.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
        <Dropzone onFilesSelected={handleFilesSelected} disabled={loading} />

        {progress && (
          <ProgressBar current={progress.current} total={progress.total} />
        )}

        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={handleFinalizeBatch}
            disabled={loading || activeBatch.results.length === 0}
            style={{
              padding: "0.5rem 1rem",
              borderRadius: "6px",
              border: "1px solid #ccc",
              background: activeBatch.results.length > 0 ? "#fff" : "#f5f5f5",
              cursor: activeBatch.results.length > 0 ? "pointer" : "not-allowed",
              color: activeBatch.results.length > 0 ? "#333" : "#aaa",
              fontSize: "0.9rem",
              fontWeight: 500,
            }}
          >
            Finalize Batch
          </button>
        </div>
      </div>

      {error && (
        <div
          style={{
            color: "var(--error)",
            textAlign: "center",
            marginTop: "1rem",
            padding: "1rem",
            background: "rgba(239, 68, 68, 0.1)",
            borderRadius: "8px",
          }}
        >
          {error}
        </div>
      )}

      {loading && !progress && (
        <div style={{ textAlign: "center", marginTop: "2rem", color: "#888" }}>
          Processing images...
        </div>
      )}

      {/* Render Finalized Batches */}
      {finalizedBatches.map((batch) => {
        const summary = calculateSummary(batch.results);
        if (!summary) return null; // Should not happen if we only finalize with results
        return (
          <div key={batch.id} style={{ marginBottom: "3rem", opacity: 0.8 }}>
            <div style={{
              borderBottom: "1px solid #eee",
              paddingBottom: "1rem",
              marginBottom: "1rem",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              color: "#666"
            }}>
              <span>Batch completed at {new Date(batch.timestamp).toLocaleTimeString()}</span>
              <span>{batch.results.length} images</span>
            </div>
            <ResultSummary {...summary} />
            <ResultsGrid results={batch.results} hasSummary={true} />
          </div>
        );
      })}

      {/* Divider if needed */}
      {finalizedBatches.length > 0 && activeBatch.results.length > 0 && (
        <hr style={{ margin: "3rem 0", border: "none", borderTop: "2px dashed #eee" }} />
      )}

      {/* Render Active Batch */}
      {(() => {
        const summary = calculateSummary(activeBatch.results);
        return (
          <div>
            {summary && <ResultSummary {...summary} />}
            <ResultsGrid results={activeBatch.results} hasSummary={Boolean(summary)} />
          </div>
        )
      })()}
    </main>
  );
}


