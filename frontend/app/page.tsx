"use client";

import { useState } from "react";
import Dropzone from "./components/Dropzone";
import ResultsGrid from "./components/ResultsGrid";

interface AnalysisResult {
  filename: string;
  originalImage: string;
  gradCamImage?: string;
  label: string;
  score: number;
  reasoning?: string;
}

export default function Home() {
  const [results, setResults] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFilesSelected = async (files: File[]) => {
    setLoading(true);
    setError(null);
    const newResults: AnalysisResult[] = [];

    // Process files sequentially or in parallel?
    // Parallel is better for UX, but let's limit concurrency if needed.
    // For now simple Promise.all

    // We append new results to existing ones or clear?
    // Let's prepend new ones.

    try {
      const promises = files.map(async (file): Promise<AnalysisResult | null> => {
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

          // Create object URL for local preview
          const objectUrl = URL.createObjectURL(file);

          return {
            filename: file.name,
            originalImage: objectUrl,
            gradCamImage: data.grad_cam_image,
            label: data.label,
            score: data.score,
            reasoning: data.reasoning,
          };
        } catch (err) {
          console.error(err);
          return null;
        }
      });

      const results = await Promise.all(promises);
      const validResults = results.filter((r): r is AnalysisResult => r !== null);

      setResults(prev => [...validResults, ...prev]);
    } catch (err) {
      setError("An error occurred while processing images.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container">
      <h1 className="title">Deepfake Verification</h1>
      <p className="subtitle">
        Upload images to detect AI manipulation using our advanced Effort-ViT model.
        Hover over results to reveal Grad-CAM heatmaps.
      </p>

      <Dropzone onFilesSelected={handleFilesSelected} disabled={loading} />

      {error && (
        <div style={{
          color: 'var(--error)',
          textAlign: 'center',
          marginTop: '1rem',
          padding: '1rem',
          background: 'rgba(239, 68, 68, 0.1)',
          borderRadius: '8px'
        }}>
          {error}
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', marginTop: '2rem', color: '#888' }}>
          Processing images...
        </div>
      )}

      <ResultsGrid results={results} />
    </main>
  );
}
