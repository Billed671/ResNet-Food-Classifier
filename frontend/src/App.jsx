import { useEffect, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL ?? "/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) return undefined;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(selected) {
    if (!selected) return;
    setPrediction(null);
    if (!selected.type.startsWith("image/")) {
      setError("Please choose an image file.");
      return;
    }
    setError(null);
    setFile(selected);
  }

  async function classify() {
    setLoading(true);
    setError(null);
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail ?? `Server responded with ${response.status}.`);
      }
      setPrediction(payload);
    } catch (err) {
      setError(err instanceof TypeError ? "Could not reach the prediction server." : err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app">
      <h1>Food Classifier</h1>
      <p className="subtitle">
        Upload a photo of a dish. The model recognises 12 Food-101 classes, from ramen to red
        velvet cake.
      </p>

      <label
        className="dropzone"
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          selectFile(event.dataTransfer.files[0]);
        }}
      >
        {previewUrl ? (
          <img src={previewUrl} alt="Selected dish" />
        ) : (
          <span>Drop an image here or click to choose one</span>
        )}
        <input
          type="file"
          accept="image/*"
          aria-label="Food image"
          onChange={(event) => selectFile(event.target.files[0])}
        />
      </label>

      <button type="button" onClick={classify} disabled={!file || loading}>
        {loading ? "Classifying…" : "Classify"}
      </button>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {prediction && (
        <section className="result" aria-live="polite">
          <h2>{prediction.label.replaceAll("_", " ")}</h2>
          <div className="confidence-bar">
            <div style={{ width: `${prediction.confidence * 100}%` }} />
          </div>
          <p>{(prediction.confidence * 100).toFixed(1)}% confidence</p>
        </section>
      )}
    </main>
  );
}
