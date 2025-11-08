// FoodFreshnessChecker.jsx
import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";
import * as mobilenet from "@tensorflow-models/mobilenet";
import "bootstrap/dist/css/bootstrap.min.css";
import "./index.css";

export default function FoodFreshnessChecker() {
  const [model, setModel] = useState(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [freshEmbeddings, setFreshEmbeddings] = useState([]);
  const [rottenEmbeddings, setRottenEmbeddings] = useState([]);
  const [message, setMessage] = useState("Upload images to start");
  const [verdict, setVerdict] = useState(null);
  const [overlayRed, setOverlayRed] = useState(false);

  const freshInputRef = useRef();
  const rottenInputRef = useRef();
  const testInputRef = useRef();
  const freshPreview = useRef();
  const rottenPreview = useRef();
  const testPreview = useRef();

  useEffect(() => {
    let canceled = false;
    async function load() {
      setLoadingModel(true);
      try {
        await tf.setBackend("webgl").catch(() => tf.setBackend("cpu"));
        await tf.ready();
        console.log("TF.js backend:", tf.getBackend());

        const m = await mobilenet.load({ version: 2, alpha: 1.0 });
        if (!canceled) setModel(m);
        console.log("✅ MobileNet loaded");
      } catch (err) {
        console.error("Failed loading MobileNet:", err);
      } finally {
        if (!canceled) setLoadingModel(false);
      }
    }
    load();
    return () => (canceled = true);
  }, []);

  function fileToImage(file, imgEl) {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(file);
      const imageElement = imgEl.current;
      imageElement.src = url;
      imageElement.onload = () => resolve();
      imageElement.onerror = (e) => reject(e);
    });
  }

  async function handleAddReferences(files, isFresh = true) {
    if (!model) return alert("Model still loading — wait a second");
    const list = Array.from(files || []);
    for (const f of list) {
      const img = document.createElement("img");
      await fileToImage(f, { current: img });
      const emb = model.infer(img, true);
      if (isFresh) {
        setFreshEmbeddings((prev) => [...prev, emb]);
        if (freshPreview.current) freshPreview.current.src = img.src;
      } else {
        setRottenEmbeddings((prev) => [...prev, emb]);
        if (rottenPreview.current) rottenPreview.current.src = img.src;
      }
    }
    setMessage(isFresh ? "Fresh references added" : "Rotten references added");
  }

  function cosineSimilarity(a, b) {
    const num = a.mul(b).sum();
    const denom = a.norm().mul(b.norm());
    const sim = num.div(denom);
    return sim.dataSync()[0];
  }

  function avgSimilarity(testEmb, refs) {
    if (!refs || refs.length === 0) return null;
    let s = 0;
    for (const r of refs) s += cosineSimilarity(testEmb, r);
    return s / refs.length;
  }

  function colorHeuristic(imgEl) {
    const canvas = document.createElement("canvas");
    const w = 160;
    const h = Math.max(40, Math.round((imgEl.naturalHeight / imgEl.naturalWidth) * w));
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(imgEl, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;
    let total = 0, brownish = 0;
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2];
      total++;
      const brightness = 0.299 * r + 0.587 * g + 0.114 * b;
      const isBrown = r > 70 && r > g + 10 && g > b - 20 && brightness < 140;
      const isDark = brightness < 60;
      if (isBrown || isDark) brownish++;
    }
    return brownish / total;
  }

  async function handleCheck(file) {
    if (!file) return;
    if (!model) return alert("Model still loading — wait a second");
    setMessage("Analyzing...");
    const img = document.createElement("img");
    await fileToImage(file, { current: img });
    if (testPreview.current) testPreview.current.src = img.src;

    const emb = model.infer(img, true);
    const simFresh = avgSimilarity(emb, freshEmbeddings);
    const simRotten = avgSimilarity(emb, rottenEmbeddings);
    const colorScore = colorHeuristic(img);

    let rottenConfidence = 0;
    let reason = "";

    if (simFresh !== null && simRotten !== null) {
      const normFresh = (simFresh + 1) / 2;
      const normRotten = (simRotten + 1) / 2;
      rottenConfidence =
        normRotten * 0.65 + (1 - normFresh) * 0.2 + colorScore * 0.15;
      reason = `simFresh=${simFresh.toFixed(3)}, simRotten=${simRotten.toFixed(3)}, color=${colorScore.toFixed(2)}`;
    } else if (simFresh !== null) {
      const normFresh = (simFresh + 1) / 2;
      rottenConfidence = (1 - normFresh) * 0.75 + colorScore * 0.25;
      reason = `simFresh=${simFresh.toFixed(3)}, color=${colorScore.toFixed(2)}`;
    } else if (simRotten !== null) {
      const normRotten = (simRotten + 1) / 2;
      rottenConfidence = normRotten * 0.85 + colorScore * 0.15;
      reason = `simRotten=${simRotten.toFixed(3)}, color=${colorScore.toFixed(2)}`;
    } else {
      rottenConfidence = colorScore;
      reason = `color=${colorScore.toFixed(2)} (no references)`;
    }

    rottenConfidence = Math.max(0, Math.min(1, rottenConfidence));
    const isRotten = rottenConfidence > 0.55;

    setVerdict({ isRotten, rottenConfidence, reason });
    setMessage(isRotten ? "Not safe to eat (likely rotten)" : "Likely fresh / edible");
    setOverlayRed(isRotten);
    emb.dispose();
  }

  function clearAll() {
    setFreshEmbeddings((prev) => { prev.forEach(t => t.dispose && t.dispose()); return []; });
    setRottenEmbeddings((prev) => { prev.forEach(t => t.dispose && t.dispose()); return []; });
    setVerdict(null);
    setMessage("Upload images to start");
    setOverlayRed(false);
    if (freshPreview.current) freshPreview.current.src = "";
    if (rottenPreview.current) rottenPreview.current.src = "";
    if (testPreview.current) testPreview.current.src = "";
    if (freshInputRef.current) freshInputRef.current.value = "";
    if (rottenInputRef.current) rottenInputRef.current.value = "";
    if (testInputRef.current) testInputRef.current.value = "";
  }

  return (
    <div className="app-shell p-4">
      <div className="container py-4">
        {/* Header */}
        <div className="app-header d-flex justify-content-between align-items-center mb-4 flex-wrap">
          <div className="d-flex align-items-center mb-3 mb-md-0">
            <img
              src={process.env.PUBLIC_URL + "/aerobay-logo.jpg"}
              alt="Aerobay Logo"
              className="company-logo me-3"
            />
            <div>
              <h1 className="h3 mb-0 accent-red fw-bold">Food Freshness Checker</h1>
              <div className="small-muted">Powered by <span className="brand">Aerobay</span> | Empowered by Innovation</div>
            </div>
          </div>
          <div>
            <button className="btn btn-outline-light me-2" onClick={clearAll}>Reset</button>
            <button className="btn btn-red shadow-lg" onClick={() => alert("Tip: provide multiple references for higher accuracy")}>Tips</button>
          </div>
        </div>

        {/* Upload Sections */}
        <div className="row g-3">
          <div className="col-md-4">
            <div className="card card-dark p-3">
              <h6 className="text-muted">Fresh reference</h6>
              <input type="file" className="form-control form-control-sm my-2" accept="image/*" multiple onChange={(e) => handleAddReferences(e.target.files, true)} ref={freshInputRef} />
              <div className="img-wrapper position-relative">
                {!freshPreview.current?.src && <div className="placeholder-text">fresh</div>}
                <img ref={freshPreview} alt="fresh" className="preview-img" />
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card card-dark p-3">
              <h6 className="text-muted">Rotten reference (optional)</h6>
              <input type="file" className="form-control form-control-sm my-2" accept="image/*" multiple onChange={(e) => handleAddReferences(e.target.files, false)} ref={rottenInputRef} />
              <div className="img-wrapper position-relative">
                {!rottenPreview.current?.src && <div className="placeholder-text">rotten</div>}
                <img ref={rottenPreview} alt="rotten" className="preview-img" />
              </div>
            </div>
          </div>

          <div className="col-md-4">
            <div className="card card-dark p-3">
              <h6 className="text-muted">Test photo</h6>
              <input type="file" className="form-control form-control-sm my-2" accept="image/*" onChange={(e) => handleCheck(e.target.files[0])} ref={testInputRef} />
              <div className="img-wrapper position-relative">
                {overlayRed && <div className="overlay-red"></div>}
                {!testPreview.current?.src && <div className="placeholder-text">test</div>}
                <img ref={testPreview} alt="test" className="preview-img" />
              </div>
            </div>
          </div>

          {/* Verdict Section */}
          <div className="col-12">
            <div className="card card-dark p-3 d-flex align-items-center justify-content-between flex-wrap">
              <div>
                <div className="small-muted">Model status: {loadingModel ? "loading..." : model ? "ready" : "failed"}</div>
                <h5 className="mt-2">{message}</h5>
                {verdict && (
                  <div className="mt-2">
                    <div className={`badge ${verdict.isRotten ? "bg-danger" : "bg-success"}`} style={{ fontSize: "1rem" }}>
                      {verdict.isRotten ? "NOT EDIBLE" : "EDIBLE"}
                    </div>
                    <div className="small-muted mt-2">Confidence: {(verdict.rottenConfidence * 100).toFixed(1)}%</div>
                    <div className="big-muted">Reason: {verdict.reason}</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <footer className="mt-4 text-center small-muted">
          © {new Date().getFullYear()} Food Freshness Checker App
        </footer>
      </div>
    </div>
  );
}
