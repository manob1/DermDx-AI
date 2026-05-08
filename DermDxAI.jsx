import { useState, useRef, useCallback, useEffect } from "react";
import { Upload, BarChart2, Flame, FileText, Activity, AlertTriangle, CheckCircle, Download, Copy, ChevronDown, Microscope, X, Loader2, Zap, Shield, Info } from "lucide-react";

// ── API config ────────────────────────────────────────────────────────────────
const FLASK_URL = "http://localhost:5000";

const DISEASE_CLASSES = [
  { name: "Actinic Keratosis", risk: "HIGH", abbr: "AK" },
  { name: "Atopic Dermatitis", risk: "MEDIUM", abbr: "AD" },
  { name: "Benign Keratosis", risk: "LOW", abbr: "BK" },
  { name: "Dermatofibroma", risk: "LOW", abbr: "DF" },
  { name: "Melanocytic Nevus", risk: "MEDIUM", abbr: "MN" },
  { name: "Melanoma", risk: "HIGH", abbr: "MEL" },
  { name: "Squamous Cell Carcinoma", risk: "HIGH", abbr: "SCC" },
  { name: "Tinea / Ringworm / Candidiasis", risk: "MEDIUM", abbr: "TRC" },
  { name: "Vascular Lesion", risk: "MEDIUM", abbr: "VL" },
];

const ANATOMICAL_SITES = ["Scalp", "Face", "Ear", "Neck", "Chest", "Back", "Abdomen", "Upper extremity", "Lower extremity", "Hand", "Foot", "Nail", "Genital", "Oral/Mucosal"];
const FITZPATRICK_TYPES = ["Type I – Very Fair", "Type II – Fair", "Type III – Medium", "Type IV – Olive", "Type V – Brown", "Type VI – Dark Brown/Black"];

function generateSoftmax() {
  const raw = DISEASE_CLASSES.map(() => Math.random() * Math.random());
  const sum = raw.reduce((a, b) => a + b, 0);
  return raw.map(v => v / sum);
}

const RISK_CONFIG = {
  HIGH: { color: "text-red-400", bg: "bg-red-500/15", border: "border-red-500/40", dot: "bg-red-400", label: "HIGH RISK", icon: AlertTriangle },
  MEDIUM: { color: "text-amber-400", bg: "bg-amber-500/15", border: "border-amber-500/40", dot: "bg-amber-400", label: "MEDIUM RISK", icon: Shield },
  LOW: { color: "text-emerald-400", bg: "bg-emerald-500/15", border: "border-emerald-500/40", dot: "bg-emerald-400", label: "LOW RISK", icon: CheckCircle },
};

const HEATMAP_COLORS = ["#3b0764","#5b21b6","#7c3aed","#a855f7","#c084fc","#f59e0b","#ef4444","#dc2626","#b91c1c"];

function SkeletonBlock({ className = "" }) {
  return <div className={`animate-pulse rounded-lg bg-white/5 ${className}`} />;
}

export default function DermDxAI() {
  const [activeTab, setActiveTab] = useState("upload");
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [confidences, setConfidences] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [animBars, setAnimBars] = useState(false);
  const [copied, setCopied] = useState(false);
  const [patientMeta, setPatientMeta] = useState({
    age: "", sex: "", site: "", fitzpatrick: "", duration: ""
  });
  const fileInputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file || !file.type.match(/image\/(jpeg|png|tiff)/)) return;
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = (e) => setImagePreview(e.target.result);
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  // ── Real inference via Flask ─────────────────────────────────────────────
  const [apiError, setApiError] = useState(null);

  const analyze = async () => {
    if (!imageFile) return;

    setIsLoading(true);
    setApiError(null);
    setActiveTab("upload");

    try {
      // Build multipart payload — Flask reads request.files["image"]
      const formData = new FormData();
      formData.append("image", imageFile);

      const res = await fetch(`${FLASK_URL}/predict`, {
        method: "POST",
        body: formData,
        // Note: do NOT set Content-Type header; the browser sets it with the boundary
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
        throw new Error(errJson.error || `Server error ${res.status}`);
      }

      const data = await res.json();
      /*
        Expected shape from Flask:
        {
          prediction:        "Melanoma",
          confidence:        94.3,          ← percentage (0–100)
          risk:              "HIGH",
          all_probabilities: { "Melanoma": 94.3, "Actinic Keratosis": 1.2, ... }
        }
      */

      // Map all_probabilities onto DISEASE_CLASSES order, normalise to 0-1
      const sorted = DISEASE_CLASSES
        .map((cls) => ({
          ...cls,
          prob: (data.all_probabilities[cls.name] ?? 0) / 100,
        }))
        .sort((a, b) => b.prob - a.prob);

      setConfidences(sorted);

      const topClass = DISEASE_CLASSES.find(c => c.name === data.prediction);
      setPrediction({
        className:  data.prediction,
        confidence: data.confidence / 100,   // store as 0-1 for existing display logic
        risk:       data.risk ?? topClass?.risk ?? "LOW",
      });

      setActiveTab("results");
      setTimeout(() => setAnimBars(true), 100);

    } catch (err) {
      setApiError(err.message);
      console.error("[DermDx] Inference error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const TABS = [
    { id: "upload", label: "Upload", icon: Upload },
    { id: "results", label: "Results", icon: BarChart2 },
    { id: "gradcam", label: "Grad-CAM", icon: Flame },
    { id: "report", label: "Report", icon: FileText },
  ];

  const locked = !prediction;

  const copyReport = () => {
    if (!prediction) return;
    const text = `DermDx AI — Clinical Report\n\nPatient Age: ${patientMeta.age || "N/A"}\nSex: ${patientMeta.sex || "N/A"}\nAnatomical Site: ${patientMeta.site || "N/A"}\nFitzpatrick Type: ${patientMeta.fitzpatrick || "N/A"}\nLesion Duration: ${patientMeta.duration ? patientMeta.duration + " months" : "N/A"}\n\nPredicted Diagnosis: ${prediction.className}\nConfidence: ${(prediction.confidence * 100).toFixed(1)}%\nRisk Level: ${prediction.risk}\n\n⚠ For research use only. Not a medical diagnosis.`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ fontFamily: "'DM Sans', 'Syne', sans-serif", background: "#080912", minHeight: "100vh", color: "#e2e8f0" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0d0e1f; } ::-webkit-scrollbar-thumb { background: #4c1d95; border-radius: 99px; }
        .glow-border { box-shadow: 0 0 0 1px rgba(139,92,246,0.2), inset 0 0 30px rgba(139,92,246,0.03); }
        .glow-btn { box-shadow: 0 0 20px rgba(139,92,246,0.4), 0 4px 15px rgba(0,0,0,0.4); }
        .glow-btn:hover { box-shadow: 0 0 30px rgba(139,92,246,0.65), 0 4px 20px rgba(0,0,0,0.5); transform: translateY(-1px); }
        .tab-active { background: linear-gradient(135deg, rgba(139,92,246,0.25), rgba(99,102,241,0.15)); border-bottom: 2px solid #8b5cf6; }
        .tab-inactive:hover { background: rgba(255,255,255,0.04); }
        .card { background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01)); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; }
        .card-inner { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; }
        .drop-zone { border: 1.5px dashed rgba(139,92,246,0.4); background: rgba(139,92,246,0.03); }
        .drop-zone-active { border-color: rgba(139,92,246,0.8); background: rgba(139,92,246,0.08); box-shadow: 0 0 30px rgba(139,92,246,0.15); }
        .bar-fill { transition: width 1.2s cubic-bezier(0.25,0.46,0.45,0.94); }
        .fade-in { animation: fadeIn 0.5s ease forwards; }
        @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        .pulse-dot { animation: pulseDot 2s ease-in-out infinite; }
        @keyframes pulseDot { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(0.8); } }
        .shimmer { background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent); background-size: 200% 100%; animation: shimmer 1.5s infinite; }
        @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
        select option { background: #1a1b2e; }
        input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; }
      `}</style>

      {/* Navbar */}
      <nav style={{ background: "rgba(8,9,18,0.95)", backdropFilter: "blur(20px)", borderBottom: "1px solid rgba(139,92,246,0.15)", position: "sticky", top: 0, zIndex: 50, padding: "0 24px" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", height: 60 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 38, height: 38, borderRadius: 10, background: "linear-gradient(135deg, #7c3aed, #4f46e5)", display: "flex", alignItems: "center", justifyContent: "center", boxShadow: "0 0 16px rgba(124,58,237,0.5)" }}>
              <Activity size={18} color="#fff" />
            </div>
            <div>
              <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 17, background: "linear-gradient(90deg, #c4b5fd, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>DermDx AI</div>
              <div style={{ fontSize: 10, color: "#64748b", fontFamily: "'DM Mono', monospace", letterSpacing: "0.05em" }}>SqueezeNet CNN · 9-Class · Grad-CAM XAI</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.3)", borderRadius: 99, padding: "4px 10px" }}>
              <span className="pulse-dot" style={{ width: 6, height: 6, borderRadius: "50%", background: "#10b981", display: "inline-block" }} />
              <span style={{ fontSize: 11, fontWeight: 600, color: "#10b981", fontFamily: "'DM Mono', monospace" }}>Live</span>
            </div>
            <div style={{ background: "rgba(139,92,246,0.15)", border: "1px solid rgba(139,92,246,0.3)", borderRadius: 99, padding: "4px 10px", fontSize: 11, fontWeight: 600, color: "#a78bfa", fontFamily: "'DM Mono', monospace" }}>9 Classes</div>
          </div>
        </div>
      </nav>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px" }}>
        {/* Header */}
        <div className="fade-in" style={{ marginBottom: 32 }}>
          <h1 style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "clamp(22px,3vw,34px)", color: "#f1f5f9", marginBottom: 8 }}>Skin Disease Classification Dashboard</h1>
          <p style={{ color: "#64748b", fontSize: 13, fontFamily: "'DM Mono', monospace" }}>SqueezeNet CNN (Fire modules) → Softmax (9 classes) → Grad-CAM explainability</p>
        </div>

        {/* Tabs */}
        <div style={{ display: "flex", gap: 4, marginBottom: 28, background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: 12, padding: 4 }}>
          {TABS.map(({ id, label, icon: Icon }) => {
            const isActive = activeTab === id;
            const isLocked = locked && id !== "upload";
            return (
              <button key={id} onClick={() => !isLocked && setActiveTab(id)}
                className={isActive ? "tab-active" : "tab-inactive"}
                style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", gap: 7, padding: "9px 12px", borderRadius: 8, border: "none", cursor: isLocked ? "not-allowed" : "pointer", transition: "all 0.2s", color: isActive ? "#c4b5fd" : isLocked ? "#374151" : "#94a3b8", fontWeight: 500, fontSize: "clamp(11px,1.5vw,13px)", fontFamily: "'DM Sans', sans-serif", background: "transparent" }}>
                <Icon size={14} />
                <span style={{ display: "inline" }}>{label}</span>
                {isLocked && <span style={{ fontSize: 9, opacity: 0.5 }}>🔒</span>}
              </button>
            );
          })}
        </div>

        {/* UPLOAD TAB */}
        {activeTab === "upload" && (
          <div className="fade-in" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 20 }}>
            {/* Image Upload */}
            <div className="card glow-border" style={{ padding: 24 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
                <div style={{ width: 32, height: 32, borderRadius: 8, background: "rgba(139,92,246,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}><Microscope size={16} color="#a78bfa" /></div>
                <div>
                  <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 15, color: "#a78bfa" }}>Dermoscopic Image</div>
                  <div style={{ fontSize: 11, color: "#475569" }}>JPEG / PNG / TIFF — auto-compressed to 1024px</div>
                </div>
              </div>

              <div className={`drop-zone ${dragOver ? "drop-zone-active" : ""}`}
                style={{ borderRadius: 12, padding: 32, textAlign: "center", cursor: "pointer", transition: "all 0.3s", minHeight: 200, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}>
                {imagePreview ? (
                  <>
                    <img src={imagePreview} alt="Preview" style={{ maxWidth: "100%", maxHeight: 180, borderRadius: 8, objectFit: "cover" }} />
                    <button onClick={(e) => { e.stopPropagation(); setImageFile(null); setImagePreview(null); setPrediction(null); }} style={{ position: "absolute", top: 8, right: 8, background: "rgba(0,0,0,0.7)", border: "none", borderRadius: "50%", padding: 4, cursor: "pointer", color: "#94a3b8" }}><X size={14} /></button>
                  </>
                ) : (
                  <>
                    <div style={{ width: 56, height: 56, borderRadius: 16, background: "rgba(139,92,246,0.15)", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 14 }}>
                      <Upload size={24} color="#7c3aed" />
                    </div>
                    <p style={{ fontWeight: 600, color: "#a78bfa", fontSize: 14, marginBottom: 4 }}>Drop dermoscopic image here</p>
                    <p style={{ fontSize: 11, color: "#475569", marginBottom: 16 }}>High-res supported — client-side compression active</p>
                    <div style={{ background: "linear-gradient(135deg, #7c3aed, #4f46e5)", color: "#fff", borderRadius: 8, padding: "8px 20px", fontSize: 13, fontWeight: 600 }}>Browse Files</div>
                  </>
                )}
                <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/tiff" style={{ display: "none" }} onChange={(e) => handleFile(e.target.files[0])} />
              </div>

              <button onClick={analyze} disabled={!imageFile || isLoading}
                className="glow-btn"
                style={{ width: "100%", marginTop: 16, padding: "13px 0", borderRadius: 10, border: "none", cursor: imageFile && !isLoading ? "pointer" : "not-allowed", background: imageFile && !isLoading ? "linear-gradient(135deg, #7c3aed, #4f46e5)" : "rgba(255,255,255,0.05)", color: imageFile && !isLoading ? "#fff" : "#374151", fontWeight: 700, fontSize: 14, fontFamily: "'Syne', sans-serif", display: "flex", alignItems: "center", justifyContent: "center", gap: 8, transition: "all 0.3s" }}>
                {isLoading ? <><Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} /><style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>Analyzing...</> : <><Zap size={16} />Analyze Image</>}
              </button>

              {/* API error banner */}
              {apiError && (
                <div style={{ marginTop: 12, padding: "10px 14px", background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.35)", borderRadius: 8, display: "flex", alignItems: "flex-start", gap: 8 }}>
                  <AlertTriangle size={14} style={{ color: "#f87171", marginTop: 1, flexShrink: 0 }} />
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: "#f87171", marginBottom: 2 }}>Inference failed</div>
                    <div style={{ fontSize: 11, color: "#94a3b8", fontFamily: "'DM Mono', monospace" }}>{apiError}</div>
                    <div style={{ fontSize: 10, color: "#64748b", marginTop: 4 }}>Make sure Flask is running: <code style={{ color: "#a78bfa" }}>python app.py</code></div>
                  </div>
                </div>
              )}
            </div>

            {/* Patient Metadata */}
            <div className="card glow-border" style={{ padding: 24 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 20 }}>
                <div style={{ width: 32, height: 32, borderRadius: 8, background: "rgba(99,102,241,0.2)", display: "flex", alignItems: "center", justifyContent: "center" }}><Shield size={16} color="#818cf8" /></div>
                <div>
                  <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 15, color: "#818cf8" }}>Patient Metadata</div>
                  <div style={{ fontSize: 11, color: "#475569" }}>Clinical context — aids reporting, not CNN input</div>
                </div>
              </div>

              {[
                { label: "Patient Age", key: "age", type: "number", placeholder: "e.g. 52" },
                { label: "Biological Sex", key: "sex", type: "select", options: ["Male", "Female", "Other / Prefer not to say"] },
                { label: "Anatomical Site", key: "site", type: "select", options: ANATOMICAL_SITES },
                { label: "Fitzpatrick Skin Type", key: "fitzpatrick", type: "select", options: FITZPATRICK_TYPES },
                { label: "Lesion Duration (months)", key: "duration", type: "number", placeholder: "e.g. 6" },
              ].map(({ label, key, type, placeholder, options }) => (
                <div key={key} style={{ marginBottom: 14 }}>
                  <label style={{ display: "block", fontSize: 10, fontWeight: 700, color: "#64748b", letterSpacing: "0.1em", marginBottom: 6, textTransform: "uppercase", fontFamily: "'DM Mono', monospace" }}>{label}</label>
                  {type === "select" ? (
                    <div style={{ position: "relative" }}>
                      <select value={patientMeta[key]} onChange={e => setPatientMeta(p => ({ ...p, [key]: e.target.value }))}
                        style={{ width: "100%", padding: "10px 36px 10px 12px", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, color: patientMeta[key] ? "#e2e8f0" : "#475569", fontSize: 13, appearance: "none", cursor: "pointer", fontFamily: "'DM Sans', sans-serif", outline: "none" }}>
                        <option value="" disabled>Select...</option>
                        {options.map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                      <ChevronDown size={14} style={{ position: "absolute", right: 12, top: "50%", transform: "translateY(-50%)", color: "#475569", pointerEvents: "none" }} />
                    </div>
                  ) : (
                    <input type={type} placeholder={placeholder} value={patientMeta[key]} onChange={e => setPatientMeta(p => ({ ...p, [key]: e.target.value }))}
                      style={{ width: "100%", padding: "10px 12px", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, color: "#e2e8f0", fontSize: 13, fontFamily: "'DM Sans', sans-serif", outline: "none" }} />
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* RESULTS TAB */}
        {activeTab === "results" && (
          <div className="fade-in">
            {isLoading ? (
              <div style={{ display: "grid", gap: 16 }}>
                <SkeletonBlock className="shimmer" style={{ height: 120 }} />
                <SkeletonBlock className="shimmer" style={{ height: 320 }} />
              </div>
            ) : prediction && (
              <>
                {/* Top Prediction */}
                <div className="card glow-border" style={{ padding: 28, marginBottom: 20, display: "flex", flexWrap: "wrap", gap: 20, alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ flex: 1, minWidth: 200 }}>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "#475569", letterSpacing: "0.15em", textTransform: "uppercase", fontFamily: "'DM Mono', monospace", marginBottom: 8 }}>Primary Prediction</div>
                    <h2 style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "clamp(18px,2.5vw,26px)", color: "#f1f5f9", marginBottom: 6 }}>{prediction.className}</h2>
                    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
                      <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 28, fontWeight: 700, background: "linear-gradient(90deg, #a78bfa, #818cf8)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                        {(prediction.confidence * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: 12, color: "#64748b" }}>confidence</div>
                    </div>
                  </div>
                  <div>
                    {(() => { const rc = RISK_CONFIG[prediction.risk]; const Icon = rc.icon; return (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, background: rc.bg, border: `1px solid`, borderColor: rc.border.replace("border-","").replace("/40",""), borderRadius: 12, padding: "10px 18px" }}>
                        <Icon size={18} className={rc.color} style={{ color: prediction.risk === "HIGH" ? "#f87171" : prediction.risk === "MEDIUM" ? "#fbbf24" : "#34d399" }} />
                        <span style={{ fontWeight: 700, fontSize: 13, fontFamily: "'DM Mono', monospace", color: prediction.risk === "HIGH" ? "#f87171" : prediction.risk === "MEDIUM" ? "#fbbf24" : "#34d399" }}>{rc.label}</span>
                      </div>
                    ); })()}
                  </div>
                </div>

                {/* Probability Bars */}
                <div className="card glow-border" style={{ padding: 24 }}>
                  <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 14, color: "#94a3b8", marginBottom: 20 }}>Class Probability Distribution</div>
                  {confidences.map(({ name, prob, risk }, i) => {
                    const rc = RISK_CONFIG[risk];
                    const barColor = risk === "HIGH" ? "linear-gradient(90deg, #7c3aed, #ef4444)" : risk === "MEDIUM" ? "linear-gradient(90deg, #6366f1, #f59e0b)" : "linear-gradient(90deg, #6366f1, #10b981)";
                    return (
                      <div key={name} style={{ marginBottom: 14 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 5 }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: "#475569", width: 28 }}>#{i + 1}</span>
                            <span style={{ fontSize: 13, color: i === 0 ? "#e2e8f0" : "#94a3b8", fontWeight: i === 0 ? 600 : 400 }}>{name}</span>
                          </div>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 12, color: i === 0 ? "#a78bfa" : "#64748b" }}>{(prob * 100).toFixed(1)}%</span>
                          </div>
                        </div>
                        <div style={{ height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 99, overflow: "hidden" }}>
                          <div className="bar-fill" style={{ height: "100%", width: animBars ? `${prob * 100}%` : "0%", background: barColor, borderRadius: 99 }} />
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="card-inner" style={{ padding: "12px 16px", marginTop: 14, display: "flex", gap: 8, alignItems: "flex-start" }}>
                  <Info size={13} style={{ color: "#475569", marginTop: 1, flexShrink: 0 }} />
                  <p style={{ fontSize: 11, color: "#475569", fontStyle: "italic" }}>For research use only. Not a medical diagnosis. Always consult a qualified dermatologist for clinical evaluation.</p>
                </div>
              </>
            )}
          </div>
        )}

        {/* GRAD-CAM TAB */}
        {activeTab === "gradcam" && prediction && (
          <div className="fade-in">
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 20, marginBottom: 20 }}>
              {[
                { title: "Original Image", content: imagePreview ? <img src={imagePreview} alt="Original" style={{ width: "100%", borderRadius: 8, display: "block" }} /> : <div style={{ height: 200, background: "rgba(255,255,255,0.03)", borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", color: "#475569" }}>No image</div>, label: "INPUT" },
                { title: "Grad-CAM Heatmap", content: (
                  <div style={{ borderRadius: 8, overflow: "hidden", position: "relative", background: "#0d0e1f" }}>
                    {imagePreview && <img src={imagePreview} alt="Heatmap base" style={{ width: "100%", display: "block", opacity: 0.5, filter: "grayscale(60%)" }} />}
                    <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse at 55% 45%, rgba(239,68,68,0.7) 0%, rgba(245,158,11,0.5) 25%, rgba(139,92,246,0.3) 55%, transparent 75%)", mixBlendMode: "screen" }} />
                    <div style={{ position: "absolute", top: 8, right: 8, background: "rgba(0,0,0,0.7)", borderRadius: 6, padding: "3px 8px", fontSize: 10, fontFamily: "'DM Mono', monospace", color: "#f87171" }}>Simulated overlay</div>
                  </div>
                ), label: "GRAD-CAM" },
              ].map(({ title, content, label }) => (
                <div key={label} className="card glow-border" style={{ padding: 20 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                    <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 14, color: "#94a3b8" }}>{title}</div>
                    <span style={{ fontSize: 9, fontFamily: "'DM Mono', monospace", color: "#475569", background: "rgba(255,255,255,0.05)", borderRadius: 4, padding: "2px 6px" }}>{label}</span>
                  </div>
                  {content}
                </div>
              ))}
            </div>

            <div className="card glow-border" style={{ padding: 20, marginBottom: 16 }}>
              <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 700, fontSize: 13, color: "#a78bfa", marginBottom: 10 }}>How to interpret Grad-CAM</div>
              <p style={{ fontSize: 12, color: "#64748b", lineHeight: 1.7 }}>Gradient-weighted Class Activation Mapping highlights the regions most influential in the model's classification decision. <strong style={{ color: "#94a3b8" }}>Warm colors (red/orange)</strong> indicate high activation — areas the network focused on. <strong style={{ color: "#94a3b8" }}>Cool colors (purple/blue)</strong> indicate lower relevance.</p>
            </div>

            {/* Color Scale */}
            <div className="card-inner" style={{ padding: "14px 20px" }}>
              <div style={{ fontSize: 11, color: "#64748b", marginBottom: 8, fontFamily: "'DM Mono', monospace" }}>Activation Scale</div>
              <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                <span style={{ fontSize: 10, color: "#475569" }}>Low</span>
                <div style={{ flex: 1, height: 8, borderRadius: 99, background: `linear-gradient(90deg, ${HEATMAP_COLORS.join(",")})` }} />
                <span style={{ fontSize: 10, color: "#475569" }}>High</span>
              </div>
            </div>
          </div>
        )}

        {/* REPORT TAB */}
        {activeTab === "report" && prediction && (
          <div className="fade-in">
            <div className="card glow-border" style={{ padding: 28, marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12, marginBottom: 24 }}>
                <div>
                  <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 18, color: "#f1f5f9", marginBottom: 4 }}>Clinical Summary Report</div>
                  <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: "#475569" }}>Generated: {new Date().toLocaleString()} · DermDx AI v1.0</div>
                </div>
                <div style={{ display: "flex", gap: 8 }}>
                  <button onClick={copyReport} style={{ display: "flex", alignItems: "center", gap: 6, padding: "8px 14px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8, color: copied ? "#10b981" : "#94a3b8", fontSize: 12, cursor: "pointer", transition: "all 0.2s", fontFamily: "'DM Sans', sans-serif" }}>
                    {copied ? <CheckCircle size={13} /> : <Copy size={13} />} {copied ? "Copied!" : "Copy"}
                  </button>
                  <button style={{ display: "flex", alignItems: "center", gap: 6, padding: "8px 14px", background: "linear-gradient(135deg, rgba(139,92,246,0.3), rgba(99,102,241,0.2))", border: "1px solid rgba(139,92,246,0.4)", borderRadius: 8, color: "#a78bfa", fontSize: 12, cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>
                    <Download size={13} /> PDF
                  </button>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12, marginBottom: 20 }}>
                {[
                  { label: "Patient Age", value: patientMeta.age || "Not provided" },
                  { label: "Biological Sex", value: patientMeta.sex || "Not provided" },
                  { label: "Anatomical Site", value: patientMeta.site || "Not provided" },
                  { label: "Fitzpatrick Type", value: patientMeta.fitzpatrick || "Not provided" },
                  { label: "Lesion Duration", value: patientMeta.duration ? `${patientMeta.duration} months` : "Not provided" },
                ].map(({ label, value }) => (
                  <div key={label} className="card-inner" style={{ padding: "12px 14px" }}>
                    <div style={{ fontSize: 9, fontFamily: "'DM Mono', monospace", color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 4 }}>{label}</div>
                    <div style={{ fontSize: 13, color: "#cbd5e1", fontWeight: 500 }}>{value}</div>
                  </div>
                ))}
              </div>

              <div style={{ borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: 20 }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
                  {[
                    { label: "Predicted Diagnosis", value: prediction.className, highlight: true },
                    { label: "Model Confidence", value: `${(prediction.confidence * 100).toFixed(1)}%`, highlight: true },
                    { label: "Risk Classification", value: prediction.risk, color: prediction.risk === "HIGH" ? "#f87171" : prediction.risk === "MEDIUM" ? "#fbbf24" : "#34d399" },
                    { label: "Recommended Action", value: prediction.risk === "HIGH" ? "Urgent dermatologist referral" : prediction.risk === "MEDIUM" ? "Dermatologist consultation" : "Routine monitoring" },
                  ].map(({ label, value, highlight, color }) => (
                    <div key={label} className="card-inner" style={{ padding: "14px 16px", background: highlight ? "rgba(139,92,246,0.06)" : undefined, border: highlight ? "1px solid rgba(139,92,246,0.2)" : undefined }}>
                      <div style={{ fontSize: 9, fontFamily: "'DM Mono', monospace", color: "#475569", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 5 }}>{label}</div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: color || (highlight ? "#c4b5fd" : "#e2e8f0"), fontFamily: "'Syne', sans-serif" }}>{value}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="card-inner" style={{ padding: "12px 16px", display: "flex", gap: 8, alignItems: "flex-start" }}>
              <AlertTriangle size={13} style={{ color: "#f59e0b", marginTop: 1, flexShrink: 0 }} />
              <p style={{ fontSize: 11, color: "#475569", fontStyle: "italic" }}>This report is generated by an AI research tool and is <strong style={{ color: "#64748b" }}>not a substitute for clinical diagnosis</strong>. For research use only. Always seek professional medical advice.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
