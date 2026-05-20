import { useState, useRef, useCallback, useEffect } from 'react'
import { analyzeImage } from './services/api'

// ─── Icons (inline SVG to avoid extra deps) ──────────────────

const IconUpload = () => (
  <svg className="w-10 h-10" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
  </svg>
)

const IconShield = ({ className = 'w-6 h-6' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
  </svg>
)

const IconScan = ({ className = 'w-6 h-6' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
  </svg>
)

const IconCpu = ({ className = 'w-6 h-6' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M8.25 3v1.5M4.5 8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 002.25-2.25V6.75a2.25 2.25 0 00-2.25-2.25H6.75A2.25 2.25 0 004.5 6.75v10.5a2.25 2.25 0 002.25 2.25zm.75-12h9v9h-9v-9z" />
  </svg>
)

const IconBrain = ({ className = 'w-6 h-6' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z" />
  </svg>
)

const IconCheck = ({ className = 'w-6 h-6' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
)

const IconX = ({ className = 'w-5 h-5' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
)

const IconWarning = ({ className = 'w-5 h-5' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
  </svg>
)

const IconRefresh = ({ className = 'w-5 h-5' }) => (
  <svg className={className} fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
    <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
  </svg>
)

// ─── Stat pill ───────────────────────────────────────────────

function StatPill({ value, label }) {
  return (
    <div className="glass-card rounded-xl px-4 py-3 text-center min-w-[90px]">
      <div className="text-xl font-bold gradient-text">{value}</div>
      <div className="text-xs text-slate-400 mt-0.5 font-medium">{label}</div>
    </div>
  )
}

// ─── Pipeline step card ───────────────────────────────────────

function PipelineStep({ step, icon, title, description, delay }) {
  return (
    <div
      className="glass-card rounded-2xl p-6 flex flex-col gap-3 opacity-initial-0 animate-fade-up"
      style={{ animationDelay: `${delay}ms`, animationFillMode: 'forwards' }}
    >
      <div className="flex items-center gap-3">
        <div className="w-9 h-9 rounded-lg bg-purple-950/60 border border-purple-800/40 flex items-center justify-center text-purple-400 flex-shrink-0">
          {icon}
        </div>
        <span className="font-mono text-xs text-purple-400/70 font-semibold tracking-widest">
          STEP {step}
        </span>
      </div>
      <h3 className="text-white font-semibold text-base leading-snug">{title}</h3>
      <p className="text-slate-400 text-sm leading-relaxed">{description}</p>
    </div>
  )
}

// ─── Confidence bar ───────────────────────────────────────────

function ConfidenceBar({ confidence, isFake }) {
  const pct = Math.round(confidence * 100)
  const color = isFake
    ? 'from-red-600 to-red-400'
    : 'from-emerald-600 to-emerald-400'

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">
          Confidence
        </span>
        <span className={`text-sm font-bold font-mono ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
          {pct}%
        </span>
      </div>
      <div className="w-full bg-white/5 rounded-full h-2 overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${color} animate-bar-fill`}
          style={{ '--target-width': `${pct}%`, width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

// ─── Detection panel ─────────────────────────────────────────

const MAX_FILE_SIZE_MB = 10
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']

function DetectionPanel() {
  const [phase, setPhase] = useState('idle')   // idle | preview | loading | result | error
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)   // { label, confidence }
  const [errorMsg, setErrorMsg] = useState('')
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  const reset = useCallback(() => {
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPhase('idle')
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setErrorMsg('')
    setDragging(false)
  }, [previewUrl])

  const handleFile = useCallback((f) => {
    if (!f) return

    if (!ACCEPTED_TYPES.includes(f.type)) {
      setErrorMsg('Unsupported format. Please upload JPG, PNG, WEBP, or BMP.')
      setPhase('error')
      return
    }

    if (f.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setErrorMsg(`File too large. Maximum size is ${MAX_FILE_SIZE_MB} MB.`)
      setPhase('error')
      return
    }

    if (previewUrl) URL.revokeObjectURL(previewUrl)
    const url = URL.createObjectURL(f)
    setFile(f)
    setPreviewUrl(url)
    setPhase('preview')
    setErrorMsg('')
  }, [previewUrl])

  const handleInputChange = (e) => handleFile(e.target.files?.[0])

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  const handleDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const handleDragLeave = () => setDragging(false)

  const handleAnalyze = async () => {
    if (!file) return
    setPhase('loading')

    try {
      const data = await analyzeImage(file)
      setResult(data)
      setPhase('result')
    } catch (err) {
      setErrorMsg(err.message || 'Something went wrong. Is your backend running?')
      setPhase('error')
    }
  }

  const isFake = result?.label === 'FAKE'

  return (
    <div className="glass-card-strong rounded-3xl p-6 w-full max-w-md mx-auto flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white font-bold text-lg leading-tight">Image Analyzer</h2>
          <p className="text-slate-400 text-xs mt-0.5">Upload a surveillance image to detect deepfakes</p>
        </div>
        <div className="w-9 h-9 rounded-xl bg-purple-950/70 border border-purple-700/40 flex items-center justify-center text-purple-400">
          <IconScan className="w-5 h-5" />
        </div>
      </div>

      {/* ── IDLE: drop zone ── */}
      {phase === 'idle' && (
        <div
          className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center gap-3 cursor-pointer transition-all duration-200
            ${dragging
              ? 'border-purple-500/70 bg-purple-900/10'
              : 'border-white/10 hover:border-purple-500/40 hover:bg-white/[0.02]'
            }`}
          onClick={() => inputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <div className={`text-slate-500 transition-colors ${dragging ? 'text-purple-400' : ''}`}>
            <IconUpload />
          </div>
          <div className="text-center">
            <p className="text-slate-300 text-sm font-medium">
              {dragging ? 'Drop to upload' : 'Drag & drop or click to browse'}
            </p>
            <p className="text-slate-500 text-xs mt-1">JPG, PNG, WEBP, BMP — max {MAX_FILE_SIZE_MB} MB</p>
          </div>
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept={ACCEPTED_TYPES.join(',')}
            onChange={handleInputChange}
          />
        </div>
      )}

      {/* ── PREVIEW: show image + analyze btn ── */}
      {phase === 'preview' && (
        <div className="flex flex-col gap-4 animate-scale-in">
          <div className="relative rounded-2xl overflow-hidden border border-white/10 bg-black/20">
            <img
              src={previewUrl}
              alt="Selected"
              className="w-full object-cover max-h-56"
            />
            <button
              onClick={reset}
              className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/60 backdrop-blur-sm flex items-center justify-center text-slate-300 hover:text-white hover:bg-black/80 transition-all"
              title="Remove image"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="flex items-center gap-2 px-1">
            <div className="w-2 h-2 rounded-full bg-emerald-400 flex-shrink-0" />
            <span className="text-slate-400 text-xs truncate font-mono">{file?.name}</span>
            <span className="text-slate-600 text-xs ml-auto flex-shrink-0">
              {(file?.size / 1024 / 1024).toFixed(2)} MB
            </span>
          </div>

          <button
            onClick={handleAnalyze}
            className="w-full py-3.5 rounded-2xl font-semibold text-sm text-white
              bg-gradient-to-r from-purple-700 to-violet-600
              hover:from-purple-600 hover:to-violet-500
              active:scale-[0.98] transition-all duration-150
              glow-purple"
          >
            Analyze Image
          </button>
        </div>
      )}

      {/* ── LOADING: spinner ── */}
      {phase === 'loading' && (
        <div className="flex flex-col items-center gap-5 py-8 animate-fade-in">
          <div className="relative w-16 h-16">
            <div className="absolute inset-0 rounded-full border-2 border-purple-900" />
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-purple-500 animate-spin-slow" />
            <div className="absolute inset-2 rounded-full border border-transparent border-t-violet-400 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '0.8s' }} />
            <div className="absolute inset-0 flex items-center justify-center text-purple-400">
              <IconScan className="w-5 h-5" />
            </div>
          </div>
          <div className="text-center">
            <p className="text-white text-sm font-medium">Analyzing image…</p>
            <p className="text-slate-500 text-xs mt-1">Running DINOv2 + MLP inference</p>
          </div>
        </div>
      )}

      {/* ── RESULT ── */}
      {phase === 'result' && result && (
        <div className={`flex flex-col gap-4 animate-scale-in rounded-2xl p-4 border
          ${isFake
            ? 'bg-red-950/20 border-red-800/30 glow-red'
            : 'bg-emerald-950/20 border-emerald-800/30 glow-green'
          }`}>

          {/* Result image + verdict */}
          <div className="relative rounded-xl overflow-hidden border border-white/10">
            <img src={previewUrl} alt="Analyzed" className="w-full object-cover max-h-48" />
            <div className={`absolute inset-0 bg-gradient-to-t
              ${isFake ? 'from-red-950/80' : 'from-emerald-950/80'} to-transparent`}
            />
            <div className="absolute bottom-3 left-3 right-3 flex items-center gap-2">
              <div className={`w-7 h-7 rounded-full flex items-center justify-center
                ${isFake ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                {isFake ? <IconX /> : <IconCheck />}
              </div>
              <span className={`font-mono font-bold text-sm tracking-widest
                ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
                {isFake ? 'DEEPFAKE DETECTED' : 'AUTHENTIC IMAGE'}
              </span>
            </div>
          </div>

          {/* Verdict label */}
          <div className="text-center">
            <div className={`text-5xl font-black font-mono tracking-tight
              ${isFake ? 'gradient-text-result-fake' : 'gradient-text-result-real'}`}>
              {result.label}
            </div>
            <div className="text-slate-400 text-xs mt-1">Classification Result</div>
          </div>

          {/* Confidence bar */}
          <ConfidenceBar confidence={result.confidence} isFake={isFake} />

          {/* Metadata strip */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] px-3 py-2">
              <div className="text-slate-500">Model</div>
              <div className="text-slate-300 font-mono font-medium mt-0.5">DINOv2 + MLP</div>
            </div>
            <div className="rounded-lg bg-white/[0.03] border border-white/[0.06] px-3 py-2">
              <div className="text-slate-500">Timestamp</div>
              <div className="text-slate-300 font-mono font-medium mt-0.5">
                {new Date().toLocaleTimeString()}
              </div>
            </div>
          </div>

          {/* Retry */}
          <button
            onClick={reset}
            className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl
              border border-white/10 text-slate-400 text-sm hover:text-white hover:border-white/20
              hover:bg-white/[0.03] transition-all"
          >
            <IconRefresh className="w-4 h-4" />
            Analyze Another Image
          </button>
        </div>
      )}

      {/* ── ERROR ── */}
      {phase === 'error' && (
        <div className="flex flex-col gap-4 animate-scale-in rounded-2xl p-4 bg-amber-950/20 border border-amber-800/30">
          <div className="flex items-start gap-3">
            <div className="text-amber-400 mt-0.5 flex-shrink-0">
              <IconWarning />
            </div>
            <div>
              <p className="text-amber-300 text-sm font-semibold">Analysis Failed</p>
              <p className="text-slate-400 text-xs mt-1 leading-relaxed">{errorMsg}</p>
            </div>
          </div>
          <button
            onClick={reset}
            className="flex items-center justify-center gap-2 w-full py-2.5 rounded-xl
              border border-white/10 text-slate-400 text-sm hover:text-white
              hover:bg-white/[0.03] transition-all"
          >
            <IconRefresh className="w-4 h-4" />
            Try Again
          </button>
        </div>
      )}
    </div>
  )
}

// ─── Main App ─────────────────────────────────────────────────

export default function App() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <div className="min-h-screen noise-bg" style={{ background: '#07071a' }}>

      {/* ── Ambient background orbs ── */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden" aria-hidden>
        <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full opacity-[0.12]"
          style={{ background: 'radial-gradient(circle, #7c3aed, transparent 70%)' }} />
        <div className="absolute top-[10%] right-[-15%] w-[500px] h-[500px] rounded-full opacity-[0.08]"
          style={{ background: 'radial-gradient(circle, #4f46e5, transparent 70%)' }} />
        <div className="absolute bottom-[5%] left-[20%] w-[400px] h-[400px] rounded-full opacity-[0.07]"
          style={{ background: 'radial-gradient(circle, #0ea5e9, transparent 70%)' }} />
      </div>

      {/* Dot grid overlay */}
      <div className="fixed inset-0 pointer-events-none dot-grid opacity-40" aria-hidden />

      {/* ── Navbar ── */}
      <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300
        ${scrolled ? 'glass-card border-b border-white/[0.06] py-3' : 'py-5'}`}>
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-700 to-violet-600 flex items-center justify-center shadow-lg">
              <IconShield className="w-4 h-4 text-white" />
            </div>
            <span className="font-bold text-white text-sm tracking-tight">ForensicAI</span>
          </div>

          <div className="hidden sm:flex items-center gap-2 text-xs text-slate-500 font-medium">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse inline-block" />
            KIT&apos;s College of Engineering, Kolhapur
          </div>

          <a
            href="#how-it-works"
            className="text-xs text-slate-400 hover:text-white transition-colors font-medium px-3 py-1.5 rounded-lg hover:bg-white/[0.05]"
          >
            How it works
          </a>
        </div>
      </header>

      {/* ══════════════════════════════════════════════════
           HERO SECTION
      ══════════════════════════════════════════════════ */}
      <section className="relative min-h-screen flex items-center pt-24 pb-20">
        <div className="max-w-7xl mx-auto px-6 w-full">
          <div className="grid lg:grid-cols-2 gap-12 xl:gap-20 items-center">

            {/* Left — project info */}
            <div className="flex flex-col gap-6">

              {/* Badge */}
              <div className="opacity-initial-0 animate-fade-up animate-delay-100"
                style={{ animationFillMode: 'forwards' }}>
                <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full
                  border border-purple-700/40 bg-purple-950/40 text-purple-300 text-xs font-semibold tracking-wider uppercase">
                  <IconShield className="w-3.5 h-3.5" />
                  B.Tech Final Year Project · CSBS · 2025
                </span>
              </div>

              {/* Title */}
              <div className="opacity-initial-0 animate-fade-up animate-delay-200"
                style={{ animationFillMode: 'forwards' }}>
                <h1 className="text-4xl sm:text-5xl xl:text-6xl font-black leading-[1.08] tracking-tight">
                  <span className="text-white">Forensic CCTV</span>
                  <br />
                  <span className="gradient-text">Deepfake</span>
                  <br />
                  <span className="text-white">Detection System</span>
                </h1>
              </div>

              {/* Description */}
              <div className="opacity-initial-0 animate-fade-up animate-delay-300"
                style={{ animationFillMode: 'forwards' }}>
                <p className="text-slate-400 text-base leading-relaxed max-w-lg">
                  A frame-level forensic screening pipeline using a frozen <span className="text-slate-300 font-medium">DINOv2 ViT-S/14</span> vision
                  encoder and a lightweight <span className="text-slate-300 font-medium">MLP classifier head</span>, evaluated explicitly
                  across CCTV degradation profiles. Forensic-ready logging for surveillance-imagery deepfake detection.
                </p>
              </div>

              {/* Stats */}
              <div className="opacity-initial-0 animate-fade-up animate-delay-400"
                style={{ animationFillMode: 'forwards' }}>
                <div className="flex flex-wrap gap-3">
                  <StatPill value="0.88" label="Val. AUC-ROC" />
                  <StatPill value="0.80" label="Val. F1" />
                  <StatPill value="10K" label="Frames" />
                  <StatPill value="5×2" label="Eval matrix" />
                </div>
              </div>

              {/* Institution + Team */}
              <div className="opacity-initial-0 animate-fade-up animate-delay-500"
                style={{ animationFillMode: 'forwards' }}>
                <div className="glass-card rounded-2xl p-4 flex flex-col gap-4">

                  <div className="flex items-start gap-3">
                    <div className="w-8 h-8 rounded-lg bg-indigo-950/60 border border-indigo-800/30 flex items-center justify-center flex-shrink-0">
                      <svg className="w-4 h-4 text-indigo-400" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4.26 10.147a60.436 60.436 0 00-.491 6.347A48.627 48.627 0 0112 20.904a48.627 48.627 0 018.232-4.41 60.46 60.46 0 00-.491-6.347m-15.482 0a50.57 50.57 0 00-2.658-.813A59.905 59.905 0 0112 3.493a59.902 59.902 0 0110.399 5.84c-.896.248-1.783.52-2.658.814m-15.482 0A50.697 50.697 0 0112 13.489a50.702 50.702 0 017.74-3.342M6.75 15a.75.75 0 100-1.5.75.75 0 000 1.5zm0 0v-3.675A55.378 55.378 0 0112 8.443m-7.007 11.55A5.981 5.981 0 006.75 15.75v-1.5" />
                      </svg>
                    </div>
                    <div>
                      <div className="text-white text-sm font-semibold leading-snug">
                        KIT&apos;s College of Engineering, Kolhapur
                      </div>
                      <div className="text-slate-500 text-xs mt-0.5">
                        Dept. of Computer Science & Business Systems
                      </div>
                    </div>
                  </div>

                  <div className="border-t border-white/[0.05] pt-3 grid grid-cols-2 gap-3 text-xs">
                    <div>
                      <div className="text-slate-600 font-medium uppercase tracking-wider mb-1.5">Team</div>
                      <div className="flex flex-col gap-1">
                        {[
                          'Aditya Salunkhe',
                          'Harshwardhan Powar',
                          'Ridhima Pore',
                          'Shivani Rawool',
                          'Sanket Desai',
                        ].map((name) => (
                          <div key={name} className="flex items-center gap-1.5 text-slate-300">
                            <div className="w-1 h-1 rounded-full bg-purple-500 flex-shrink-0" />
                            {name}
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-600 font-medium uppercase tracking-wider mb-1.5">Guided by</div>
                      <div className="text-slate-300">Mr. J. S. Pujari</div>
                      <div className="text-slate-600 text-[11px] mt-0.5">Project Supervisor</div>
                      <div className="mt-3 text-slate-600 font-medium uppercase tracking-wider mb-1.5">HOD</div>
                      <div className="text-slate-300">Dr. M. R. Hudagi</div>
                      <div className="text-slate-600 text-[11px] mt-0.5">Head of Department</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right — detection panel */}
            <div className="opacity-initial-0 animate-fade-up animate-delay-300 lg:animate-delay-100"
              style={{ animationFillMode: 'forwards' }}>
              <div className="animate-float">
                <DetectionPanel />
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════
           HOW IT WORKS
      ══════════════════════════════════════════════════ */}
      <section id="how-it-works" className="relative py-24 border-t border-white/[0.05]">
        <div className="max-w-7xl mx-auto px-6">

          {/* Section header */}
          <div className="text-center mb-14">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full
              border border-indigo-700/40 bg-indigo-950/40 text-indigo-300 text-xs font-semibold tracking-wider uppercase mb-4">
              Detection Pipeline
            </div>
            <h2 className="text-3xl sm:text-4xl font-black text-white">
              How It <span className="gradient-text">Works</span>
            </h2>
            <p className="text-slate-400 mt-3 max-w-xl mx-auto text-sm leading-relaxed">
              A five-stage screening pipeline processes your surveillance image from raw pixels to a
              forensic-ready authenticity verdict with a tamper-evident audit trail.
            </p>
          </div>

          {/* Steps grid */}
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            <PipelineStep
              step={1}
              icon={<IconUpload />}
              title="Image Upload"
              description="Upload a CCTV or surveillance image. Accepted formats: JPG, PNG, WEBP, BMP up to 10 MB."
              delay={0}
            />
            <PipelineStep
              step={2}
              icon={<IconScan className="w-5 h-5" />}
              title="Face Crop"
              description="RetinaFace (with MTCNN fallback) locates the strongest face, expands by 1.3× margin, and resizes to 224×224."
              delay={100}
            />
            <PipelineStep
              step={3}
              icon={<IconCpu className="w-5 h-5" />}
              title="Forensic Enhancement"
              description="Optional deterministic pre-processing: CLAHE on L channel + bilateral denoise + unsharp mask. No learned prior, cannot hallucinate."
              delay={200}
            />
            <PipelineStep
              step={4}
              icon={<IconBrain className="w-5 h-5" />}
              title="DINOv2 + MLP"
              description="A frozen ViT-S/14 self-supervised encoder extracts 384-d CLS features; a 99K-parameter MLP head outputs a real/fake logit."
              delay={300}
            />
            <PipelineStep
              step={5}
              icon={<IconShield className="w-5 h-5" />}
              title="Verdict & Audit Trail"
              description="Binary classification with confidence + ViT attention rollout heatmap. SHA-256 of input and model logged for forensic chain-of-custody."
              delay={400}
            />
          </div>

          {/* Tech tags */}
          <div className="mt-10 flex flex-wrap gap-2 justify-center">
            {[
              'DINOv2 ViT-S/14', 'MLP head', 'Attention Rollout', 'RetinaFace', 'MTCNN',
              'CLAHE', 'PyTorch', 'transformers', 'Albumentations',
              'Degradation-stratified eval', 'SHA-256 CoC',
            ].map((tag) => (
              <span
                key={tag}
                className="px-3 py-1 rounded-full text-xs font-medium
                  bg-white/[0.04] border border-white/[0.07] text-slate-400
                  hover:text-slate-200 hover:border-white/[0.14] transition-colors"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="border-t border-white/[0.05] py-8">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs text-slate-600">
          <span>
            © 2025 Forensic CCTV Deepfake Detection System · KIT&apos;s CoEK, Kolhapur
          </span>
          <span className="font-mono">
            DINOv2 ViT-S/14 + MLP · Val. AUC-ROC 0.88
          </span>
        </div>
      </footer>

    </div>
  )
}
