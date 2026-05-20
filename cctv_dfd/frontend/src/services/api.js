// ─────────────────────────────────────────────────────────────
//  BACKEND CONFIGURATION
//  Set VITE_API_URL in your .env file to point at your backend.
//  Default fallback: http://localhost:5000
//
//  Your backend endpoint must accept:
//    POST /predict
//    Content-Type: multipart/form-data
//    Body field:   image  (the uploaded file)
//
//  Expected JSON response format:
//    { "label": "REAL" | "FAKE",  "confidence": 0.93 }
//    confidence is a number between 0.0 and 1.0
// ─────────────────────────────────────────────────────────────

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

export async function analyzeImage(imageFile) {
  const formData = new FormData()
  formData.append('image', imageFile)

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let message = `Server error (${response.status})`
    try {
      const err = await response.json()
      if (err.message || err.error) message = err.message || err.error
    } catch (_) {}
    throw new Error(message)
  }

  const data = await response.json()

  // Normalise: label to uppercase, confidence clamped 0–1
  return {
    label: String(data.label).toUpperCase(),
    confidence: Math.min(1, Math.max(0, Number(data.confidence))),
  }
}
