# Forensic CCTV Deepfake Detection System — Frontend

> React + Tailwind CSS frontend for the B.Tech Final Year Project  
> KIT's College of Engineering, Kolhapur — CSBS Department — 2025

---

## Prerequisites

Before you start, make sure the following are installed on your machine:

| Tool | Version | Download |
|------|---------|----------|
| Node.js | v18 or higher | https://nodejs.org |
| npm | v9 or higher (ships with Node) | — |

To check if Node is already installed, open a terminal and run:
```
node -v
npm -v
```

---

## Project Setup (First Time Only)

**1. Open a terminal in the project folder**

Right-click inside `F:\AI Face Detection Page` in File Explorer → "Open in Terminal"  
Or in VS Code: Terminal → New Terminal

**2. Install dependencies**
```
npm install
```
This downloads React, Vite, Tailwind CSS, and all other packages into a `node_modules` folder.

---

## Running the Website on Localhost

```
npm run dev
```

The site will be live at:
```
http://localhost:3000
```

Open that URL in your browser. The page hot-reloads whenever you save a file — no need to restart.

To stop the server: press `Ctrl + C` in the terminal.

---

## Connecting Your Backend (Python / Flask / FastAPI)

The frontend calls your backend at one endpoint:

```
POST /predict
Content-Type: multipart/form-data
Field name:   image   (the uploaded image file)
```

### Step 1 — Set your backend URL

Create a file called `.env` in the project root (copy from `.env.example`):

```
VITE_API_URL=http://localhost:5000
```

Change `5000` to whatever port your Python backend runs on.

### Step 2 — Make sure your backend returns this JSON format

```json
{
  "label": "FAKE",
  "confidence": 0.93
}
```

- `label` must be either `"REAL"` or `"FAKE"` (case-insensitive, the frontend normalises it)
- `confidence` is a decimal between `0.0` and `1.0` (e.g. `0.87` = 87%)

### Step 3 — Enable CORS on your backend

The browser will block requests to a different port unless your backend allows it.

**Flask example:**
```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow all origins during development

@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    # ... run your model ...
    return jsonify({"label": "FAKE", "confidence": 0.93})
```
Install: `pip install flask-cors`

**FastAPI example:**
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    # ... run your model ...
    return {"label": "FAKE", "confidence": 0.93}
```

---

## File Structure

```
F:\AI Face Detection Page\
│
├── index.html              ← App entry point (HTML shell)
├── package.json            ← Project dependencies & scripts
├── vite.config.js          ← Vite dev server + proxy config
├── tailwind.config.js      ← Tailwind theme (colours, animations)
├── postcss.config.js       ← PostCSS config for Tailwind
├── .env.example            ← Template for environment variables
├── .env                    ← Your local env (create this yourself)
│
└── src/
    ├── main.jsx            ← React entry point
    ├── App.jsx             ← Entire single-page application
    ├── index.css           ← Tailwind directives + global styles
    │
    └── services/
        └── api.js          ← Backend API integration (edit this)
```

### The only file you need to edit to connect your backend:

**`src/services/api.js`** — contains the `analyzeImage()` function.

If your endpoint is named differently (e.g. `/api/detect` instead of `/predict`), change line 27:
```js
const response = await fetch(`${API_BASE_URL}/predict`, {   // ← change /predict here
```

If your backend field name is different (e.g. `file` instead of `image`), change line 21:
```js
formData.append('image', imageFile)   // ← change 'image' to match your backend
```

---

## Building for Production

```
npm run build
```

This outputs a production-ready static build to the `dist/` folder.  
You can host it on any static server (Nginx, Apache, Netlify, Vercel, etc.).

To preview the production build locally:
```
npm run preview
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `npm install` fails | Make sure Node 18+ is installed. Delete `node_modules` and try again. |
| Page loads but upload does nothing | Your backend is not running. Start your Python server first. |
| "Analysis Failed" error | Check the browser console (F12) for the actual error. Usually a CORS issue or wrong port. |
| CORS error in browser | Add `flask-cors` or FastAPI `CORSMiddleware` to your backend (see above). |
| Backend not receiving the file | Make sure your Flask/FastAPI route reads `request.files['image']` — the field name must match. |
| Port 3000 already in use | Change it in `vite.config.js`: `server: { port: 3001 }` |

---

## Team

| Name | Roll No. |
|------|----------|
| Aditya Shivaji Salunkhe | 05 |
| Harshwardhan Haridas Powar | 30 |
| Ridhima Sachin Pore | 47 |
| Shivani Ladaji Rawool | 52 |
| Sanket Satyajit Desai | 62 |

**Guide:** Mr. J. S. Pujari  
**HOD:** Dr. M. R. Hudagi  
**Co-Guide:** Mrs. Yashaswini Kadiyal  
**Institution:** KIT's College of Engineering, Kolhapur (Empowered Autonomous)  
**Department:** Computer Science & Business Systems
