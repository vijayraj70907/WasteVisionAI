# WasteVision AI — Complete Project Documentation

> **Model Version:** v2.0 — Multi-class Waste Specialist  
> **Validation Accuracy:** 99.48%  
> **Run Command:** `python app.py`  
> **App URL:** `http://127.0.0.1:5000`

---

# PART 1 — PROBLEM STATEMENT & MOTIVATION

## 1.1 The Problem

Improper disposal of biomedical and general waste is one of the most pressing environmental and public health challenges in the world today.

**In hospitals and healthcare facilities:**
- Used syringes discarded in general bins cause needle-stick injuries to sanitation workers
- Expired medications flushed down drains contaminate water supplies
- Infectious and hazardous waste mixed with regular garbage spreads disease

**In homes and communities:**
- People throw glass, plastic, and paper into the same bin, making recycling impossible
- Most individuals do not know the correct bin colour or disposal procedure for each waste type
- Awareness campaigns have limited reach and fail to create lasting behaviour change

**The scale of the problem:**
- The World Health Organization estimates that 85% of hospital waste is non-hazardous but only 15% is infectious or hazardous — yet improper labelling and sorting puts ALL of it at risk
- India alone generates over 600 tonnes of biomedical waste per day
- Less than 30% of plastic waste globally is collected for recycling
- Needle-stick injuries affect over 1 million healthcare workers annually

## 1.2 Why Existing Solutions Fall Short

| Existing Approach | Limitation |
|---|---|
| Printed posters on walls | Static, ignored, not context-aware |
| Annual training workshops | Knowledge fades; no real-time help |
| Manual sorting by staff | Expensive, error-prone, health risk |
| Barcode scanners | Require pre-labelled items; can't identify unlabelled waste |
| Sensor-based smart bins | Very expensive; require infrastructure changes |

## 1.3 The Proposed Solution

**WasteVision AI** is an intelligent, camera-driven waste classification system that:

1. **Identifies** the waste type in real time from a camera or photo
2. **Tells** the user exactly how to dispose of it safely
3. **Warns** loudly when hazardous waste (syringes) is detected
4. **Rewards** users for correct disposal behaviour to build lasting habits
5. **Shows** the community's collective impact to create social motivation
6. **Guides** users to the nearest physical disposal facility

The key insight is that **behaviour change requires three things:** knowledge, motivation, and community. WasteVision AI delivers all three.

---

# PART 2 — PROJECT OVERVIEW

## 2.1 What is WasteVision AI?

WasteVision AI is a full-stack web application powered by a deep learning model (MobileNetV2) that classifies 7 categories of waste from camera or uploaded images. It runs entirely in the browser — users just open a webpage and point their phone or webcam at waste.

## 2.2 Who is it for?

| User Type | How they use it |
|---|---|
| Hospital nurses and staff | Quick check before disposal — is this safe to throw in the regular bin? |
| Sanitation workers | Identify unknown waste without touching it |
| Pharmacy counter staff | Identify returned medicine waste type |
| General public | Learn recycling rules for glass, paper, plastic |
| Students and researchers | Educational tool for waste awareness |
| Healthcare administrators | Track and report waste disposal metrics |

## 2.3 Core Value Proposition

- **Zero hardware cost** — works on any smartphone or laptop with a camera
- **No app to install** — runs in the browser
- **Instant results** — less than 2 seconds per prediction
- **Safe for everyone** — tells you what NOT to touch before you touch it
- **99.48% accuracy** on 7 waste classes

---

# PART 3 — TECHNOLOGY STACK

## 3.1 Backend

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Web Framework | Flask | 3.1.3 | Routes, templates, session management |
| ML Framework | TensorFlow | 2.21.0 | Model loading and inference |
| Image Processing | Pillow | 10.3.0 | Open, resize, convert camera frames |
| Password Security | Werkzeug | 3.1.3 | PBKDF2-SHA256 password hashing |
| PDF Generation | ReportLab | 4.2.5 | Export scan history as A4 PDF |
| AI Tips API | Google Gemini 2.0 Flash | — | Generate disposal instructions |
| Database | SQLite | built-in | Store users, predictions, feedback |
| Production Server | Gunicorn | 26.0.0 | WSGI server for deployment |

## 3.2 Frontend

| Component | Technology | Purpose |
|---|---|---|
| Markup | HTML5 | Page structure |
| Styling | CSS3 (inline per template) | Dark-theme UI, animations |
| Logic | Vanilla JavaScript | Camera access, fetch API, speech |
| Charts | Chart.js 4.4.2 (CDN) | Dashboard doughnut and line charts |
| Icons | Lucide Icons (CDN) | Sidebar and button icons |
| Fonts | Inter via Google Fonts | Typography |
| Maps | Google Maps JS API + Places | Disposal site finder |

## 3.3 AI & Data

| Component | Detail |
|---|---|
| Model architecture | MobileNetV2 (ImageNet pre-trained) |
| Custom head | GlobalAveragePooling2D → BatchNorm → Dense(256) → Dropout(0.4) → Dense(128) → Dropout(0.3) → Dense(7, Softmax) |
| Input size | 224 × 224 × 3 (RGB) |
| Output | 7-class probability vector |
| Training data | 3,879 images across 7 classes |
| Val accuracy | **99.48%** |
| AI tip source | Google Gemini 2.0 Flash API (with hardcoded fallbacks) |
| Dataset licence | CC BY 4.0 (RealWaste), CC BY-SA 4.0 (RHWC) |

---

# PART 4 — SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────┐
│                    USER BROWSER                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Camera   │  │ Upload   │  │ Charts / Maps /  │  │
│  │ (WebRTC) │  │ File     │  │ Speech / Audio   │  │
│  └────┬─────┘  └────┬─────┘  └──────────────────┘  │
└───────┼─────────────┼──────────────────────────────-┘
        │ base64      │ multipart/form-data
        ▼             ▼
┌─────────────────────────────────────────────────────┐
│              FLASK APPLICATION (app.py)              │
│                                                      │
│  ┌────────────┐   ┌──────────────┐  ┌────────────┐  │
│  │ /predict   │   │ Auth Routes  │  │ Page Routes│  │
│  │ /predict-  │   │ /login       │  │ /          │  │
│  │ frame      │   │ /signup      │  │ /dashboard │  │
│  └─────┬──────┘   └──────────────┘  └────────────┘  │
│        │                                             │
│  ┌─────▼──────┐   ┌──────────────┐  ┌────────────┐  │
│  │ preprocess │   │   SQLite DB  │  │  Gemini    │  │
│  │ PIL image  │──▶│ predictions  │  │  API call  │  │
│  └─────┬──────┘   │ users        │  │ (cached)   │  │
│        │          │ feedback     │  └────────────┘  │
│  ┌─────▼──────┐   └──────────────┘                  │
│  │MobileNetV2 │                                      │
│  │ .h5 model  │                                      │
│  └─────┬──────┘                                      │
│        │ JSON response                               │
└────────┼────────────────────────────────────────────-┘
         ▼
  {prediction, confidence, disposal_tip,
   is_unknown, model_version, eco_points, badge}
```

### Data flow for a single prediction:
1. User points camera at waste or uploads a photo
2. Browser captures a JPEG frame (1.5s interval for live, or on button click for upload)
3. Frame sent to Flask as base64 JSON (live) or multipart form (upload)
4. PIL opens the image, resizes to 224×224, applies MobileNetV2 preprocessing
5. TensorFlow runs inference — 7 probability values returned
6. If top probability < 50% → return "Unknown / Not Sure"
7. If top probability ≥ 50% → look up class name from `class_names.json`
8. If confidence ≥ 75% → save to `predictions` table + award eco-points to user
9. Fetch disposal tip from in-process cache (or call Gemini API if not cached)
10. Return JSON to browser
11. Browser updates UI: result card, confidence bar, tip card, hazard banner, points toast

---

# PART 5 — PROJECT STRUCTURE

```
WasteVision_AI-main/
│
├── app.py                          # Core application — all routes, model, DB
├── train_model.py                  # Trains MobileNetV2 (2-phase fine-tuning)
├── augment_dataset.py              # Augments dataset images
├── download_new_classes.py         # Downloads Glass/Paper/Plastic images
├── convert_model.py                # Converts .h5 to .keras format
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version (3.11.11)
├── history.db                      # SQLite database (auto-created at startup)
│
├── model/
│   ├── biowaste_best_model.h5      # Trained model weights (7 classes)
│   └── class_names.json            # Index → class label mapping
│
├── dataset/                        # Training data (one folder per class)
│   ├── UnsedTablets/               # 517 images
│   ├── UnusedSyringe/              # 669 images
│   ├── WasteSyringe/               # 604 images
│   ├── WasteTablets/               # 589 images
│   ├── Glass/                      # 500 images (RealWaste CC BY 4.0)
│   ├── Paper/                      # 500 images (RealWaste CC BY 4.0)
│   └── Plastic/                    # 500 images (RealWaste CC BY 4.0)
│
├── uploads/                        # Images uploaded via /predict (auto-created)
│
├── static/
│   ├── css/style.css               # Global stylesheet (mostly overridden by templates)
│   ├── js/script.js                # Legacy JS file
│   └── images/                     # logo.png, bins.png, syringe.png, tablets.png, etc.
│
└── templates/                      # Jinja2 HTML templates (one per page)
    ├── index.html                  # Home/Landing page
    ├── live_detection.html         # Live camera detection
    ├── upload_image.html           # Image upload + batch
    ├── detection_history.html      # Scan history table
    ├── history_detail.html         # Single scan detail
    ├── dashboard.html              # Personal analytics
    ├── leaderboard.html            # Top users ranking
    ├── disposal_finder.html        # Google Maps disposal finder
    ├── login_signup.html           # Combined auth page
    ├── waste_guide.html            # Waste segregation guide
    ├── about.html                  # Project info
    └── contact.html                # Team + feedback form
```

---

# PART 6 — DATABASE DESIGN

## 6.1 predictions table
Stores every detection that passes the 75% confidence threshold.

```sql
CREATE TABLE predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT,       -- uploaded file name or "live_camera"
    predicted_class TEXT,       -- e.g., "Waste Syringe"
    confidence      REAL,       -- 0.0 to 100.0
    source          TEXT,       -- "upload" or "live"
    created_at      TEXT        -- "YYYY-MM-DD HH:MM:SS"
);
```

## 6.2 users table
Stores registered accounts. Eco-points and badge are updated after every qualifying scan.

```sql
CREATE TABLE users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL,
    email       TEXT    UNIQUE NOT NULL,
    password    TEXT    NOT NULL,   -- PBKDF2-SHA256 hash, never plain text
    eco_points  INTEGER NOT NULL DEFAULT 0,
    badge       TEXT    NOT NULL DEFAULT 'Newcomer'
);
```

## 6.3 feedback table
Stores contact form submissions.

```sql
CREATE TABLE feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    email       TEXT,
    rating      TEXT,
    feedback    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

# PART 7 — WASTE CLASSES SUPPORTED

| Index | Class Label | Type | Hazardous | Disposal Method |
|---|---|---|---|---|
| 0 | Glass Waste | General Recycling | No | Rinse, remove lids, glass recycling bin |
| 1 | Paper Waste | General Recycling | No | Flatten, keep dry, paper recycling bin |
| 2 | Plastic Waste | General Recycling | No | Rinse, check recycling number, blue bin |
| 3 | Unused Tablets | Medical | No | Return to pharmacy take-back programme |
| 4 | Unused Syringe | Medical | **YES** | Cap needle, sharps container, clinic hand-off |
| 5 | Waste Syringe | Medical | **YES** | Rigid sharps container, hospital disposal site |
| 6 | Waste Tablets | Medical | No | Sealable bag, pharmacy take-back |

**Hazardous classes** (Unused Syringe, Waste Syringe) trigger:
- Pulsing red video border
- Red flashing hazard banner
- Web Audio API two-tone alarm (repeating every 4 seconds)

---

# PART 8 — FEATURE DEEP DIVE

## 8.1 Home Page (`/`)

The landing page is the first thing every visitor sees. It does not require login.

**Purpose:** Introduce the project, showcase what it can do, and funnel users into using live detection or upload.

**Key sections:**

**Hero area** — A bold headline "Detect. Segregate. Protect." with two call-to-action buttons:
- *Live Detection* — opens the camera page
- *Upload Image* — opens the upload page

**Community Impact Stats bar** — Four live tiles that auto-refresh every 30 seconds:
- Total Scans: how many waste items have been detected by ALL users combined
- Hazardous Flagged: total syringes identified (serious health risk prevented)
- Top This Week: the most-detected waste class in the last 7 days with scan count
- CO₂ Saved: estimated 0.5 kg CO₂ equivalent per scan (reflects environmental benefit)

All tiles animate with a smooth count-up effect on page load. This creates a sense that the platform is alive and active.

**What We Detect grid** — Image cards showing each waste category type to educate visitors before they try the tool.

---

## 8.2 Login & Sign Up (`/login-signup`)

**Purpose:** Allow users to create accounts so their eco-points, badges, and scan history are tracked across sessions.

**Why accounts are needed:**
Without user accounts, eco-points and leaderboard rankings cannot be attributed to individuals, and the gamification system loses meaning. The history and dashboard pages also require user identity to show personalised data.

**Login tab:**
- Enter email and password
- Password is verified against a PBKDF2-SHA256 hash stored in the database
- On success: session stores user_id, username, eco_points, badge
- On failure: a clear inline error message appears (no page reload)

**Sign Up tab:**
- Enter username, email, and password
- Duplicate emails show an inline error
- Passwords are **never stored as plain text** — Werkzeug hashes them before insertion

**Design choice:** Combined login/signup as a single page reduces friction. Users don't have to navigate between pages.

---

## 8.3 Live Camera Detection (`/live-detection`)

**Purpose:** Real-time waste identification using the device camera.

**Requires login:** Yes. Login is required so that eco-points from live detections are attributed to the correct user account.

**How it works step by step:**
1. User clicks **Start Camera** — browser's permission dialog appears
2. On permission grant, the video stream starts in the left panel
3. A JavaScript `setInterval` fires every **1,500 milliseconds**
4. Each tick: the current video frame is drawn onto a hidden `<canvas>`, converted to a base64 JPEG string
5. The string is sent as JSON to `/predict-frame`
6. Flask decodes the base64, opens it with PIL, resizes to 224×224, preprocesses, runs inference
7. The JSON result is returned and the UI updates

**Result panel (right side):**
- Detected class name in large bold text
- Animated confidence progress bar (green-to-yellow gradient)
- Confidence percentage: e.g., "Confidence: 94.72%"
- AI Disposal Tip card (green) with instructions specific to that class
- Read Aloud button (🔊) — speaks the result + tip using browser SpeechSynthesis
  - Auto-fires when the class **changes** — does not speak every 1.5 seconds to avoid annoyance
- Model version badge

**Unknown state:** If confidence < 50%, the prediction turns yellow and shows:
*"⚠️ Confidence below 50% — model is unsure. Reposition item for a clearer view."*
Nothing is saved, no points are awarded.

**Hazard alert system (for syringes):**
When Unused Syringe or Waste Syringe is detected:
- Red pulsing outline appears around the video feed
- A red banner appears: *"⚠️ HAZARDOUS WASTE DETECTED — Do NOT touch with bare hands!"*
- Web Audio API plays a two-tone alarm (880 Hz → 660 Hz square wave, 0.18 volume)
- Alarm repeats every 4 seconds while the hazardous class remains detected
- All alerts clear automatically when a non-hazardous class is detected

This system exists because syringes carry bloodborne pathogens (HIV, Hepatitis B/C). An incorrect disposal can cause life-threatening needle-stick injuries. The multi-sensory alert (visual + audio + text) ensures the warning is not missed.

**Eco-points:** Awarded automatically for every detection saved (confidence ≥ 75%):
- Live detection = +5 points
- Confidence ≥ 90% = additional +2 bonus points

---

## 8.4 Upload Image (`/upload-image`)

**Purpose:** Classify one or multiple waste images from saved photos.

**Requires login:** No — anyone can use this without an account. Points are only awarded if logged in.

**Single file mode:**
1. Click the dashed drop zone or browse
2. Preview appears
3. Click Predict
4. Result card updates instantly with class, confidence, disposal tip, voice button

**Batch mode (multiple files):**
1. Select multiple files using Ctrl+Click in the file browser
2. Click Predict
3. Each file is processed **sequentially** (in order)
4. A progress bar shows: *"Processing 3 of 7: syringe_photo.jpg"*
5. Results appear as a card grid, colour-coded:
   - Green bar = safe class, confident prediction
   - Yellow bar = unknown / low confidence
   - Red bar = hazardous class
6. Each card shows: thumbnail, filename, class label, confidence bar

**Why batch mode?** Healthcare facilities often photograph multiple items at once for audit purposes. Batch upload removes the need to process each image individually.

**Export PDF row** (at the bottom): Date pickers to filter range → Download PDF Report button. The PDF is generated server-side by ReportLab and downloaded directly.

---

## 8.5 Detection History (`/detection-history`)

**Purpose:** View, review, and manage all past scans.

**Requires login:** Yes.

**Table columns:**
- File Name (uploaded image or "live_camera")
- Detected Item (class label, or yellow "❓ Unknown / Not Sure" badge)
- Category (Hazardous Waste ☣ / Infectious Waste ☣ / Sharps ☣)
- Confidence (colour-coded: green ≥ 90%, yellow ≥ 50%, red < 50%)
- Date & Time
- Delete button (🗑) — auth-guarded, only logged-in users can delete

**Pagination:** 10 records per page to keep the page fast. Arrow navigation and numbered page links.

**Why is history important?**
- Hospitals need audit trails of waste disposal records
- Users can track their scanning behaviour over time
- Provides accountability — who detected what, when

**Model version display** and **Export PDF row** at the bottom of the table.

---

## 8.6 My Dashboard (`/dashboard`)

**Purpose:** Personal analytics — a visual summary of the user's scanning activity.

**Requires login:** Yes.

**4 Stat Cards:**
| Card | What it shows | Why it matters |
|---|---|---|
| Total Scans | Your all-time saved detections | Measures engagement |
| CO₂ Saved | scans × 0.5 kg (estimate) | Tangible environmental impact |
| Eco Points | Your current point balance | Gamification progress |
| Active Days | Days with detections in last 30 | Habit formation tracking |

**Doughnut Chart — Waste Categories:**
Shows the breakdown of everything you have scanned by class. This tells you at a glance which waste types appear most in your environment — useful for targeted training.

**Line Chart — Scans Over Time:**
Shows daily scan counts for the last 30 days. Helps identify whether scanning behaviour is consistent or sporadic.

**Carbon Impact callout:** Below the line chart, a green pill badge states the estimated total CO₂ saved. This reinforces the environmental value of the user's actions.

---

## 8.7 Eco Points — Why They Exist

### The behavioural problem
Knowing what to do and actually doing it consistently are very different things. Research in behaviour change consistently shows that **intrinsic motivation alone is insufficient** for building long-term habits. People need:

1. **Immediate feedback** — did I do the right thing?
2. **Progress tracking** — am I improving?
3. **Social comparison** — how do I compare to others?
4. **Rewards** — what do I get for doing the right thing?

Eco-points address all four.

### How points are earned

| Action | Points |
|---|---|
| Live camera detection (confidence ≥ 75%) | +5 points |
| Image upload detection (confidence ≥ 75%) | +3 points |
| Bonus: confidence ≥ 90% (either mode) | +2 additional points |
| Low-confidence or unknown detections | 0 points |

### Why these specific values?
- Live detection earns more (+5) than upload (+3) because live detection is harder — the user has to bring the waste to the camera, which implies more active engagement with disposal.
- The bonus for high confidence (+2) rewards users who provide clean, clear images, which helps the system work correctly.
- No points for unknown/low-confidence detections prevents gaming the system by rapidly scanning random items.

### Where points appear in the UI
1. **Topbar points pill** — visible on every page. Shows "120 pts · Green Warrior" and links to the Leaderboard.
2. **Points toast notification** — a popup in the bottom-right corner immediately after earning: "+5 eco pts 🌱 Total: 120 · Green Warrior". Disappears after 3.2 seconds.
3. **Dashboard stat card** — running total.
4. **Leaderboard** — rank among all users.

---

## 8.8 Badges — Why They Exist

### The problem with points alone
Raw numbers ("you have 120 points") are motivating up to a point, but they do not create identity. Badges give users a **named role** that they identify with and want to protect or upgrade.

### The four badge tiers

| Badge | Threshold | Symbol | Identity |
|---|---|---|---|
| Newcomer | 0 points | 🌱 | Just getting started |
| Green Warrior | 50 points | ♻️ | Actively recycling |
| Recycling Champion | 150 points | 🏆 | Consistent, skilled |
| Zero Waste Hero | 350 points | 🌍 | Role model for others |

### Why these threshold values?
- **50 points for Green Warrior**: Roughly 10 live scans. A new user reaches this within their first serious session — provides early positive reinforcement so users don't quit before seeing progress.
- **150 points for Recycling Champion**: Requires sustained engagement across multiple days (~30 scans). Filters casual from committed users.
- **350 points for Zero Waste Hero**: A significant milestone that typically takes weeks of daily use. Creates an aspirational goal.

### Badge psychology
Once a user earns "Green Warrior", they are unlikely to stop scanning — doing so would mean losing progress toward the next badge. This is called the **sunk cost motivation** and is a well-documented driver in gamified systems (Duolingo streaks, fitness app achievements, etc.).

---

## 8.9 Leaderboard — Why It Exists

### The social motivation problem
Individual eco-points motivate the individual, but they do not create community pressure. The Leaderboard converts private behaviour into a **public social signal**.

### What is shown
- Top 20 users ranked by total eco-points
- Gold 🥇 / Silver 🥈 / Bronze 🥉 medals for top 3
- Username, badge, and point total for each entry
- Visible to everyone — no login required to view

### Why the Leaderboard matters
1. **Social proof** — seeing "Zero Waste Hero" next to a username tells everyone that high-scoring behaviour is possible and aspirational
2. **Competitive motivation** — users who are close to a higher rank are strongly motivated to close the gap
3. **Community identity** — the leaderboard turns isolated individuals into a visible community of environmentally responsible people
4. **Accountability** — being ranked publicly creates mild accountability to maintain or improve one's position
5. **Recruitment** — when a new user sees the leaderboard, they immediately understand what the platform values and want to participate

### Why it is public (no login required)
Making the leaderboard public maximises social pressure. Anyone — even a non-registered visitor — can see it. This also serves as a recruitment tool: "Join WasteVision AI and see your name here."

---

## 8.10 Find Disposal Site (`/disposal-finder`)

**Purpose:** Bridge the gap between knowing how to dispose of waste and actually being able to do it. "Return to a pharmacy" is useless advice if the user doesn't know where the nearest pharmacy is.

**Requires login:** No.

**How it works:**
1. User selects waste type from dropdown (e.g., "Unused Syringe → Hospital / Clinic")
2. Clicks **Find Nearby Sites**
3. Browser's geolocation API fetches user coordinates
4. If `GOOGLE_MAPS_API_KEY` is set:
   - Google Places `nearbySearch` runs within 8 km radius
   - Dark-styled interactive map appears with up to 10 numbered markers
   - Results panel lists facility name, address, distance, open/closed status
   - Clicking a result pans the map to that location
5. If no API key: clear fallback text advice is shown with the detected coordinates

**Waste type to search keyword mapping:**
| Dropdown Selection | Places Keyword | Type |
|---|---|---|
| Unused Tablets | pharmacy medicine disposal | pharmacy |
| Unused Syringe | hospital clinic sharps bin | hospital |
| Waste Syringe | hospital clinic sharps bin | hospital |
| Waste Tablets | pharmacy medicine disposal | pharmacy |
| General Biomedical | biomedical waste recycling | establishment |

---

## 8.11 Waste Guide (`/waste-guide`)

A static reference page. No login required.

**Contents:**
- Detailed disposal instructions for each of the 4 medical waste classes
- Bin colour-coding guide:
  - 🟡 Yellow — infectious waste
  - 🔴 Red — contaminated plastic
  - 🔵 Blue — glass/recyclable
  - 🟢 Green — general waste
- Safety guidelines for handling medical waste

---

## 8.12 About (`/about`)

Describes the project: mission, features, model accuracy (99%), tech stack, and waste categories. Educational reference for stakeholders and evaluators.

---

## 8.13 Contact (`/contact`)

**Team members and roles:**
- Soumili Debnath
- Sayani Chatterjee
- Subham Biswas
- Rishita Chakraborty
- Saswata Sur
- **Mentor:** Dr. Tathagata Roy Chowdhury

**Feedback form:** Name, Email, Rating (1–5), Message. Submissions are saved to the `feedback` table. Allows the team to collect user experience data.

---

# PART 9 — SECURITY DESIGN

## 9.1 Password Security
- **Werkzeug `generate_password_hash`** uses PBKDF2-SHA256 with a random salt
- Passwords are **never stored in plain text** and **never logged**
- `check_password_hash` performs constant-time comparison to prevent timing attacks

## 9.2 Route Protection
Routes that expose personal data or allow destructive actions require authentication:
- `/live-detection` — requires login (points attribution)
- `/detection-history` — requires login (personal data)
- `/dashboard` — requires login (personal analytics)
- `/export-pdf` — requires login (personal data download)
- `/delete-history/<id>` — requires login (destructive action)

## 9.3 Input Validation
- `/predict` rejects files not in {png, jpg, jpeg, gif, webp}
- PIL image open is wrapped in try/except — broken files return HTTP 400
- All SQL queries use parameterised `?` placeholders — no string interpolation
- `secure_filename()` used on all uploaded filenames

## 9.4 Session Management
- Flask session with a secret key (`SECRET_KEY` env var, change before production)
- Session stores: user_id, username, eco_points, badge
- `session.clear()` on logout

---

# PART 10 — API REFERENCE

## 10.1 All Endpoints

| Method | URL | Auth | Description |
|---|---|---|---|
| GET | `/` | No | Home page |
| GET | `/home` | No | Redirect to home |
| GET | `/live-detection` | **Yes** | Live camera page |
| GET | `/upload-image` | No | Upload page |
| GET | `/detection-history?page=N` | **Yes** | History table |
| GET | `/dashboard` | **Yes** | Analytics dashboard |
| GET | `/api/dashboard-data` | **Yes** | JSON dashboard data |
| GET | `/leaderboard` | No | Top 20 users |
| GET | `/disposal-finder` | No | Maps finder page |
| GET | `/waste-guide` | No | Waste guide |
| GET | `/about` | No | About page |
| GET | `/contact` | No | Contact page |
| GET | `/login-signup` | No | Auth page |
| POST | `/signup` | No | Register account |
| POST | `/login` | No | Authenticate |
| GET | `/logout` | No | Clear session |
| POST | `/predict` | No | Predict uploaded image |
| POST | `/predict-frame` | No | Predict camera frame |
| GET | `/stats` | No | Community stats JSON |
| GET | `/export-pdf?from=&to=` | **Yes** | Download PDF report |
| GET | `/history/<int:id>` | No | Single scan detail |
| GET | `/delete-history/<int:id>` | **Yes** | Delete a scan |
| POST | `/submit-feedback` | No | Save feedback |

## 10.2 POST /predict — Request & Response

**Request:** `multipart/form-data` with field `file`

**Response JSON:**
```json
{
  "prediction":    "Waste Syringe",
  "confidence":    94.72,
  "disposal_tip":  "Do NOT recap a used syringe. Drop it directly into a rigid, puncture-proof sharps container...",
  "is_unknown":    false,
  "model_version": "v2.0 — Multi-class Waste Specialist",
  "eco_points":    120,
  "badge":         "Green Warrior"
}
```

**Unknown response (confidence < 50%):**
```json
{
  "prediction":    "Unknown / Not Sure",
  "confidence":    34.21,
  "disposal_tip":  "",
  "is_unknown":    true,
  "model_version": "v2.0 — Multi-class Waste Specialist",
  "eco_points":    118,
  "badge":         "Green Warrior"
}
```

## 10.3 POST /predict-frame — Request & Response

**Request:** JSON body
```json
{ "image": "data:image/jpeg;base64,/9j/4AAQ..." }
```

**Response:** Same format as `/predict`

## 10.4 GET /stats — Response

```json
{
  "total_scans":     312,
  "hazardous_count": 84,
  "top_class_week":  "Waste Syringe",
  "top_class_count": 46,
  "co2_saved":       156.0
}
```

---

# PART 11 — THE AI MODEL IN DETAIL

## 11.1 Why MobileNetV2?

MobileNetV2 was chosen over larger models (ResNet, VGG, EfficientNet) for three reasons:

1. **Speed**: It runs inference in ~200ms on CPU — fast enough for real-time camera processing
2. **Size**: The model is only ~14 MB — can be served over a web server and loaded quickly
3. **Accuracy**: Despite its compact size, fine-tuned MobileNetV2 achieves >99% accuracy on focused waste classification tasks

## 11.2 Transfer Learning

MobileNetV2 was pre-trained on ImageNet (1.28M images, 1000 classes). Its convolutional layers have already learned to detect edges, textures, shapes, and object parts. We freeze these layers and only train our custom classification head — dramatically reducing the amount of training data needed.

## 11.3 Two-Phase Training

**Phase 1 — Top-layer training (base frozen)**
- The MobileNetV2 base is frozen (no weight updates)
- Only the custom head (BatchNorm, Dense layers, Dropout) is trained
- Learning rate: 1e-3 (Adam)
- Result: model learns the basic mapping from features → 7 classes
- Best accuracy: ~93.9% after 2 epochs

**Phase 2 — Fine-tuning (last 30 layers unfrozen)**
- The top 30 layers of MobileNetV2 are unfrozen
- Lower learning rate (1e-4) to avoid destroying ImageNet features
- The model now learns features specific to our waste dataset
- Result: accuracy climbs to **99.48%** by epoch 9

## 11.4 Confidence Thresholds

Three thresholds control the system's behaviour:

| Threshold | Value | Effect |
|---|---|---|
| Display threshold | 50% | Below this → "Unknown / Not Sure" |
| History threshold | 75% | Below this → not saved, no points |
| Bonus threshold | 90% | Above this → +2 bonus points |

These values are intentional:
- 50% prevents the system from confidently labelling random objects
- 75% ensures only meaningful, accurate detections contribute to history and metrics
- 90% rewards users who provide ideal images (good lighting, clear view, steady hand)

## 11.5 Model Files

| File | Description |
|---|---|
| `model/biowaste_best_model.h5` | Keras HDF5 weights — loaded at startup |
| `model/class_names.json` | `{"0":"Glass Waste","1":"Paper Waste",...}` — auto-generated by train_model.py |

The JSON file is critical: it decouples the model from the application code. When you retrain with new classes, just run `train_model.py` — it overwrites the JSON, and the app picks up the new classes automatically without any code changes.

---

# PART 12 — GEMINI AI DISPOSAL TIPS

## 12.1 What it does

After every confident prediction, WasteVision AI provides specific, expert-quality disposal instructions tailored to the exact waste class detected. These are not generic warnings — they include:
- Specific safety steps (e.g., wear gloves, do not recap needles)
- The correct container type (e.g., rigid sharps container, sealable bag)
- Where to take it (hospital, pharmacy, recycling centre)

## 12.2 How it works

1. After prediction, `get_disposal_tip(predicted_class)` is called
2. The function first checks `_tip_cache` (in-process Python dict)
3. If cached → return immediately (sub-millisecond)
4. If not cached and `GEMINI_API_KEY` is set → call Gemini 2.0 Flash API
5. Prompt: *"You are a medical waste safety expert. Give concise, practical safe-disposal instructions (3-4 sentences) for: {class}. Focus on safety steps, correct bin/container type, and where to take it. Plain text only."*
6. Response cached for the lifetime of the server process
7. If API unavailable or key not set → hardcoded fallback tips returned

## 12.3 Fallback tips

All 7 classes have hand-written fallback tips baked into `app.py`. The app never shows an error to the user — fallback is seamless and indistinguishable from API-generated tips.

## 12.4 Enable Gemini
```powershell
$env:GEMINI_API_KEY = "your_key_from_google_ai_studio"
python app.py
```

---

# PART 13 — PDF EXPORT SYSTEM

**Library:** ReportLab 4.2.5  
**Format:** A4, landscape margins 2cm all sides

**Report sections:**
1. Title bar: "WasteVision AI — Detection Report" in green
2. User name and generation timestamp
3. Optional date range (from URL query params `?from=YYYY-MM-DD&to=YYYY-MM-DD`)
4. Horizontal rule in green
5. Summary table (5 rows): Total Scans, Hazardous Items, Eco Points, Badge, CO₂ Saved
6. Full scan history table with 6 columns: #, Filename, Detected Class, Confidence, Source, Date
   - Column header row: dark background with white bold text
   - Alternating row colours (white / light green)
   - Hazardous rows: light red background, red class name text
7. Footer: model version + "Hazardous items highlighted in red"

**Usage:**
- Via Upload Image page: set date pickers → click Download PDF Report
- Via Detection History page: same controls
- Direct URL: `http://127.0.0.1:5000/export-pdf?from=2025-01-01&to=2025-12-31`

---

# PART 14 — COMMUNITY STATS SYSTEM

## Why community stats on the Home page?

Individual stats motivate individuals. Community stats motivate communities.

When a visitor sees "3,847 total scans" and "CO₂ Saved: 1,923 kg", they understand:
- They are not the only person using this tool
- The platform has real-world impact at scale
- Joining adds to something bigger than themselves

This is called **social proof** — one of the most powerful drivers of human behaviour.

## How it is calculated

Stats are computed fresh from the database on every `/stats` request:

```
total_scans     = SELECT COUNT(*) FROM predictions

hazardous_count = SELECT COUNT(*) FROM predictions
                  WHERE predicted_class IN ('Unused Syringe','Waste Syringe')

top_class_week  = SELECT predicted_class, COUNT(*) as cnt
                  FROM predictions
                  WHERE DATE(created_at) >= (today - 7 days)
                    AND predicted_class != 'Unknown / Not Sure'
                  GROUP BY predicted_class
                  ORDER BY cnt DESC
                  LIMIT 1

co2_saved       = total_scans × 0.5 kg
```

## Auto-refresh

The Home page calls `/stats` on load and then every 30 seconds using `setInterval`. Each update triggers the count-up animation on all 4 tiles, making the page feel live and dynamic.

---

# PART 15 — TRAINING PIPELINE (STEP BY STEP)

## Step 1 — Add images to dataset folders

For existing classes, images are already in `dataset/`. For new classes:
```
dataset/
  Glass/      ← add glass waste images here
  Paper/      ← add paper waste images here
  Plastic/    ← add plastic waste images here
```

## Step 2 — Download new class images (automated)

```bash
python download_new_classes.py
```

This script:
- Checks for Kaggle API credentials (env var `KAGGLE_API_TOKEN`, `~/.kaggle/kaggle.json`, or interactive login)
- Alternatively, downloads from HuggingFace parquet shards without any credentials
- Copies 500 images each for Glass, Paper, Plastic into the dataset folders
- Validates that each folder has enough images before proceeding

## Step 3 — Augment the dataset

```bash
python augment_dataset.py
```

This script:
- Loops through every class folder under `dataset/`
- Skips folders with > 499 images (already sufficient)
- For each original image, generates **25 augmented variants** using:
  - Rotation up to ±30°
  - Zoom up to 20%
  - Width/height shift up to 20%
  - Shear up to 20%
  - Brightness range [0.7, 1.3]
  - Random horizontal flip
- Saves augmented images with `aug_` prefix so originals are preserved
- Safe to re-run — will not re-augment already-augmented folders

## Step 4 — Train the model

```bash
python train_model.py
```

This script:
- Auto-detects all class folders under `dataset/` — no hardcoded class list
- Builds a `folder_name → human_label` mapping
- Splits data 80/20 train/validation
- Phase 1: trains only the custom head (15 epochs, lr=1e-3)
- Phase 2: unfreezes last 30 MobileNetV2 layers and fine-tunes (10 epochs, lr=1e-4)
- Saves the best-performing checkpoint to `model/biowaste_best_model.h5`
- Writes `model/class_names.json` with the final index→label mapping
- Prints instructions to update `MODEL_VERSION` in `app.py`

Expected training time:
- CPU only (no GPU): ~20–40 minutes
- With CUDA GPU: ~5–10 minutes

## Step 5 — Restart the app

```bash
python app.py
```

No other changes needed. `app.py` reads `class_names.json` at startup and automatically reflects whatever classes were trained.

---

# PART 16 — INSTALLATION & RUN COMMANDS

## 16.1 Requirements

- Python 3.11 or 3.12
- pip
- A webcam or camera (for live detection)
- ~2 GB disk space (for model + datasets)

## 16.2 Install from scratch

```bash
# Clone the repository
git clone https://github.com/debnathsoumili782-ui/WasteVision_AI.git
cd WasteVision_AI-main

# Install all dependencies
pip install -r requirements.txt
```

## 16.3 Run the application

```bash
python app.py
```

Open browser at: **http://127.0.0.1:5000**

## 16.4 Run with optional features enabled

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY      = "your_gemini_key"
$env:GOOGLE_MAPS_API_KEY = "your_maps_key"
$env:SECRET_KEY          = "a_strong_random_secret"
python app.py
```

```bash
# Linux / macOS
export GEMINI_API_KEY="your_gemini_key"
export GOOGLE_MAPS_API_KEY="your_maps_key"
export SECRET_KEY="a_strong_random_secret"
python app.py
```

## 16.5 Run on a custom port

```bash
# Linux / macOS
PORT=8080 python app.py

# Windows PowerShell
$env:PORT=8080; python app.py
```

## 16.6 Run in debug mode (development only)

```bash
# Linux / macOS
FLASK_DEBUG=1 python app.py

# Windows PowerShell
$env:FLASK_DEBUG=1; python app.py
```

## 16.7 Retrain the model end-to-end

```bash
# Step 1: Download Glass/Paper/Plastic training images
python download_new_classes.py

# Step 2: Augment all classes that need more images
python augment_dataset.py

# Step 3: Train (takes 20-40 min on CPU)
python train_model.py

# Step 4: Restart Flask
python app.py
```

## 16.8 Dependencies (requirements.txt)

```
Flask==3.1.3
gunicorn==26.0.0
tensorflow==2.15.1
numpy==1.26.4
opencv-python-headless==4.10.0.84
Pillow==10.4.0
h5py==3.10.0
reportlab==4.2.5
werkzeug==3.1.3
```

---

# PART 17 — DEVELOPMENT PHASES SUMMARY

## Phase 1 — Security Fix + Core Engagement

**Problem solved:** The original app stored passwords in plain text — a critical security vulnerability. Users had no reason to keep coming back.

**What was built:**
- Replaced plain-text passwords with Werkzeug PBKDF2 hashing
- Eco-points system: +5 live, +3 upload, +2 bonus for confidence ≥ 90%
- Four badge tiers (Newcomer → Zero Waste Hero)
- Leaderboard with top 20 users
- Hazard alert banner + Web Audio API alarm for syringes
- Points toast notifications
- Auth guard on delete-history route

## Phase 2 — Intelligence + Insight

**Problem solved:** The app classified waste but didn't tell users what to DO with it. Users had no visibility into their own data.

**What was built:**
- Google Gemini AI integration for class-specific disposal instructions
- Personal analytics dashboard (Chart.js doughnut + line charts)
- Carbon impact estimates on dashboard
- Voice announcement (SpeechSynthesis API) — auto-fires on class change
- Read Aloud button on result cards

## Phase 3 — Real-World Reach + Scale

**Problem solved:** The app was still only useful for individuals in isolation. It needed community features, location awareness, and reporting.

**What was built:**
- Community stats bar on Home page (4 tiles, 30s auto-refresh)
- Google Maps disposal site finder with geolocation
- PDF export (ReportLab, date-range, hazard highlighting)
- Batch image upload with results grid
- Confidence threshold UI (unknown state, yellow styling, warning note)
- Model version badge in all result panels
- `/stats` public endpoint

## Model Upgrade — v1 (4 classes) → v2 (7 classes)

**Problem solved:** The model only covered medical waste, missing the most common recyclable materials.

**What was built:**
- Downloaded 500 CC BY 4.0 images each for Glass, Paper, Plastic
- Balanced dataset: 500–669 images per class, 3,879 total
- Retrained with two-phase MobileNetV2 fine-tuning
- **Final accuracy: 99.48%**
- `class_names.json` pattern: zero code changes required on any future retrain

---

# PART 18 — AUTHORS & CREDITS

## Development Team

| Name | Role |
|---|---|
| Soumili Debnath | Lead Developer |
| Sayani Chatterjee | Developer |
| Subham Biswas | Developer |
| Rishita Chakraborty | Developer |
| Saswata Sur | Developer |

**Project Mentor:** Dr. Tathagata Roy Chowdhury

## Dataset Credits

| Dataset | Source | Licence | Used For |
|---|---|---|---|
| Original medical waste images | Custom collection | Project-internal | Syringes, Tablets (4 classes) |
| RealWaste | UCI Machine Learning Repository / Kaggle | CC BY 4.0 | Glass, Paper, Plastic images |
| RHWC | Kaggle | CC BY-SA 4.0 | Additional recycling images |

**Licence:** This project is developed for academic and educational purposes.

---

*End of Document — WasteVision AI v2.0*
