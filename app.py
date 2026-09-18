from flask import Flask, render_template, request, jsonify, redirect, url_for, session, Response
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from io import BytesIO
import base64
import sqlite3
import os
import datetime
import urllib.request
import urllib.error
import json
import numpy as np
import tensorflow as tf
import keras

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.environ.get("SECRET_KEY", "WasteVision_AI_secret_key_change_in_prod")

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "model", "biowaste_best_model.h5")
CLASS_MAP_PATH= os.path.join(BASE_DIR, "model", "class_names.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH       = os.path.join(BASE_DIR, "history.db")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = keras.models.load_model(MODEL_PATH, compile=False)

with open(CLASS_MAP_PATH, encoding="utf-8") as _f:
    _raw = json.load(_f)
CLASS_NAMES: dict[int, str] = {int(k): v for k, v in _raw.items()}

NUM_CLASSES = len(CLASS_NAMES)

MODEL_VERSION        = "v2.0 — Multi-class Waste Specialist"
CONFIDENCE_THRESHOLD = 50.0
HISTORY_MIN_CONFIDENCE = 75.0
HAZARDOUS_CLASSES    = {"Unused Syringe", "Waste Syringe"}

# ---------------------------------------------------------------------------
# Badge tier logic
# ---------------------------------------------------------------------------
BADGE_TIERS = [
    (0,    "Newcomer",           "🌱"),
    (50,   "Green Warrior",      "♻️"),
    (150,  "Recycling Champion", "🏆"),
    (350,  "Zero Waste Hero",    "🌍"),
]

def get_badge(points: int) -> tuple[str, str]:
    """Return (badge_name, badge_emoji) for the given point total."""
    badge_name, badge_emoji = BADGE_TIERS[0][1], BADGE_TIERS[0][2]
    for threshold, name, emoji in BADGE_TIERS:
        if points >= threshold:
            badge_name, badge_emoji = name, emoji
    return badge_name, badge_emoji


# ---------------------------------------------------------------------------
# Gemini AI disposal-tip helper
# ---------------------------------------------------------------------------

# Hardcoded fallback tips used when the API is unavailable or the key is unset
FALLBACK_TIPS: dict[str, str] = {
    "Unused Tablets": (
        "Store unused tablets in their original packaging in a cool, dry place. "
        "Return them to a pharmacy or authorised medicine take-back programme. "
        "Never flush tablets down the toilet or throw them in general waste."
    ),
    "Unused Syringe": (
        "Cap the needle immediately after use to prevent needle-stick injuries. "
        "Place the capped syringe in a puncture-resistant sharps container. "
        "Seal and label the container, then hand it to a healthcare facility or "
        "authorised sharps disposal point. Do NOT place in household recycling."
    ),
    "Waste Syringe": (
        "Do NOT recap a used syringe. Drop it directly into a rigid, puncture-proof "
        "sharps container. When the container is ¾ full, seal it and deliver it to "
        "a hospital, clinic, or approved hazardous-waste collection site. "
        "Wear protective gloves throughout handling."
    ),
    "Waste Tablets": (
        "Do not crush or dissolve expired tablets. Place them in a sealable bag "
        "and return to a pharmacy medicine take-back programme. "
        "If no take-back is available, mix with an undesirable substance (e.g., coffee grounds) "
        "in a sealed bag and dispose in household waste — never in recycling or compost."
    ),
    "Plastic Waste": (
        "Rinse plastic containers before disposal to remove food residue. "
        "Check the recycling number (1–7) on the bottom — numbers 1 and 2 are widely accepted "
        "by kerbside recycling programmes. Place clean plastics in your blue recycling bin. "
        "Avoid single-use plastic bags — take them to supermarket collection points instead."
    ),
    "Paper Waste": (
        "Flatten cardboard boxes and remove any plastic tape before recycling. "
        "Shredded paper and tissue/paper towels generally cannot be recycled — place in general waste. "
        "Keep paper dry; wet or greasy paper (e.g., pizza boxes) goes in compost or general waste. "
        "Place clean, dry paper and cardboard in your recycling bin."
    ),
    "Glass Waste": (
        "Rinse glass bottles and jars thoroughly. Lids and caps are usually a different material — "
        "remove them and recycle separately if metal, or dispose in general waste if plastic. "
        "Never put broken glass loose into a recycling bin; wrap it in newspaper and label 'broken glass' "
        "before placing in general waste to protect waste handlers."
    ),
}

# In-process cache so we only hit the API once per class per server lifetime
_tip_cache: dict[str, str] = {}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent?key={key}"
)

def get_disposal_tip(predicted_class: str) -> str:
    """
    Return an AI-generated disposal tip for the given waste class.
    Results are cached in-process. Falls back to FALLBACK_TIPS on any error.
    """
    if predicted_class in _tip_cache:
        return _tip_cache[predicted_class]

    if not GEMINI_API_KEY:
        tip = FALLBACK_TIPS.get(predicted_class, "Dispose of this item at a certified biomedical waste facility.")
        _tip_cache[predicted_class] = tip
        return tip

    prompt = (
        f"You are a medical waste safety expert. "
        f"Give concise, practical safe-disposal instructions (3-4 sentences) for: {predicted_class}. "
        f"Focus on safety steps, correct bin/container type, and where to take it. Plain text only."
    )
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 200, "temperature": 0.3}
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            GEMINI_URL.format(key=GEMINI_API_KEY),
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        tip = body["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as exc:
        print(f"[Gemini] API error for '{predicted_class}': {exc}")
        tip = FALLBACK_TIPS.get(predicted_class, "Dispose of this item at a certified biomedical waste facility.")

    _tip_cache[predicted_class] = tip
    return tip


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Prediction history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            predicted_class TEXT,
            confidence REAL,
            source TEXT,
            created_at TEXT
        )
    """)

    # Users table — eco_points and badge added with safe ALTER if missing
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            eco_points INTEGER NOT NULL DEFAULT 0,
            badge TEXT NOT NULL DEFAULT 'Newcomer'
        )
    """)

    # Migrate: add eco_points column to existing DBs that predate this schema
    try:
        cur.execute("ALTER TABLE users ADD COLUMN eco_points INTEGER NOT NULL DEFAULT 0")
    except Exception:
        pass  # column already exists

    # Migrate: add badge column to existing DBs that predate this schema
    try:
        cur.execute("ALTER TABLE users ADD COLUMN badge TEXT NOT NULL DEFAULT 'Newcomer'")
    except Exception:
        pass  # column already exists

    # Feedback table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            rating TEXT,
            feedback TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

init_db()


# ---------------------------------------------------------------------------
# Image preprocessing & inference
# ---------------------------------------------------------------------------
def preprocess_pil_image(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr


def predict_array(arr):
    preds = model.predict(arr, verbose=0)[0]
    class_index = int(np.argmax(preds))
    confidence  = float(preds[class_index]) * 100

    if class_index not in CLASS_NAMES:
        return "Unknown / Not Sure", round(confidence, 2)

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_class = "Unknown / Not Sure"
    else:
        predicted_class = CLASS_NAMES[class_index]

    print(f"[predict] class={class_index} label={predicted_class} conf={confidence:.1f}%")
    return predicted_class, round(confidence, 2)


# ---------------------------------------------------------------------------
# History + eco-points helper
# ---------------------------------------------------------------------------
def save_history(filename, predicted_class, confidence, source):
    """
    Insert a prediction row and, if a user is logged in, award eco-points:
      • Live detection  → +5 pts
      • Upload          → +3 pts
      • Confidence ≥ 90 → +2 bonus pts
    Then recompute and persist the badge tier.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (filename, predicted_class, confidence, source, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (filename, predicted_class, confidence, source, now)
    )

    # Award eco-points to the logged-in user (if any)
    user_id = session.get("user_id")
    if user_id:
        points = 5 if source == "live" else 3
        if confidence >= 90:
            points += 2

        cur.execute("SELECT eco_points FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        if row:
            new_total = row[0] + points
            badge_name, _ = get_badge(new_total)
            cur.execute(
                "UPDATE users SET eco_points = ?, badge = ? WHERE id = ?",
                (new_total, badge_name, user_id)
            )
            # Keep session values fresh so the topbar updates immediately
            session["eco_points"] = new_total
            session["badge"] = badge_name

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/home")
def home():
    return redirect(url_for("index"))


@app.route("/live-detection")
def live_detection():
    if "user_id" not in session:
        return redirect(url_for("login_signup"))
    return render_template("live_detection.html")


@app.route("/upload-image")
def upload_image():
    return render_template("upload_image.html")


@app.route("/detection-history")
def detection_history():
    if "user_id" not in session:
        return redirect(url_for("login_signup"))

    page = request.args.get("page", 1, type=int)
    per_page = 10
    offset = (page - 1) * per_page

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
        (per_page, offset)
    )
    rows = cur.fetchall()

    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]
    conn.close()

    total_pages = max(1, (total + per_page - 1) // per_page)

    return render_template(
        "detection_history.html",
        rows=rows,
        page=page,
        total_pages=total_pages
    )


@app.route("/leaderboard")
def leaderboard():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT username, eco_points, badge FROM users ORDER BY eco_points DESC LIMIT 20"
    )
    leaders = cur.fetchall()
    conn.close()
    return render_template("leaderboard.html", leaders=leaders)


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login_signup"))
    return render_template("dashboard.html")


@app.route("/api/dashboard-data")
def dashboard_data():
    """
    Returns JSON for the personal analytics dashboard:
      - counts_by_class : { class_name: count }
      - scans_by_date   : [ { date: "YYYY-MM-DD", count: N } ]  (last 30 days)
      - total_scans     : int
      - carbon_impact   : float  (kg CO₂ saved — 0.5 kg per scan)
      - user_points     : int
      - user_badge      : str
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Counts by predicted class (all-time, all users — gives global insight)
    cur.execute("""
        SELECT predicted_class, COUNT(*) as cnt
        FROM predictions
        GROUP BY predicted_class
        ORDER BY cnt DESC
    """)
    counts_by_class = {row[0]: row[1] for row in cur.fetchall()}

    # Daily scan counts for the last 30 days
    since = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
    cur.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as cnt
        FROM predictions
        WHERE DATE(created_at) >= ?
        GROUP BY day
        ORDER BY day ASC
    """, (since,))
    scans_by_date = [{"date": row[0], "count": row[1]} for row in cur.fetchall()]

    cur.execute("SELECT COUNT(*) FROM predictions")
    total_scans = cur.fetchone()[0]

    conn.close()

    carbon_impact = round(total_scans * 0.5, 1)   # 0.5 kg CO₂ saved per scan

    return jsonify({
        "counts_by_class": counts_by_class,
        "scans_by_date":   scans_by_date,
        "total_scans":     total_scans,
        "carbon_impact":   carbon_impact,
        "user_points":     session.get("eco_points", 0),
        "user_badge":      session.get("badge", "Newcomer"),
    })


@app.route("/waste-guide")
def waste_guide():
    return render_template("waste_guide.html")


@app.route("/guide")
def guide():
    return redirect(url_for("waste_guide"))


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/login-signup")
def login_signup():
    return render_template("login_signup.html")


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    if not username or not email or not password:
        return render_template("login_signup.html", error="All fields are required.")

    hashed = generate_password_hash(password)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users(username, email, password, eco_points, badge) VALUES(?, ?, ?, 0, 'Newcomer')",
            (username, email, hashed)
        )
        conn.commit()
        return redirect(url_for("login_signup"))
    except sqlite3.IntegrityError:
        return render_template("login_signup.html", signup_error="An account with that email already exists.")
    finally:
        conn.close()


@app.route("/login", methods=["POST"])
def login():
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT id, username, password, eco_points, badge FROM users WHERE email = ?",
        (email,)
    )
    user = cur.fetchone()
    conn.close()

    if user and check_password_hash(user[2], password):
        session["user_id"]    = user[0]
        session["username"]   = user[1]
        session["eco_points"] = user[3]
        session["badge"]      = user[4]
        return redirect(url_for("index"))

    return render_template("login_signup.html", login_error="Invalid email or password.")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Prediction routes
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error": "No file uploaded"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "File type not allowed. Use PNG, JPG, JPEG, GIF, or WEBP."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        img = Image.open(save_path)
        arr = preprocess_pil_image(img)
    except Exception:
        return jsonify({"error": "Could not read image file."}), 400

    predicted_class, confidence = predict_array(arr)
    is_unknown = (predicted_class == "Unknown / Not Sure")

    if not is_unknown:
        save_history(filename, predicted_class, confidence, "upload")
    disposal_tip = get_disposal_tip(predicted_class) if not is_unknown else ""

    return jsonify({
        "prediction":    predicted_class,
        "confidence":    confidence,
        "disposal_tip":  disposal_tip,
        "is_unknown":    is_unknown,
        "model_version": MODEL_VERSION,
        "eco_points":    session.get("eco_points", 0),
        "badge":         session.get("badge", "Newcomer")
    })


@app.route("/predict-frame", methods=["POST"])
def predict_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")

    if not image_data:
        return jsonify({"error": "No image data"}), 400

    try:
        _header, encoded = image_data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes))
        arr = preprocess_pil_image(img)
        predicted_class, confidence = predict_array(arr)
        is_unknown = (predicted_class == "Unknown / Not Sure")

        if not is_unknown and confidence > HISTORY_MIN_CONFIDENCE:
            save_history("live_camera", predicted_class, confidence, "live")

        disposal_tip = get_disposal_tip(predicted_class) if not is_unknown else ""

        return jsonify({
            "prediction":    predicted_class,
            "confidence":    confidence,
            "disposal_tip":  disposal_tip,
            "is_unknown":    is_unknown,
            "model_version": MODEL_VERSION,
            "eco_points":    session.get("eco_points", 0),
            "badge":         session.get("badge", "Newcomer")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Misc routes
# ---------------------------------------------------------------------------
@app.route("/history/<int:id>")
def history_detail(id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions WHERE id = ?", (id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        return "History record not found", 404

    return render_template("history_detail.html", row=row)


@app.route("/delete-history/<int:id>")
def delete_history(id):
    if "user_id" not in session:
        return redirect(url_for("login_signup"))

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE id = ?", (id,))
    conn.commit()
    conn.close()

    return redirect(url_for("detection_history"))


@app.route("/submit-feedback", methods=["POST"])
def submit_feedback():
    name     = request.form.get("name", "")
    email    = request.form.get("email", "")
    rating   = request.form.get("rating", "")
    feedback = request.form.get("feedback", "")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO feedback(name, email, rating, feedback) VALUES(?, ?, ?, ?)",
        (name, email, rating, feedback)
    )
    conn.commit()
    conn.close()

    return redirect(url_for("contact"))


# ---------------------------------------------------------------------------
# Community stats  (public — no auth required)
# ---------------------------------------------------------------------------
@app.route("/stats")
def stats():
    """
    Returns JSON community stats:
      - total_scans      : int  (all-time, all users)
      - hazardous_count  : int  (syringe classes all-time)
      - top_class_week   : str  (most-detected class in the last 7 days)
      - top_class_count  : int
      - co2_saved        : float  (kg)
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM predictions")
    total_scans = cur.fetchone()[0]

    hazardous_classes = list(HAZARDOUS_CLASSES)
    placeholders = ",".join("?" * len(hazardous_classes))
    cur.execute(
        f"SELECT COUNT(*) FROM predictions WHERE predicted_class IN ({placeholders})",
        hazardous_classes
    )
    hazardous_count = cur.fetchone()[0]

    week_ago = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    cur.execute("""
        SELECT predicted_class, COUNT(*) as cnt
        FROM predictions
        WHERE DATE(created_at) >= ?
          AND predicted_class != 'Unknown / Not Sure'
        GROUP BY predicted_class
        ORDER BY cnt DESC
        LIMIT 1
    """, (week_ago,))
    row = cur.fetchone()
    top_class_week  = row[0] if row else "No data yet"
    top_class_count = row[1] if row else 0

    conn.close()

    return jsonify({
        "total_scans":     total_scans,
        "hazardous_count": hazardous_count,
        "top_class_week":  top_class_week,
        "top_class_count": top_class_count,
        "co2_saved":       round(total_scans * 0.5, 1),
    })


# ---------------------------------------------------------------------------
# Disposal Location Finder page
# ---------------------------------------------------------------------------
@app.route("/disposal-finder")
def disposal_finder():
    gmaps_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    return render_template("disposal_finder.html", gmaps_key=gmaps_key)


# ---------------------------------------------------------------------------
# Export PDF report
# ---------------------------------------------------------------------------
@app.route("/export-pdf")
def export_pdf():
    if "user_id" not in session:
        return redirect(url_for("login_signup"))

    # Optional date-range filter from query params (YYYY-MM-DD)
    date_from = request.args.get("from", "")
    date_to   = request.args.get("to",   "")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    if date_from and date_to:
        cur.execute("""
            SELECT filename, predicted_class, confidence, source, created_at
            FROM predictions
            WHERE DATE(created_at) BETWEEN ? AND ?
            ORDER BY id DESC
        """, (date_from, date_to))
    else:
        cur.execute("""
            SELECT filename, predicted_class, confidence, source, created_at
            FROM predictions ORDER BY id DESC
        """)
    rows = cur.fetchall()

    cur.execute("SELECT eco_points, badge FROM users WHERE id = ?", (session["user_id"],))
    user_row = cur.fetchone() or (0, "Newcomer")
    conn.close()

    # Build the PDF with reportlab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Table, TableStyle, Paragraph,
            Spacer, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        return (
            "<h2>reportlab not installed.</h2>"
            "<p>Run <code>pip install reportlab</code> then restart the server.</p>",
            500
        )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    green  = colors.HexColor("#18b83f")
    red    = colors.HexColor("#e03333")
    dark   = colors.HexColor("#07101f")

    title_style = ParagraphStyle(
        "title", parent=styles["Title"],
        textColor=green, fontSize=22, spaceAfter=4
    )
    sub_style = ParagraphStyle(
        "sub", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#444"), spaceAfter=2
    )
    hdr_style = ParagraphStyle(
        "hdr", parent=styles["Heading2"],
        textColor=dark, fontSize=13, spaceAfter=6, spaceBefore=14
    )

    story = []
    story.append(Paragraph("WasteVision AI — Detection Report", title_style))
    story.append(Paragraph(f"User: {session.get('username', 'N/A')}", sub_style))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sub_style
    ))
    if date_from and date_to:
        story.append(Paragraph(f"Date range: {date_from}  →  {date_to}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=green, spaceAfter=10))

    # Summary stats
    story.append(Paragraph("Summary", hdr_style))
    total = len(rows)
    hazardous = sum(1 for r in rows if r[1] in HAZARDOUS_CLASSES)
    eco_pts, badge = user_row
    summary_data = [
        ["Total Scans", str(total)],
        ["Hazardous Items", str(hazardous)],
        ["Eco Points", str(eco_pts)],
        ["Badge", badge],
        ["CO₂ Saved (est.)", f"{round(total * 0.5, 1)} kg"],
    ]
    summary_tbl = Table(summary_data, colWidths=[8*cm, 8*cm])
    summary_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#e8faf0")),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 11),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f6fbf8")]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#c8e6d0")),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_tbl)
    story.append(Spacer(1, 0.4*cm))

    # Scan history table
    story.append(Paragraph("Full Scan History", hdr_style))
    if rows:
        tbl_data = [["#", "Filename", "Detected Class", "Confidence", "Source", "Date"]]
        for i, r in enumerate(rows, 1):
            conf_str = f"{r[2]:.1f}%"
            tbl_data.append([str(i), r[0][:28], r[1], conf_str, r[3], r[4]])

        col_w = [1.2*cm, 5.5*cm, 4.5*cm, 2.2*cm, 1.8*cm, 3.8*cm]
        tbl = Table(tbl_data, colWidths=col_w, repeatRows=1)
        row_styles = [
            ("BACKGROUND",    (0, 0), (-1, 0),  dark),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f9f6")]),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#d0e8da")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ALIGN",         (3, 1), (3, -1),  "CENTER"),
        ]
        # Highlight hazardous rows in light red
        for i, r in enumerate(rows, 1):
            if r[1] in HAZARDOUS_CLASSES:
                row_styles.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#ffeaea")))
                row_styles.append(("TEXTCOLOR",  (2, i), (2,  i), red))
        tbl.setStyle(TableStyle(row_styles))
        story.append(tbl)
    else:
        story.append(Paragraph("No scan records found for the selected date range.", styles["Normal"]))

    story.append(Spacer(1, 0.6*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#c0d8c8")))
    story.append(Paragraph(
        "WasteVision AI — Model v1.0 Medical Waste Specialist | "
        "Hazardous items highlighted in red.",
        ParagraphStyle("foot", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#888"), alignment=TA_CENTER, spaceBefore=6)
    ))

    doc.build(story)
    buf.seek(0)

    fname = f"WasteVision_Report_{session.get('username', 'user')}_{datetime.date.today()}.pdf"
    return Response(
        buf.read(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
