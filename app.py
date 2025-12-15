from flask import Flask, render_template, request, redirect, url_for, flash, session
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os
from flask import jsonify

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Dummy user storage
users = {}

# ===== LOAD MODEL =====
MODEL_PATH = "pitfall_model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("✅ Loaded trained pitfall model successfully.")
else:
    print("⚠️ Model not found. Using dummy logistic regression for testing.")
    model = LogisticRegression()
    X_dummy = np.random.rand(10, 11)
    y_dummy = np.random.randint(0, 2, 10)
    model.fit(X_dummy, y_dummy)

# ================= ROUTES =================

@app.route("/")
def home():
    return redirect(url_for("login"))

# ---- REGISTER ----
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username in users:
            flash("Username already exists!", "error")
        else:
            users[username] = {"username": username, "password": password}
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
    return render_template("registration.html")

# ---- LOGIN ----
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = users.get(username)
        if user and user["password"] == password:
            session["username"] = username
            return redirect(url_for("select_city"))
        else:
            flash("Invalid credentials!", "error")
    return render_template("login.html")

# ---- LOGOUT ----
@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))

# ---- SELECT CITY ----
@app.route("/select_city", methods=["GET", "POST"])
def select_city():
    if "username" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        state = request.form.get("state")
        city = request.form.get("city")
        if not state or not city:
            flash("Please select both state and city!", "error")
            return redirect(url_for("select_city"))

        session["state"] = state
        session["city"] = city
        flash(f"State: {state}, City: {city} selected successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("select_city.html")

# ---- DASHBOARD ----
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "username" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    if "state" not in session or "city" not in session:
        flash("Please select your state and city first.", "warning")
        return redirect(url_for("select_city"))

    username = session["username"]
    user_data = users.get(username, {})

    prediction = None
    probability = None
    risk_label = None
    risk_icon = None

    if request.method == "POST":
        try:
            print("Form Data:", request.form)

            # Collect all 11 features
            features = [
                float(request.form["TerrainType"]),
                float(request.form["Weather"]),
                float(request.form["RockDensity"]),
                float(request.form["SurfaceRoughness"]),
                float(request.form["MoisturePct"]),
                float(request.form["SlopeDeg"]),
                float(request.form["TrafficLoad"]),
                float(request.form["DrainageQuality"]),
                float(request.form["DepthCm"]),
                float(request.form["SoilHardness"]),
                float(request.form["VibrationLevel"]),
            ]

            features_array = np.array([features])
            print("Features Collected:", features_array)

            # --- PREDICTION LOGIC ---
            prob = model.predict_proba(features_array)[0][1]  # Probability of class 1
            class_pred = 1 if prob >= 0.5 else 0

            prediction = int(class_pred)
            probability = round(prob * 100, 2)

            # --- PITFALL RISK LABEL ---
            if prob < 0.3:
                risk_label = "Low Pitfall Risk"
                risk_icon = "✅"
            elif 0.3 <= prob < 0.7:
                risk_label = "Moderate Pitfall Risk"
                risk_icon = "⚠️"
            else:
                risk_label = "High Pitfall Risk"
                risk_icon = "🔴"

            print(f"Predicted: {prediction}, Probability: {probability}%")

        except Exception as e:
            print("❌ Error during prediction:", str(e))
            flash(f"Error: {str(e)}", "error")

    return render_template(
        "dashboard.html",
        prediction=prediction,
        probability=probability,
        risk_label=risk_label,
        risk_icon=risk_icon,
        user=user_data,
        state=session.get("state"),
        city=session.get("city")
    )

# ---- CHANGE PASSWORD ----
@app.route("/change_password", methods=["GET", "POST"])
def change_password():
    if "username" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    username = session["username"]
    user_data = users.get(username, {})

    if request.method == "POST":
        old_password = request.form["old_password"]
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]

        if user_data.get("password") != old_password:
            flash("Old password is incorrect!", "error")
        elif new_password != confirm_password:
            flash("New passwords do not match!", "error")
        else:
            users[username]["password"] = new_password
            flash("Password changed successfully!", "success")
            return redirect(url_for("dashboard"))

    return render_template("change_password.html", user=user_data)

# ---- CHATBOT PAGE ----
@app.route("/chatbot")
def chatbot_page():
    if "username" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))
    return render_template("chatbot.html")

# ---- CHATBOT RESPONSE API ----
@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        user_msg = request.json.get("message", "").lower()
        print("User message:", user_msg)

        if "slope" in user_msg:
            reply = "Slope refers to the steepness or incline of the ground surface — usually measured in degrees or percentage."
        elif "rainfall" in user_msg:
            reply = "Rainfall affects soil stability by increasing water content and reducing cohesion."
        elif "pitfall" in user_msg:
            reply = "A pitfall refers to a hazard or issue that could cause slope failure or instability."
        else:
            reply = "Hmm 🤔 I’m not sure about that yet, but I’ll learn soon!"

        return jsonify({"reply": reply})
    except Exception as e:
        print("Error:", e)
        return jsonify({"reply": f"⚠️ Error: {str(e)}"})
# ====================
if __name__ == "__main__":
    app.run(debug=True)
