from flask import Flask, render_template, request, jsonify, redirect
import joblib
import numpy as np
import os

app = Flask(__name__)


# =============================================================
# LOAD MODEL + VECTORIZER
# =============================================================

print("📄 Loading model & vectorizer...")

# ---- Load Model ----
if os.path.exists("trained_model.sav"):
    model = joblib.load("trained_model.sav")
    print("✅ Model loaded successfully")
else:
    raise FileNotFoundError("❌ trained_model.sav NOT FOUND in project folder")

# ---- Load Vectorizer ----
if os.path.exists("vectorizer.pkl"):
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ vectorizer.pkl loaded successfully")
else:
    raise FileNotFoundError("❌ vectorizer.pkl NOT FOUND in project folder")



# =============================================================
# ROUTES
# =============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Check if it's a form submission or JSON
        if request.is_json:
            data = request.get_json()
            tweets = data.get("tweets", [])
        else:
            # Form submission
            tweet_text = request.form.get("tweet", "").strip()
            if not tweet_text:
                return jsonify({"error": "No tweet received"}), 400
            tweets = [tweet_text]

        print("\n===== REQUEST RECEIVED =====")
        print("TWEETS:", tweets)

        if not tweets:
            return jsonify({"error": "No tweet received"}), 400

        # Transform text using vectorizer
        X = vectorizer.transform(tweets)
        print("VECTOR SHAPE:", X.shape)

        # Predictions
        preds = model.predict(X)
        print("PREDICTIONS:", preds)

        # Confidence Scores
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                conf = [float(max(row)) for row in probs]
            else:
                conf = [None] * len(preds)
        except:
            conf = [None] * len(preds)

        # Fixed JSON serialization
        results = []
        for t, p, c in zip(tweets, preds, conf):
            results.append({
                "text": t,
                "sentiment": int(p),
                "score": float(c) if c is not None else None
            })

        print("FINAL RESULT:", results)
        
        # If JSON request, return JSON
        if request.is_json:
            return jsonify({"predictions": results})
        
        # If form submission, redirect to result page
        from urllib.parse import quote
        res = results[0]
        confidence = f"{res['score']*100:.1f}%" if res['score'] else "N/A"
        return redirect(f"/result?tweet={quote(res['text'])}&sentiment={res['sentiment']}&confidence={quote(confidence)}")

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500



@app.route("/result")
def result_page():
    tweet = request.args.get("tweet", "")
    sentiment = request.args.get("sentiment", "Unknown")
    confidence = request.args.get("confidence", "N/A")

    return render_template(
        "result.html",
        tweet=tweet,
        sentiment=sentiment,
        confidence=confidence
    )



if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000")
    app.run(debug=True)