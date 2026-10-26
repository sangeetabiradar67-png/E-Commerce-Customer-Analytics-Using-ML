from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
import json
from flask import Flask, jsonify

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RFM_FILE = RESULTS_DIR / "rfm.csv"
REC_FILE = RESULTS_DIR / "recommendations.csv"

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "frontend" / "templates"),
    static_folder=str(BASE_DIR / "frontend" / "static"),
)

# Simple cache: re-read a CSV only when the file changes on disk
_cache = {}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    mtime = path.stat().st_mtime
    cached = _cache.get(path)
    if cached is None or cached[0] != mtime:
        df = pd.read_csv(path)
        if "CustomerID" in df.columns:
            df["CustomerID"] = df["CustomerID"].astype(int)
        _cache[path] = (mtime, df)
    return _cache[path][1]


def to_records(df: pd.DataFrame):
    """DataFrame -> list of dicts, with NaN converted to None so JSON is valid."""
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@app.route("/")
def home():
    return render_template("dashboard.html")


@app.route("/upload")
def upload_page():
    # The old file-picker version of the dashboard
    return render_template("index.html")


@app.route("/api/summary")
def summary():
    rfm = load_csv(RFM_FILE)
    if rfm.empty:
        return jsonify(error="rfm.csv not found. Run export_for_dashboard.py first."), 404

    data = {
        "customers": int(len(rfm)),
        "total_revenue": round(float(rfm["Monetary"].sum()), 2),
        "avg_recency": round(float(rfm["Recency"].mean()), 1),
        "avg_frequency": round(float(rfm["Frequency"].mean()), 2),
    }
    if "Segment" in rfm.columns:
        data["segments"] = rfm["Segment"].value_counts().to_dict()
    return jsonify(data)


@app.route("/api/rfm")
def rfm_data():
    rfm = load_csv(RFM_FILE)
    if rfm.empty:
        return jsonify(error="rfm.csv not found"), 404

    limit = request.args.get("limit", default=100, type=int)
    sort_by = request.args.get("sort", default="Monetary")
    if sort_by not in rfm.columns:
        sort_by = "Monetary"

    segment = request.args.get("segment")
    if segment and "Segment" in rfm.columns:
        rfm = rfm[rfm["Segment"] == segment]

    rfm = rfm.sort_values(sort_by, ascending=False).head(limit)
    return jsonify(to_records(rfm))


@app.route("/api/customer/<int:customer_id>")
def customer(customer_id):
    rfm = load_csv(RFM_FILE)
    rec = load_csv(REC_FILE)

    row = rfm[rfm["CustomerID"] == customer_id] if not rfm.empty else rfm
    if row.empty:
        return jsonify(error=f"Customer {customer_id} not found"), 404

    recs = rec[rec["CustomerID"] == customer_id] if not rec.empty else rec
    return jsonify(profile=to_records(row)[0], recommendations=to_records(recs))


@app.route("/api/recommendations/<int:customer_id>")
def recommendations(customer_id):
    rec = load_csv(REC_FILE)
    if rec.empty:
        return jsonify(error="recommendations.csv not found"), 404
    return jsonify(to_records(rec[rec["CustomerID"] == customer_id]))


@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not found"), 404

@app.route("/evaluation")
def evaluation():
    result_file = (
        Path(__file__).resolve().parent
        / "results"
        / "recommendation_evaluation.json"
    )

    if not result_file.exists():
        return jsonify({"error": "Evaluation results not found"}), 404

    with open(result_file, "r", encoding="utf-8") as file:
        results = json.load(file)

    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)