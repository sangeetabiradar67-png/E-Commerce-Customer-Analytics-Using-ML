"""
Serves the dashboard and your exported results.

Run from the src folder:
    pip install flask
    python dashboard_app.py
Then open http://127.0.0.1:5000
"""
from pathlib import Path 


from flask import Flask, abort, send_from_directory

BASE = Path(__file__).resolve().parent
TEMPLATES = BASE / "frontend" / "templates"
RESULTS = BASE / "results"
ALLOWED = {"rfm.csv"}

app = Flask(__name__, static_folder=str(BASE / "frontend" / "static"))

@app.route("/")
def dashboard():
    return send_from_directory(TEMPLATES, "dashboard.html")


@app.route("/results/<name>")
def results(name):
    if name not in ALLOWED:
        abort(404)
    return send_from_directory(RESULTS, name, mimetype="text/csv")

if __name__ == "__main__":
    app.run(debug=True)