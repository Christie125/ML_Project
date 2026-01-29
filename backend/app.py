from flask import Flask, Blueprint, request, jsonify, render_template
from api import score_api

app = Flask(__name__)

app.register_blueprint(score_api)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
        app.run(port=5000, debug=True)