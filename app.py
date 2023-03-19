from flask import Flask, render_template, url_for, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/get-match', methods=['POST'])
def index():
    response = request
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return request

if __name__ == "__main__":
    app.run(debug=True)