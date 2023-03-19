from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app, support_credentials=True)



@app.route('/get-match', methods=['POST'])
@cross_origin(support_credentials=True)
def index():
    # breakpoint()
    response = request.get_json()
    return jsonify({"what": response})

if __name__ == "__main__":
    app.run()