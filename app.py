from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from pybase64 import urlsafe_b64decode, b64decode
from io import BytesIO
from PIL import Image
import numpy as np


app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/get-match', methods=['POST'])
@cross_origin(support_credentials=True)
def index():
    ref_sample = get_b64string(request.get_json()['ref_sample'])
    test_samples = [get_b64string(s) for s in request.get_json()['test_samples']]
    breakpoint()
    return jsonify({"what": response})

def get_b64string(s):
    return s.split(",")[1]

def get_image_ndarray(base64_image_string):
    image = Image.open(BytesIO(b64decode(base64_image_string)))
    return np.array(image)


if __name__ == "__main__":
    app.run()
