from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from pybase64 import urlsafe_b64decode, b64decode, b64encode_as_string
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

    ref_ndarray = get_image_ndarray(ref_sample)
    test_ndarrays = [get_image_ndarray(s) for s in test_samples]

    ref_2darray = cvt_to_2darray(ref_ndarray, pick_last_value_in_list)
    test_2darrays = [cvt_to_2darray(s, pick_last_value_in_list) for s in test_ndarrays]
    image = Image.fromarray(ref_2darray)
    image = image.resize((len(ref_ndarray), len(ref_ndarray[0])))
    image = image.convert('L')
    image.save("result.png")
    image = open("./result.png", 'rb')
    image = image.read()
    # breakpoint()
    response = b64encode_as_string(image)
    # breakpoint()
    return jsonify({"match": "data:image/png;base64,"+response})

def get_b64string(s):
    return s.split(",")[1]

def pick_last_value_in_list(a_list):
    return a_list[-1]

def get_image_ndarray(base64_image_string):
    image = Image.open(BytesIO(b64decode(base64_image_string)))
    image = image.resize((28, 28))
    return np.array(image)

def cvt_to_2darray(numpy_ndarray, func):
    result = np.empty((len(numpy_ndarray), len(numpy_ndarray[0])))

    for row in range(len(numpy_ndarray)):
        for col in range(len(numpy_ndarray[0])):
            result[row, col] = func(numpy_ndarray[row, col])
    
    return result


if __name__ == "__main__":
    app.run()
