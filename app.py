from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from pybase64 import urlsafe_b64decode, b64decode
from io import BytesIO
from PIL import Image
import numpy as np
import datetime

from perspective_module import Perspective
from associations_module import Associations
from compare_module import Compare
from sample_class_module import Sample

app = Flask(__name__)
CORS(app, support_credentials=True)


@app.route('/get-match', methods=['POST'])
@cross_origin(support_credentials=True)
def index():
    ref_sample = get_b64string(request.get_json()['ref_sample'])
    test_samples = [get_b64string(s) for s in request.get_json()['test_samples']]

    ref_sample_b64 = request.get_json()['ref_sample']
    test_samples_b64 = [s for s in request.get_json()['test_samples']]

    save_search([ref_sample_b64, test_samples_b64[0], test_samples_b64[1]])

    ref_ndarray = get_image_ndarray(get_b64string(ref_sample_b64))
    test_ndarrays = [get_image_ndarray(get_b64string(s)) for s in test_samples_b64]

    ref_2darray = cvt_to_2darray(ref_ndarray, pick_last_value_in_list)
    test_2darrays = [cvt_to_2darray(s, pick_last_value_in_list) for s in test_ndarrays]
    return jsonify(test_samples_b64[compare_samples(ref_2darray, test_2darrays)])

def save_search(str_array):
    current_time = datetime.datetime.now()
    with open('./search_history.txt', 'w') as f:
        f.write(str(current_time))
        f.write("\n")
        f.write("\n")

        for b64_image_string in str_array:
            f.write(b64_image_string)
            f.write("\n")
            f.write("\n")

        f.write("\n")
        f.write("\n")
        f.write("\n")

def get_b64string(s):
    return s.split(",")[1]

def pick_last_value_in_list(a_list):
    return a_list[-1]

def get_image_ndarray(base64_image_string):
    image = Image.open(BytesIO(b64decode(base64_image_string)))
    return np.array(image.resize((10, 10)))

def cvt_to_2darray(numpy_ndarray, func):
    result = np.empty((len(numpy_ndarray), len(numpy_ndarray[0])))

    for row in range(len(numpy_ndarray)):
        for col in range(len(numpy_ndarray[0])):
            result[row, col] = func(numpy_ndarray[row, col])
    
    return result

def compare_samples(ref_2darray, test_2darrays):
    ref_association = Associations(Sample(ref_2darray).convertToAngle())
    test_1_association = Associations(sampleImage=Sample(test_2darrays[0]).convertToAngle(), usePlainDataframes=True, includeDistance=True, limitDistance=30)
    test_2_association = Associations(sampleImage=Sample(test_2darrays[1]).convertToAngle(), usePlainDataframes=True, includeDistance=True, limitDistance=30)

    comp1 = ref_association.compare(test_1_association)
    comp2 = ref_association.compare(test_2_association)

    # breakpoint()
 
    dist1 = comp1.getDistanceToNeighbors(reference='column')
    dist2 = comp2.getDistanceToNeighbors(reference='column')

    return int(dist1 < dist2)

if __name__ == "__main__":
    app.run()
