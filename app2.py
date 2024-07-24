from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
app = Flask(__name__)

coco = 'coco.names'
ssd = 'frozen_inference_graph.pb'
frozen = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

def is_person(img):    
    if img is None:
        return True
    img = cv2.resize(img, (640, 640))
    if os.path.exists(coco):
        with open(coco, "rt") as f:
            class_names = f.read().rstrip("\n").split("\n")
    else:
        return False

    if not os.path.exists(ssd) or not os.path.exists(frozen):
        return False

    net = cv2.dnn_DetectionModel(frozen, ssd)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    class_ids, confs, bbox = net.detect(img, confThreshold=0.6, nmsThreshold=0.4)

    if len(class_ids) != 0:
        for class_id, conf, _ in zip(class_ids.flatten(), confs, bbox):
            if class_names[class_id - 1] == 'person' and conf > 0.45:
                return True

    return False

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'Failed to read the image file'}), 400

    person_detected = is_person(img)
    return jsonify({'person_present': person_detected})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
