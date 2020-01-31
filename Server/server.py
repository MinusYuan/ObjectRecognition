import cv2
import os
import sys
import json
import utils
import config
import numpy as np
from flask import Flask, request, jsonify
from time import time, strftime, localtime
from werkzeug.utils import secure_filename

os.environ['GLOG_minloglevel'] = '2' # No show caffe layer message
try:
    caffe_installed = 1
    import caffe
except ImportError:
    caffe_installed = 0

app = Flask(__name__)

def NOW(type = 0):
    if not type:
        return strftime(config.time_format, localtime())
    else:
        return strftime(config.time_format1, localtime())

def showLabel():
    print("------------------ Table ------------------")
    for i, label in label_table.items():
        print("  --> {}: {}".format(i, label))
    print("-------------------------------------------")

def getqueue():
    global queue
    now = NOW()[:6]
    if now != _date: # When date changes, clear all old data
        global date, jsons
        queue, jsons = -1, {} # Avoid memory leak, it should be cleaned everyday.
        date = now
    queue += 1
    return queue

def get_net(model_path):
    timer = time()
    JOIN = JOIN_
    # Check model first
    files = os.listdir(model_path)
    model = filter(lambda x: x.endswith('caffemodel'), files)
    deploy = filter(lambda x: 'deploy' in x and x.endswith('prototxt'), files)
    assert len(model) == 1, 'In {} folder, there are {} caffemodels'.format(model_path, len(model))
    assert len(deploy) == 1, 'In {} folder, there are {} deploy protxts'.format(model_path, len(deploy))
    detection_layer = utils.getDetectionLayer(JOIN(model_path, deploy[0]))
    if caffe_installed:
        caffe.set_mode_gpu()
    
        # Create a model
        net = caffe.Net(JOIN(model_path, deploy[0]), # defines the structure of the model
                        JOIN(model_path, model[0]), # contains the trained weights
                        caffe.TEST) # use test mode (e.g., don't perform dropout)
        # net.blobs['data'].reshape(1,3,config.image_resize,config.image_resize)

        # Transformer
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', np.array(config.model_mean)) # mean pixel
        transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
        print('Elapsed time of loading model: {}'.format(time() - timer))
        return net, transformer, detection_layer
    else:
        net = cv2.dnn.readNetFromCaffe(JOIN(model_path, deploy[0]), 
                                       JOIN(model_path, model[0]))
        return net, 0, 0

def detect_by_path(img_path):
    net, transformer = net_ssd, transformer_ssd
    if caffe_installed:
        caffe.set_mode_gpu()
        image = caffe.io.load_image(img_path)
        net.blobs['data'].data[...] = transformer.preprocess('data', image)
        # Forward pass.
        dets = net.forward()[detection_layer]
    else:
        image = cv2.imread(img_path)
        blob = cv2.dnn.blobFromImage(
	                  cv2.resize(image, (config.input_width, config.input_height)), 1.0,
	                  (config.input_width, config.input_height), config.model_mean, 
	                  swapRB=False, crop=False)
        net.setInput(blob)
        dets = net.forward()
    h, w, _ = image.shape
    return h, w, dets[0,0,:,1:7]

def check_results(h, w, dets, queue):
    global jsons
    ans = [[str(label_table[det[0]]),
            max(0, int(w*det[2])),
            max(0, int(w*det[4])),
            max(0, int(h*det[3])),
            max(0, int(h*det[5]))]\
            for det in dets if det[1] >= 0.6]
    jsons[queue] = ans

@app.route('/detect', methods=['POST'])
def detect_api():
    print('detect Called')
    msg = 'Sucess'
    queue = -1
    try:
        JOIN = JOIN_
        if request.method == 'POST':
            global jsons
            queue = str(getqueue())
            addr = request.remote_addr
            print("[{}] Remote addr: {}, Queue: {}".format(NOW(), addr, queue))
            upload_path = os.path.join(config.upload_path, addr)
            utils.checkDir(upload_path)
            for f in request.files.getlist('files'):
                print("[{}] HANDLING IMAGE: {}".format(NOW(), f.filename))
                filename = JOIN(upload_path, secure_filename(f.filename))
                f.save(filename)
                timer = time()
                h, w, dets = detect_by_path(filename)

                check_results(h, w, dets, queue)
                print("[+] SSD Detection time: {}".format(time() - timer))
                print("[{}] Save result of {}".format(NOW(), f.filename))
    except Exception as e:
        print(e)
        msg = 'INTERNAL ERROR'
    return jsonify({'queue': queue})

@app.route('/askResult', methods=['GET', 'POST'])
def askResult_api():
    global jsons
    if request.method == 'POST':
        global jsons
        addr = request.remote_addr
        queue = str(json.load(request.files['data'])['queue'])
        print("[{}] Asked queue No.{} from {}".format(NOW(), queue, addr))
        result = jsons.pop(queue)
    return jsonify({'result': result})

if __name__ == '__main__':
    # Initialize parameter
    queue, jsons = -1, {}
    _date = NOW()[:6]
    JOIN_ = os.path.join
    ip = utils.getIPAddr()
    
    # Get ssd net
    timer = time()
    net_ssd, transformer_ssd, detection_layer = get_net(config.model_path)
    label_table = utils.readLabelMap(config.table_path)

    showLabel()
    app.run(debug=False, host=ip, port=config.port, threaded=True)









