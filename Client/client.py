import os
import cv2
import json
import utils
import requests
from config import *
from time import time

if __name__ == '__main__':
  JOIN = os.path.join
  images = [img for img in os.listdir(image_dir) if img.upper().endswith('.JPG')]
  utils.checkDir(output_dir)
  timer = time()
  for image in images:
    print('image: {}'.format(image))
    
    # detect
    url = base_url + '/detect'
    img = cv2.imread(JOIN(image_dir, image))
    files = [('files', open(JOIN(image_dir, image), 'rb'))] # For Object
    ret = requests.post(url=url, files=files).json()
    queue = ret['queue']
    
    # askResult
    url = base_url + '/askResult'
    files = {'data': json.dumps({'queue': queue})}
    ret = requests.post(url=url, files=files)
    results = ret.json()['result']
    for r in results:
        cv2.rectangle(img, (r[1], r[3]), (r[2], r[4]), [0, 255, 0], 2)
        cv2.putText(img, r[0], (r[1], r[4]), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(JOIN(output_dir, image), img)
  time_diff = time() - timer
  print("Total elapsed time: {}, average time: {}".format(time_diff, time_diff / len(images)))
