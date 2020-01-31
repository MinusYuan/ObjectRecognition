import os
import cv2
import json
import requests
from config import *
from time import time

if __name__ == '__main__':
  count = 0
  video = os.popen('ls /dev | grep video').read().strip()
  if video:
    index = int(video.split('\n')[0].split('video')[1])
    cap = cv2.VideoCapture(index)
    filename = 'image.jpg'
    
    timer = time()
    while True:
      _, image = cap.read()
      cv2.imwrite(filename, image)
      
      url = base_url + '/detect'
      files = [('files', open(filename, 'rb'))]
      ret = requests.post(url=url, files=files).json()
      queue = ret['queue']
      
      # askResult
      url = base_url + '/askResult'
      files = {'data': json.dumps({'queue': queue})}
      ret = requests.post(url=url, files=files)
      results = ret.json()['result']
      for r in results:
          cv2.rectangle(image, (r[1], r[3]), (r[2], r[4]), [0, 255, 0], 2)
          cv2.putText(image, r[0], (r[1], r[4]), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
      cv2.imshow("Detection", image)
      if cv2.waitKey(20) == ord('q'):
        os.remove(filename)
        break
      count += 1
    time_diff = time() - timer
    print("Total elapsed time: {}, average time: {}".format(time_diff, time_diff / count))
