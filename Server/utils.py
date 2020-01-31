import os

def checkDir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


def readTxt(txtfile, useDict = 1):
    with open(txtfile, 'r') as txt:
        if useDict:
            return {i: label.strip() for i, label in enumerate(txt.readlines(), 1)}
        else:
            return [label.strip() for label in txt.readlines()]


def getDetectionLayer(prototxt):
    lines = readTxt(prototxt, 0)
    detection_types = [line.split(': ')[1] for line in lines if 'DetectionOutput' in line]
    assert len(detection_types) == 1, "In the deploy, it needs only one detection output layer"
    detection_name = [line.split(': ')[1].strip('\"') for line in lines if 'name:' in line][-1]
    
    return detection_name


def getIPAddr():
    from socket import socket, AF_INET, SOCK_DGRAM
    addr = "127.0.0.1"
    try:
        s = socket(AF_INET, SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addr = s.getsockname()[0]
    except socket.error:
        pass
    s.close()
    return addr
    
def readLabelMap(prototxt):
    lines = readTxt(prototxt, 0)
    lines = [line.split(': ')[1].strip('\"') for line in lines if 'display' in line]
    return dict(zip(range(len(lines)), lines))
