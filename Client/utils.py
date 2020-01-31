import os

def checkDir(folder):
  if not os.path.isdir(folder):
    os.makedirs(folder)

