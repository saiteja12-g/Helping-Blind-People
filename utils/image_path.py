import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import cv2
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs

def get_image_path(video_path):
    
    image_paths = []
    
    cap = cv2.VideoCapture(video_path)
    arr = np.empty((0, 1944), int)
    D = dict()
    count = 0

    while cap.isOpened():
    
        ret, frame = cap.read()
    
        if ret == True:
            if count%50==0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                file_name = 'frames/'+'frame'+ str(count) +'.png'
                cv2.imwrite(file_name, frame_rgb)
                image_paths.append(file_name)
            count+=1
        else:
            break
    return image_paths

