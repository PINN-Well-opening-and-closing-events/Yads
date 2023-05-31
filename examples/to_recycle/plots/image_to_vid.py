"""
import cv2
import numpy as np

# choose codec according to format needed
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('video.avi', fourcc, 1, (width, height))

for j in range(0, 5):
    img = cv2.imread(str(i) + '.png')
    video.write(img)

cv2.destroyAllWindows()
video.release()
"""
