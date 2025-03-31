import cv2 
import numpy as np 
import uuid 
import os 
import time 

IMAGES_PATH = 'images/image_data'
labels = ['hello', 'thanks','please', 'namaste','sorry','yes','no']
number_images = 50

# Create directories for each label and capture images
for label in labels:
    os.makedirs(os.path.join(IMAGES_PATH, label), exist_ok=True)
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgname in range(number_images):
        ret, frame = cap.read()
        imagename = os.path.join(IMAGES_PATH, label, '{}_{}.jpg'.format(label, str(uuid.uuid1())))
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
