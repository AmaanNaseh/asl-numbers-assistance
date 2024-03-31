import os
import cv2
import numpy as np
import urllib.request

dataset = './dataset'

if not os.path.exists(dataset):
    os.makedirs(dataset)

total_classes = 10
class_size = 100

#cam = cv2.VideoCapture(0)

url = "http://192.168.0.105:8080/shot.jpg"

for i in range(total_classes):
    if not os.path.exists(os.path.join(dataset, str(i))):
        os.makedirs(os.path.join(dataset, str(i)))

    print('Creating Dataset for Class : {0}'.format(i))

    while True:
        imgPath = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
        img =  cv2.imdecode(imgNp, -1)
        
        #_, img = cam.read()

        cv2.putText(img, 'Press "Q" to Start Capturing !', (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                    cv2.LINE_AA)
        
        cv2.imshow('Dataset Creation', img)

        if cv2.waitKey(25) == ord('q'):
            break

    cnt = 0

    while cnt < class_size:
        imgPath = urllib.request.urlopen(url)
        imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
        img =  cv2.imdecode(imgNp, -1)
        
        #_, img = cam.read()
        
        cv2.imshow('Dataset Creation', img)

        cv2.waitKey(25)

        cv2.imwrite(os.path.join(dataset, str(i), '{}.jpg'.format(cnt)), img)

        cnt += 1

        if cv2.waitKey(25) == 27:
            break

print("Dataset Creation is Successful")

#cam.release()
cv2.destroyAllWindows()