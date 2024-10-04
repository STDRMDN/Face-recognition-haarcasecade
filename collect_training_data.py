import numpy as np   #libary Digunakan untuk komputasi array dan operasi numerik.
from PIL import Image #library Digunakan untuk manipulasi gambar.
import os, cv2 #library Mengelola interaksi dengan sistem operasi, seperti membuat dan menghapus direktori atau file. 
#cv2Merupakan modul OpenCV untuk pengolahan gambar dan video.
import pickle #library 

def generate_dataset(img, name, img_id): #
    cv2.imwrite(f"data/{name}_{img_id}.jpg", img)  #mengambil gambar dari webcam dengan nama file yang mengikuti format di atas dan di simpan ke folder data

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # mengonversi gambar ke grayscale, karena deteksi wajah lebih efektif pada gambar grayscale.
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors) # mendeteksi wajah dalam gambar berdasarkan classifier yang diberikan
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords

def detect(img, faceCascade, img_id, name):
    color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0), "white": (255, 255, 255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
    
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        generate_dataset(roi_img, name, img_id)

    return img

user_name = input("Enter the name of the user: ")
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
img_id = 0

while True:
    if img_id % 50 == 0:
        print("Collected ", img_id, " images")
    ret, img = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    img = detect(img, faceCascade, img_id, user_name)
    cv2.imshow("face detection", img)
    img_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
