import numpy as np # Digunakan untuk memanipulasi array
from PIL import Image #Library untuk memproses gambar
import os, cv2 #Digunakan untuk berinteraksi dengan sistem file, Digunakan untuk pengolahan gambar dan operasi pengenalan wajah
import pickle #Digunakan untuk menyimpan objek Python

def train_classifier(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)] # mengambil dan menggabungkan 
    faces = [] #menyimpan data wajah
    names = [] #menyimpan nama 

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        name = os.path.split(image)[1].split("_")[0]

        faces.append(imageNp)
        names.append(name)

    unique_names = list(set(names))
    name_to_id = {name: idx for idx, name in enumerate(unique_names)}
    ids = [name_to_id[name] for name in names]

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    with open('name_to_id.pkl', 'wb') as f:
        pickle.dump(name_to_id, f)
    
    clf.write("classifier.xml")

train_classifier("data")
