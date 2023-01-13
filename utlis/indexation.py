import os
import cv2
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications import vgg16

def generateSIFT(file):
    features1 = []
    if not os.path.isdir("SIFT"):
        os.mkdir("SIFT")

    img = cv2.imread(file)
    #featureSum = 0
    sift = cv2.SIFT_create()  
    kps , des = sift.detectAndCompute(img,None)

    num_image = file.split('/')[3].split('.')[0]
    np.savetxt("SIFT/"+str(num_image)+".txt" ,des)
    features1.append((data,feature))
    return ((file,des))

    

def generateORB(filenames):
    features1 = []
    if not os.path.isdir("ORB"):
        os.mkdir("ORB")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        orb = cv2.ORB_create()
        key_point1,descrip1 = orb.detectAndCompute(img,None)
        data = os.path.join(filenames, path)
        num_image = path.split('_')[4].split('.')[0]
        np.savetxt("ORB/"+str(num_image)+".txt" ,descrip1 )
        features1.append((data,descrip1))
        i+=1
    print("indexation ORB terminée !!!!")

    return features1



def indexationModel(file):
  model = vgg16.VGG16()
  if not os.path.isdir("modelIndex"):
        os.mkdir("modelIndex")
  print('oups')

  file_name = os.path.basename(file) # load an image from file
  image = load_img(file, target_size=(224, 224))
  image = img_to_array(image)
  # reshape data for the model
  print('coucou')
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = preprocess_input(image)
  feature = model.predict(image) # predict the probability
  feature = np.array(feature[0])
  num_image = file.split('/')[3].split('.')[0]
  np.savetxt("modelIndex"+"/"+ num_image +".txt",feature)
  return ((file,feature))
