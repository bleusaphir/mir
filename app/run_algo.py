
import csv
import time
from matplotlib import pyplot as plt
import numpy as np
import os
from utlis import  recherche
import warnings
warnings.filterwarnings('ignore')
import cv2

import json
import operator

start = time.time()
FEATURES_DIR = os.path.join(os.getcwd(), 'app', 'features')
SAVE_DIR = "static/"


def Compute_RP(top,nom_image_requete, nom_images_non_proches): 
  rappel_precision=[]
  position1=nom_image_requete[:3]

  #Preprocessing
  for j in range(top):
    position2=nom_images_non_proches[j][:3]
    if position1==position2:
      rappel_precision.append("pertinant") 
    else:
      rappel_precision.append("non pertinant")

  #Evaluation
  nper = rappel_precision.count("pertinant")
  if nper == 0:
    return 0,0,0,0
  R = nper/top
  P = nper/top
  AP = 0
  count = 0
  for i in range(top): 
    if rappel_precision[i] == "pertinant":
      count += 1
      AP += count/(i+1)
  
  AP /= nper
  RP = rappel_precision[:nper].count("pertinant")/nper

  return R,P,AP,RP
def recherche_f(image_req,top, features1, sim, descriptor):
  with open('dict_name.json', 'r') as f:
    dict_names = json.load(f)

  # FileNotFoundError: [Errno 2] No such file or directory: '4_5_singes_baboon_4504_ORB.txt.npy'

  # features1[image_req][0] = 4_5_singes_baboon_4504_ORB.txt.npy
  # On veut 4_5_singes_baboon_4504.jpg

  # base --> good :
  # good = '_'.join(base.split('.')[0].split('_')[:-1]) + '.jpg'
  # avec base = features1[image_req][0]

  good_name = '_'.join(features1[image_req][0].split('.')[0].split('_')[:-1]) + '.jpg'
  
  base_image = SAVE_DIR + dict_names[good_name]
  voisins = getkVoisins(features1, features1[image_req],top, sim)
  #print(voisins)
  nom_images_proches = []
  nom_images_non_proches = []
  for k in range(top):
    name = '_'.join(voisins[k][0].split('.')[0].split('_')[:-1]) + '.jpg'
  
    nom_images_non_proches.append(name)
    nom_images_proches.append(SAVE_DIR + dict_names[name])




  return nom_images_proches, nom_images_non_proches, base_image

def Compute_RP(top,nom_image_requete, nom_images_non_proches): 
  rappel_precision=[]
  position1=nom_image_requete[:3]

  rp = []
  #Preprocessing
  for j in range(top):
    position2=nom_images_non_proches[j][:3]
    if position1==position2:
      rappel_precision.append("pertinant") 
    else:
      rappel_precision.append("non pertinant")

  #Evaluation
  nper = rappel_precision.count("pertinant")
  nonper = rappel_precision.count("non pertinant")
  if nper == 0:
    return 0,0,0,0
  R = nper/top
  P = nper/nper
  AP = 0
  count = 0
  for i in range(top): 
    if rappel_precision[i] == "pertinant":
      count += 1
      AP += count/(i+1)
    rp.append(str((count/(i+1))*100) + " " + str((count/(top))*100))
  AP /= nper
  RP = rappel_precision[:nper].count("pertinant")/nper

  


  with open("rp.csv", 'w') as s:
    for a in rp:
      s.write(str(a) + '\n')

  return R,P,AP,RP

def Display_RP(fichier, input, des):
  x = []
  y = []
  with open(fichier) as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
      x.append(float(row[0]))
      y.append(float(row[1]))
      fig = plt.figure()
  plt.plot(y,x,'C1', label= des )
  plt.xlabel('Rappel')
  plt.ylabel('Précison')
  plt.title("R/P")
  plt.legend()
  nameImg = SAVE_DIR + input.split('.')[0] +"_" + des + "_RP.png"
  plt.savefig("app/" + nameImg)

  return nameImg

def getkVoisins(lfeatures, test, k, sim) :
  ldistances = []
  for i in range(len(lfeatures)):
    dist = distance_f(test[1], lfeatures[i][1], distanceName = sim)
    ldistances.append((lfeatures[i][0], lfeatures[i][1], dist))
  ldistances.sort(key=operator.itemgetter(2))
  lvoisins = []
  for i in range(k):
    lvoisins.append(ldistances[i])
  return lvoisins

def distance_f(l1,l2,distanceName):
    if distanceName=="Euclidienne":
        distance = recherche.euclidianDistance(l1,l2)
    elif distanceName in ["Correlation","Chi carre","Intersection","Bhattachayrya"]:
        if distanceName=="Correlation":
            methode=cv2.HISTCMP_CORREL
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Chi carre":
            distance = recherche.chiSquareDistance(l1,l2)
        elif distanceName=="Intersection":
            methode=cv2.HISTCMP_INTERSECT
            distance = cv2.compareHist(np.float32(l1), np.float32(l2), methode)
        elif distanceName=="Bhattacharyya":
            distance = recherche.bhatta(l1,l2)
    elif distanceName=="Brute force":
        distance = recherche.bruteForceMatching(l1,l2)
    elif distanceName=="Flann":
        distance= recherche.flann(l1,l2)
    return distance

# indexationMethod = {
#   'SIFT' : indexation.generateSIFT,
#   'ORB' : indexation.generateORB,
#   'Model' : indexation.indexationModel
# }
def json_numpy_obj_hook(dct):
  import base64
  """
  Decodes a previously encoded numpy ndarray
  with proper shape and dtype
  :param dct: (dict) json encoded ndarray
  :return: (ndarray) if input was an encoded ndarray
  """
  if isinstance(dct, dict) and '__ndarray__' in dct:
      data = base64.b64decode(dct['__ndarray__'])
      return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
  return dct
def load(*args, **kwargs):
    kwargs.setdefault('object_hook', json_numpy_obj_hook)
    return json.load(*args, **kwargs)

def getIndex(x, input, descriptor):
  for i, x in enumerate(os.listdir(descriptor)):
    if '_'.join(x.split('.')[0].split('_')[:-1]) == input.split('.')[0] :
      
      return i

def search(input, top, descriptor, sim):

  import json
  features = []
  #features_file =  os.path.join(FEATURES_DIR, descriptor + '_features.txt')
  for i, x in enumerate(os.listdir(descriptor)):
    features.append((x, np.load(descriptor + '/' + x, allow_pickle=True)))
    if '_'.join(x.split('.')[0].split('_')[:-1]) == input.split('.')[0] :
      
      index_input = i

  print(input)
  img_list, img_non_proches, base_image = recherche_f(index_input, top, features, sim, descriptor)

  print("Temps moyen de recherche : " + str(int((time.time() - start) / top)) + " secondes.")

  R,P,AP,RP = Compute_RP(top, input, img_non_proches)
  nameIMG = Display_RP("rp.csv", input, descriptor)
  print(f"R = {R} \t P = {P} \t AP = {AP} \t RP = {RP}")

  return img_list, base_image, nameIMG, [R,P,AP,RP]




