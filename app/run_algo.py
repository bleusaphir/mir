
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
import os
from utlis import indexation, recherche
import warnings
warnings.filterwarnings('ignore')
import cv2
import argparse
import json
import operator
parser = argparse.ArgumentParser()
parser.add_argument("--index", type=str, default='ORB', help="indexation method")
parser.add_argument("--search", type=str, default='Euclidienne', help="search method")
parser.add_argument("--top", type=int, default='20', help="top 20 or 50")
parser.add_argument("--input-dir", default= 'MIR_DATASETS_B/', help="input directory")
args = parser.parse_args()

FEATURES_DIR = os.path.join(os.getcwd(), 'app', 'features')
SAVE_DIR = 'app/static/'

def recherche_f(image_req,top, features1):
  with open('dict_name.json', 'r') as f:
    dict_names = json.load(f)

  # FileNotFoundError: [Errno 2] No such file or directory: '4_5_singes_baboon_4504_ORB.txt.npy'

  # features1[image_req][0] = 4_5_singes_baboon_4504_ORB.txt.npy
  # On veut 4_5_singes_baboon_4504.jpg

  # base --> final :
  # final = '_'.join(base.split('.')[0].split('_')[:-1]) + '.jpg'
  # avec base = features1[image_req][0]

  good_name = '_'.join(features1[image_req][0].split('.')[0].split('_')[:-1]) + '.jpg'

  base_image = dict_names[good_name]
  voisins = getkVoisins(features1, features1[image_req],top)
  #print(voisins)
  nom_images_proches = []
  nom_images_non_proches = []
  for k in range(top):
    name = '_'.join(voisins[k][0].split('.')[0].split('_')[:-1]) + '.jpg'

    nom_images_proches.append(dict_names[name])
  plt.figure(figsize=(5, 5))




  plt.imshow(imread(base_image), cmap='gray', interpolation='none')  
  plt.title("Image requête")

  plt.savefig(SAVE_DIR + good_name)


  plt.figure(figsize=(25, 25))
  plt.subplots_adjust(hspace=0.2, wspace=0.2)
  for j in range(top):
    plt.subplot(int(top/4),int(top/5),j+1)
    print(nom_images_proches[j])
    plt.imshow(imread(nom_images_proches[j]), cmap='gray', interpolation='none')
    nom_images_non_proches.append(os.path.splitext(os.path.basename(nom_images_proches[j]))[0])
    title = "Image proche n°"+str(j)
    plt.title(title)

    path_final = good_name.split('.')[0].split('.')[0] + '_result' + '.png'
    plt.savefig(SAVE_DIR + good_name.split('.')[0].split('.')[0] + '_result')
  return [good_name, path_final]


def getkVoisins(lfeatures, test, k) :
  ldistances = []
  for i in range(len(lfeatures)):
    dist = recherche.flann(test[1], lfeatures[i][1])
    ldistances.append((lfeatures[i][0], lfeatures[i][1], dist))
  ldistances.sort(key=operator.itemgetter(2))
  lvoisins = []
  for i in range(k):
    lvoisins.append(ldistances[i])
  return lvoisins

def distance_f(l1,l2,distanceName):
    if distanceName=="Euclidienne":
        distance = recherche.euclidianDistance(l1,l2)
    elif distanceName in ["Correlation","Chi carre","Intersection","Bhattacharyya"]:
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

indexationMethod = {
  'SIFT' : indexation.generateSIFT,
  'ORB' : indexation.generateORB,
  'Model' : indexation.indexationModel
}
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

def search(input, top, descriptor):
  import json
  features = []
  #features_file =  os.path.join(FEATURES_DIR, descriptor + '_features.txt')
  for i, x in enumerate(os.listdir(descriptor)):
    features.append((x, np.load(descriptor + '/' + x, allow_pickle=True)))
    if x.split('.') == input :
      index_input = i

  img_list = recherche_f(i, top, features)

  return img_list



def main():
  for path, subdirs, files in os.walk(args.input_dir):
    for name in files:
      features = indexationMethod[args.index](path)
      file = os.path.join(path, name)
      images = [1, 3, 4]
      for input in images :
        nom_image_requete, nom_images_proches, nom_images_non_proches = recherche_f(input, 20, features)
        
      break
  plt.show()

if __name__ == '__main__':
  main()

