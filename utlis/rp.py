def Compute_RP(RP_file, top,nom_image_requete, nom_images_non_proches):
  text_file = open(RP_file, "w")
  rappel_precision=[]
  rp = []
  position1=int(nom_image_requete)//100
  for j in range(top):
    position2=int(nom_images_non_proches[j])//100
    if position1==position2:
      rappel_precision.append("pertinant")
    else:
      rappel_precision.append("non pertinant")
  for i in range(top):
    j=i
    val=0
    while j>=0:
      if rappel_precision[j]=="pertinant":
        val+=1
      j-=1
    rp.append(str((val/(i+1))*100)+" "+str((val/top)*100))
    with open(RP_file, 'w') as s:
      for a in rp:
        s.write(str(a) + '\n')
        
def Display_RP(fichier):
  x = []
  y = []
  with open(fichier) as csvfile:
    plots = csv.reader(csvfile, delimiter=' ')
    for row in plots:
      x.append(float(row[0]))
      y.append(float(row[1]))
      fig = plt.figure()
  plt.plot(y,x,'C1', label="VGG16" );
  plt.xlabel('Rappel')
  plt.ylabel('Précison')
  plt.title("R/P")
  plt.legend()