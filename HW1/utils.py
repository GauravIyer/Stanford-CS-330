import os
import random
import numpy as np
import matplotlib.pyplot as plt

def get_images(image_dict,num_classes,labels,nb_samples,shuffle=True):
  classes=random.sample(list(image_dict.keys()),num_classes)
  sampler=lambda x:random.sample(x,nb_samples)
  images_labels=[]
  for c,l in zip(classes,labels):
    for i in sampler(image_dict[c]):
      images_labels.append((i,l))
  if shuffle:
    random.shuffle(images_labels)
  return images_labels

def image_file_to_array(filename,dim_input=784):
  image=plt.imread(filename)
  image=image.reshape([dim_input])
  image=image.astype(np.float32)/255.
  image=1.0-image
  return image

def load_images(paths):
  image_dict={}
  for p in paths:
    images=[image_file_to_array(i) for i in os.scandir(p)]
    image_dict[p]=images
  return image_dict