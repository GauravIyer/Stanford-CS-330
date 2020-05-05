import os
import random
import numpy as np
from utils import get_images,load_images,image_file_to_array

class DataGenerator(object):
  def __init__(self,root,num_classes,num_samples_per_class):
    self.num_classes=num_classes
    self.num_samples_per_class=num_samples_per_class
    self.root=root
    data_folder=self.root
    character_paths=[]
    for family in os.listdir(data_folder):
      if os.path.isdir(os.path.join(data_folder,family)):
        for character in os.listdir(os.path.join(data_folder,family)):
          if (os.path.isdir(os.path.join(data_folder,family,character))):
            character_paths.append(os.path.join(data_folder,family,character))
    random.shuffle(character_paths)
    num_train=1200
    self.metatrain_paths=character_paths[:num_train]
    self.metatest_paths=character_paths[num_train:]
    self.metatrain_images=load_images(self.metatrain_paths)
    self.metatest_images=load_images(self.metatest_paths)
    print("Data Loaded.")

  def sample_batch(self,batch_type,batch_size):
    if batch_type=="train":
      images_dict=self.metatrain_images
    elif batch_type=="test":
      images_dict=self.metatest_images
    all_image_batches=[]
    all_label_batches=[]
    one_hot_labels=np.identity(self.num_classes)
    for _ in range(batch_size):
      images_labels=get_images(images_dict,self.num_classes,one_hot_labels,self.num_samples_per_class)
      images=[i for i,l in images_labels]
      labels=[l for i,l in images_labels]
      images=np.vstack(images).reshape((self.num_samples_per_class,self.num_classes,-1))
      labels=np.vstack(labels).reshape((self.num_samples_per_class,self.num_classes,-1))
      all_image_batches.append(images)
      all_label_batches.append(labels)
    all_image_batches=np.stack(all_image_batches).astype(np.float32)
    all_label_batches=np.stack(all_label_batches).astype(np.float32)
    return all_image_batches,all_label_batches