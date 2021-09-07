import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

def load_images(dir):
    images = []
    for name in os.listdir(dir):
        if name.endswith('jpg'):  
            path = dir + name
            try: 
                image = np.array(Image.open(path)) 
                resized = np.expand_dims(cv2.resize(image, (200, 200)), axis=0)
                if resized.shape == (1, 200, 200, 3):
                    images.append(resized)
            except:
                continue
    return images

def get_target_images(target_name): 
    img_names = os.listdir('../data/targets/'+target_name)
    x = [] 

    for name in tqdm(img_names):
        path = os.path.join('../data/targets/' + target_name, name)
        image = np.expand_dims(np.array(Image.open(path)), axis=0)
        x.append(image)
         

    X = np.concatenate(x, axis=0) 
    return X

def get_attr_images(attr, trg):
    ax = [] 
    ax = load_images(os.path.join('../../../../data/attr/race/'+attr, trg)) 
    return ax