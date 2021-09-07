import numpy as np
import tensorflow as tf
import os
from PIL import Image
import random
from tqdm import tqdm
import cv2
import math
import keras
import sys
sys.path.append((os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('FEAT.ipynb')))))))))
from models.load_model import *
from load_data import *
from get_embeddings import *
from feat import *

def test(output):
    for model_name in output:
        print('-'*10)
        print(model_name)
        print('-'*10)

        dict_XY, dict_AB, dict_X, dict_Y, dict_A, dict_B = output[model_name]
        cossims = construct_cossim_lookup(dict_XY, dict_AB)

        pval = p_val_permutation_test(dict_X, dict_Y, dict_A, dict_B, 100000, cossims=cossims, parametric=False)
        print("p-value : %f"%pval)

        esize = effect_size(dict_X, dict_Y, dict_A, dict_B, cossims=cossims)
        print("effect size : %f"%esize)
        print()

if __name__=="__main__":
    # 1. load models
    openface = OpenFace()
    arcface = ArcFace()
    vggface = VggFace()
    deepface = DeepFace()
    facenet = FaceNet()
    deepid = DeepID()
    models = {'openface':openface, 'arcface':arcface, 'vggface':vggface, 'deepface':deepface, 'facenet':facenet, 'deepid':deepid}


    # 2. load images
    trg_1 = input('Please enter the first target name : ') 
    trg_2 = input('Please enter the second target name : ') 
    
    attr_1 = input('Please enter the first attribute name : ') 
    attr_2 = input('Please enter the second attribute name : ') 

    X = get_target_images(trg_1)
    Y = get_target_images(trg_2)
    AX = get_attr_images(attr_1, trg_1)
    AY = get_attr_images(attr_1, trg_1)
    BX = get_attr_images(attr_1, trg_1)
    BY = get_attr_images(attr_1, trg_1) 

    output = forward(models, X, Y, AX, AY, BX, BY)
    test(output)