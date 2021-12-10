#change skin tone
#input : path
#output : np.array
from skin.skinDetection import change_skin
from skin.noFaceSkinDetection import obtain_skin_color
from skin.faceDetection import get_skin_color
from PIL import Image
import io
import numpy as np

def change_skin_tone(path, color):
    changed = change_skin(path, color , '')
    if len(changed) == 0:
        return []

    image = Image.open(io.BytesIO(changed))
    temp = image.resize((200,200))

    original = Image.open(path)
    original = original.resize((200,200))

    if image==None:
        return []

    if np.sum(np.array(original)-np.array(temp))==0: # no difference
        return []
    
    return np.array(image)