from .detect_features import *
from .transform_skin import change_skin_tone
import numpy as np         
import dlib

# make nose, lip thinner
# input, output : np.array
def change_nose(img, level):
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects)==0:
        return []
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)

        output,facial_features_cordinates = visualize_facial_landmarks(image, shape)
    im = image.copy()
  
    # 1. narrow nose
    nose = facial_features_cordinates['Nose']
    x1 = np.min(nose[:,0])
    x2 = np.max(nose[:,0])
    y1 = np.min(nose[:,1])
    y2 = np.max(nose[:,1])
    
    nose_part = im[:,x1:x2]
    
    #resize nose
    width_level = 1 - (1-0.5)* 0.25 * level
    new_nose = cv2.resize(nose_part, None, fx = width_level, fy = 1, interpolation = cv2.INTER_CUBIC) 

    
    whited = np.concatenate((im[:,:x1], new_nose, + im[:,x2:]), axis=1)
    
    return whited 

def change_lip(img, level):
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects)==0:
        return []
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = shape_to_numpy_array(shape)

        output,facial_features_cordinates = visualize_facial_landmarks(image, shape)

    im = image.copy()
    
    # 2. narrow lip
    Mouth = facial_features_cordinates['Mouth']
    x_min = np.min(Mouth[:,0])
    x_max = np.max(Mouth[:,0])
    y_min = np.min(Mouth[:,1])
    y_max = np.max(Mouth[:,1])
    
    lip_part = im[y_min:y_max, :]
    
    narrow_level = 1 - (1-0.6) * 0.25 * level
    new_lip = cv2.resize(lip_part, None, fx = 1, fy = narrow_level, interpolation = cv2.INTER_CUBIC) 
    
    whited = np.concatenate((im[:y_min,:], new_lip, + im[y_max:,:]), axis=0)
    
    return whited 

#input : input image path, output image path
#output : void -> save transformed in path
def change_total(input_path, output_path, level=4):
    black_skin =  np.array([79,48,25])
    white_skin = np.array([255,226,214])
    change_color = tuple(black_skin + (white_skin - black_skin) * level * 0.25)
    try:
        skin_changed = change_skin_tone(input_path, change_color)
    except:
        return
    try:
        nose_changed = change_nose(skin_changed, level)
    except:
        return
    try:
        total_changed = change_lip(nose_changed, level)
        total_changed = cv2.cvtColor(total_changed.astype('uint8'), cv2.COLOR_BGR2RGB)
        #save
        cv2.imwrite(output_path, total_changed)
    except:
        return 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')