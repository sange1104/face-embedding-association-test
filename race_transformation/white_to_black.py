from .detect_features import *
from .transform_skin import change_skin_tone
import numpy as np 
import dlib

# make nose, lip thicker
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
  
    # 1. widen nose
    nose = facial_features_cordinates['Nose']
    x1 = np.min(nose[:,0])
    x2 = np.max(nose[:,0])
    y1 = np.min(nose[:,1])
    y2 = np.max(nose[:,1])
    x1-=8
    x2+=8
    y1+=15

    roi = im[y1:y2, x1:x2]
    
    if len(roi) == 0:
        x1+=8
        x2-=8
        y1-=15
        roi = im[y1:y2, x1:x2]
        if len(roi) == 0:
            return []
    
    #resize nose
    width_level = 1 + (1.5-1) * 0.25 * level
    resize_roi = cv2.resize(roi, None, fx = width_level, fy = 1, interpolation = cv2.INTER_CUBIC) #indentation wrong
    
    newimg_res = im.copy()
    center_x = (x1 + x2)  /2
    center_y = (y1 + y2)  /2
    new_nose_half_w = resize_roi.shape[1] / 2
    newimg_res[y1:y2, int(center_x-new_nose_half_w):int(center_x+new_nose_half_w)] = resize_roi
    
    # blur the concatenated part
    appended_l = newimg_res[y1:y2, int(center_x-new_nose_half_w)-5:int(center_x-new_nose_half_w)+3].copy()
    appended_r = newimg_res[y1:y2, int(center_x+new_nose_half_w)-3:int(center_x+new_nose_half_w)+5].copy()
    
    if len(appended_l)==0 or len(appended_r)==0:
        return []

    try:
        for i in range(5):
            appended_l = cv2.GaussianBlur(appended_l, (5,5), 0)
            appended_r = cv2.GaussianBlur(appended_r, (5,5), 0)
    except:
        return []

    newimg_res[y1:y2, int(center_x-new_nose_half_w)-5:int(center_x-new_nose_half_w)+3] = appended_l
    newimg_res[y1:y2, int(center_x+new_nose_half_w)-3:int(center_x+new_nose_half_w)+5] = appended_r

    
    kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    
    # nose lip blur 
    newimg_res = cv2.GaussianBlur(newimg_res,(5,5),0)
    newimg_res = cv2.GaussianBlur(newimg_res,(5,5),0)
    newimg_res = cv2.GaussianBlur(newimg_res,(5,5),0)
    
    # image sharpening
    newimg_res = cv2.filter2D(newimg_res,-1,kernel_sharpen_1)
    newimg_res = cv2.cvtColor(newimg_res.astype('uint8'), cv2.COLOR_BGR2RGB)
    
    return newimg_res 

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
    
    # 2. widen lip
    Mouth = facial_features_cordinates['Mouth']
    x_min = np.min(Mouth[:,0])
    x_max = np.max(Mouth[:,0])
    y_min = np.min(Mouth[:,1])
    y_max = np.max(Mouth[:,1])
    
    lip = im[y_min:y_max, x_min:x_max]
    
    widen_level = 1 + (1.6-1) * 0.25 * level
    resize_lip = cv2.resize(lip, None, fx = 1.05, fy = widen_level, interpolation = cv2.INTER_CUBIC)  
    newimg_res = im.copy()
    center_x = (x_min + x_max)  /2
    center_y = (y_min + y_max)  /2
    new_lip_half_h = resize_lip.shape[0] / 2
    new_lip_half_w = resize_lip.shape[1] / 2

    newimg_res[int(center_y-new_lip_half_h):int(center_y+new_lip_half_h),int(center_x-new_lip_half_w):int(center_x+new_lip_half_w)] = resize_lip
    
    # blur the concatenated part
    appended_u = newimg_res[int(center_y-new_lip_half_h)-5:int(center_y-new_lip_half_h)+3,int(center_x-new_lip_half_w)-5:int(center_x+new_lip_half_w)+3 ].copy()
    appended_d = newimg_res[int(center_y+new_lip_half_h)-3:int(center_y+new_lip_half_h)+5,int(center_x-new_lip_half_w)-3:int(center_x+new_lip_half_w)+5 ].copy()

    for i in range(5):
        appended_u = cv2.GaussianBlur(appended_u, (3,3), 0)
        appended_d = cv2.GaussianBlur(appended_d, (3,3), 0)

    newimg_res[int(center_y-new_lip_half_h)-5:int(center_y-new_lip_half_h)+3,int(center_x-new_lip_half_w)-5:int(center_x+new_lip_half_w)+3] = appended_u
    newimg_res[int(center_y+new_lip_half_h)-3:int(center_y+new_lip_half_h)+5,int(center_x-new_lip_half_w)-3:int(center_x+new_lip_half_w)+5] = appended_d
    
    kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    
    # nose lip blur 
    newimg_res = cv2.GaussianBlur(newimg_res,(5,5),0)
    newimg_res = cv2.GaussianBlur(newimg_res,(5,5),0)
    newimg_res = cv2.GaussianBlur(newimg_res,(5,5),0)
    
    # image sharpening
    newimg_res = cv2.filter2D(newimg_res,-1,kernel_sharpen_1)
    newimg_res = cv2.cvtColor(newimg_res.astype('uint8'), cv2.COLOR_BGR2RGB)

    return newimg_res
 
#input : input image path, output image path
#output : void -> save transformed in path
def change_total(input_path, output_path, level=4):
    black_skin =  np.array([79,48,25])
    white_skin = np.array([255,226,214])
    change_color = tuple(white_skin - (white_skin - black_skin) * level * 0.25)
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