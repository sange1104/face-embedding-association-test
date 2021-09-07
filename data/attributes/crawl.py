from google_images_download import google_images_download   
import os
import dlib
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


def get_data(attr_list):
    for i in attr_list:
        response = google_images_download.googleimagesdownload()   #class instantiation

        arguments = {"keywords":"asian "+i,"limit":100,"print_urls":True, "chromedriver":'../chromedriver.exe'}   #put your chromedriver path
        paths = response.download(arguments)  

def get_frontal_face(attr_list):
    face_detector = dlib.get_frontal_face_detector()

    for j,i in enumerate(attr_list):
        query = i
        img_dir = './downloads/asian '+query +'/'
        n = 0
        for img in os.listdir(img_dir):
            path = os.path.join(img_dir,img)
            try:
                image = np.array(Image.open(path))
                try:
                    faces = face_detector(image)
                    exist = 1
                    if len(faces) == 0:
                        exist = 0
                except:
                    exist = 0

                if exist == 1 and n < 10:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(img_dir + 'as_' + query + '_%d.jpg'%n, image)
                    n += 1
                os.remove(path)
            except:
                os.remove(path)
        print(j, '. ', img_dir, ' > ', len(os.listdir(img_dir)))

def mv_dir(attr_name, attr_list):
    new_path = './data/'+str(attr_name)
    try: 
        os.makedirs(new_path) 
    except OSError: 
        if not os.path.isdir(new_path): 
            raise

    for j,i in enumerate(attr_list):
        query = i
        img_dir = './downloads/asian '+query +'/'
        for img in os.listdir(img_dir):
            path = os.path.join(img_dir,img)
            image = cv2.imread(path)
            save_path = os.path.join(new_path, img)
            cv2.imwrite(save_path, image) 
        print('finished > ',query)


if __name__=="__main__":
    # set query keywords for each attribute
    career = ['executive', 'executives', 'management', 'managements', 'professional', 'corporation', 'corporations', 'salary', 'salaries', 'office', 'offices', 'business', 'businesses', 'career', 'careers']
    family = ['home', 'homes', 'parent', 'parents', 'child', 'children', 'family', 'families', 'cousin', 'cousins', 'marriage', 'marriages', 'wedding', 'weddings', 'relative', 'relatives']

    pleasant = ['caress', 'caresses', 'freedom', 'health', 'love', 'peace', 'cheer', 'friend',
            'friends', 'heaven', 'loyal', 'pleasure', 'diamond', 'diamonds', 'gentle', 'honest',
            'lucky', 'rainbow', 'rainbows', 'diploma', 'diplomas', 'gift', 'gifts', 'honor',
            'honors', 'miracle', 'miracles', 'sunrise', 'sunrises', 'family', 'families', 'happy',
            'laughter', 'paradise', 'paradises', 'vacation', 'vacations']
    unpleasant = ['abuse', 'abuses', 'crash', 'crashes', 'filth', 'murder', 'murders',
                'sickness', 'sicknesses', 'accident', 'accidents', 'death', 'deaths',
                'grief', 'poison', 'poisons', 'stink', 'assault', 'assaults', 'disaster',
                'disasters', 'hatred', 'pollute', 'tragedy', 'tragedies', 'bomb', 'bombs', 'divorce',
                'divorces', 'jail', 'jails', 'poverty', 'ugly', 'cancer', 'cancers', 'evil', 
                'kill', 'rotten', 'vomit']

    likable = ['agreeable', 'fair', 'honest', 'trustworthy', 'selfless', 'accommodating', 'likable', 'liked']
    unlikable = ['abrasive', 'conniving', 'manipulative', 'dishonest', 'selfish', 'pushy', 'unlikable', 'unliked']
                
    competent = ['competent', 'productive', 'effective', 'ambitious', 'active', 'decisive', 'strong', 'tough', 'bold', 'assertive']
    incompetent = ['incompetent', 'unproductive', 'ineffective', 'unambitious', 'passive', 'indecisive', 'weak', 'gentle', 'timid', 'unassertive']

    attr_dict = {'career':career, 'family':family, 'pleasant':pleasant, 'unpleasant':unpleasant, 'likable':likable, 'unlikable':unlikable, 'competent':competent, 'incompetent':incompetent}
    

    while True:
        attr_name = input('Please enter attribute name for crawl("q" for quit) : ') 
        if attr_name=='q':
            break
        elif attr_name not in attr_dict.keys():
            print('You can enter one of these attributes...')
            print(attr_dict.keys())
        else:
            get_data(attr_dict[attr_name])
            get_frontal_face(attr_dict[attr_name])
            mv_dir(attr_name, attr_dict[attr_name])
