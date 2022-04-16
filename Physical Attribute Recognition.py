# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 22:21:47 2021

@author: marce
"""

import os
import cv2
import sys
import glob
import logging
import argparse
import numpy as np
import mxnet as mx 
import pandas as pd
import pytesseract
import pickle
from pathlib import Path
from dotenv import load_dotenv
import model.emotion.detectemotion as ime
from mxnet_moon.lightened_moon import lightened_moon_feature
from PIL import Image
from deepface import DeepFace
from io import BytesIO
from IPython import get_ipython
import tensorflow as tf
import pandas as pd
import os
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from scripts.models import FacePrediction
import glob
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN
import logging
import os
import GPUtil

#%%

# Pytesseract for text recognition
pytesseract.pytesseract.tesseract_cmd= r"...\tesseract.exe"

logging.basicConfig(level=logging.DEBUG)

#Skip Warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


print(mx.__version__) 
#OpenCV Version

#print (cv2.__version__) 

#%% Loading Enviroment

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


pathImg = os.getenv("D:/DIP") #Processed images from DIP process

"""Creating columns to dataframe"""


data_image=list()
data_date= list()
data_age=list()
data_brand=list()
data_num_person=list()
data_gender=list()
data_ethnicity=list()
data_emotion=list()
data_bmi=list()
data_bmi2=list()

#%% BMI Predictor model by Chaoran Liu

devices='gpu'
mode = 'predict' #'train' or 'predict'

model_type = 'vgg16'
model_tag = 'seqMT'

model_id = '{:s}_{:s}'.format(model_type, model_tag)

model_dir = r"...\saved_model\model_7_best.h5"
bs = 5

epochs = 2
freeze_backbone = True # True => transfer learning; False => train from scratch

import pandas as pd
import os
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from scripts.models import FacePrediction
import glob

allimages = os.listdir(r"...\face_aligned")
train = pd.read_csv(r"...\train.csv")
valid = pd.read_csv(r"...\valid.csv")

train = train.loc[train['index'].isin(allimages)]
valid = valid.loc[valid['index'].isin(allimages)]

# create metrics, model dirs
Path(r"...\metrics").mkdir(parents = True, exist_ok = True)
Path(r"...\saved_model").mkdir(parents = True, exist_ok = True)

es = EarlyStopping(patience=3)
ckp = ModelCheckpoint(model_dir, save_best_only=True, save_weights_only=True, verbose=1)
tb = TensorBoard('./tb/%s'%(model_id))
callbacks = [es, ckp]

model_bmi = FacePrediction(img_dir = r"...\face_aligned", model_type = model_type)
model_bmi.define_model(freeze_backbone = freeze_backbone)

if mode == 'train':
    model_history = model_bmi.train(train, valid, bs = bs, epochs = epochs, callbacks = callbacks)
else:
    model_bmi.load_weights(model_dir)
    
physical_devices = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)

#%% Functions

# Face boxes detector funtion

def face_boxes(filename): #faces= result_list

    with tf.device('/device:GPU:0'):

        # Filename= Path of image
    
        # load image from file
        pixels = pyplot.imread(filename)
        
        # create the detector, using default weights
        detector = MTCNN()
        
        # detect faces in the image
        faces = detector.detect_faces(pixels)
        
        # face boxes list
        boxes=list()
        
        for result in faces:
            if result['confidence']>=0.95:
            
                # get coordinates
                x, y, width, height = result['box']
                
                boxes.append([x,y,x+width,y+height])
            else:
                continue
            
        return(boxes)
   


# Gender, Age and Race predictor

def Gender_Age_Race(image,faceBox): 
    
    with tf.device('/device:GPU:0'):
        padding=20
        
        image=cv2.imread(image, 1 )
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Getting the face from image to predict 
        face=image[max(0,faceBox[1]-padding):min(faceBox[3]+padding,image.shape[0]-1),max(0,faceBox[0]-padding):min(faceBox[2]+padding, image.shape[1]-1)]
        
        # cv2.imshow('image',face)
        # cv2.waitKey(0)

        # predict Age, Gender and race
        
        obj = DeepFace.analyze(face, actions = ['age','race', 'gender'],enforce_detection=False)
        
        df_age_gender= str (obj["age"])+" years old "+ str(obj["gender"])
        age=str(obj["age"])
        gender= str(obj["gender"])
        race_detected="ethnicity: "+str (obj["race"])
        
        return age,gender,race_detected
     
# Detecting Brands on Image

def brands_on_image(image_path):
    
    with tf.device('/device:GPU:0'):
        
        try:
            
            target=list()
            brands=list()
            
            img = Image.open(image_path) 
            
            splitter= image_path.split("-")
            
            splitter=splitter[-1][0:-4].upper()
            if splitter.isupper():
                brands.append(splitter)                                    
                
            # Passing the image object to  
            # image_to_string() function 
            # This function will 
            # extract the text from the image 
            
            text = pytesseract.image_to_string(img)
            
            frase= text.split("\n")
    
            for i in frase:
                if ".cl" in i:
                    target.append(i)
                else:
                    continue
            
            for k in target:
                splitting= k.split(" ")
                for j in splitting:
                    if ".cl" in j:
                        brands.append(j)
                    # elif ".com" in j:
                    #         brands.append(j)
                    else:
                        continue
            
            return(brands)
        
        except:
            
            return("No detected Brands on Image")
        
        
        

# Main Function

save_path=[]


""" Function for gender detection,age detection and emotion detection"""            
def main():

    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    img_dir = os.path.join(pathImg)
    
    if os.path.exists(img_dir):
        names = os.listdir(pathImg)      
        
        # print(type(names),len(names),names[0]) #filenames in folder
        img_paths = [name for name in names]
        # print((type(img_paths),len(img_paths),img_paths[0]))
    
        sorting= sorted(names)     

        target= [name for name in names]  #targetting date
        
        # target= [name for name in names if name.startswith("2016-12")]
        
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

            
        with tf.device('/device:GPU:0'):
    
            for imge in target:

                paths = (img_dir + os.sep + imge).replace("\\","/")
                            
                print("Image Path and type",paths,type(paths))
                          
                
                image = cv2.imread(paths)

                
                
                img = cv2.imread(paths, 1)
                
                try:   
                    faceBoxes=face_boxes(paths)
                
                except:
                    continue
                
                
                if len(faceBoxes)==0:
                    print("No face detected")
                    continue
                
                
                
                print("HERE FACEBOXES:",faceBoxes)
                # Adding bmi to image
                    
                # Loop throuth the coordinates
                
                # Loop throuth the coordinates
                # model_bmi.predict_bmifaces(path)
                
                counter=0
                try:
                    bmi_prediction,faces,boxes = model_bmi.predict_faces2(paths)
                
                except:
                    print("error in resizing ")
                    continue
                        
                
                if not faceBoxes:
                    print("No face detected")
                    continue
                
                print("Print faceboxes",faceBoxes)
                # print("Print boxes", boxes_i)
                
                for faceBox in faceBoxes:
                    # imagen = Image.fromarray(resultImg)
                    # imagen.show()
                    
                    # Add Brand detected
                    
                    try:
                        text_on_image= brands_on_image(paths)
                        if len(text_on_image)==0:
                            break
                        else:
                            data_brand.append(text_on_image)
    
                        print("#====Detected Brands on Image====#")
                        print(text_on_image)
                    except:
                        data_brand.append("null")
                    
                    
                    # Add num person
                    try:
                        data_num_person.append(counter)
                        
                    except:
                        data_num_person.append("null")
                    
                    
                    
                    # Add path to data collection
                    
                    try: 
                        data_image.append(imge)
                    
                    except:
                        data_image.append("null")
                        
                        
                    
                    # Add brand to data collection               
                    
                    
                    
                    # Predict Ages and genders
    
                    print("#====Detected Age, Gender and Race====#")
                    try:
                        gender,age,race_detected = Gender_Age_Race(paths,faceBox)
                        data_gender.append(gender)
                        data_age.append(age)
                        data_ethnicity.append(race_detected)
                    
                        print('Gender',gender)
                        print('Age',age)
                        print ('Race',race_detected)

                    except:
                        print("Error")
                        data_gender.append("null")
                        data_age.append("null")
                        data_ethnicity.append("null")
                    
                    
                     # Predict BMI2
                        
                    
                    print("#====Detected BMI2===========#")
                    
                    
                    
                    print ("Bmi_PREDICTION", bmi_prediction )
                    
                    try:    
                        data_bmi2.append(bmi_prediction[counter][0][0])
                        print('BMI', bmi_prediction[counter][0][0])
                        
                    except:
                        print("Error")
                        data_bmi2.append("null")
                        

                    
                     #  Add Date to index 
                    print("#====Detected Date Images===========#")
                    
                    try:
                        data_date.append(imge[:10])
                        
                    except:
                        print("Error")
                        data_date.append("null")
                    
                    
                    
                    
    
#%%  
    
#====Data Collection===========#
# cv2.destroyAllWindows()  
    
    
data = {'Image name': data_image,
            'Date': data_date,
            'Brand': data_brand,
            'Num Person': data_num_person,
            'Gender': data_gender,
            'Age': data_age,
            'Ethnicity': data_ethnicity,
            'BMI2': data_bmi2,
            }
    
atr= [i for i in data.keys()]


#%% SAVING MODEL




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="predict the face attribution of one input image")
    parser.add_argument('--gpus', type=str, help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--pad', type=float, nargs='+',
                                  help="pad (left,top,right,bottom) for face detection region")
    args = parser.parse_args()
       
    logging.info(args)
    
    main()




#%%  

data_collection = pd.DataFrame(data)



data_collection.to_csv(r"\data.csv",index=False)
