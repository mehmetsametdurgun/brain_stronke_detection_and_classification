from keras import backend as K
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import pandas as pd
import keras
import numpy as np
import os 
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image as im
import SimpleITK as sitk    #conda install -c simpleitk simpleitk
                            #conda install -c simpleitk/label/dev simpleitk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

img_rows, img_cols = 256,256
dim=(img_rows, img_cols)

#%%
#yarışma verisi yolu
data_path="C:/Users/samet/OneDrive/Masaüstü/teknofest_test/yarisma_verisi/"


dicom_images_names = os.listdir(data_path)


path_dicom_images = []
for name in dicom_images_names:
    path_dicom_images.append("yarisma_verisi//" + name)
    
#beyin dataframe
df_dicom = pd.DataFrame(dicom_images_names,columns=['name'])
df_dicom["path"] = path_dicom_images
df_dicom = df_dicom.sort_values(by=['name'])

ids = []
for name in dicom_images_names:
    ids.append(name.split(".")[0])  

#%%
dicom_img=[]

for dicom_name in df_dicom["path"]:
    # SimpleITK ile tek dosya oku
    reader = sitk.ImageFileReader() 
    reader.SetFileName(dicom_name)
    image = reader.Execute()

    # SimpleITK image'i numpy array'e cevir
    npimage = sitk.GetArrayFromImage(image)

    skull_windowLevel=100
    skull_windowWidth=0

    windowLevel = 50
    windowWidth = 100

    # [WL - WW / 2, WL + WW / 2] aralığını [0, 1.0] aralığına taşı
    windowstart = windowLevel - windowWidth / 2
    npgray = (npimage[0] - windowstart) / windowWidth
    npgray = np.clip(npgray, 0, 1.0)
    npgray = npgray * 255

    skull_windowstart = skull_windowLevel - skull_windowWidth / 2
    skull_npgray = (npimage[0] - skull_windowstart) / skull_windowWidth
    skull_npgray = np.clip(skull_npgray, 0, 1.0)
    skull_npgray = skull_npgray * 255

    npgray=npgray-skull_npgray
    
    npgray = npgray.astype(np.uint8)
    img = cv2.cvtColor(npgray,cv2.COLOR_GRAY2RGB)    
    resized = cv2.resize(img,(256,256), interpolation = cv2.INTER_AREA)
    dicom_img.append(resized)


    
#%%
#3 dim dicom_list 
dicom_img = np.array(dicom_img,dtype="float32")
dicom_img = dicom_img/255

#dicom_img_test will use on main model
dicom_img_test = np.array([])
dicom_img_test = dicom_img.copy()

#preparing data for skull model
mean = np.mean(dicom_img)
std = np.std(dicom_img)

dicom_img -= mean
dicom_img /= std
dicom_img += (150/255)

#%%
#Creating model
K.set_image_data_format('channels_last')

smooth = 1.0

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#1.0 eklendi
def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)

def unet(batch_norm=False,channel=3,output=1):
    
    inputs = Input((img_rows, img_cols,channel))

    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    if batch_norm:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    if batch_norm:
        conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = Activation('relu')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    if batch_norm:
        conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    if batch_norm:
        conv2 = BatchNormalization(axis=3)(conv2)
    conv2 = Activation('relu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    if batch_norm:
        conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    if batch_norm:
        conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = Activation('relu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    if batch_norm:
        conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    if batch_norm:
        conv4 = BatchNormalization(axis=3)(conv4)
    conv4 = Activation('relu')(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    if batch_norm:
        conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    if batch_norm:
        conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = Activation('relu')(conv5)

    up6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4], axis=3)

    conv6 = Conv2D(256, (3, 3), padding='same')(up6)
    if batch_norm:
        conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    conv6 = Conv2D(256, (3, 3), padding='same')(conv6)
    if batch_norm:
        conv6 = BatchNormalization(axis=3)(conv6)
    conv6 = Activation('relu')(conv6)

    up7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)

    conv7 = Conv2D(128, (3, 3), padding='same')(up7)
    if batch_norm:
        conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    if batch_norm:
        conv7 = BatchNormalization(axis=3)(conv7)
    conv7 = Activation('relu')(conv7)

    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)

    conv8 = Conv2D(64, (3, 3), padding='same')(up8)
    if batch_norm:
        conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    conv8 = Conv2D(64, (3, 3), padding='same')(conv8)
    if batch_norm:
        conv8 = BatchNormalization(axis=3)(conv8)
    conv8 = Activation('relu')(conv8)

    up9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)

    conv9 = Conv2D(32, (3, 3), padding='same')(up9)
    if batch_norm:
        conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    conv9 = Conv2D(32, (3, 3), padding='same')(conv9)
    if batch_norm:
        conv9 = BatchNormalization(axis=3)(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(output, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer = Adam(lr = 1e-3), loss = "binary_crossentropy", metrics = [dice_coef])

    return model

model_for_crop = unet()

#%%
#ResUNetPlusPlus-with-CRF and TTA 
model_for_crop.load_weights('C://Users//samet//OneDrive//Masaüstü//teknofest_test//modeller//weights_128.h5')

dependencies = {
    'dice_coef': dice_coef
}
 
model_for_kanama = tf.keras.models.load_model('C://Users//samet//OneDrive//Masaüstü//teknofest_test//modeller//kanama_weihts.h5', custom_objects = dependencies)
model_for_iskemi = tf.keras.models.load_model("C://Users//samet//OneDrive//Masaüstü//teknofest_test//modeller//tikayici_segmentation_weihts.h5", custom_objects = dependencies)

#%%
cropped_img=[]

for x in range(dicom_img.shape[0]):    
    örnek = np.expand_dims(dicom_img[x],axis=0)
    pred = model_for_crop.predict(örnek)
    kernel = np.ones((3,3), dtype = np.uint8)
    dilation = cv2.dilate(pred[0][:,:,:],kernel,iterations = 3)

    copy = dicom_img_test[x].copy()
    
    dilation = (255.0 / dilation.max() * (dilation - dilation.min())).astype(np.float32)
    _, dilation = cv2.threshold(dilation,100,255,cv2.THRESH_BINARY)

    dilation=dilation/255
      
    corr=np.zeros((256,256,3))

    if dilation.sum()<50:
        corr=copy
    else:
        corr[dilation==1]=copy[dilation==1]
        corr[dilation!=1]=0
    
    #normal
    if corr.sum()<2500:
        rescaled = (255.0 / copy.max() * (copy - copy.min())).astype(np.uint8)
    else:
        rescaled = (255.0 / corr.max() * (corr - corr.min())).astype(np.uint8)
        

    crop_ilk=rescaled.astype(np.float32)/255
    
    cropped_img.append(crop_ilk)

cropped_img = np.array(cropped_img,dtype="float32")

#%%
seg_preds=[]
file_path = r"C:\Users\samet\OneDrive\Masaüstü\teknofest_test\segment"
kernel = np.ones((2,2),np.uint8)

for x in range(cropped_img.shape[0]):
    örnek = np.expand_dims(cropped_img[x],axis=0)
    
    pred_kanama = model_for_kanama.predict(örnek)
    pred_kanama[pred_kanama>(1*29/100)]=2
    pred_kanama[pred_kanama<(1*29/100)]=0

    pred_iskemi = model_for_iskemi.predict(örnek)
    pred_iskemi[pred_iskemi>(1*29/100)]=1
    pred_iskemi[pred_iskemi<(1*29/100)]=0

    pred = pred_iskemi[0] + pred_kanama[0]


    os.chdir(file_path) 
    cv2.imwrite(ids[x] +".png", pred) 
    
#%%

os.chdir("C:\\Users\samet\\OneDrive\\Masaüstü\\teknofest_test\\")

see_path = "C:/Users/samet/OneDrive/Masaüstü/teknofest_test/segment/"


see_names = os.listdir(see_path)


see = []
for name in see_names:
    see.append("segment//" + name)
    
#beyin dataframe
see_df = pd.DataFrame(see_names,columns=['name'])
see_df["path"] = see
see_df = see_df.sort_values(by=['name'])



see_img=[]
for name_seg in see_df["path"]:
    img_seg=cv2.imread(name_seg,0)
    img_seg=cv2.resize(img_seg,dim)
    see_img.append(img_seg)

see_img = np.array(see_img,dtype="float32")


#%%
for i in range(5): 
    plt.figure()
    plt.imshow(cropped_img[i])
    plt.figure()
    plt.imshow(see_img[i])

#
