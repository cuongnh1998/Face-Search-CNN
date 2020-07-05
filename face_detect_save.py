import cv2
import glob
import pandas as pd
from imageio import imread,imsave,imwrite
from skimage.transform import resize
from tqdm import tqdm
import dlib

img_path = glob.glob("image/*")
print(img_path)
img_data = pd.DataFrame(columns=['image','label','name'])

for i,train_path in tqdm(enumerate(img_path)):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        img_data.loc[len(img_data)]=[image,i,name]
        
print(img_data)
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
for img_path in img_data.image:

    image = imread(img_path)

    print("Processing : " + img_path)

    faces_cnn = cnn_face_detector(image,1)
    faceRect = faces_cnn[0]
    
    x1 = faceRect.rect.left()
    y1 = faceRect.rect.top()
    x2 = faceRect.rect.right()
    y2 = faceRect.rect.bottom()

    face = image[y1:y2,x1:x2]
    imsave(img_path,face)
    print ("Done : " + img_path)
    