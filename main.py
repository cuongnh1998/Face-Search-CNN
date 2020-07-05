import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from imageio import imread
from scipy.spatial import distance
from keras.models import load_model
import pandas as pd
from tqdm import tqdm
import dlib
from model import create_model
from align import AlignDlib
import glob
import imutils
import os

# INITIALIZE MODELS
nn4_small2 = create_model()
# Tóm tắt kiến trúc của model
nn4_small2.summary()
# đây là gọi kiến trúc mạng có sẵn
nn4_small2.load_weights('weights/nn4.small2.v1.h5')
# load bo du liey 68 dac trung de quay mat
alignment = AlignDlib('shape_predictor_68_face_landmarks.dat')

#  chạy ảnh trong csdl
train_paths = glob.glob("image/*")
print(train_paths)
train_paths_v2 = glob.glob("image/csdl/*")
nb_classes = len(train_paths)

df_train = pd.DataFrame(columns=['index','image', 'label', 'name'])

index = 0
for i,train_path in enumerate(train_paths):
    name = train_path.split("\\")[-1]
    images = glob.glob(train_path + "/*")
    for image in images:
        df_train.loc[len(df_train)]=[index,image,i,name]
        index +=1
print(df_train)
# can chinh lai mat
def align_face(face):
    print(face.shape)
    (h,w,c) = face.shape
    bb = dlib.rectangle(0, 0, w, h)
    print('BB',bb)
    return alignment.align(96, face, bb,landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
# lấy dữ liệu để tính embedding vector
def load_and_align_images(filepaths):
    aligned_images = []
    for filepath in filepaths:
        print(filepath)
        print("Processing : " + filepath)
        img = cv2.imread(filepath)
        print(img,'???')
        aligned = align_face(img)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
            
    return np.array(aligned_images)
    
def calc_embs(filepaths, batch_size=64):
    pd = []
    for start in tqdm(range(0, len(filepaths), batch_size)):
        aligned_images = load_and_align_images(filepaths[start:start+batch_size])
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_images)))
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)

    return np.array(embs)
# xoay mat cua anh dau vao
def align_faces(faces):
    aligned_images = []
    for face in faces:
        aligned = align_face(face)
        aligned = (aligned / 255.).astype(np.float32)
        aligned = np.expand_dims(aligned, axis=0)
        aligned_images.append(aligned)
        
    return aligned_images
# tinh em 
def calc_emb_test(faces):
    pd = []
    aligned_faces = align_faces(faces)
    if(len(faces)==1):
        pd.append(nn4_small2.predict_on_batch(aligned_faces))
    elif(len(faces)>1):
        pd.append(nn4_small2.predict_on_batch(np.squeeze(aligned_faces)))
    #embs = l2_normalize(np.concatenate(pd))
    embs = np.array(pd)
    #print(embs)
    return np.array(embs)

# Bắt đầu quá trình trainning ảnh đầu
label2idx = []
for i in tqdm(range(len(train_paths))):
    label2idx.append(np.asarray(df_train[df_train.label == i].index))

print(label2idx)
# sau khi tính emb của tập train sẽ lưu vào train_embs.npy
# train_embs = calc_embs(df_train.image)
# np.save("train_embs.npy", train_embs)

# trả về 1 mảng nối các vector
train_embs = np.load('train_embs.npy')
train_embs = np.concatenate(train_embs)
print('Vector cua csdl',list(train_embs))
for a in list(train_embs):
    print(len(a))
# cho ảnh đầu vào và tìm emb của input sau đó tính khoảng cách euclide của vector input với từng vector trong tập train
test = 0
test_paths = glob.glob("test_image/*.jpg")
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
for path in test_paths:
    test_image = cv2.imread(path)
    show_image = test_image.copy()
    # print("Processing : " + path)

    faces_cnn = cnn_face_detector(test_image, 1)
    faceRect = faces_cnn[0]

    x1 = faceRect.rect.left()
    y1 = faceRect.rect.top()
    x2 = faceRect.rect.right()
    y2 = faceRect.rect.bottom()

    faces = []
    face = test_image[y1:y2,x1:x2]
    print("Done : " + path)
    faces.append(face)
    # tinh embs vector cho anh tim kiem
    test_embs = calc_emb_test(faces)

    test_embs = np.concatenate(test_embs)

    print('vector anh tim kiem',list(test_embs))
    test_distances = []
    min_distances = 5
    # lay so hang cua ma tran do va chay for 
    # shape [0] return so hang
    for i in range(test_embs.shape[0]):
        distances = []#luu khoanh cac giua cac embs vector
        for j in range(len(train_paths)):
            for k in label2idx[j]:
                # test_distances.append(distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1)))
                a = distance.euclidean(test_embs[i].reshape(-1), train_embs[k].reshape(-1))
                print(a)
                if min_distances > a:
                    min_distances = a
                    test = k
            print("Index của ảnh giống nhất trong csdl là: ", test)
            print("Khoảng cách nhỏ nhất giữa hai embs là: ", min_distances)
            print("Đường dẫn của ảnh giống nhất là :", df_train.loc[test].values[1])

# show ra ảnh đầu vào và ảnh giống nhất
enumm = 0
for path in train_paths_v2:
    get_image = cv2.imread(path)
    if enumm == test:
        #print(path)
        get_image = imutils.resize(get_image, width=360)
        str1 = df_train.loc[test].values[1]
        str2 = str1.split("\\", 2)
        os.chdir('C:/Users/ninhh/OneDrive/Desktop/project_DPT_v2/test_image')
        a = []
        a = os.listdir()
        show_image = imutils.resize(show_image, width=360)
        cv2.imshow(a[0],show_image)
        cv2.waitKey(0)
        cv2.imshow(str2[2], get_image)
        cv2.waitKey(0)
        break
    else:
        enumm += 1

