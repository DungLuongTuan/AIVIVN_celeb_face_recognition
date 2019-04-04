"""
    arcface extract feature + linearSVC classifier
"""

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
import pandas as pd
import numpy as np
import subprocess
import pickle
import random
import pdb
import os


def load_test_data(test_emb_path):
    test_df = pd.read_csv("data/raw_data/sample_submission.csv")
    test_data = []
    test_filename = []

    for image_name in test_df.image.values:
        emb_path = os.path.join(test_emb_path, image_name[:-4] + ".npy")
        emb = np.load(emb_path)
        test_data.append(emb)
        test_filename.append(image_name)

    return test_data, test_filename

def main():
    #   load test data
    print("load test emb")
    test_data, test_images = load_test_data("/home/tittit/challenges/AIVIVN/celeb_face_recognition/data/emb_image/test")

    #   load SVM model
    print("load SVM model")
    with open("models/model.pkl", "rb") as f:
        clf = pickle.load(f)

    cnt = 0
    step = 2000
    pred_images = []
    pred_labels = []
    pred_functs = []

    while cnt*step < len(test_data):
        print(cnt, end = "\r")
        data = test_data[cnt*step: int(cnt + 1)*step]
        images = test_images[cnt*step: int(cnt + 1)*step]
        pred_label = clf.predict(data)
        pred_funct = clf.decision_function(data)
        for i, image in enumerate(images):
            funct = list(pred_funct[i])
            index = list(np.arange(1000))

            funct, index = zip(*sorted(zip(funct, index), key = lambda x: -x[0]))
            pred_images.append(image)
            pred_labels.append(index[0])
            pred_functs.append(funct[0])
        cnt += 1
    pred_functs, pred_images, pred_labels = zip(*sorted(zip(pred_functs, pred_images, pred_labels), key = lambda x: -x[0]))

    threshold = 0
    for funct, image, label in zip(pred_functs, pred_images, pred_labels):
        if funct > threshold:
            if not os.path.exists("data/embedding/add"):
                os.makedirs("data/embedding/add")
            label_path = os.path.join("data/embedding/add", str(label))
            if not os.path.exists(label_path):
                os.makedirs(label_path)
            new_file_path = os.path.join(label_path, image[:-4] + ".npy")
            old_file_path = os.path.join("data/embedding/test", image[:-4] + ".npy")
            subprocess.call(["cp " + old_file_path + " " + new_file_path], shell = True)

if __name__ == '__main__':
    main()