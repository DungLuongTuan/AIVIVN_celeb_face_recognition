"""
    arcface extract feature + linearSVC classifier
"""

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics
import pandas as pd
import numpy as np
import argparse
import pickle
import random
import pdb
import os

def load_train_data(rate, train_emb_path):
    labels = os.listdir(train_emb_path)
    full_data = []
    full_label = []
    for label in labels:
        train_label_path = os.path.join(train_emb_path, label)
        files = os.listdir(train_label_path)
        for file in files:
            file_path = os.path.join(train_label_path, file)
            emb = np.load(file_path)
            full_data.append(emb)
            full_label.append(int(label))
    #   separate train and dev set
    c = list(zip(full_data, full_label))
    random.shuffle(c)
    full_data, full_label = zip(*c)

    train_data = full_data[:int(rate*len(full_data))]
    train_label = full_label[:int(rate*len(full_data))]
    dev_data = full_data[int(rate*len(full_data)):]
    dev_label = full_data[int(rate*len(full_data)):]
    #   return
    return train_data, train_label, dev_data, dev_label


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


def main(args):
    #   load train data
    print("load train emb")
    train_data, train_label, _, _ = load_train_data(rate = 1.0, train_emb_path = "data/embedding/train")

    #   load add data
    if args.mode == "add":
        print("load add emb")
        add_data, add_label, _, _ = load_train_data(rate = 1.0, train_emb_path = "data/embedding/add")
        train_data += add_data
        train_label += add_label
        
    #   load test data
    print("load test emb")
    test_data, test_images = load_test_data("data/embedding/test")

    #   train new SVM model
    print("train new SVM model")
    if not os.path.exists("models/model.pkl"):
        Cs = [1., 10., 15., 20., 100., 1000.]
        parameters = {'C':Cs}
        svm = LinearSVC(multi_class = 'ovr')
        clf = GridSearchCV(svm, parameters, cv=5)
        clf.fit(X = train_data + add_data, y = train_label + add_label)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(clf, f)
    else:
        with open("models/model.pkl", "rb") as f:
            clf = pickle.load(f)

    print("eval on train set")
    true_pred = 0
    for data, label in zip(train_data, train_label):
        pred_label = clf.predict([data])[0]
        if pred_label == label:
            true_pred += 1
    print("accuracy on train: ", true_pred/len(train_data))

    print("predict label for test set")
    threshold = -0.55

    cnt = 0
    step = 2000

    f = open("submission.csv", "w")
    f.write("image,label\n")
    while cnt*step < len(test_data):
        print(cnt, end = "\r")
        data = test_data[cnt*step: int(cnt + 1)*step]
        images = test_images[cnt*step: int(cnt + 1)*step]
        pred_label = clf.predict(data)
        pred_funct = clf.decision_function(data)
        for i, image in enumerate(images):
            funct = list(pred_funct[i])
            index = list(np.arange(1000))

            funct_plus = list(pred_funct[i])
            funct_plus.append(threshold)
            index_plus = list(np.arange(1001))

            funct_plus, index_plus = zip(*sorted(zip(funct_plus, index_plus), key = lambda x: -x[0]))
            funct, index = zip(*sorted(zip(funct, index), key = lambda x: -x[0]))

            res = ""
            if index_plus[0] == 1000:
                for j in range(5):
                    res += str(index_plus[j]) + " "
            else:
                for j in range(4):
                    res += str(index[j]) + " "
                    if j == 1:
                        res += "1000 "
            f.write(image + "," + res[:-1] + "\n")
        cnt += 1
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="normal", help="'normal' for use train data or 'add' for use train + add data")
    args = parser.parse_args()
    main(args)