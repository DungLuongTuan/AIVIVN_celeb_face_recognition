import pandas as pd 
import subprocess
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default = "/home/tittit/data/challenges/AIVIVN/vn_celeb_face_recognition")
args = parser.parse_args()

train_path = os.path.join(args.data_path, "train")
train_csv  = os.path.join(args.data_path, "train.csv")

#	prepare train data
if not os.path.exists("data/image"):
	os.makedirs("data/image")
if not os.path.exists("data/image/train"):
	os.makedirs("data/image/train")

train_df = pd.read_csv(train_csv)
for image, label in zip(train_df.image.values, train_df.label.values):
	label_path = os.path.join("data/image/train", str(label))
	if not os.path.exists(label_path):
		os.makedirs(label_path)
	subprocess.call(["cp " +  os.path.join(train_path, image) + " " + label_path], shell = True)

#	prepare test data
test_path = os.path.join(args.data_path, "test")
test_images = os.listdir(test_path)
if not os.path.exists("data/image/test"):
	os.makedirs("data/image/test")

for image in test_images:
	subprocess.call(["cp " + os.path.join(test_path, image) + " data/image/test/" + image], shell = True)
