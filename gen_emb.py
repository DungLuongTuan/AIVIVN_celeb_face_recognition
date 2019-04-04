import sys
sys.path.append('insightface/deploy')

import numpy as np
import face_embedding
import argparse
import cv2
import sys
import os

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/tittit/challenges/AIVIVN/celeb_face_recognition/insightface/models/model-r100-ii/model,0000', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_embedding.FaceModel(args)

if not os.path.exists("data/embedding"):
	os.makedirs("data/embedding")

#	generate train emb
if not os.path.exists("data/embedding/train"):
	os.makedirs("data/embedding/train")
train_inp_path = "data/image/train"
train_out_path = "data/embedding/train"
labels = os.listdir(train_inp_path)
cnt = 0
for label in labels:
	inp_label_path = os.path.join(train_inp_path, label)
	out_label_path = os.path.join(train_out_path, label)
	if not os.path.exists(out_label_path):
		os.makedirs(out_label_path)
	images = os.listdir(inp_label_path)
	for image in images:
		cnt += 1
		print(cnt, end = "\r")
		image_path = os.path.join(inp_label_path, image)
		emb_path = os.path.join(out_label_path, image[:-4])
		img = cv2.imread(image_path)
		feat = model.get_feature(img)
		np.save(emb_path, feat)

#	generate test emb
if not os.path.exists("data/embedding/test"):
	os.makedirs("data/embedding/test")
test_inp_path = "data/image/test"
test_out_path = "data/embedding/test"
images = os.listdir(test_inp_path)
cnt = 0
for image in images:
	cnt += 1
	print(cnt, end = "\r")
	image_path = os.path.join(test_inp_path, image)
	emb_path = os.path.join(test_out_path, image[:-4])
	img = cv2.imread(image_path)
	feat = model.get_feature(img)
	np.save(emb_path, feat)