import os
import random
import trimesh
import glob
import numpy as np
import open3d as o3d
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

class PointNet():
    def __init__(self):
        tf.random.set_random_seed(1)
        self.train_points = []
        self.test_points = []
        self.train_labels = []
        self.test_labels = []


    def parse_dataset(self, num_demo, num_points=2048):

        file_list = os.listdir(self.DATA_DIR)
        print("processing demontration number: ", num_demo)
        tf.random.shuffle(file_list)
        train_files = file_list[:30]
        test_files = file_list[31:41]

        for f in train_files:
            f = os.path.join(self.DATA_DIR, f)
            pcd = o3d.io.read_point_cloud(f, format='pcd')
            out_array = np.asarray(pcd.points)

            self.train_points.append(out_array)
            self.train_labels.append(num_demo)

        for f in test_files:
            f = os.path.join(self.DATA_DIR, f)
            pcd = o3d.io.read_point_cloud(f, format='pcd')
            out_array = np.asarray(pcd.points)
            
            self.test_points.append(out_array)
            self.test_labels.append(num_demo)

        # self.all_train_points.append(train_points)
        # self.all_test_points.append(test_points)
        # self.all_train_labels.append(train_labels)
        # self.all_test_labels.append(test_labels)

    def augment(self, points, label):
        # jitter points
        points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
        # shuffle points
        points = tf.random.shuffle(points)
        return points, label

    def main(self):
        for i in range(0, 5):
            self.DATA_DIR = "/home/venky/dl_tensorflow/models/demonstration_" + str(i) + "/top_clouds"
            self.parse_dataset(i)

        NUM_CLASSES = 10
        BATCH_SIZE = 32

        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_points, self.train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(self.test_points), tf.convert_to_tensor(self.test_labels)))

        train_dataset = train_dataset.shuffle(len(self.train_points)).map(augment).batch(BATCH_SIZE)
        test_dataset = test_dataset.shuffle(len(self.test_points)).batch(BATCH_SIZE)

pointnet = PointNet()
pointnet.main()