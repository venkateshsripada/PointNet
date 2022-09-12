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

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

class PointNet():
    def __init__(self):
        tf.random.set_random_seed(1)
        self.train_points = []
        self.test_points = []
        self.train_labels = []
        self.test_labels = []


    def parse_dataset(self, num_demo, num_points=196000):

        file_list = os.listdir(self.DATA_DIR)
        print("processing demontration number: ", num_demo)
        tf.random.shuffle(file_list)
        train_files = file_list[:30]
        test_files = file_list[31:41]

        for f in train_files:
            f = os.path.join(self.DATA_DIR, f)
            pcd = o3d.io.read_point_cloud(f, format='pcd')            
            out_array = np.asarray(pcd.points)
            tensor_train = tf.convert_to_tensor(out_array[:num_points])

            self.train_points.append(tensor_train)
            if num_demo <= num_points:
                self.train_labels.append(num_demo)

        for f in test_files:
            f = os.path.join(self.DATA_DIR, f)
            pcd = o3d.io.read_point_cloud(f, format='pcd')
            out_array = np.asarray(pcd.points)
            tensor_test = tf.convert_to_tensor(out_array[:num_points])
            
            self.test_points.append(tensor_test)
            if num_demo <= num_points:
                self.test_labels.append(num_demo)

        # print("Train points: ",tf.shape(self.train_points))
        # print("Train labels: ", tf.shape(self.train_labels))
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

    # Build model
    def conv_bn(self, x, filters):
        x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)

    def dense_bn(self, x, filters):
        x = layers.Dense(filters)(x)
        x = layers.BatchNormalization(momentum=0.0)(x)
        return layers.Activation("relu")(x)
        
    # T-Net
    def tnet(self, inputs, num_features):

        # Initalise bias as the indentity matrix
        bias = keras.initializers.Constant(np.eye(num_features).flatten())
        reg = OrthogonalRegularizer(num_features)

        x = self.conv_bn(inputs, 32)
        x = self.conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = self.dense_bn(x, 128)
        x = layers.Dense(
            num_features * num_features,
            kernel_initializer="zeros",
            bias_initializer=bias,
            activity_regularizer=reg,
        )(x)
        feat_T = layers.Reshape((num_features, num_features))(x)
        # Apply affine transformation to input features
        return layers.Dot(axes=(2, 1))([inputs, feat_T])

    def main(self):
        for i in range(0, 5):
            self.DATA_DIR = "/home/venky/dl_tensorflow/models/demonstration_" + str(i) + "/top_clouds"
            self.parse_dataset(i)

        NUM_POINTS = 196000
        NUM_CLASSES = 20
        BATCH_SIZE = 32

        train_dataset = tf.data.Dataset.from_tensor_slices((self.train_points, self.train_labels))
        test_dataset = tf.data.Dataset.from_tensor_slices((self.test_points, self.test_labels))

        train_dataset = train_dataset.shuffle(len(self.train_points)).map(self.augment).batch(BATCH_SIZE)
        test_dataset = test_dataset.shuffle(len(self.test_points)).batch(BATCH_SIZE)


        inputs = keras.Input(shape=(NUM_POINTS, 3))

        x = self.tnet(inputs, 3)
        x = self.conv_bn(x, 32)
        x = self.conv_bn(x, 32)
        x = self.tnet(x, 32)
        x = self.conv_bn(x, 32)
        x = self.conv_bn(x, 64)
        x = self.conv_bn(x, 512)
        x = layers.GlobalMaxPooling1D()(x)
        x = self.dense_bn(x, 256)
        x = layers.Dropout(0.3)(x)
        x = self.dense_bn(x, 128)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
        # Get weights of last layer
        weights = model.layers[-1].get_weights()[0]
        print(model.layers[-1].name, weights)
        model.summary()

        return weights


pointnet = PointNet()
pointnet.main()