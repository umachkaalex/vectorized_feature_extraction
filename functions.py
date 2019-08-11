import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import History, ModelCheckpoint
from keras.layers import Dense, Activation, Conv2D, Flatten, \
     BatchNormalization, MaxPooling2D, Dropout
from keras.utils import to_categorical as OHE
from numpy import linalg as LA
from PIL import Image
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class PrepareData:
    def __init__(self, img_fldr):
        # internals
        self.all_files = 0
        self.X = []
        self.Y = np.array
        self.datasets = {}
        self.classes = {}
        self.img_shapes = {}
        self.train_f = []
        self.test_f = []
        self.nmb_of_classes = int
        # properties
        self.img_fldr = img_fldr


    def run(self):
        self.fill_x_y()
        x_shape = self.X.shape
        self.img_shapes['img_width'] = x_shape[2]
        self.img_shapes['img_height'] = x_shape[1]

        self.define_shapes()
        self.datasets['X_train'] = self.X
        self.datasets['Y_train'] = self.Y
        print('X train shape: ' + str(self.X.shape))
        print('Y train shape: ' + str(self.Y.shape))

        self.all_files = 0
        self.X = []

        self.fill_x_y(train=False)
        self.define_shapes()
        self.datasets['X_test'] = self.X
        self.datasets['Y_test'] = self.Y
        print('X test shape: ' + str(self.X.shape))
        print('Y test shape: ' + str(self.Y.shape))

        self.datasets['classes'] = self.classes

        return self.datasets

    def define_shapes(self):
        x_shape = self.X.shape
        if len(x_shape) == 3:
            self.X = self.X.reshape(x_shape[0], x_shape[1], x_shape[2], 1)
            self.img_shapes['channels'] = 1
        else:
            self.img_shapes['channels'] = x_shape[1]
        self.datasets['img_shapes'] = self.img_shapes

    def count_files(self, train):
        self.train_f = os.listdir(self.img_fldr + 'train/')
        self.test_f = os.listdir(self.img_fldr + 'test/')
        try:
            assert self.train_f == self.test_f, "Classes mismatch"
        except AssertionError as e:
            print(e)
        else:
            self.nmb_of_classes = len(self.train_f)

        for i in range(0, self.nmb_of_classes):
            if train:
                cur_folder = self.img_fldr + 'train/' + self.train_f[i] + '/'
                self.classes[i] = self.train_f[i]
            else:
                cur_folder = self.img_fldr + 'test/' + self.test_f[i] + '/'
            files = os.listdir(cur_folder)
            self.all_files += len(files)

        self.Y = np.zeros([self.all_files, 1])

    def fill_x_y(self, train=True):
        self.count_files(train)
        cur_file = 0
        for i in tqdm(range(0, self.nmb_of_classes)):
            if train:
                cur_folder = self.img_fldr + 'train/' + self.train_f[i] + '/'
            else:
                cur_folder = self.img_fldr + 'test/' + self.test_f[i] + '/'
            files = os.listdir(cur_folder)
            for ii in range(len(files)):
                file = files[ii]
                im = Image.open(cur_folder + str(file))
                np_im = np.array(im)
                self.X.append(np_im)
                self.Y[cur_file, 0] = i
                cur_file += 1

        self.X = np.asarray(self.X)


def extract_features(data, idx):
    # number of images
    m = data.shape[0]
    # width of the image
    w = data.shape[1]
    # height of the image
    h = data.shape[2]
    # reshape data to run vectorized calculations
    data = data.T.reshape([w, h, m])
    # divide pixel color numbers by 255
    data = data / 255
    # initialize array to store extracted features with 3 columns for first 3 ones.
    new_data = np.zeros([m, 3])
    # calculate the coefficients of variation (CV) of diagonal...
    new_data[:, 0] = np.std(np.diagonal(data), axis=1)/np.mean(np.diagonal(data), axis=1)
    # and anti-diagonal pixels of image...
    new_data[:, 1] = np.std(np.diagonal(np.rot90(data)), axis=1)/np.mean(np.diagonal(np.rot90(data)), axis=1)
    # define "shapes" of pictures
    new_data[:, 2] = LA.norm(np.mean(data, axis=0), axis=0)/LA.norm(np.mean(data, axis=1), axis=0)
    #############
    # next 4 features are CVs of all pixels of each quadrant.
    # define quadrant's shape: in this case 14x14 pixels
    quad_shape = int(w/2)
    # number of quadrants per 0 and 1 axis. Now it will be 2 per "width" and per "height"
    n_quad = w//quad_shape
    # next: total number of feature's we'l get.
    # it equals the number of 14x14 smaller shapes into which 28x28 shape could be splitted.
    n_feat = np.square(int((w//quad_shape)))
    # reshape data to size: (2, 14, 2, 14, 60000): 2x2 quadrants with 14x14 shape each
    # and 60000 depth (for train dataset).
    resh_data = data.copy().reshape(n_quad, quad_shape, n_quad, quad_shape, m)
    # calculate std for each image in each quadrant
    feat_std = np.std(np.std(resh_data, axis=3), axis=1).reshape(n_feat, m).T
    # calculate mean for each image in each quadrant
    feat_mean = np.mean(np.mean(resh_data, axis=3), axis=1).reshape(n_feat, m).T
    # get 4 new features that represent division of std by mean (CV) for each of 4 quadrants.
    feat = feat_std/feat_mean
    # stack new features to the previous 3.
    new_data = np.hstack([new_data, feat])
    #############
    # get diagonal pixels from each of 4 quadrants
    get_diag = np.diagonal(resh_data, axis1=1, axis2=3).reshape([n_feat, m, quad_shape])
    # get anti-diagonal pixels from each of 4 quadrants
    get_diag_rot = np.diagonal(resh_data, axis1=1, axis2=3).reshape([n_feat, m, quad_shape])
    # calculate CV for diagonals
    feat = np.std(get_diag, axis=2).T/np.mean(get_diag, axis=2).T
    # calculate CV for anti-diagonals
    feat_rot = np.std(get_diag_rot, axis=2).T/np.mean(get_diag_rot, axis=2).T
    # calculate CV for sum of diagonals and anti-diagonals values
    sum_feat = (np.std(get_diag, axis=2).T+np.std(get_diag_rot, axis=2).T)/\
               (np.mean(get_diag, axis=2).T+np.mean(get_diag_rot, axis=2).T)
    # stack new features to the previous.
    new_data = np.hstack([new_data, feat, feat_rot, sum_feat])
    ########
    # calculate CV for sum of rows and columns pixels
    feat = (np.std(data, axis=0)+np.std(data, axis=1))/(np.mean(data, axis=0)+np.mean(data, axis=1))
    # stack new features to the previous.
    new_data = np.hstack([new_data, feat.T])
    ########
    # next features without comments, because it's hard to describe in words their meaning
    # you can sort out by yourself how they are calculated

    def calc_feat(feat, axis):
        return (np.mean(feat, axis=axis)).reshape(-1, 1)
    # next 90 features
    for i in range(1, int(w / 2) + 2):
        feat = np.std(data[:i, :, :], axis=0)
        new_data = np.hstack([new_data, calc_feat(feat, 0)])

        feat = np.std(data[:, :i, :], axis=0)
        new_data = np.hstack([new_data, calc_feat(feat, 0)])

        feat = np.std(data[-i:, :, :], axis=0)
        new_data = np.hstack([new_data, calc_feat(feat, 0)])

        feat = np.std(data[:, -i:, :], axis=0)
        new_data = np.hstack([new_data, calc_feat(feat, 0)])

        feat = (np.std(np.diagonal(data, offset=-i), axis=1) / np.mean(np.diagonal(data, offset=-i), axis=1)).reshape(-1, 1)
        new_data = np.hstack([new_data, feat])

        feat = (np.std(np.diagonal(data, offset=i), axis=1) / np.mean(np.diagonal(data, offset=i), axis=1)).reshape(-1, 1)
        new_data = np.hstack([new_data, feat])

    #########
    # next 57 features

    for r in range(w - 1):
        cur_2 = np.zeros([m, w - 1])
        cur_3 = np.zeros([m, w - 2])

        for c in range(h - 1):
            cur_2[:, c] = np.std(LA.norm(data[r:r + 2, c:c + 2, :], axis=0), axis=0)

            if r < w - 2 and c < h - 2:
                cur_3[:, c] = np.std(LA.norm(data[r:r + 3, c:c + 3, :], axis=0), axis=0)

        new_data = np.hstack([new_data, calc_feat(cur_2, 1)])
        new_data = np.hstack([new_data, calc_feat(cur_3, 1)])

    ######
    # next 13 features
    rows = h - 1
    cols = w - 1
    for r in range(int(rows / 2)):
        cur_1 = np.zeros([m, int(cols / 2)])
        for c in range(int(cols / 2)):
            cur_1[:, c] = LA.norm(data[(r, r, rows - r, rows - r), (c, cols - c, c, cols - c), :], axis=0)
        new_data = np.hstack([new_data, np.std(cur_1, axis=1).reshape(-1, 1)])

    # check how many NaN values were created
    print('Total Non NaN values: ' + str(np.count_nonzero(~np.isnan(new_data))))
    print('Total NaN values: ' + str(np.count_nonzero(np.isnan(new_data))))
    # convert NaNs to zeros
    new_data = np.nan_to_num(new_data)
    if len(idx) == 0:
        _, idx = np.unique(new_data, axis=1, return_index=True)
    new_data = new_data[:, idx]
    # new_data = pd.DataFrame(data=new_data)
    # new_data = new_data.T.drop_duplicates().T.values
    print(new_data.shape)

    return new_data, idx


def MiniVGGNet(dataset, epochs, verb):
    # define number of classes
    n_cls = dataset['y_train'].shape[1]
    # clear session
    K.clear_session()
    # collect history
    history = History()
    # save weights with best val accuracy
    checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_acc', verbose=verb, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, history]
    # start building model
    model = Sequential()
    # first convolutional layer
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(dataset['train_shape'][1:])))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=-1))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25, seed=0))

    # second convolutional layer
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25, seed=0))

    # fully connected layer
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5, seed=0))

    # soft classifier
    model.add(Dense(n_cls))
    model.add(Activation('softmax'))

    # compile model
    opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # train model
    model.fit(dataset['X_train'], dataset['y_train'],
              validation_data=[dataset['X_test'], dataset['y_test']],
              epochs=epochs, batch_size=64, verbose=verb, callbacks=callbacks_list)

    return model, history


def train_predict(dataset, savepath, name, verb=2, epochs=100):
    trained, cur_history = MiniVGGNet(dataset, epochs, verb)
    trained.load_weights('weights.hdf5')
    y_pred = trained.predict(dataset['X_test'])
    cur_pred_df = pd.DataFrame(data=y_pred)
    cur_pred_df.to_csv(savepath + 'cur_pred_' + str(name) + '.csv', index=False)
    y_pred = np.argmax(y_pred, axis=1)
    cur_acc = accuracy_score(np.argmax(dataset['y_test'], axis=1), y_pred)
    print(str(name) + ' accurancy: ' + str(cur_acc))
    cur_history = cur_history.history['val_acc']

    return cur_history, name


def extr_feat(dataset):
    X_train = dataset['X_train']
    X_test = dataset['X_test']
    mtr = X_train.shape[0]
    mts = X_test.shape[0]
    w = X_train.shape[1]
    h = X_train.shape[2]
    d = X_train.shape[3]

    # check if data gray or RGB
    if d > 1:
        # if data not gray - convert to gray
        data = np.zeros([mtr, w, h, 1])
        for i in tqdm(range(mtr)):
            img = X_train[i, :, :, :]
            data[i, :, :, 0] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        data = X_train

    # extract features from MNIST test dataset
    feat_train_data, idx = extract_features(data, [])

    # the same procedure for test dataset
    if d > 1:
        data = np.zeros([mts, w, h, 1])
        for i in tqdm(range(X_test.shape[0])):
            img = X_test[i, :, :, :]
            data[i, :, :, 0] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        data = X_test

    feat_test_data, idx = extract_features(data, idx)

    return feat_train_data, feat_test_data


def reshape_as_orig(feat_train, feat_test, dataset):
    train_shape = dataset['train_shape'].copy()
    test_shape = dataset['test_shape'].copy()
    w = train_shape[1]
    h = train_shape[2]
    train_shape[-1] = 1
    test_shape[-1] = 1
    # get shape of created features array
    feat_shape = feat_train.shape
    # how many times the feature array as pixels
    # can be totally "placed into" image shape
    dupl_num = w * h // feat_shape[1]
    # define the amount of zero columns on the left side
    start_zeros = (w * h % feat_shape[1]) // 2
    # define the amount of zero columns on the right side
    end_zeros = w * h % feat_shape[1] - start_zeros
    # open new feature array with the same shape as original img array
    X_train_feat = np.zeros([feat_train.shape[0], start_zeros])
    # concatenate to starting zero features - extracted ones
    for i in range(dupl_num):
        X_train_feat = np.hstack([X_train_feat, feat_train])
    # create ending zero columns array
    end_zeros_arr = np.zeros([feat_train.shape[0], end_zeros])
    # close new feature array with ending zero columns
    X_train_feat = np.hstack([X_train_feat, end_zeros_arr])
    # reshape new feature array to original train image dataset
    X_train_feat = X_train_feat.reshape(train_shape)

    # the same procedure for test dataset
    X_test_feat = np.zeros([feat_test.shape[0], start_zeros])
    for i in range(dupl_num):
        X_test_feat = np.hstack([X_test_feat, feat_test])
    end_zeros_arr = np.zeros([feat_test.shape[0], end_zeros])
    X_test_feat = np.hstack([X_test_feat, end_zeros_arr])
    X_test_feat = X_test_feat.reshape(test_shape)

    return X_train_feat, X_test_feat


def extract_edges(X_train, X_test, lower=50, upper=150):
    mtr = X_train.shape[0]
    mts = X_test.shape[0]
    w = X_train.shape[1]
    h = X_train.shape[2]
    d = X_train.shape[3]

    # create "container" to store edges
    edge_train_data = np.zeros([len(X_train), w * h])
    # start looping over dataset images
    for i in tqdm(range(edge_train_data.shape[0])):
        # check if image more than 1 layer in depth
        if d > 1:
            img = cv2.cvtColor(X_train[i, :, :, :], cv2.COLOR_BGR2GRAY)  # convert to gray
            img = img.astype(np.uint8).reshape(w, h)
        else:
            img = X_train[i, :, :, :].astype(np.uint8).reshape(w, h)
        # define edges in the images
        edge_train_data[i, :] = cv2.Canny(img, lower, upper).flatten()
    # the same procedure for the test data
    edge_test_data = np.zeros([len(X_test), w * h])
    for i in tqdm(range(edge_test_data.shape[0])):
        if d > 1:
            img = cv2.cvtColor(X_test[i, :, :, :], cv2.COLOR_BGR2GRAY)
            img = img.astype(np.uint8).reshape(w, h)
        else:
            img = X_test[i, :, :, :].astype(np.uint8).reshape(w, h)
        edge_test_data[i, :] = cv2.Canny(img, lower, upper).flatten()

    # convert extracted edge data to original image dataset shapes
    edge_train_data = edge_train_data.reshape(mtr, w, h, 1)
    edge_test_data = edge_test_data.reshape(mts, w, h, 1)

    return edge_train_data, edge_test_data


def reshape_data(X_train, y_train, X_test, y_test, savepath):
    w = X_train.shape[1]  # image width
    h = X_train.shape[2]  # image height

    if len(X_train.shape) == 4:
        d = X_train.shape[3]  # depth for color images
    if len(X_train.shape) == 3:
        d = 1  # depth for gray images

    # length of train data
    mtr = X_train.shape[0]
    # length of test data
    mts = X_test.shape[0]

    train_shape = [mtr, w, h, d]
    test_shape = [mts, w, h, d]
    # reshape data to use in VGG model and feature extraction
    X_train = X_train.reshape(train_shape)
    X_test = X_test.reshape(test_shape)
    # convert train labels to one-hot vectors
    y_train = OHE(y_train)
    # save test non one-hot labels to csv to use later in accuracy comparing
    y_test_df = pd.DataFrame(data=y_test, columns=['target'])
    y_test_df.to_csv(str(savepath) + 'mnist_test.csv', index=False)
    # convert train labels to one-hot vectors
    y_test = OHE(y_test)

    dataset = dict({'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'train_shape': train_shape,
                    'test_shape': test_shape})

    return dataset
