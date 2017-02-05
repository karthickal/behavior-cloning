# import all required libraries
import csv
import os
from math import degrees
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from scipy import misc
from sklearn.model_selection import train_test_split

# data file constants
DATA_DIRECTORY = './data/'
DATA_CSV = 'driving_log.csv'

# random seed
RANDOM_SEED = 99

# hyperparameters
VALIDATION_FRACTION = 0.33
BATCH = 128
EPOCHS = 15
DATA_MULTIPLIER = 2

# image pre-processing constants
IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_CHANNELS = 3
IMAGE_TOP_ROI = 50
IMAGE_BOTTOM_ROI = 140

# data generation constants
SIDE_CAM_OFFSET = 0.4  # left camera and right camera offset
SHIFT_X_OFFSET = 0.003
ANGLE_THRESHOLD = 0.01
ANGLE_BOUNDS = 0.05


def load_raw_data(csv_file, directory):
    '''
    Function to load the raw csv data consisting of the camera image paths and steering angles
    :param csv_file: The CSV filename that contains the data
    :param directory: The directory name that contains the data
    :return: Returns a tuple consisting of the camera images path and the steering angle
    '''

    # get the CSV path and initalize varaibles
    csv_path = os.path.join(directory, csv_file)
    X = []
    y = []

    # open the CSV file
    with open(csv_path, 'r') as csv_input:
        reader = csv.DictReader(csv_input)
        for row in reader:
            # image is stored under the 'center', 'left', 'right' header
            record = row

            # steering data is stored under the 'steering' header
            steering = float(row['steering'])

            # add data to the list
            X.append(record)
            y.append(steering)

    # return the loaded data
    return X, y


def load_image(path):
    '''
    Function to load the image as an array from the given path
    :param path:
    :return: Returns the image in the path as a numpy array of float32
    '''
    return misc.imread(path).astype(np.float32)


def preprocess_image(x):
    '''
    Function to pre-process the image before training or prediction.
    This function crops the region of interest from the image and resies it. It also
    normalizes the image so the calculations are smaller

    :param x: A numpy array of the image
    :return: A numpy array comprising of the pre-processed image
    '''
    # remove the top and bottom from the image which is not relevant and resize the image
    x_resized = x[IMAGE_TOP_ROI:IMAGE_BOTTOM_ROI:, :, ]
    x_resized = misc.imresize(x_resized, size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    # normalize the image so the calculations are smaller
    a = -0.5
    b = 0.5
    min = 0
    max = 255
    x_normalized = a + (((x_resized - min) * (b - a)) / (max - min))

    return x_normalized


def get_left_cam_image(x, y):
    '''
    Function to load the left camera image as a numpy array and add relevant
    offset to the camera data
    :param x: The row from CSV containing the paths to the images
    :param y: The steering angle for the images
    :return: A tuple of the image as a numpy array and the offset angle
    '''
    _x = load_image(os.path.join(DATA_DIRECTORY, x['left'].strip()))

    # add the side camera offset value to the left camera data
    _y = y + SIDE_CAM_OFFSET

    return _x, _y


def get_right_cam_image(x, y):
    '''
    Function to load the right camera image as a numpy array and add relevant
    offset to the camera data
    :param x: The row from CSV containing the paths to the images
    :param y: The steering angle for the images
    :return: A tuple of the image as a numpy array and the offset angle
    '''
    _x = load_image(os.path.join(DATA_DIRECTORY, x['right'].strip()))

    # remove the side camera offset value to the left camera data
    _y = y - SIDE_CAM_OFFSET

    return _x, _y


def get_center_cam_image(x, y):
    '''
    Function to load the center camera image as a numpy array and add relevant
    offset to the camera data
    :param x: The row from CSV containing the paths to the images
    :param y: The steering angle for the images
    :return: A tuple of the image as a numpy array and the offset angle
    '''
    _x = load_image(os.path.join(DATA_DIRECTORY, x['center'].strip()))

    # no offset required for the angle data
    _y = y

    return _x, _y


def flip_image(x_data, y_data):
    '''
    Function to flip the image along the vertical axis and change signs of the steering angle
    :param x_data: The numpy array consisting of the image data
    :param y_data: The original angle of the image
    :return: A tuple of the flipped image as a numpy array and the new steering angle
    '''
    x_data = np.fliplr(x_data)

    # if angle is 0.00 do no switch the signs
    if "%.2f" % y_data != "0.00":
        y_data = y_data * -1

    return x_data, y_data


def shift_left(x_data, y_data, shift_off=SHIFT_X_OFFSET):
    '''
    Function to shift the data to the left and relevant offset to the original steering angle
    :param x_data: A numpy array of the image to be shifted
    :param y_data: The original angle
    :param shift_off: The offset used to modify the steering angle
    :return: A tuple of the shifted image and the modified steering angle
    '''

    # choose a random integer between 0 and 25 to shift the image using opencv
    rows, cols, ch = x_data.shape
    trans_wid = np.random.randint(0, 25)
    trans_matrix = np.float32([[1, 0, -trans_wid], [0, 1, 0]])
    translated_image = cv2.warpAffine(x_data, trans_matrix, (cols, rows))

    # reduce the steering angle by each pixel of shift and return the data
    return translated_image, y_data - (trans_wid * shift_off)


def shift_right(x_data, y_data, shift_off=SHIFT_X_OFFSET):
    '''
    Function to shift the data to the right and relevant offset to the original steering angle
    :param x_data: A numpy array of the image to be shifted
    :param y_data: The original angle
    :param shift_off: The offset used to modify the steering angle
    :return: A tuple of the shifted image and the modified steering angle
    '''

    # choose a random integer between 0 and 25 to shift the image using opencv
    rows, cols, ch = x_data.shape
    trans_wid = np.random.randint(0, 25)
    trans_matrix = np.float32([[1, 0, trans_wid], [0, 1, 0]])
    translated_image = cv2.warpAffine(x_data, trans_matrix, (cols, rows))

    # increase the steering angle by each pixel of shift and return the data
    return translated_image, y_data + (trans_wid * shift_off)


def get_angle_tr_list(original_angle):
    '''
    Function to generate a list of angles that will be considered as the same data
    Ex: if original angle is 0.05 then the list would be [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    This is used to ensure a uniform data generation strategy
    :param original_angle:
    :return: The similar angles list
    '''
    gen_angles_list = np.arange(original_angle - ANGLE_BOUNDS, original_angle + ANGLE_BOUNDS, ANGLE_THRESHOLD)
    gen_angles_round_list = ['%.2f' % elem for elem in gen_angles_list]

    return gen_angles_round_list


def train_data_generator(X, y, batch=BATCH, fraction=1):
    '''

    :param X: A list containing the camera image paths
    :param y: A list of corresponding steering angles
    :param batch: The batch size to be used for training
    :return: A tuple of image lists and steering angles of length = batch
    '''
    x_batch = []
    y_batch = []

    # maintain the list of all generated angles
    generated_angles = {}
    while True:

        # get a random data from the list
        data = np.random.randint(0, len(X))
        _x, _y = X[data], y[data]

        # select a camera image based on a random choice
        cam_choice = np.random.randint(0, 3)
        if cam_choice == 0:
            x_data, y_data = get_left_cam_image(_x, _y)
        elif cam_choice == 1:
            x_data, y_data = get_center_cam_image(_x, _y)
        else:
            x_data, y_data = get_right_cam_image(_x, _y)

        # translate the image based on a random choice
        translate_choice = np.random.randint(0, 2)
        if translate_choice != 0:
            # if translation is required randomly select between left shift and right shift
            shift_choice = np.random.randint(0, 2)
            if shift_choice == 0:
                x_data, y_data = shift_left(x_data, y_data)
            else:
                x_data, y_data = shift_right(x_data, y_data)

        # flip the image based on a random choice
        flip_choice = np.random.randint(0, 2)
        if flip_choice != 0 and y_data != 0:
            x_data, y_data = flip_image(x_data, y_data)

        # check if angle generated has already met its threshold value
        rounded_angle = round(y_data, 2)
        angle_key = '%.2f' % rounded_angle
        # get the similar angles list
        tr_list = get_angle_tr_list(rounded_angle)
        total_angle_count = 0
        # sum the count of all angles generated in the similar list
        for check_angle in tr_list:
            if check_angle in generated_angles:
                total_angle_count = total_angle_count + generated_angles[check_angle]
            else:
                generated_angles[check_angle] = 0
        # if sum is greater than threshold value discard the angle and continue,
        if total_angle_count <= batch * fraction:
            if angle_key == "-0.00":
                angle_key = "0.00"
            generated_angles[angle_key] = generated_angles[angle_key] + 1
        else:
            continue

        # preprocess the generated image
        x_data = preprocess_image(x_data)

        # add the image and steering angle to the batch
        x_batch.append(x_data)
        y_batch.append(y_data)

        # yield the batch if the limit is reached
        if len(x_batch) >= batch:
            generated_angles = {}
            yield np.asarray(x_batch), np.asarray(y_batch)


def validation_data_generator(X, y, batch=BATCH):
    '''
    Function to generate the validation data
    :param X: A list containing the camera image paths
    :param y: A list of corresponding steering angles
    :param batch: The batch size to be used for training
    :return: A tuple of image lists and steering angles of length = batch
    '''
    x_batch = []
    y_batch = []

    while True:

        # randomly select a data from the list
        data = np.random.randint(0, len(X))
        _x, _y = X[data], y[data]

        # get the center camera image and preprocess it
        x_data, y_data = get_center_cam_image(_x, _y)
        x_data = preprocess_image(x_data)

        # add the data to the batch and yield if limit is reached
        x_batch.append(x_data)
        y_batch.append(y_data)

        if len(x_batch) >= batch:
            yield np.asarray(x_batch), np.asarray(y_batch)


def summarize_data_frequency(angle_data):
    '''
    Utility function to visualize the generated angle data
    :param angle_data: A list of generated angle data
    :return: The minimum and maximum degree found
    '''

    # plot a histogram of the data
    plt.figure(figsize=(9, 6))
    plt.ylabel("frequency", fontsize=14)
    plt.xlabel("steering angle (in radians)", fontsize=14)
    plt.hist(angle_data, 180)
    plt.show()

    # get the most frequent, max angle and min angle value from the data
    values, counts = np.unique(angle_data, return_counts=True)
    most_frequent = values[np.argmax(counts)]
    most_frequent_count = counts[np.argmax(counts)]
    min_radians = np.amin(angle_data)
    max_radians = np.amax(angle_data)
    min_degrees = degrees(min_radians)
    max_degrees = degrees(max_radians)

    print("---- summary of data ----")
    print("{} occurred for {} times".format(most_frequent, most_frequent_count))
    print("minimum radians was {} while maximum was {}".format(min_radians, max_radians))
    print("minimum degree was {} while maximum was {}".format(min_degrees, max_degrees))

    return min_degrees, max_degrees


# setup the model and return it
def get_model():

    # load the model and weights from the path if it exists and return it
    if os.path.exists(os.path.join('.', 'model.json')):
        with open(os.path.join('.', 'model.json'), 'r') as model_file:
            json_model = model_file.read()
            model_net = model_from_json(json_model)
            model_net.load_weights('model.h5', by_name=True)

            print("Loaded model from file")
            return model_net

    # create a new Keras Sequential Model
    model_net = Sequential()

    # Add the input block of convolution layer, A 1x1 convolutional layer with a filter size of 3
    model_net.add(Convolution2D(3, 1, 1, border_mode='same', activation='relu', name='input_conv1',
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))

    # Add the first block of convolution layer, A 3x3 convolution of 64 filter size with a max pooling layer of 2x2
    # The convolution layer is activated by the relu function
    # Output after pooling would be 16x16x64 when image size is 32x32x3
    model_net.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1'))
    model_net.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Add the second block of convolution layer, A 3x3 convolution of 32 filter size with a max pooling layer of 2x2
    # The convolution layer is activated by the relu function
    # Output after pooling would be 8x8x32 when image size is 32x32x3
    model_net.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block2_conv1'))
    model_net.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Add the third block of convolution layer, A 3x3 convolution of 16 filter size with a max pooling layer of 2x2
    # The convolution layer is activated by the relu function
    # Output after pooling would be 4x4x16 when image size is 32x32x3
    model_net.add(Convolution2D(16, 3, 3, activation='relu', border_mode='same', name='block3_conv1'))
    model_net.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # flatten the image to (None, 256)}
    model_net.add(Flatten(name='flat'))

    # Add 4 decreasing Fully Connected layers each activated by the Exponential Linear Unit function
    # The ELU output is dropped out to 50% to avoid overfitting
    model_net.add(Dense(1024, name='fc1'))
    model_net.add(ELU())
    model_net.add(Dropout(0.5, name='fc1_dropout'))
    model_net.add(Dense(128, name='fc2'))
    model_net.add(ELU())
    model_net.add(Dropout(0.5, name='fc2_dropout'))
    model_net.add(Dense(64, name='fc3'))
    model_net.add(ELU())
    model_net.add(Dropout(0.5, name='fc3_dropout'))
    model_net.add(Dense(32, name='fc4'))
    model_net.add(ELU())
    model_net.add(Dropout(0.5, name='fc4_dropout'))

    # Add the final output layer that gives the predicted steering angle in radians
    model_net.add(Dense(1, name='output', init='zero'))

    print("Created a new model")

    return model_net


def predict_angles(x, y, count=5):
    '''
    Utility function to randomly predict some angles
    :param x: A list of records containing the image paths
    :param y: A list of steering angles
    :param count: The count of data to predict
    :return: None
    '''
    out = 0
    model_net = get_model()
    while out <= count:
        ind = np.random.randint(0, len(x))
        _x, _y = x[ind], y[ind]
        out = out + 1

        # get the center camera image and use the model to predict the steering angle
        x_data, y_data = get_center_cam_image(_x, _y)
        angle = model_net.predict(np.expand_dims(preprocess_image(x_data), axis=0), batch_size=1)

        print("Predicted angle is {}, while actual is {}".format(angle, _y))


def train_model(model, x_train, y_train, x_val, y_val):

    # print a summary of the model
    model.summary()

    # compile the model with an Adam optimizer and a mean squared error loss function to predict steering angle in radians
    model.compile(optimizer=Adam(1e-5), loss='mse', metrics=['mean_squared_error'])

    # train the model for the number of epochs defined
    epoch = 0
    while True:

        print("Running Epoch {} out of {}".format(epoch + 1, EPOCHS))
        print("Predicting some angles before training")
        predict_angles(x_train, y_train)

        # for each of the epoch train the model using the generators
        # also reduce the fraction of similar angles available after every epoch
        history = model.fit_generator(
            train_data_generator(x_train, y_train, batch=BATCH, fraction=(1 - (0.1* ((epoch+1)%10)))),
            validation_data=validation_data_generator(x_val, y_val),
            nb_epoch=1,
            nb_val_samples=BATCH,
            verbose=1,
            samples_per_epoch=(DATA_MULTIPLIER * len(x_train))
        )

        print("Predicting some angles after training")

        # save the model for later use
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
            print("Saved model to disk")

        # serialize weights to HDF5
            model.save_weights("model.h5")
        print("Saved weights to disk")

        # predict some angles after training
        predict_angles(x_train, y_train)

        # increment the epoch and break if limit is reached
        epoch = epoch + 1
        if epoch >= EPOCHS:
            break


if __name__ == '__main__':

    # set the random seed
    np.random.seed(RANDOM_SEED)

    # load the image paths and steering angles and split into training and validation data
    X, y = load_raw_data(DATA_CSV, DATA_DIRECTORY)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_FRACTION, random_state=RANDOM_SEED)
    del X, y

    # load the model and print the summary
    model = get_model()
    model.summary()

    # train the model
    train_model(model, X_train, y_train, X_val, y_val)
