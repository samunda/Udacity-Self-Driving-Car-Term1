import csv
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Convolution2D, Dropout, Flatten, Dense


# get the list of data samples in the csv file
def get_list_of_data(driving_log_file, use_flipped_image, use_left_right_images, correction):
    samples = []

    with open(driving_log_file) as csvfile:
        reader = csv.reader(csvfile)

        for line in reader:

            # discard csv header line
            if line[0] == 'center':
                continue

            center_image_name = line[0].split('/')[-1]
            center_angle = float(line[3])

            # format: [file name, flip left-right?, angle]
            samples.append([center_image_name, False, center_angle])

            if use_flipped_image:
                samples.append([center_image_name, True, -center_angle])

            if use_left_right_images:
                left_image_name = line[1].split('/')[-1]
                right_image_name = line[2].split('/')[-1]

                left_angle = center_angle + correction
                right_angle = center_angle - correction

                samples.append([left_image_name, False, left_angle])
                samples.append([right_image_name, False, right_angle])

                if use_flipped_image:
                    samples.append([left_image_name, True, -left_angle])
                    samples.append([right_image_name, True, -right_angle])

    return samples


# train and save the resulting model
def train(driving_log_file, use_flipped_image, use_left_right_images, correction, batch_size, num_epochs, drop_prob):
    samples = get_list_of_data(driving_log_file, use_flipped_image, use_left_right_images, correction)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    postfix = 'batch_size={}_epochs={}_correction={}_drop={}'.format(batch_size, num_epochs, correction, drop_prob)

    print(postfix)

    def generator(samples, batch_size):
        num_samples = len(samples)

        # loop forever so the generator never terminates
        while 1:

            sklearn.utils.shuffle(samples)

            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []

                for batch_sample in batch_samples:
                    image_name = image_data_dir + batch_sample[0]
                    image = cv2.imread(image_name)

                    flip = batch_sample[1]
                    angle = batch_sample[2]

                    if flip:
                        images.append(np.fliplr(image))
                    else:
                        images.append(image)

                    angles.append(angle)

                X_train = np.array(images)
                y_train = np.array(angles)

                yield sklearn.utils.shuffle(X_train, y_train)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    print(keras.__version__)

    model = Sequential()

    # pre-process data
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

    # crop top 70 pixels and bottom 25 pixels
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))

    model.add(Convolution2D(64, 3, 3, activation='relu'))

    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(drop_prob))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dropout(drop_prob))

    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    samples_per_epoch = len(train_samples)
    nb_val_samples = len(validation_samples)

    history_object = model.fit_generator(train_generator,
                                         samples_per_epoch=samples_per_epoch,
                                         validation_data=validation_generator,
                                         nb_val_samples=nb_val_samples,
                                         nb_epoch=num_epochs,
                                         verbose=1)

    # save model
    model.save('model_{}.h5'.format(postfix))

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')

    # save figure
    plt.savefig('loss_{}.png'.format(postfix))
    plt.close('all')


if __name__ == '__main__':
    driving_log_file = 'data/driving_log.csv'
    image_data_dir = 'data/IMG/'

    use_flipped_image = True

    list_of_drop_probs = [0.5]
    list_of_epochs = [4, 8]
    list_of_corrections = [0.18, 0.20]

    batch_size = 64

    # train models for each combination of parameters
    for drop_prob in list_of_drop_probs:

        for num_epochs in list_of_epochs:

            for correction in list_of_corrections:

                if correction == 0.0:
                    use_left_right_images = False
                else:
                    use_left_right_images = True

                train(driving_log_file, use_flipped_image, use_left_right_images, correction, batch_size, num_epochs,
                      drop_prob)
