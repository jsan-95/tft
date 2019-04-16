import os
import cv2
import numpy as np

DATADIR = "data/images_filtered/"
VALIDATIONDIR = "data/test/test_folder/"
CATEGORIES = []
CATEGORIES_VALIDATION = []
IMG_SIZE = 50
training_data = []
validation_data = []

if __name__ == '__main__':
    def createTrainingData():
        i = 0
        for fn in os.listdir(DATADIR):
            if os.path.isdir(DATADIR + fn):
                CATEGORIES.append(fn)
                i += 1
            if i == 2:
                break
        print(CATEGORIES)
        for category in CATEGORIES:
            class_num = CATEGORIES.index(category)
            path = os.path.join(DATADIR, category)

            # for class_path in os.listdir(path):
            #     if class_path.endswith('.jpeg'):
            #         img = os.path.join(path, class_path)
            #         img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            #         normalizedImg = np.zeros((IMG_SIZE, IMG_SIZE))
            #         normalizedImg = cv2.normalize(img_array, normalizedImg, 0, 255, cv2.NORM_MINMAX)
            #         new_array = cv2.resize(normalizedImg, (IMG_SIZE, IMG_SIZE))
            #         training_data.append([new_array, class_num])  # add this to our training_data

            for class_path in os.listdir(path):
                 path_instead = os.path.join(path, class_path)
                 print(path_instead)
                 for img in os.listdir(path_instead):

                     img_array = cv2.imread(os.path.join(path_instead, img), cv2.IMREAD_GRAYSCALE)
                     new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                     training_data.append([new_array, class_num])  # add this to our training_data


    def createValidationData():
        for category in CATEGORIES:
            CATEGORIES_VALIDATION.append(category)

        for category in CATEGORIES_VALIDATION:
            class_num = CATEGORIES_VALIDATION.index(category)
            path_image = os.path.join(VALIDATIONDIR, category + '.jpg')
            print(path_image)
            img_array = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
            normalizedImg = np.zeros((IMG_SIZE, IMG_SIZE))
            normalizedImg = cv2.normalize(img_array, normalizedImg, 0, 255, cv2.NORM_MINMAX)
            new_array = cv2.resize(normalizedImg, (IMG_SIZE, IMG_SIZE))
            validation_data.append([new_array, class_num])  # add this to our training_data

    createTrainingData()
    createValidationData()

    x_train = []
    y_train = []
    x_validation = []
    y_validation = []

    for features, label in training_data:
        x_train.append(features)
        y_train.append(label)

    for features, label in validation_data:
        x_validation.append(features)
        y_validation.append(label)

    x_train = np.array(x_train)
    x_validation = np.array(x_validation)


    def resize_data(data):
        data_upscaled = np.zeros((data.shape[0], IMG_SIZE, IMG_SIZE, 1))
        return data_upscaled


    x_train_resized = resize_data(x_train)
    x_test_resized = resize_data(x_validation)

    from keras.applications import vgg16

    def create_vgg16():
        model = vgg16.VGG16(include_top=True, weights=None, input_tensor=None, input_shape=(IMG_SIZE, IMG_SIZE, 1), pooling='avg',
                            classes=1)
        return model


    vgg16_model = create_vgg16()
    vgg16_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc', 'mse'])
    vgg16_model.summary()

    y_train = np.array(y_train)
    y_validation = np.array(y_validation)

    print(x_train_resized.shape)
    print(y_train.shape)
    print(x_test_resized.shape)
    print(y_validation.shape)

    vgg16 = vgg16_model.fit(x=x_train_resized,
                            y=y_train, batch_size=32,
                            epochs=10,
                            verbose=1,
                            validation_data=(x_test_resized, y_validation),
                            shuffle=True)
