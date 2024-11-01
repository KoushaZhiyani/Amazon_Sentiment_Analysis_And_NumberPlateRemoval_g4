import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np


def standardize_input(image):
    standard_im = cv2.resize(image, (1100, 600))

    return standard_im


def estimate_label(rgb_image, threshold):
    average = avg_brightness(rgb_image)

    predicted_label = 0

    if average > threshold:
        predicted_label = 1

    return predicted_label


def avg_brightness(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    sum_brightness = np.sum(hsv[:, :, 2])
    area = 600 * 1100.0

    avg = sum_brightness / area

    return avg


def classify_image_with_estimator(image_path, threshold=99):
    image = cv2.imread(image_path)

    standardized_image = standardize_input(image)

    predicted_label = estimate_label(standardized_image, threshold)

    return predicted_label


if __name__ == '__main__':
    image_dir_training = "../../images/training/"
    image_dir_test = "../../images/test/"


    def load_dataset(image_dir):
        im_list = []
        image_types = ["day", "night"]

        for im_type in image_types:
            for file in glob.glob(os.path.join(image_dir, im_type, "*")):
                im = mpimg.imread(file)

                if not im is None:
                    im_list.append((im, im_type))

        return im_list


    IMAGE_LIST = load_dataset(image_dir_training)


    def encode(label):
        return 1 if label == 'day' else 0


    def preprocess(image_list):
        standard_list = []

        for item in image_list:
            image = item[0]
            label = item[1]

            standardized_im = standardize_input(image)

            binary_label = encode(label)

            standard_list.append((standardized_im, binary_label))

        return standard_list


    STANDARDIZED_LIST = preprocess(IMAGE_LIST)
    image_num = 0
    selected_image = STANDARDIZED_LIST[image_num][0]
    selected_label = STANDARDIZED_LIST[image_num][1]

    plt.imshow(selected_image)
    print("Shape: " + str(selected_image.shape))
    print("Label [1 = day, 0 = night]: " + str(selected_label))

    image_num = 190
    test_im = STANDARDIZED_LIST[image_num][0]

    avg = avg_brightness(test_im)

    print('Avg brightness: ' + str(avg))

    plt.imshow(test_im)

    TEST_IMAGE_LIST = load_dataset(image_dir_test)

    STANDARDIZED_TEST_LIST = preprocess(TEST_IMAGE_LIST)

    random.shuffle(STANDARDIZED_TEST_LIST)


    def get_misclassified_images(test_images, threshold):
        misclassified_images_labels = []

        for image in test_images:

            im = image[0]
            true_label = image[1]

            predicted_label = estimate_label(im, threshold)

            if predicted_label != true_label:
                misclassified_images_labels.append((im, predicted_label, true_label))

        return misclassified_images_labels


    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST, threshold=99)

    total = len(STANDARDIZED_TEST_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total

    print('Accuracy: ' + str(accuracy))
    print("Number of misclassified images = " + str(len(MISCLASSIFIED)) + ' out of ' + str(total))