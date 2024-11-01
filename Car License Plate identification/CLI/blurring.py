import cv2 as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
from helpers import extract_argument_value


input_path = extract_argument_value(sys.argv, 'input-path')
output_path = extract_argument_value(sys.argv, 'output-path')

def mean_average_precision(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersect_mins = tf.maximum(y_true[..., :2], y_pred[..., :2])
    intersect_maxs = tf.minimum(y_true[..., 2:], y_pred[..., 2:])
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.0)
    intersection = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_area = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
    pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])

    union = true_area + pred_area - intersection
    iou = tf.where(union > 0, intersection / union, tf.zeros_like(intersection))

    return tf.reduce_mean(iou)


seg_model = load_model("Car License Plate identification/Model/MobileNetV2_95_50layer.keras", custom_objects={"mean_average_precision": mean_average_precision})


for img_file in os.listdir(input_path):
    if img_file.endswith('.png') or img_file.endswith('.jpg'):
        path = os.path.join(input_path, img_file)
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        Original_image = img.copy()
        O_x, O_y = img.shape[:2]
        ratio_x, ratio_y = (224 / O_y), (224 / O_x)
        img = cv.resize(img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        img = img / 255.0
        x = seg_model.predict(img)[0]
        xmin, xmax = round(x[0] / ratio_x) - 3, round(x[2] / ratio_x) + 3
        ymin, ymax = round(x[1] / ratio_y) - 3, round(x[3] / ratio_y) + 3

        plate = Original_image[ymin: ymax, xmin: xmax]
        blur_plate = cv.GaussianBlur(plate, (101, 101), 0)
        Original_image[ymin: ymax, xmin: xmax] = blur_plate
        print(f"{output_path}/{img_file}")
        cv.imwrite(f"{output_path}/{img_file}", Original_image)
