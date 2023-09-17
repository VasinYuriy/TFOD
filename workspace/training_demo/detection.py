import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np


def init(PATH_TO_LABELS, PATH_TO_CFG, PATH_TO_CKPT):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging


    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)
    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)


    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

    return ckpt, detection_model, category_index


@tf.function
def _detect_fn(image, detection_model):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


def webcam_rec(camnum, PATH_TO_LABELS, PATH_TO_CFG, PATH_TO_CKPT):
    cap = cv2.VideoCapture(camnum)
    ckpt, detection_model, category_index = init(PATH_TO_LABELS, PATH_TO_CFG, PATH_TO_CKPT)
    while True:
        # Read frame from camera
        ret, image_np = cap.read()

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = _detect_fn(input_tensor, detection_model)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'][0].numpy(),
              (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.30,
              agnostic_mode=False)

        # Display output
        cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def image_rec(image, PATH_TO_LABELS, PATH_TO_CFG, PATH_TO_CKPT):
    ckpt, detection_model, category_index = init(PATH_TO_LABELS, PATH_TO_CFG, PATH_TO_CKPT)
    image_np_expanded = np.expand_dims(image, axis=0)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = _detect_fn(input_tensor, detection_model)
    label_id_offset = 1
    image_np_with_detections = image.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
              image_np_with_detections,
              detections['detection_boxes'][0].numpy(),
              (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
              detections['detection_scores'][0].numpy(),
              category_index,
              use_normalized_coordinates=True,
              max_boxes_to_draw=200,
              min_score_thresh=.30,
              agnostic_mode=False)

    # Display output
    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))
    cv2.waitKey(0)


if __name__ == '__main__':
    PATH_TO_LABELS = 'annotations/label_map.pbtxt'
    PATH_TO_CFG = 'exported-models/my_model1/pipeline.config'
    PATH_TO_CKPT = 'exported-models/my_model1/checkpoint/'
    img = cv2.imread('images/rofl/img1.jpg')
    image_rec(
        img,
        PATH_TO_LABELS,
        PATH_TO_CFG,
        PATH_TO_CKPT
    )
    # webcam_rec(
    #     0,
    #     PATH_TO_LABELS,
    #     PATH_TO_CFG,
    #     PATH_TO_CKPT
    # )
