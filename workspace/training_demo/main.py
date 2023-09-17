import sys
import cv2

sys.path.append("../scripts/validation")
import detection


if __name__ == '__main__':
    PATH_TO_LABELS = 'annotations/label_map.pbtxt'
    PATH_TO_CFG = 'exported-models/my_model1/pipeline.config'
    PATH_TO_CKPT = 'exported-models/my_model1/checkpoint/'
    image = cv2.imread('images/rofl/img1.jpg')
    detection.webcam_rec(0, PATH_TO_LABELS, PATH_TO_CFG, PATH_TO_CKPT)
