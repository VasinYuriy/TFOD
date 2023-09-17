import os
import cv2
from absl import app
from absl import flags
from datetime import datetime


FLAGS = flags.FLAGS
flags.DEFINE_string('img_dir', None, 'Path to write outputs.')
flags.mark_flag_as_required('img_dir')


def create_dir(image_path):
    if not os.path.exists(image_path):
        os.mkdir(image_path)


def get_pictures(image_path):
    create_dir(image_path)
    cam = cv2.VideoCapture(0)

    while True:
        result, image = cam.read()

        if not result:
            print('Видео недоступно')
            break
        else:
            cv2.imshow("video", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                image_name = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                image_name = image_name.replace(' ', '_')
                image_name = image_name.replace('/', '_')
                image_name = image_name.replace(':', '')
                save_path = image_path + '/' + image_name + '.png'
                cv2.imwrite(save_path, image)
                print('Saved at:{}'.format(save_path))
            elif key == ord('q'):
                break


def main(_):
    image_path = FLAGS.img_dir
    get_pictures(image_path)


if __name__ == "__main__":
    app.run(main)
