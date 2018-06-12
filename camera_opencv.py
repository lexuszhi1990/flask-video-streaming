import cv2
from base_camera import BaseCamera
from refinedet_live import refinedet_wrapper

class Camera(BaseCamera):
    video_source = 0

    def __init__(self):
        self.detector = refinedet_wrapper()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        while True:
            # read current frame
            _, img = camera.read()


            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
