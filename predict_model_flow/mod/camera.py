# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import cv2
import time
import numpy
import logging
import threading
import queue


from typing import Optional
try:
    from vimba import *
except:
	pass


class CAMERA():
    def __init__(self,args):
        self.frame = None
        self.args = args

    def open(self):
        if self.args.open_camera:
            self.cap = cv2.VideoCapture(0) # 開啟連接在USB上的相機鏡頭
        if self.args.predict_video_path:
            self.cap = cv2.VideoCapture(self.args.predict_video_path) # 直接讀取video
    
    def read(self): 
        try:
            is_success, frame = self.cap.read()
            logging.debug("Sucess")

        except cv2.error:
            logging.debug("continue")

        if not is_success:
            logging.debug("End")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 將opencv讀取到的影像由BGR轉RGB
        
        return is_success, frame
    
    def is_opened(self):
        if not self.cap.isOpened():
            raise "Please Check the camera id."
        return self.cap.isOpened()
        
    def release(self):
        self.cap.release()

    def read_image(self):
        self.simple_image = cv2.imread(self.args.predict_picture_path) # 直接讀取video
        return self.simple_image

    def opuput_setting(self, path, width = 1439, high = 1080):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v') #codec
        if not os.path.exists(path):
            os.makedirs(path)
            time.sleep(1)
            self.out = cv2.VideoWriter(path + "/"+ f"output_{int(time.time())}.mp4", self.fourcc, 20.0, (width, high))

    def output_release(self):
        self.out.release()

    def opuput_save_frame(self, index, frame, path):
        if not os.path.exists(path):
            os.makedirs(path)
            time.sleep(1)
        cv2.imwrite(path + "/" +str(index) + ".png", frame)

    def output_save_video(self, frame):
            self.out.write(frame)

class VIMBA_CAMERA():
    def get_camera(self, camera_id: Optional[str]) -> Camera:
        with Vimba.get_instance() as vimba:
            if camera_id:
                try:
                    return vimba.get_camera_by_id(camera_id)

                except VimbaCameraError:
                    abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

            else:
                cams = vimba.get_all_cameras()
                if not cams:
                    abort('No Cameras accessible. Abort.')

                return cams[0]

    def setup_camera(self, cam: Camera, fw):
        with cam:
            # Try to enable automatic exposure time setting
            try:
                cam.load_settings(fw, PersistType.All)
                print("Load FW setting sucess!!")

            except (AttributeError, VimbaFeatureError):
                print('Camera {}: Failed to set Feature \'ExposureAuto\'.'.format(
                            cam.get_id()))

    def resize_if_required(self, frame: Frame) -> numpy.ndarray:
        # Helper function resizing the given frame, if it has not the required dimensions.
        # On resizing, the image data is copied and resized, the image inside the frame object
        # is untouched.
        FRAME_HEIGHT = 608
        FRAME_WIDTH = 608
        cv_frame = frame.as_opencv_image()
        if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
            print(frame.get_height(), frame.get_width())
            cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
            cv_frame = cv_frame[..., numpy.newaxis]

        return cv_frame

class Handler:
    def __init__(self, frame_q):
        self.shutdown_event = threading.Event()
        self.frame_q = frame_q
        
    def __call__(self, cam: Camera, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            logging.debug('{} acquired {}'.format(cam, frame))
           
            self.frame_q.put(frame.as_opencv_image())
            # cv2.imshow(msg.format(cam.get_name()), frame.as_opencv_image())
        cam.queue_frame(frame)

