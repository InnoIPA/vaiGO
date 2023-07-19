# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import cv2

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GObject
except:
    pass

import time

class DP(object):
    def __init__(self, width=1920, height=1080):
        Gst.init(None)
        self.width = width
        self.height = height
        self.number_frames = 0
        self.fps = 60
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds

        self.launch_string = 'appsrc name=source is-live=true format=GST_FORMAT_TIME ' \
                             ' caps=video/x-raw,format=BGR,width=%s,height=%s,framerate=60/1 ' \
                             '! kmssink driver-name=xlnx plane-id=39 sync=false fullscreen-overlay=true' % (self.width, self.height)

        self.pipeline = Gst.parse_launch(self.launch_string)
        self.appsrc = self.pipeline.get_child_by_name('source')
        self.pipeline.set_state(Gst.State.PLAYING)

        # create canvas
        color=(0, 0, 0)
        canvas = np.ones((self.height, self.width, 3), dtype="uint8")
        canvas[:] = color
        self.canvas = canvas

    def imshow(self, frame):
        if (frame.shape[0] < self.height and frame.shape[1] == self.width) or (frame.shape[0] == self.height and frame.shape[1] < self.width) or (frame.shape[0] < self.height and frame.shape[1] < self.width):
            background = self.canvas
            background[0:frame.shape[0], 0:frame.shape[1]] = frame
            frame = background
        elif frame.shape[0] > self.height or frame.shape[1] > self.width:
            frame = cv2.resize(frame, (self.width, self.height))

        data = frame.tostring()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)

        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = timestamp

        self.number_frames += 1
        retval = self.appsrc.emit('push-buffer', buf)