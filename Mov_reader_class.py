from tkinter import filedialog as fd
import cv2
import numpy as np
from matplotlib.pyplot import *


class MoveReader:
    def __init__(self):
        self.open_file()
        self.init_figure()

    def open_file(self):
        self.filename = fd.askopenfilename(initialdir="C:/Users/user/OneDrive - Technion/Desktop/НИКИТА-ANDERNACH")
        cap = cv2.VideoCapture(self.filename)
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((self.frameCount, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True

        while (fc < self.frameCount and ret):
            ret, buf[fc] = cap.read()
            fc += 1
        cap.release()
        self.raw_array = buf[:, 90:630, 90:1150]
        self.blue_part = self.raw_array[:, :, :, 2]
        self.red_part = self.raw_array[:, :, :, 0]
        self.greed_part = self.raw_array[:, :, :, 1]
        threashold = 254
        particles_markers = (self.blue_part > 254) & (self.greed_part < threashold * 0.66) & (
                self.red_part < threashold * 0.66)
        particle_frame_numbers_list = []
        paricle_frame_list = []
        for i, frame in enumerate(particles_markers):
            if frame.max() > 0:
                particle_frame_numbers_list.append(i)
                paricle_frame_list.append(frame)
        self.particle_frame_numers_array = np.array(particle_frame_numbers_list)
        print(f'The video containce {self.particle_frame_numers_array.size} frames with traces')
        self.particle_frame_array = np.array(paricle_frame_list)

    def init_figure(self):
        self.fig, self.ax = subplots()
        self.ax.imshow(self.raw_array[0])
        self.ax.set_title(f"frame {0}")
        self.frame_index = 0
        self.ax.axis('equal')

        def on_scroll(event):
            increment = 1 if event.button == 'up' else -1
            if (self.frame_index + increment < 0) | (
                    self.frame_index + increment >= self.particle_frame_numers_array.size):
                increment = 0
            self.frame_index += increment
            self.ax.set_title(f"frame {self.particle_frame_numers_array[self.frame_index]}")
            self.ax.imshow(self.particle_frame_array[self.frame_index])
            draw()

        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        show()
