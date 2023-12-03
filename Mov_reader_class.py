from tkinter import filedialog as fd
import cv2
import numpy as np
from matplotlib.pyplot import *
import skimage
import pandas as pd

threashold_red = 209
threashold_green = 50
threashold_blue = 50


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

        particles_markers = (self.blue_part > threashold_red) & (self.greed_part < threashold_green) & (
                self.red_part < threashold_blue)
        particle_frame_numbers_list = []
        paricle_frame_list = []
        self.labels_list = []
        labels_count_list = []
        self.region_props_list = []
        for i, frame in enumerate(particles_markers):
            if frame.max() > 0:
                particle_frame_numbers_list.append(i)
                paricle_frame_list.append(frame)
                labels, label_count = skimage.measure.label(frame, background=0, return_num=True)
                # region_props = pd.DataFrame(skimage.measure.regionprops_table(labels))
                # self.region_props_list.append(region_props)
                self.labels_list.append(labels)
                labels_count_list.append(label_count)
        self.labels_count = np.array(labels_count_list)
        self.particles_count = self.labels_count.sum()
        self.particles_frequency = self.particles_count / self.frameCount * 30.0
        self.particle_frame_numers_array = np.array(particle_frame_numbers_list)
        print(f'The video timing is {self.frameCount / 30.0} sec')
        print(f'The video contains {self.frameCount} frames')
        print(f'The video contains {self.particle_frame_numers_array.size} frames with traces')
        print(f'The video catches {self.particles_count} particles')
        print(f'The video catches {self.particles_frequency} particles/second')
        self.particle_frame_array = np.array(paricle_frame_list)

    def init_figure(self):
        self.fig, self.ax = subplots(1, 2)
        self.ax[1].imshow(self.labels_list[0])
        self.ax[1].grid()
        self.ax[0].grid()
        self.ax[1].set_title(f"frame {0}")
        self.frame_index = 0

        def on_scroll(event):
            increment = 1 if event.button == 'up' else -1
            if (self.frame_index + increment < 0) | (
                    self.frame_index + increment >= self.particle_frame_numers_array.size):
                increment = 0
            self.frame_index += increment
            self.ax[0].set_title(
                f"frame {self.particle_frame_numers_array[self.frame_index]}")
            self.ax[1].set_title(f"{self.labels_count[self.frame_index]} particles")
            self.ax[1].imshow(self.labels_list[self.frame_index])
            self.ax[0].imshow(self.raw_array[self.particle_frame_numers_array[self.frame_index]])
            draw()

        def on_key_press(event):
            increment = 0
            if event.key in ['right', 'up']:
                increment = 1
            if event.key in ['left', 'down']:
                increment = -1
            if (self.frame_index + increment < 0) | (
                    self.frame_index + increment >= self.particle_frame_numers_array.size):
                increment = 0
            self.frame_index += increment
            self.ax[0].set_title(
                f"frame {self.particle_frame_numers_array[self.frame_index]}")
            self.ax[1].set_title(f"{self.labels_count[self.frame_index]} particles")
            self.ax[1].imshow(self.labels_list[self.frame_index])
            self.ax[0].imshow(self.raw_array[self.particle_frame_numers_array[self.frame_index]])
            draw()

        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        show()
