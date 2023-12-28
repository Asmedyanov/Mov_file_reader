from tkinter import filedialog as fd
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import skimage
import pandas as pd
import numpy as np
import os

threshold_red = 209
threshold_green = 50
threshold_blue = 50
threshold_area = 1  # pix
initialdir = "C:/Users/Nikita Asmedianov/Desktop/НИКИТА-ANDERNACH"


class MoveReader:
    def __init__(self):
        self.open_file()
        # self.scrolling()
        self.show_current_frame()
        self.make_report_folder()
        self.frame_by_frame_analysis()

    def frame_by_frame_analysis(self):
        intensity = np.zeros(self.frameCount)
        self.new_order = [2, 1, 0]
        self.frames_with_particle_list = []
        self.frame_numbers_with_particle_list = []
        self.frame_property_list = []  # number, number of particles
        self.particle_property_df = pd.DataFrame(
            columns=['Frame', 'Time', 'Area', 'X', 'Y'])  # frame, x, y (in cleaned frame), radius
        self.frame_df = pd.DataFrame(
            columns=['Frame', 'Time', 'Particles', 'Intensity'])
        self.frame_current = 0
        processed_part_previous = 0
        for frame_index in range(self.frameCount):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            r, frame = self.cap.read()
            frame_clean = frame[self.top_limit:self.bottom_limit, :, self.new_order].astype(np.uint8)
            frame_clean = np.delete(frame_clean, self.ax0_index_to_remove, axis=0)
            frame_clean = np.delete(frame_clean, self.ax1_index_to_remove, axis=1)

            intensity[frame_index] = np.sum(frame_clean)
            self.frame_avg_intensity_current = intensity[frame_index] / self.pixelCount
            self.particle_analysis(frame_clean)
            self.frame_current += 1
            processed_part = int(self.frame_current * 100.0 / self.frameCount)
            if processed_part > processed_part_previous:
                print(f'Processed part is {processed_part} %')
                processed_part_previous = processed_part
        self.intenity = intensity / self.pixelCount
        intensity_df = pd.DataFrame({
            'Frame': np.arange(self.frameCount),
            'Time': np.arange(self.frameCount) / self.FPS,
            'Intensity': self.intenity
        })
        intensity_df.to_csv(f'{self.rep_dir}/intensity_full_report.csv')
        plot(self.intenity)
        xlabel('frame number')
        ylabel('AVG intensity')
        title('Common intensity')
        tight_layout()
        savefig(f'{self.rep_dir}/intensity_common.png')
        self.intenity.tofile(f'{self.rep_dir}/intensity_common.csv', sep=',')
        clf()
        self.particle_property_df.to_csv(f'{self.rep_dir}/particle_properties.csv')
        self.frame_df.to_csv(f'{self.rep_dir}/frame_with_particle_properties.csv')
        time_hist, time_bin = np.histogram(self.particle_property_df['Area'].values, bins=100)
        area_hist, area_bin = np.histogram(self.particle_property_df['Area'].values, bins=30)
        x_hist, x_bin = np.histogram(self.particle_property_df['X'].values, bins=30)
        y_hist, y_bin = np.histogram(self.particle_property_df['Y'].values, bins=30)
        df_time = pd.DataFrame({
            'Time_bin': time_bin[1:],
            'Time_hist': time_hist,
        })
        df_area = pd.DataFrame({
            'Area_bin': area_bin[1:],
            'Area_hist': area_hist,
        })
        df_x = pd.DataFrame({
            'x_bin': x_bin[1:],
            'x_hist': x_hist,
        })
        df_y = pd.DataFrame({
            'y_bin': y_bin[1:],
            'y_hist': y_hist,
        })
        df_time.to_csv(f'{self.rep_dir}/Histogram_time.csv')
        df_area.to_csv(f'{self.rep_dir}/Histogram_area.csv')
        df_x.to_csv(f'{self.rep_dir}/Histogram_x.csv')
        df_y.to_csv(f'{self.rep_dir}/Histogram_y.csv')
        fig, ax = subplots(ncols=2, nrows=2)

        ax[0, 0].set_title('Time histogram')
        ax[0, 0].set_xlabel('Sec')
        ax[0, 1].set_title('Area histogram')
        ax[0, 1].set_xlabel('Pix')
        ax[1, 0].set_title('X histogram')
        ax[1, 0].set_xlabel('Pix')
        ax[1, 1].set_title('Y histogram')
        ax[1, 1].set_xlabel('Pix')
        ax[0, 0].set_ylabel('Particles')
        ax[1, 0].set_ylabel('Particles')
        ax[0, 0].hist(self.particle_property_df['Time'], bins=100)
        ax[0, 1].hist(self.particle_property_df['Area'], bins=30)
        ax[1, 0].hist(self.particle_property_df['X'], bins=30)
        ax[1, 1].hist(self.particle_property_df['Y'], bins=30)
        tight_layout()
        savefig(f'{self.rep_dir}/histogram.png')
        clf()
        close()

        rep_text = open(f'{self.rep_dir}/report.txt', 'w')
        rep_text.write(f'{self.file_name_short}\n')
        rep_text.write(f'The video timing is {self.frameCount / self.FPS} sec\n')
        rep_text.write(f'The video FPS is {self.FPS}\n')
        rep_text.write(f'The video contains {self.frameCount} frames\n')
        rep_text.write(f'The video contains {len(self.frame_df)} frames with traces\n')
        rep_text.write(f'The video catches {len(self.particle_property_df)} particles\n')
        rep_text.write(
            f'The video catches {len(self.particle_property_df) * self.FPS / self.frameCount} particles/second\n')
        rep_text.write(f'The video AVG intensity is {self.intenity.mean()}\n')
        rep_text.close()

    def particle_analysis(self, raw_array):
        blue_part = raw_array[:, :, 1]
        red_part = raw_array[:, :, 0]
        greed_part = raw_array[:, :, 2]
        particles_markers = ((red_part > threshold_red) & (greed_part < threshold_green) & (
                blue_part < threshold_blue)).astype(np.uint8)
        particles_markers_criteria = particles_markers.max()
        if particles_markers_criteria > 0:
            labels, label_count = skimage.measure.label(particles_markers, background=0, return_num=True)
            properties_list = skimage.measure.regionprops(labels)
            criteria = 0
            for properties in properties_list:
                local_criteria = int(properties.area > threshold_area)
                criteria += local_criteria
                if local_criteria == 0:
                    continue
                new_row = [
                    self.frame_current,
                    self.frame_current / self.FPS,
                    properties['area'],
                    properties['centroid'][1],
                    properties['centroid'][0]
                ]
                self.particle_property_df.loc[len(self.particle_property_df)] = new_row
                new_row = [
                    self.frame_current,
                    self.frame_current / self.FPS,
                    label_count,
                    self.frame_avg_intensity_current
                ]
                self.frame_df.loc[len(self.frame_df)] = new_row

            if criteria == 0:
                return
            fig, ax = subplots(2)
            ax[0].imshow(raw_array)
            ax[1].imshow(labels, cmap='flag_r')
            ax[0].grid()
            ax[1].grid()
            ax[0].set_ylabel(f'Original frame {self.frame_current}')
            ax[1].set_ylabel(f'{label_count} particles')
            tight_layout()
            savefig(f'{self.particle_frame_dir}/frame_{self.frame_current}.png')
            '''if label_count > 8:
                show()'''
            plt.clf()

    def make_report_folder(self):
        file_name_split = self.filename.split('/')
        self.file_name_short = file_name_split[-1].split('.')[-2]
        file_dir = file_name_split[:-1]
        self.file_dir = '/'.join(file_dir)
        os.chdir(self.file_dir)
        self.rep_dir = f'Report_{self.file_name_short}'
        self.particle_frame_dir = f'{self.rep_dir}/particle_frames'
        os.makedirs(self.rep_dir, exist_ok=True)
        os.makedirs(self.particle_frame_dir, exist_ok=True)

    def make_bin_file(self):
        bin_file = open(f'{self.rep_dir}/full.bin', 'ab')
        for frame_index in range(self.frameCount):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            r, frame = self.cap.read()
            frame_clean = frame[self.top_limit:self.bottom_limit]
            frame_clean = np.delete(frame_clean, self.ax0_index_to_remove, axis=0)
            frame_clean = np.delete(frame_clean, self.ax1_index_to_remove, axis=1)
            frame_clean = frame_clean.astype(np.int8)
            frame_clean.tofile(bin_file)
        bin_file.close()

    def intensity_report_common(self):
        intensity = np.zeros(self.frameCount)
        for frame_index in range(self.frameCount):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            r, frame = self.cap.read()
            frame_clean = frame[self.top_limit:self.bottom_limit]
            frame_clean = np.delete(frame_clean, self.ax0_index_to_remove, axis=0)
            frame_clean = np.delete(frame_clean, self.ax1_index_to_remove, axis=1)
            intensity[frame_index] = np.sum(frame_clean)
        self.intenity = intensity / self.pixelCount
        plot(self.intenity)
        xlabel('frame number')
        ylabel('AVG intensity')
        title('Common intensity')
        savefig(f'{self.rep_dir}/intensity_common.png')
        self.intenity.tofile(f'{self.rep_dir}/intensity_common.csv', sep=',')
        clf()

    def show_current_frame(self):
        self.fig, self.ax = subplots(1, 2)
        self.ax[0].imshow(self.frame)
        self.ax[1].imshow(self.frame_clean)
        self.ax[1].grid()
        self.ax[0].grid()
        self.ax[1].set_title(f"frame {self.fc}")
        show()

    def save_report(self):
        file_name_split = self.filename.split('/')
        file_name_short = file_name_split[-1].split('.')[-2]
        file_dir = file_name_split[:-1]
        self.file_dir = '/'.join(file_dir)
        os.chdir(self.file_dir)
        rep_dir = f'Report_{file_name_short}'
        os.makedirs(rep_dir, exist_ok=True)
        rep_text = open(f'{rep_dir}/report.txt', 'w')
        rep_text.write(f'{file_name_short}\n')
        rep_text.write(f'The video timing is {self.frameCount / self.FPS} sec\n')
        rep_text.write(f'The video FPS is {self.FPS}\n')
        rep_text.write(f'The video contains {self.frameCount} frames\n')
        rep_text.write(f'The video contains {self.particle_frame_numers_array.size} frames with traces\n')
        rep_text.write(f'The video catches {self.particles_count} particles\n')
        rep_text.write(f'The video catches {self.particles_frequency} particles/second\n')
        rep_text.close()
        frames_df = pd.DataFrame({
            'Frame': self.particle_frame_numers_array,
            'Particles': self.labels_count
        })
        frames_df.to_csv(f'{rep_dir}/Frame_particle_table.csv')
        full_particle_number = np.zeros(self.frameCount)
        full_particle_number[self.particle_frame_numers_array] = self.labels_count
        full_particle_number_10_list = np.array_split(full_particle_number, int(self.frameCount / (10.0 * self.FPS)))
        particle_number_10_list = []
        start_10 = []
        stop_10 = []
        for i, a in enumerate(full_particle_number_10_list):
            start_10.append(i * 10)
            stop_10.append((i + 1) * 10)
            particle_number_10_list.append(a.sum())
        particle_number_10_df = pd.DataFrame({
            'start': start_10,
            'stop': stop_10,
            'particles': particle_number_10_list
        })
        particle_number_10_df.to_csv(f'{rep_dir}/particle_number_10_list.csv')

    def scrolling(self):
        self.filename = fd.askopenfilename(initialdir=initialdir)
        self.cap = cv2.VideoCapture(self.filename)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fc = 0
        self.new_order = [2, 1, 0]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fc)
        r, frame = self.cap.read()
        self.frame = frame[:, :, self.new_order]
        frame_clean = self.frame[50:670]
        ax0_index_to_remove_small = np.concatenate([np.arange(127, 132), np.arange(489, 494)], dtype=int)
        ax0_index_to_remove_big = np.concatenate([np.arange(23, 29), np.arange(585, 590)], dtype=int)
        ax1_index_to_remove_small = np.concatenate([np.arange(318, 324), np.arange(959, 964)], dtype=int)
        ax1_index_to_remove_big = np.concatenate([np.arange(79, 84), np.arange(1196, 1202)], dtype=int)
        criteria_small = frame[ax0_index_to_remove_small, :, 2].std()
        criteria_big = frame[ax0_index_to_remove_big, :, 2].std()
        '''if criteria_small > criteria_big:
            self.ax0_index_to_remove = ax0_index_to_remove_small
            self.ax1_index_to_remove = ax1_index_to_remove_small
        else:
            self.ax0_index_to_remove = ax0_index_to_remove_big
            self.ax1_index_to_remove = ax1_index_to_remove_big'''
        self.ax0_index_to_remove = np.concatenate([ax0_index_to_remove_small, ax0_index_to_remove_big], dtype=int)
        self.ax1_index_to_remove = np.concatenate([ax1_index_to_remove_small, ax1_index_to_remove_big], dtype=int)
        frame_clean = np.delete(frame_clean, self.ax0_index_to_remove, axis=0)
        frame_clean = np.delete(frame_clean, self.ax1_index_to_remove, axis=1)
        self.frame_clean = frame_clean
        self.fig, self.ax = subplots(1, 2)

        self.ax[0].imshow(self.frame.swapaxes(0, 1))
        self.ax[1].imshow(self.frame_clean.swapaxes(0, 1))
        self.ax[1].grid()
        self.ax[0].grid()
        self.ax[1].set_title(f"frame {self.fc}")
        self.frame_index = 0
        tight_layout()

        def on_key_press(event):
            increment = 0
            if event.key in ['right', 'up']:
                increment = 1
            if event.key in ['left', 'down']:
                increment = -1
            if (self.fc + increment < 0) | (
                    self.fc + increment >= self.frameCount):
                increment = 0
            self.fc += increment
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fc)
            r, frame = self.cap.read()
            self.frame = frame[:, :, self.new_order]

            frame_clean = self.frame[50:670]
            frame_clean = np.delete(frame_clean, self.ax0_index_to_remove, axis=0)
            frame_clean = np.delete(frame_clean, self.ax1_index_to_remove, axis=1)
            self.frame_clean = frame_clean
            self.ax[0].set_title(
                f"frame {self.fc} with frame")
            self.ax[1].set_title(f"frame {self.fc} no frame")
            self.ax[1].imshow(self.frame_clean.swapaxes(0, 1))
            self.ax[0].imshow(self.frame.swapaxes(0, 1))
            draw()

        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        show()

    def open_file(self):
        self.filename = fd.askopenfilename(initialdir=initialdir)
        self.cap = cv2.VideoCapture(self.filename)
        self.frameCount = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pixelCount = self.frameWidth * self.frameHeight
        self.fc = 0
        self.new_order = [2, 1, 0]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.fc)
        r, frame = self.cap.read()
        self.frame = frame[:, :, self.new_order]
        self.top_limit = 50
        self.bottom_limit = 670
        frame_clean = self.frame[self.top_limit:self.bottom_limit]
        ax0_index_to_remove_small = np.concatenate([np.arange(127, 132), np.arange(489, 494)], dtype=int)
        ax0_index_to_remove_big = np.concatenate([np.arange(23, 29), np.arange(585, 590)], dtype=int)
        ax1_index_to_remove_small = np.concatenate([np.arange(318, 324), np.arange(959, 964)], dtype=int)
        ax1_index_to_remove_big = np.concatenate([np.arange(79, 84), np.arange(1196, 1202)], dtype=int)
        self.ax0_index_to_remove = np.concatenate([ax0_index_to_remove_small, ax0_index_to_remove_big], dtype=int)
        self.ax1_index_to_remove = np.concatenate([ax1_index_to_remove_small, ax1_index_to_remove_big], dtype=int)
        frame_clean = np.delete(frame_clean, self.ax0_index_to_remove, axis=0)
        frame_clean = np.delete(frame_clean, self.ax1_index_to_remove, axis=1)
        self.frame_clean = frame_clean

    def open_file_old(self):
        self.filename = fd.askopenfilename(initialdir=initialdir)
        cap = cv2.VideoCapture(self.filename)
        self.frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.FPS = int(cap.get(cv2.CAP_PROP_FPS))
        self.frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # buf = np.empty((self.frameCount, self.frameHeight, self.frameWidth, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        self.new_order = [2, 1, 0]
        particle_frame_numbers_list = []
        particle_frame_list = []
        particle_frame_raw_list = []
        labels_list = []
        labels_count_list = []
        small_rep = 0
        small_rep_old = 0
        while (fc < self.frameCount and ret):
            ret, buf = cap.read()
            raw_array = buf[90:630, 90:1150]
            raw_array = raw_array[:, :, self.new_order]
            blue_part = raw_array[:, :, 0]
            red_part = raw_array[:, :, 2]
            greed_part = raw_array[:, :, 1]
            particles_markers = (blue_part > threshold_red) & (greed_part < threshold_green) & (
                    red_part < threshold_blue)
            if particles_markers.max() > 0:
                particle_frame_numbers_list.append(fc)
                particle_frame_list.append(particles_markers)
                labels, label_count = skimage.measure.label(particles_markers, background=0, return_num=True)
                labels_list.append(labels)
                labels_count_list.append(label_count)
                particle_frame_raw_list.append(raw_array)
            small_rep = fc * 10 // self.frameCount
            if small_rep > small_rep_old:
                print(f'Process {small_rep * 10} %')
                small_rep_old = small_rep
            fc += 1
        cap.release()
        self.labels_count = np.array(labels_count_list)
        self.labels_array = np.array(labels_list)
        self.particles_count = self.labels_count.sum()
        self.particles_frequency = self.particles_count / self.frameCount * self.FPS
        self.particle_frame_numers_array = np.array(particle_frame_numbers_list)
        self.particle_raw_frame_array = np.array(particle_frame_raw_list)
        print(f'The video timing is {self.frameCount / self.FPS} sec')
        print(f'The video FPS is {self.FPS}')
        print(f'The video contains {self.frameCount} frames')
        print(f'The video contains {self.particle_frame_numers_array.size} frames with traces')
        print(f'The video catches {self.particles_count} particles')
        print(f'The video catches {self.particles_frequency} particles/second')
        self.particle_frame_array = np.array(particle_frame_list)

    def init_figure(self):
        self.fig, self.ax = subplots(1, 2)
        self.ax[0].imshow(self.particle_raw_frame_array[self.particle_frame_numers_array[0]])
        self.ax[1].imshow(self.labels_array[self.particle_frame_numers_array[0]])
        self.ax[1].grid()
        self.ax[0].grid()
        self.ax[1].set_title(f"frame {self.particle_frame_numers_array[0]}")
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
            self.ax[1].imshow(self.labels_array[self.frame_index])
            self.ax[0].imshow(self.particle_raw_frame_array[self.frame_index])
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
            self.ax[1].imshow(self.labels_array[self.frame_index])
            self.ax[0].imshow(self.particle_raw_frame_array[self.frame_index])
            draw()

        self.fig.canvas.mpl_connect('scroll_event', on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', on_key_press)
        show()
