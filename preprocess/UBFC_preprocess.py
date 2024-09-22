# -*- coding:utf-8 -*-
# @File   : UBFC_preprocess.py
# @Time   : 2022/9/7 18:35
# @Author : Zhang Xinyu
import os
import warnings
import numpy as np
import dlib
import tqdm
import argparse
from torch.utils.data import Dataset
import cv2 as cv

warnings.filterwarnings("ignore")
detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

def get_args_parser():
    parser = argparse.ArgumentParser('UBFC preprocessing', add_help=False)
    # Main params.
    parser.add_argument('--data-path', default='/home/linxi/data/rPPG/UBFC', type=str,
                        help="""Please specify path to the 'UBFC' as input.""")
    parser.add_argument('--infos-path', default='/home/linxi/data/UBFC-infos-SKD', type=str,
                        help="""Please specify path to the 'UBFC-infos' as input.""")

    parser.add_argument('--frame-path', default='/home/linxi/data/ubfc-frame-SKD/frame_list', type=str,
                        help="""Please specify path to the 'frame_list' as output.""")
    parser.add_argument('--face-data-path', default='/home/linxi/data/ubfc-face-SKD/data', type=str,
                        help="""Please specify path to the 'face_data' as output.""")
    parser.add_argument('--face-img-path', default='/home/linxi/data/ubfc-face-SKD/img', type=str,
                        help="""Please specify path to the 'face_img' as output.""")
    return parser


def the_only_face(frame_in, scale=3, maxlenth=40):
    for i in range(len(frame_in)):
        rects = detector(frame_in[i], 0)
        lens = len(rects)
        if lens == 0:
            the_only_rect = (False, None)
        elif lens == 1:
            the_only_rect = (True, rects[0].rect)
        else:
            axis_x = int(frame_in[i].shape[0] / 2)
            axis_y = int(frame_in[i].shape[1] / 2)
            distances = [0.0 for x in range(lens)]
            for i in range(lens):
                rects_axis_x = int((rects[i].rect.right() - rects[i].rect.left()) / 2)
                rects_axis_y = int((rects[i].rect.right() - rects[i].rect.left()) / 2)
                distances[i] = (rects_axis_x - axis_x) ** 2 + (rects_axis_y - axis_y) ** 2

            min_distance_index = distances.index(min(distances))
            the_only_rect = (True, rects[min_distance_index].rect)
        if the_only_rect[0]:
            the_just_face_rect = the_only_rect[1]
            t0, b0, l0, r0 = the_just_face_rect.top(), the_just_face_rect.bottom(), the_just_face_rect.left(), the_just_face_rect.right()
            maxh = (b0 - t0) / scale
            maxw = (r0 - l0) / scale
            finaladdh = max(maxh, maxlenth)
            finaladdw = max(maxw, maxlenth)
            tf, bf, lf, rf = max(0, t0 - int(finaladdh*1.4)), min(frame_in[i].shape[0], b0 + int(finaladdh)), \
                             max(0, l0 - int(finaladdw)), min(frame_in[i].shape[1], r0 + int(finaladdw))
            return tf, bf, lf, rf
        else:
            continue
    return None


class Dataset_ubfc_generate(Dataset):
    def __init__(self, args, person_number, prefix=0, cache_discard=True):
        super().__init__()
        self.args = args
        self.image_size = 131
        self.margin = 20
        self.video_path = args.data_path
        self.info_pool = []
        self.collect_info(args.infos_path, person_number)
        if cache_discard:
            self.del_cache()
        self.prefix = prefix

    def __getitem__(self, index):
        info = self.info_pool[index]
        txt_path = info[0]
        person_number = info[1]
        video_path = os.path.join(self.video_path, os.path.join(person_number, "vid.avi"))

        with open(txt_path, "r") as f:
            info_str = f.readline()
            start_place, end_place, hr = info_str.split("_")[0], info_str.split("_")[1], info_str.split("_")[2]
            start_place, end_place = int(start_place), int(end_place)

            i = str(self.prefix)
            frame_list = self.read_video(video_path, start_place, end_place, person_number, i)

            if self.args.frame_path:
                save_path = os.path.join(self.args.frame_path, '_'.join([i, person_number,
                                                                        str(start_place),
                                                                        str(end_place), hr]) + '_.npy')
                frame_list_save = np.array(frame_list)
                np.save(save_path, frame_list_save)

            self.prefix += 1
        return

    def __len__(self):
        return len(self.info_pool)

    def collect_info(self, info_path, person_number):
        for p in os.listdir(info_path):
            if p in person_number:
                total_dir = os.path.join(info_path, p)
                txt_name = os.listdir(total_dir)[0]
                self.info_pool.append((os.path.join(total_dir, txt_name), p))

    def read_video(self, video_path, start_place_this_time, end_place_this_time, person_number, i):
        cap = cv.VideoCapture(video_path)
        success = True
        frame_list = []
        while success:
            success, frame = cap.read()
            if frame is not None:
                frame_list.append(frame)
        frame_list = frame_list[start_place_this_time:end_place_this_time]
        tf, bf, lf, rf = the_only_face(frame_list)

        if self.args.face_data_path:
            save_path = os.path.join(self.args.face_data_path, '_'.join([i, person_number,
                                                                         str(start_place_this_time),
                                                                         str(end_place_this_time)]) + '.npy')
            np.save(save_path, [tf, bf, lf, rf])

        img = frame_list[0].copy()
        cv.rectangle(img, (lf, tf), (rf, bf), (0, 0, 255), 3)
        if self.args.face_img_path:
            save_path = os.path.join(self.args.face_img_path, '_'.join([i, person_number,
                                                                        str(start_place_this_time),
                                                                        str(end_place_this_time)]) + '.png')

            cv.imwrite(save_path, img)

        for i in range(len(frame_list)):
            frame_list[i] = cv.resize(frame_list[i][tf: bf, lf: rf], (self.image_size + self.margin, self.image_size + self.margin))

        return frame_list

    def del_cache(self):
        paths = [self.args.face_data_path, self.args.face_img_path, self.args.frame_path]
        for path in paths:
            if path:
                if not os.path.exists(path):
                    os.makedirs(path)
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    os.remove(c_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('UBFC preprocessing', parents=[get_args_parser()])
    args = parser.parse_args()
    total_set = set(list(range(1, 50)))
    person_name = [rf"p{i}" for i in total_set]
    datasets = Dataset_ubfc_generate(args, person_name, prefix=0, cache_discard=True)
    for i in tqdm.tqdm(range(len(datasets))):
        datasets.__getitem__(i)


