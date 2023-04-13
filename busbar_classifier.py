import os
import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn import linear_model

from fileloader import Fileloader
from RANSAC import RANSAC

class Detector():
    def __init__(self, defect_distance_thr):
        self.defect_distance_thr = defect_distance_thr
        self.roi_idx = 0
        self.dist_b_max = 0
        self.dist_b_min = 0
        self.b_max = 0
        self.b_min = 0

    def run(self, heightData):
        # heightData = np.where(heightData < -9.5000, 0, heightData)
        # if np.count_nonzero(heightData < -9.5) > 0:
        #     print(np.count_nonzero(heightData < -9.5))
        #     return
        isDefect = False
        # csv_root = r'D:\sample\톱텍\불량 데이터'
        # fileloader = Fileloader(csv_path, 'csv')
        # y_value = pd.read_csv(filePath, index_col=None, header=None, encoding='CP949')  # index_col,header 첫번째행이 칼럼이름이라면 header=0
        y_value = np.expand_dims(heightData, axis=1)
        x_value = []

        # normalization
        x_norm_max = 200
        x_value = [i/x_norm_max for i in range(y_value.shape[0])] # 컴프리헨션으로 바꿈

        x_value = np.expand_dims(np.asarray(x_value), axis=1)
        xy_value = np.concatenate([x_value, y_value], 1)

        # homogeneous transformation
        z_value = np.ones(shape=(x_value.shape[0], 1), dtype=np.int8)
        xyz_value = np.concatenate([xy_value, z_value], 1)
        # for i in range(xyz_value.shape[0]):
        #     xyz_value[i] = self.__translate(xyz_value[i], deg=10, trans=[0,0])
        #     # xyz_value[i] = self.__translate(xyz_value[i], deg=0, trans=[+175/x_norm_max, -1.85])
        xy_tilt_value = xyz_value[:, 0:2]

        # get degree
        cluster_idx = []
        bottom_left = np.array([0.0, 0.0])
        bottom_right = np.array([0.0, 0.0])
        a_boundary = 20
        b_boundary = 75
        for idx in range(xy_tilt_value.shape[0]):
            if idx < a_boundary:
                cluster_idx.append(0)
                bottom_left = np.add(bottom_left, xy_tilt_value[idx])
            elif idx >= xy_tilt_value.shape[0] - a_boundary:
                cluster_idx.append(1)
                bottom_right = np.add(bottom_right, xy_tilt_value[idx])
            elif idx >= xy_tilt_value.shape[0] / 2 - b_boundary and idx < xy_tilt_value.shape[0] / 2 + b_boundary:
                cluster_idx.append(2)
            else:
                cluster_idx.append(3)
        #TODO : 추출한 좌우바닥의 기울기가 n이상일 경우 pass

        bottom_left = bottom_left / a_boundary
        bottom_right = bottom_right / a_boundary
        tilt_deg = np.arctan((bottom_left[1] - bottom_right[1]) / (bottom_left[0] - bottom_right[0]))
        # print("tilt_deg", tilt_deg / np.pi * 180)

        # rotate
        xyz_value = np.concatenate([xy_tilt_value, z_value], 1)
        for i in range(xyz_value.shape[0]):
            xyz_value[i] = self.__translate(xyz_value[i], deg=-tilt_deg / np.pi * 180, trans=[0,0])
            # xyz_value[i] = self.__translate(xyz_value[i], deg=0, trans=[+175/x_norm_max, -1.85])
        xy_compensation_value = xyz_value[:, 0:2]

        # # clustering
        # eps = 0.08i_
        # min_samples = 15
        # cluster_idx = self.__use_DBSCAN(xy_value, eps=0.8, min_samples=15)

        # line extractor
        bottom_idx = self.__extract_idxes(xy_tilt_value, bottom_left, bottom_right, threshold=0.04)
        if 1 in bottom_idx:
            bottom_pts = self.__extract_points(xy_compensation_value, bottom_idx, 1)
        else:
            bottom_pts = xy_compensation_value[:, :]
        # bottom_idx_clustered = self.__use_DBSCAN(bottom_pts, eps=0.2, min_samples=15)
        bottom_idx_clustered = self.__use_DBSCAN(bottom_pts, eps=0.3, min_samples=15)
        if not 0 in bottom_idx_clustered:
            # TODO: bottom_idx1에 0이 없으면 return
            print("null")
            return "null"
        # print("bottom_idx_clustered : ", bottom_idx_clustered)
        bottom_pts1 = self.__extract_points(bottom_pts, bottom_idx_clustered, 0) # 0 : 왼쪽
        bottom_pts2 = self.__extract_points(bottom_pts, bottom_idx_clustered, np.unique(bottom_idx_clustered)[-1]) # 마지막 index : 오른쪽

        bottom_line_x1 = np.array([bottom_pts1[bottom_pts1[:, 0].argmin(), 0] * x_norm_max, bottom_pts1[bottom_pts1[:, 0].argmax(), 0] * x_norm_max])
        bottom_line_y1 = np.array([
            (bottom_pts1[bottom_pts1[:, 0].argmin(), 1] + bottom_pts1[bottom_pts1[:, 0].argmax(), 1]) / 2,
            (bottom_pts1[bottom_pts1[:, 0].argmin(), 1] + bottom_pts1[bottom_pts1[:, 0].argmax(), 1]) / 2])
        bottom_line_x2 = np.array([bottom_pts2[bottom_pts2[:, 0].argmin(), 0] * x_norm_max, bottom_pts2[bottom_pts2[:, 0].argmax(), 0] * x_norm_max])
        bottom_line_y2 = np.array([
            (bottom_pts2[bottom_pts2[:, 0].argmin(), 1] + bottom_pts2[bottom_pts2[:, 0].argmax(), 1]) / 2,
            (bottom_pts2[bottom_pts2[:, 0].argmin(), 1] + bottom_pts2[bottom_pts2[:, 0].argmax(), 1]) / 2])

        # get offset & roi
        bottom_width_left = bottom_line_x1[1] - bottom_line_x1[0]
        bottom_width_right = bottom_line_x2[1] - bottom_line_x2[0]
        bottom_line_x = [np.mean(bottom_pts1[:,0] * x_norm_max), np.mean(bottom_pts2[:,0] * x_norm_max)]
        bottom_line_y = [np.mean(bottom_pts1[:,1]), np.mean(bottom_pts2[:,1])]
        offset = (bottom_width_left - bottom_width_right)/2

        roi_idx = []
        b_boundary = 75
        for idx in range(xy_compensation_value.shape[0]):
            if idx >= xy_compensation_value.shape[0] / 2 - b_boundary + offset and idx < xy_compensation_value.shape[0]/2 + b_boundary + offset:
                roi_idx.append(0)
            else:
                roi_idx.append(1)

        # get min max & distancce
        center_pts = self.__extract_points(xy_compensation_value, roi_idx, 0)
        b_max = [center_pts[center_pts[:, 1].argmax(), 0] * x_norm_max, center_pts[center_pts[:, 1].argmax(), 1]]
        b_min = [center_pts[center_pts[:, 1].argmin(), 0] * x_norm_max, center_pts[center_pts[:, 1].argmin(), 1]]
        bottom = np.mean(bottom_line_y)
        dist_b_max = b_max[1] - bottom
        dist_b_min = b_min[1] - bottom

        # line_eq = np.dot(np.linalg.inv([[bottom_left[0], 1], [bottom_right[0], 1]]), [bottom_left[1], bottom_right[1]])

        cluster_idx = np.expand_dims(np.asarray(cluster_idx), axis=1)
        xy_value[:, 0] = xy_value[:, 0] * x_norm_max
        xy_tilt_value[:, 0] = xy_tilt_value[:, 0] * x_norm_max
        xy_compensation_value[:, 0] = xy_compensation_value[:, 0] * x_norm_max
        xy_value = np.concatenate([xy_value, cluster_idx], 1)


        # # show plot
        # # fig = plt.figure(figsize=(16, 10))
        # # ax = fig.add_subplot(2, 2, 1)
        # # ax.scatter(xy_value[:, 0], xy_value[:, 1], c=cluster_idx, s=50, cmap='Dark2')
        # # plt.title("Original")
        # # plt.xlabel('sequence', size=10)
        # # plt.ylabel('height(mm)', size=10)
        #
        # fig = plt.figure(figsize=(16, 10))
        # ax = fig.add_subplot(2, 2, 1)
        # ax.scatter(xy_tilt_value[:, 0], xy_tilt_value[:, 1], c=cluster_idx, s=20, cmap='Dark2')
        # plt.title('bottom_left = {}, bottom_right = {}, a_boundary = {}'.format(bottom_left, bottom_right, a_boundary), size=10)
        # plt.xlabel('sequence', size=10)
        # plt.ylabel('height(mm)', size=10)
        #
        # ax = fig.add_subplot(2, 2, 2)
        # ax.scatter(xy_compensation_value[:, 0], xy_compensation_value[:, 1], c=bottom_idx, s=20, cmap='Dark2')
        # ax.plot(bottom_line_x1, bottom_line_y1, color='cornflowerblue', linewidth=1, label='RANSAC regressor')
        # ax.plot(bottom_line_x2, bottom_line_y2, color='cornflowerblue', linewidth=1, label='RANSAC regressor')
        # plt.title('left_width = {0:0.2f}, right_width = {1:0.2f}'.format(bottom_width_left, bottom_width_right),size=10)
        # plt.xlabel('sequence', size=10)
        # plt.ylabel('height(mm)', size=10)
        #
        # ax = fig.add_subplot(2, 2, 3)
        # ax.scatter(xy_compensation_value[:, 0], xy_compensation_value[:, 1], c=roi_idx, s=20, cmap='Dark2')
        # ax.scatter(b_max[0], b_max[1], c=2, s=80, cmap='spring')
        # ax.scatter(b_min[0], b_min[1], c=3, s=80, cmap='gist_heat')
        # ax.plot(bottom_line_x, bottom_line_y, '--', color='black', linewidth=2, label='RANSAC regressor')
        # plt.title('offset = {0:0.2f},   a = {1:0.3f},  b_max = {2:0.3f},  b_min = {3:0.3f}'.format(offset, bottom, b_max[1], b_min[1]),size=10)
        # plt.xlabel('sequence', size=10)
        # plt.ylabel('height(mm)', size=10)
        #
        # ax = fig.add_subplot(2, 2, 4)
        # ax.scatter(xy_compensation_value[:, 0], xy_compensation_value[:, 1], c=roi_idx, s=20, cmap='Dark2')
        # ax.scatter(b_max[0], b_max[1], c=2, s=80, cmap='spring')
        # ax.scatter(b_min[0], b_min[1], c=3, s=80, cmap='gist_heat')
        # ax.plot(bottom_line_x, bottom_line_y, '--', color='black', linewidth=2, label='RANSAC regressor')
        # if dist_b_max < 0.5 or dist_b_min < 0.5:
        #     plt.title('불량, b_max_dist = {1:0.3f},   b_min_dist = {2:0.3f}'.format(bottom, dist_b_max, dist_b_min), size=10)
        # else:
        #     plt.title('양품, b_max_dist = {1:0.3f},   b_min_dist = {2:0.3f}'.format(bottom, dist_b_max, dist_b_min), size=10)
        # plt.xlabel('sequence', size=10)
        # plt.ylabel('height(mm)', size=10)
        # # plt.show()
        # # return
        #
        # cv2.imshow("x", x)
        # cv2.waitKey(0)
        #
        # if dist_b_max < 0.5 or dist_b_min < 0.5:
        #     plt.savefig(csv_root + '/' + 'result' + '/' + '{}_result'.format(csv_class) + '/' + 'defect' + '/' + os.path.splitext(os.path.split(filePath)[-1])[-2] + '.png')
        # else:
        #     plt.savefig(csv_root + '/' + 'result' + '/' + '{}_result'.format(csv_class) + '/' + 'good' + '/' + os.path.splitext(os.path.split(filePath)[-1])[-2] + '.png')
        #
        # plt.savefig(csv_root + '/' + '{}_result'.format(csv_class) + '/' + os.path.splitext(os.path.split(filePath)[-1])[-2] + '.png')

        # print("xy_tilt_value : ", xy_tilt_value)
        # print("xy_compensation_value : ", xy_compensation_value)
        # print("center_pts : ", center_pts)
        # print(roi_idx)
        # print("dist_b_max : {}, dist_b_min : {}, offset : {}".format(dist_b_max, dist_b_min, offset))

        if dist_b_min < 0:
            defect_class = "OneSidedWelding"
        elif dist_b_min < self.defect_distance_thr and dist_b_max >= self.defect_distance_thr:
            defect_class = "BlowHole"
        elif dist_b_min < self.defect_distance_thr and dist_b_max < self.defect_distance_thr:
            defect_class = "OffWelding w/o Tab"
        else:
            defect_class = "WeldLength"

        self.roi_idx = roi_idx
        self.dist_b_max = dist_b_max
        self.dist_b_min = dist_b_min
        self.b_max = center_pts[:, 1].max()
        self.b_min = center_pts[:, 1].min()
        self.a = bottom

        return defect_class


    def __translate(self, mat, deg=0, trans = [0,0]):
        rad = np.pi * (deg / 180)
        rot = np.array([[np.cos(rad), -np.sin(rad), trans[0]], [np.sin(rad), np.cos(rad), trans[1]], [0, 0, 1]])
        xyz_value = np.dot(rot, mat)

        return xyz_value


    def __use_DBSCAN(self, xy_value, eps=0.08, min_samples=15):

        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(xy_value)
        cluster_idx = model.fit_predict(xy_value)

        # 이상치 번호는 -1, 클러스터 최대 숫자까지 iteration
        return cluster_idx


    def __extract_idxes(self, xy_tilt_value, bottom_left, bottom_right, threshold=0.4):
        bottom_idx = [0 for i in range(xy_tilt_value.shape[0])]
        line_eq = np.dot(np.linalg.inv([[bottom_left[0], 1], [bottom_right[0], 1]] + np.identity(2)*0.0001), [bottom_left[1], bottom_right[1]])
        for idx in range(xy_tilt_value.shape[0]):
            dist = abs(line_eq[0] * xy_tilt_value[idx, 0] - xy_tilt_value[idx, 1] + line_eq[1]) / math.sqrt(pow(line_eq[0], 2) + 1)
            if dist > threshold:
                continue
            bottom_idx[idx] = 1

        return bottom_idx

    def __extract_points(self, xy_compensation_value, idxes, find=2):
        tmp = []
        for idx in range(len(idxes)):
            if idxes[idx] != find:
                continue
            tmp.append(idx)
        points = xy_compensation_value[tmp, :]
        return points