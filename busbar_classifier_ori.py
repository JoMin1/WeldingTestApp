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

def main():
    count = 0
    csv_root = r'D:\sample\톱텍\불량 데이터'
    csv_classes = ['1-1', '1-2', '1-3', '1-4', '2-1', '2-2', '2-3', '2-4', '3-1', '3-2', '3-3', '3-4', '4-1', '4-2',
                   '4-3', '4-4', '5-1', '5-2', '5-3', '5-4']
    csv_classes = ['1-1', '1-2', '1-3', '1-4', '2-1', '2-2', '2-3', '2-4', '3-1', '3-2', '3-3', '3-4', '4-1', '4-2',
                   '4-3', '4-4', '5-2', '5-3', '5-4']
    for csv_class in csv_classes:
        csv_path = os.path.join(csv_root, csv_class)
        fileloader = Fileloader(csv_path, 'csv')
        os.makedirs(csv_root + '/' + 'result' + '/' + '{}_result'.format(csv_class) + '/' + 'defect', exist_ok=True)
        os.makedirs(csv_root + '/' + 'result' + '/' + '{}_result'.format(csv_class) + '/' + 'good', exist_ok=True)

        for filePath in fileloader.filePaths:
            print(filePath)
            # filePath = r"C:\Users\mjlee\Downloads\불량 데이터\2-1\85.csv"
            y_value = pd.read_csv(filePath, index_col=None, header=None, encoding='CP949') # index_col,header 첫번째행이 칼럼이름이라면 header=0
            x_value = []

            # normalization
            x_norm_max = 200
            @TODO : list => numpy
            for i in range(y_value.values.shape[0]):
                x_value.append(i / x_norm_max)
            x_value = np.expand_dims(np.asarray(x_value), axis=1)
            xy_value = np.concatenate([x_value, y_value], 1)

            # homogeneous transformation
            z_value = np.ones(shape=(x_value.shape[0], 1), dtype=np.int8)
            xyz_value = np.concatenate([xy_value, z_value], 1)
            for i in range(xyz_value.shape[0]):
                xyz_value[i] = translate(xyz_value[i], deg=10, trans=[0,0])
                # xyz_value[i] = translate(xyz_value[i], deg=0, trans=[+175/x_norm_max, -1.85])
            xy_tilt_value = xyz_value[:, 0:2]

            # tilt_compenstion
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

            # get degree and rotate
            bottom_left = bottom_left / a_boundary
            bottom_right = bottom_right / a_boundary
            tilt_deg = np.arctan((bottom_left[1] - bottom_right[1]) / (bottom_left[0] - bottom_right[0])) # compensation
            # print("tilt_deg", tilt_deg / np.pi * 180)

            xyz_value = np.concatenate([xy_tilt_value, z_value], 1)
            for i in range(xyz_value.shape[0]):
                xyz_value[i] = translate(xyz_value[i], deg=-tilt_deg / np.pi * 180, trans=[0,0])
                # xyz_value[i] = translate(xyz_value[i], deg=0, trans=[+175/x_norm_max, -1.85])
            xy_compensation_value = xyz_value[:, 0:2]

            # # clustering
            # eps = 0.08
            # min_samples = 15
            # cluster_idx = use_DBSCAN(xy_value, eps=0.8, min_samples=15)

            # line extractor
            bottom_idx = extract_idxes(xy_tilt_value, bottom_left, bottom_right, threshold=0.04)
            bottom_pts = extract_points(xy_compensation_value, bottom_idx, 1)
            #TODO : cluster 3개인 경우 가운데 제거
            bottom_idx1 = use_DBSCAN(bottom_pts, eps=0.2, min_samples=15)
            print(bottom_idx1)
            print(np.unique(bottom_idx1))
            bottom_pts1 = extract_points(bottom_pts, bottom_idx1, 0)
            bottom_pts2 = extract_points(bottom_pts, bottom_idx1, np.unique(bottom_idx1)[-1])

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
                    roi_idx.append(1)
                else:
                    roi_idx.append(0)

            # get min max & distancce
            center_pts = extract_points(xy_compensation_value, roi_idx, 1)
            b_max = np.array([center_pts[center_pts[:, 1].argmax(), 0] * x_norm_max, center_pts[center_pts[:, 1].argmax(), 1]])
            b_min = ([center_pts[center_pts[:,1].argmin(), 0] * x_norm_max, center_pts[center_pts[:,1].argmin(), 1]])
            bottom = np.mean(bottom_line_y)
            dist_b_max = b_max[1] - bottom
            dist_b_min = b_min[1] - bottom



            # line_eq = np.dot(np.linalg.inv([[bottom_left[0], 1], [bottom_right[0], 1]]), [bottom_left[1], bottom_right[1]])

            cluster_idx = np.expand_dims(np.asarray(cluster_idx), axis=1)
            xy_value[:, 0] = xy_value[:, 0] * x_norm_max
            xy_tilt_value[:, 0] = xy_tilt_value[:, 0] * x_norm_max
            xy_compensation_value[:, 0] = xy_compensation_value[:, 0] * x_norm_max
            xy_value = np.concatenate([xy_value, cluster_idx], 1)

            # show plot
            # fig = plt.figure(figsize=(16, 10))
            # ax = fig.add_subplot(2, 2, 1)
            # ax.scatter(xy_value[:, 0], xy_value[:, 1], c=cluster_idx, s=50, cmap='Dark2')
            # plt.title("Original")
            # plt.xlabel('sequence', size=10)
            # plt.ylabel('height(mm)', size=10)

            fig = plt.figure(figsize=(16, 10))
            ax = fig.add_subplot(2, 2, 1)
            ax.scatter(xy_tilt_value[:, 0], xy_tilt_value[:, 1], c=cluster_idx, s=20, cmap='Dark2')
            plt.title('bottom_left = {}, bottom_right = {}, a_boundary = {}'.format(bottom_left, bottom_right, a_boundary), size=10)
            plt.xlabel('sequence', size=10)
            plt.ylabel('height(mm)', size=10)

            ax = fig.add_subplot(2, 2, 2)
            ax.scatter(xy_compensation_value[:, 0], xy_compensation_value[:, 1], c=bottom_idx, s=20, cmap='Dark2')
            ax.plot(bottom_line_x1, bottom_line_y1, color='cornflowerblue', linewidth=1, label='RANSAC regressor')
            ax.plot(bottom_line_x2, bottom_line_y2, color='cornflowerblue', linewidth=1, label='RANSAC regressor')
            plt.title('left_width = {0:0.2f}, right_width = {1:0.2f}'.format(bottom_width_left, bottom_width_right),size=10)
            plt.xlabel('sequence', size=10)
            plt.ylabel('height(mm)', size=10)

            ax = fig.add_subplot(2, 2, 3)
            ax.scatter(xy_compensation_value[:, 0], xy_compensation_value[:, 1], c=roi_idx, s=20, cmap='Dark2')
            ax.scatter(b_max[0], b_max[1], c=2, s=80, cmap='spring')
            ax.scatter(b_min[0], b_min[1], c=3, s=80, cmap='gist_heat')
            ax.plot(bottom_line_x, bottom_line_y, '--', color='black', linewidth=2, label='RANSAC regressor')
            plt.title('offset = {0:0.2f},   a = {1:0.3f},  b_max = {2:0.3f},  b_min = {3:0.3f}'.format(offset, bottom, b_max[1], b_min[1]),size=10)
            plt.xlabel('sequence', size=10)
            plt.ylabel('height(mm)', size=10)

            ax = fig.add_subplot(2, 2, 4)
            ax.scatter(xy_compensation_value[:, 0], xy_compensation_value[:, 1], c=roi_idx, s=20, cmap='Dark2')
            ax.scatter(b_max[0], b_max[1], c=2, s=80, cmap='spring')
            ax.scatter(b_min[0], b_min[1], c=3, s=80, cmap='gist_heat')
            ax.plot(bottom_line_x, bottom_line_y, '--', color='black', linewidth=2, label='RANSAC regressor')
            if dist_b_max < 0.5 or dist_b_min < 0.5:
                plt.title('불량, b_max_dist = {1:0.3f},   b_min_dist = {2:0.3f}'.format(bottom, dist_b_max, dist_b_min), size=10)
            else:
                plt.title('양품, b_max_dist = {1:0.3f},   b_min_dist = {2:0.3f}'.format(bottom, dist_b_max, dist_b_min), size=10)
            plt.xlabel('sequence', size=10)
            plt.ylabel('height(mm)', size=10)
            # plt.show()
            # return

            if dist_b_max < 0.5 or dist_b_min < 0.5:
                plt.savefig(csv_root + '/' + 'result' + '/' + '{}_result'.format(csv_class) + '/' + 'defect' + '/' + os.path.splitext(os.path.split(filePath)[-1])[-2] + '.png')
            else:
                plt.savefig(csv_root + '/' + 'result' + '/' + '{}_result'.format(csv_class) + '/' + 'good' + '/' + os.path.splitext(os.path.split(filePath)[-1])[-2] + '.png')

            # plt.savefig(csv_root + '/' + '{}_result'.format(csv_class) + '/' + os.path.splitext(os.path.split(filePath)[-1])[-2] + '.png')

            count += 1
            print("count : ",  count)


def translate(mat, deg=0, trans = [0,0]):
    rad = np.pi * (deg / 180)
    rot = np.array([[np.cos(rad), -np.sin(rad), trans[0]], [np.sin(rad), np.cos(rad), trans[1]], [0, 0, 1]])
    xyz_value = np.dot(rot, mat)

    return xyz_value


def use_DBSCAN(xy_value, eps=0.08, min_samples=15):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(xy_value)
    cluster_idx = model.fit_predict(xy_value)

    # 이상치 번호는 -1, 클러스터 최대 숫자까지 iteration
    return cluster_idx


def extract_idxes(xy_tilt_value, bottom_left, bottom_right, threshold=0.4):
    bottom_idx = [0 for i in range(xy_tilt_value.shape[0])]
    line_eq = np.dot(np.linalg.inv([[bottom_left[0], 1], [bottom_right[0], 1]]), [bottom_left[1], bottom_right[1]])
    print("line_eq : ", line_eq)
    for idx in range(xy_tilt_value.shape[0]):
        dist = abs(line_eq[0] * xy_tilt_value[idx, 0] - xy_tilt_value[idx, 1] + line_eq[1]) / math.sqrt(pow(line_eq[0], 2) + 1)
        if dist > threshold:
            continue
        bottom_idx[idx] = 1

    return bottom_idx

def extract_points(xy_compensation_value, idxes, find=2):
    tmp = []
    for idx in range(len(idxes)):
        if idxes[idx] != find:
            continue
        tmp.append(idx)
    idxes = xy_compensation_value[tmp, :]
    return idxes



if __name__ == '__main__':
    main()


3

