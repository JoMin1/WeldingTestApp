import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
import copy
import math
import shutil
from tqdm import tqdm

import pyqtgraph as pg
from PySide6.QtGui import Qt
from PySide6.QtGui import QImage, QPixmap, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtWidgets
from PySide6.QtCore import QFile, QIODevice, Signal, QEvent, QObject, QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox

from busbar_classifier import Detector
from defectCounter import DefectCounter

import threading

class WeldingTestApp(threading.Thread):

    def __init__(self, window):

        self.__mainWindow = window
        self.__mainWindow.setWindowFlags(Qt.WindowTitleHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)

        self.init_data()
        self.init_connect_event()

        super().__init__()
        threading.Thread.__init__(self)

    def show(self):

        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
        self.__mainWindow.show()

    def init_data(self):
        self.srcImage_ori = None
        self.set_slit()
        self.divide_val = 360 # TODO : groupbox에서 직접 roi를 입력으로
        self.lower_offset = 300
        self.higher_offset = 400

        self.count_defect_thr = 5 # TODO : 15개 이상일때, 결함으로 보는 것이 아니었낭?
        self.start_height_thr = 0.15
        self.defect_profile_roi = []
        self.slit_cursur = [0, 0, 0, 0]
        self.defect_dist_thr = float(self.__mainWindow.lineEdit_defect_dist_thr.text())
        self.start_dist_thr = float(self.__mainWindow.lineEdit_start_dist_thr.text())
        self.detector = Detector(self.defect_dist_thr)

        # self.paint_color = {"null": (255, 150, 50), "BlowHole": (150, 255, 50),
        #                     "OneSidedWelding": (50, 150, 255), "OffWeldingWithoutTab": (50, 255, 150)}
        self.paint_color = {"null": (0, 0, 0), "BlowHole": (0, 255, 0),
                    "OneSidedWelding": (255, 0, 0), "OffWeldingWithoutTab": (255, 255, 0)}

    def init_connect_event(self):
        self.__mainWindow.pushButton_open_image.clicked.connect(self.event_clicked_pushButton_open_image)
        self.__mainWindow.pushButton_roi_prev.clicked.connect(self.event_clicked_pushButton_roi_prev)
        self.__mainWindow.pushButton_roi_next.clicked.connect(self.event_clicked_pushButton_roi_next)
        self.__mainWindow.pushButton_roi_next_play.clicked.connect(self.event_clicked_pushButton_roi_next_play)
        self.__mainWindow.listWidget.currentItemChanged.connect(self.select_changed_listwidget)

        # TODO : 중복 된 것이 많으니까 => 따로 button 생성 함수를 빼도 될 듯
        self.__mainWindow.pushButton_slit1.setStyleSheet('background-color: #f0f0f0;')
        self.__mainWindow.pushButton_slit2.setStyleSheet('background-color: #f0f0f0;')
        self.__mainWindow.pushButton_slit3.setStyleSheet('background-color: #f0f0f0;')
        self.__mainWindow.pushButton_slit4.setStyleSheet('background-color: #f0f0f0;')
        self.__mainWindow.pushButton_slit1.clicked.connect(self.event_clicked_pushButton_slit1)
        self.__mainWindow.pushButton_slit2.clicked.connect(self.event_clicked_pushButton_slit2)
        self.__mainWindow.pushButton_slit3.clicked.connect(self.event_clicked_pushButton_slit3)
        self.__mainWindow.pushButton_slit4.clicked.connect(self.event_clicked_pushButton_slit4)

        # TODO : qlineedit lineEdit_slit1_ROI_rightBottomX
        self.__mainWindow.lineEdit_slit1_ROI_leftTopX.textChanged.connect(self.event_textchanged_slit1_leftTopX)
        self.__mainWindow.lineEdit_slit1_ROI_leftTopY.textChanged.connect(self.event_textchanged_slit1_leftTopY)
        self.__mainWindow.lineEdit_slit1_ROI_rightBottomX.textChanged.connect(self.event_textchanged_slit1_rightBottomX)
        self.__mainWindow.lineEdit_slit1_ROI_rightBottomY.textChanged.connect(self.event_textchanged_slit1_rightBottomY)
        self.__mainWindow.lineEdit_slit2_ROI_leftTopX.textChanged.connect(self.event_textchanged_slit2_leftTopX)
        self.__mainWindow.lineEdit_slit2_ROI_leftTopY.textChanged.connect(self.event_textchanged_slit2_leftTopY)
        self.__mainWindow.lineEdit_slit2_ROI_rightBottomX.textChanged.connect(self.event_textchanged_slit2_rightBottomX)
        self.__mainWindow.lineEdit_slit2_ROI_rightBottomY.textChanged.connect(self.event_textchanged_slit2_rightBottomY)
        self.__mainWindow.lineEdit_slit3_ROI_leftTopX.textChanged.connect(self.event_textchanged_slit3_leftTopX)
        self.__mainWindow.lineEdit_slit3_ROI_leftTopY.textChanged.connect(self.event_textchanged_slit3_leftTopY)
        self.__mainWindow.lineEdit_slit3_ROI_rightBottomX.textChanged.connect(self.event_textchanged_slit3_rightBottomX)
        self.__mainWindow.lineEdit_slit3_ROI_rightBottomY.textChanged.connect(self.event_textchanged_slit3_rightBottomY)
        self.__mainWindow.lineEdit_slit4_ROI_leftTopX.textChanged.connect(self.event_textchanged_slit4_leftTopX)
        self.__mainWindow.lineEdit_slit4_ROI_leftTopY.textChanged.connect(self.event_textchanged_slit4_leftTopY)
        self.__mainWindow.lineEdit_slit4_ROI_rightBottomX.textChanged.connect(self.event_textchanged_slit4_rightBottomX)
        self.__mainWindow.lineEdit_slit4_ROI_rightBottomY.textChanged.connect(self.event_textchanged_slit4_rightBottomY)
        # self.QtWidgets = pg.PlotWidget()
        # self.__mainWindow.centralwidget.graphWidget(pg.PlotWidget())

    def event_clicked_any_pushButton_slit(self, n):
        clicked_roi_depth = 20
        if n==0:
            self.__mainWindow.pushButton_slit1.setStyleSheet('background-color: #929292;')
            self.__mainWindow.pushButton_slit2.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit3.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit4.setStyleSheet('background-color: #f0f0f0;')
            if self.slit_roi_detected:
                self.__paint_click_roi(self.slit_roi_detected[0], color=(0, 255, 0), depth=clicked_roi_depth)
        elif n==1:
            self.__mainWindow.pushButton_slit1.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit2.setStyleSheet('background-color: #929292;')
            self.__mainWindow.pushButton_slit3.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit4.setStyleSheet('background-color: #f0f0f0;')
            if self.slit_roi_detected:
                self.__paint_click_roi(self.slit_roi_detected[1], color=(0, 255, 0), depth=clicked_roi_depth)
        elif n==2:
            self.__mainWindow.pushButton_slit1.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit2.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit3.setStyleSheet('background-color: #929292;')
            self.__mainWindow.pushButton_slit4.setStyleSheet('background-color: #f0f0f0;')
            if self.slit_roi_detected:
                self.__paint_click_roi(self.slit_roi_detected[2], color=(0, 255, 0), depth=clicked_roi_depth)
        else:
            self.__mainWindow.pushButton_slit1.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit2.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit3.setStyleSheet('background-color: #f0f0f0;')
            self.__mainWindow.pushButton_slit4.setStyleSheet('background-color: #929292;')
            if self.slit_roi_detected:
                self.__paint_click_roi(self.slit_roi_detected[3], color=(0, 255, 0), depth=clicked_roi_depth)
        self.show_image0(self.srcImage_roi) # TODO : AttributeError 예외처리 필요

    def event_clicked_pushButton_slit1(self):
        print("====== slit1 ======")
        self.slitPushed = 0
        self.slit_cursur = [0, 0, 0, 0]
        self.event_clicked_any_pushButton_slit(self.slitPushed)
        self.__update_defectList(0)
        self.__slit_button_toggle()
    def event_clicked_pushButton_slit2(self):
        print("====== slit2 ======")
        self.slitPushed = 1
        self.slit_cursur = [0, 0, 0, 0]
        self.event_clicked_any_pushButton_slit(self.slitPushed)
        self.__update_defectList(1)
        self.__slit_button_toggle()
    def event_clicked_pushButton_slit3(self):
        print("====== slit3 ======")
        self.slitPushed = 2
        self.slit_cursur = [0, 0, 0, 0]
        self.event_clicked_any_pushButton_slit(self.slitPushed)
        self.__update_defectList(2)
        self.__slit_button_toggle()
    def event_clicked_pushButton_slit4(self):
        print("====== slit4 ======")
        self.slitPushed = 3
        self.slit_cursur = [0, 0, 0, 0]
        self.event_clicked_any_pushButton_slit(self.slitPushed)
        self.__update_defectList(3)
        self.__slit_button_toggle()
    def __slit_button_toggle(self):
        self.__mainWindow.pushButton_slit1.setChecked(self.slitPushed == 0)
        self.__mainWindow.pushButton_slit2.setChecked(self.slitPushed == 1)
        self.__mainWindow.pushButton_slit3.setChecked(self.slitPushed == 2)
        self.__mainWindow.pushButton_slit4.setChecked(self.slitPushed == 3)

    # FIXME : textchanged 된 것 반영하는 부분.. 아니 너무 복잡한데;
    def event_textchanged_slit1_leftTopX(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit1_leftTopY(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit1_rightBottomX(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit1_rightBottomY(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit2_leftTopX(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit2_leftTopY(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit2_rightBottomX(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit2_rightBottomY(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit3_leftTopX(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit3_leftTopY(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit3_rightBottomX(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit3_rightBottomY(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit4_leftTopX(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit4_leftTopY(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit4_rightBottomX(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def event_textchanged_slit4_rightBottomY(self):
        self.__update_roi_values()
        self.__repaint_roi()
    def __update_roi_values(self):
        self.points_roi[0][0] = int(float(self.__mainWindow.lineEdit_slit1_ROI_leftTopX.text()))
        self.points_roi[0][1] = int(float(self.__mainWindow.lineEdit_slit1_ROI_leftTopY.text()))
        self.points_roi[0][2] = int(float(self.__mainWindow.lineEdit_slit1_ROI_rightBottomX.text()))
        self.points_roi[0][3] = int(float(self.__mainWindow.lineEdit_slit1_ROI_rightBottomY.text()))
        self.points_roi[1][0] = int(float(self.__mainWindow.lineEdit_slit2_ROI_leftTopX.text()))
        self.points_roi[1][1] = int(float(self.__mainWindow.lineEdit_slit2_ROI_leftTopY.text()))
        self.points_roi[1][2] = int(float(self.__mainWindow.lineEdit_slit2_ROI_rightBottomX.text()))
        self.points_roi[1][3] = int(float(self.__mainWindow.lineEdit_slit2_ROI_rightBottomY.text()))
        self.points_roi[2][0] = int(float(self.__mainWindow.lineEdit_slit3_ROI_leftTopX.text()))
        self.points_roi[2][1] = int(float(self.__mainWindow.lineEdit_slit3_ROI_leftTopY.text()))
        self.points_roi[2][2] = int(float(self.__mainWindow.lineEdit_slit3_ROI_rightBottomX.text()))
        self.points_roi[2][3] = int(float(self.__mainWindow.lineEdit_slit3_ROI_rightBottomY.text()))
        self.points_roi[3][0] = int(float(self.__mainWindow.lineEdit_slit4_ROI_leftTopX.text()))
        self.points_roi[3][1] = int(float(self.__mainWindow.lineEdit_slit4_ROI_leftTopY.text()))
        self.points_roi[3][2] = int(float(self.__mainWindow.lineEdit_slit4_ROI_rightBottomX.text()))
        self.points_roi[3][3] = int(float(self.__mainWindow.lineEdit_slit4_ROI_rightBottomY.text()))
        self.slit_roi_detected = copy.deepcopy(self.points_roi)
        self.slit_roi_detected[0][1] = self.points_roi[0][1] + self.higher_offset
        self.slit_roi_detected[0][3] = self.points_roi[0][3] - self.lower_offset
        self.slit_roi_detected[1][1] = self.points_roi[1][1] + self.higher_offset
        self.slit_roi_detected[1][3] = self.points_roi[1][3] - self.lower_offset
        self.slit_roi_detected[2][1] = self.points_roi[2][1] + self.higher_offset
        self.slit_roi_detected[2][3] = self.points_roi[2][3] - self.lower_offset
        self.slit_roi_detected[3][1] = self.points_roi[3][1] + self.higher_offset
        self.slit_roi_detected[3][3] = self.points_roi[3][3] - self.lower_offset
        
    def __repaint_roi(self):
        self.srcImage_roi = copy.deepcopy(self.refImage_roi)
        for rois in self.points_roi:
            rois = [int(i) for i in rois]
            cv2.rectangle(self.srcImage_roi, (rois[0], rois[1]),
                      (rois[2], rois[3]), (100, 0, 100), 7)
            cv2.rectangle(self.srcImage_roi, (rois[0], rois[1]+self.higher_offset),
                      (rois[2], rois[3]-self.lower_offset), (0, 255, 0), 7)
        self.srcImage_roi_to_show = copy.deepcopy(self.srcImage_roi)
        self.show_image0(self.srcImage_roi)

    def event_clicked_pushButton_roi_prev(self):
        self.slit_cursur[self.slitPushed] -= 1
        step = round((self.slit_roi_detected[self.slitPushed][3] - self.slit_roi_detected[self.slitPushed][1]) / self.divide_val)
        if self.__mainWindow.listWidget.currentItem():
            cursur = int(self.__mainWindow.listWidget.currentItem().text()) + (self.slit_cursur[self.slitPushed] * step)
        else:
            cursur = self.slit_cursur[self.slitPushed] * step

        if cursur < 0:
            self.slit_cursur[self.slitPushed] += 1
            return
        self.__mainWindow.label_index_roi.setText(str(self.slit_roi_detected[self.slitPushed][1] + cursur))
        defect_class = self.detector.run(
            self.srcImage_ori[self.slit_roi_detected[self.slitPushed][1] + cursur,
            self.slit_roi_detected[self.slitPushed][0]:self.slit_roi_detected[self.slitPushed][2]])
        self.__update_profile_result(cursur, defect_class)
        self.__paint_cursur(self.slit_roi_detected[self.slitPushed], cursur)
        self.show_image1(self.srcImage_ori[self.slit_roi_detected[self.slitPushed][1] + cursur,
                         self.slit_roi_detected[self.slitPushed][0]:self.slit_roi_detected[self.slitPushed][2]],
                         self.detector.roi_idx)

    def event_clicked_pushButton_roi_next(self):
        self.slit_cursur[self.slitPushed] += 1
        step = round((self.slit_roi_detected[self.slitPushed][3] - self.slit_roi_detected[self.slitPushed][1]) / self.divide_val)
        if self.__mainWindow.listWidget.currentItem():
            cursur = int(self.__mainWindow.listWidget.currentItem().text()) + (self.slit_cursur[self.slitPushed] * step)
        else:
            cursur = self.slit_cursur[self.slitPushed] * step

        if cursur >= self.slit_roi_detected[self.slitPushed][3] - self.slit_roi_detected[self.slitPushed][1]:
            self.slit_cursur[self.slitPushed] -= 1
            return

        self.__mainWindow.label_index_roi.setText(str(self.slit_roi_detected[self.slitPushed][1] + cursur))
        defect_class = self.detector.run(
            self.srcImage_ori[self.slit_roi_detected[self.slitPushed][1] + cursur,
            self.slit_roi_detected[self.slitPushed][0]:self.slit_roi_detected[self.slitPushed][2]])
        self.__update_profile_result(cursur, defect_class)
        self.__paint_cursur(self.slit_roi_detected[self.slitPushed], cursur)
        self.show_image1(self.srcImage_ori[self.slit_roi_detected[self.slitPushed][1] + cursur,
                         self.slit_roi_detected[self.slitPushed][0]:self.slit_roi_detected[self.slitPushed][2]],
                         self.detector.roi_idx)

    def __update_profile_result(self, cursur, defect_class):
        self.__mainWindow.label_index_roi.setText(str(self.slit_roi_detected[self.slitPushed][1] + cursur))
        self.__mainWindow.lineEdit_defect_class.setText(defect_class)
        self.__mainWindow.lineEdit_Point_a.setText("{0:.3f}".format(self.detector.a))
        self.__mainWindow.lineEdit_Point_b_max.setText("{0:.3f}".format(self.detector.b_max))
        self.__mainWindow.lineEdit_Point_b_min.setText("{0:.3f}".format(self.detector.b_min))
        self.__mainWindow.lineEdit_distance_max.setText("{0:.3f}".format(self.detector.dist_b_max))
        self.__mainWindow.lineEdit_distance_min.setText("{0:.3f}".format(self.detector.dist_b_min))

    def __paint_cursur(self, slit_roi_detected, cursur=0, color=(0, 0, 255), thickness=6):
        tmp = cv2.line(copy.deepcopy(self.srcImage_roi), (slit_roi_detected[0], slit_roi_detected[1] + cursur),
                                     (slit_roi_detected[2], slit_roi_detected[1] + cursur), color, thickness)

        self.show_image0(tmp)
    def __paint_cursur_accumulate(self, slit_roi_detected, cursur=0, color=(0, 0, 255), thickness=6):
        cv2.line(self.srcImage_roi, (slit_roi_detected[0], slit_roi_detected[1] + cursur),
                 (slit_roi_detected[2], slit_roi_detected[1] + cursur), color, thickness)

    def event_clicked_pushButton_roi_next_play(self):
        print("play")
        # TODO : 이미 파일이 있을 시엔 전부 삭제해주고 다시 저장하게 함
        if os.path.isdir(self.saveFilePath):
            dir_list = os.listdir(self.saveFilePath)
            for d in dir_list:
                remove_dir = os.path.join(self.saveFilePath, d)
                shutil.rmtree(remove_dir)
                os.mkdir(remove_dir)
        self.srcImage_roi = copy.deepcopy(self.srcImage_roi_to_show)

        self.defect_dist_thr = float(self.__mainWindow.lineEdit_defect_dist_thr.text())
        self.start_dist_thr = float(self.__mainWindow.lineEdit_start_dist_thr.text())
        self.count_weldLength_thr = int(self.__mainWindow.lineEdit_WeldLength_Count.text())
        self.count_BlowHole_thr = int(self.__mainWindow.lineEdit_BlowHole_Count.text())
        self.count_OneSidedWelding_thr = int(self.__mainWindow.lineEdit_OneSidedWelding_Count.text())
        self.count_OffWeldingWithoutTab_thr = int(self.__mainWindow.lineEdit_OffWeldingWithoutTab_Count.text())

        for idx,i in enumerate(tqdm(range(len(self.slit_roi_detected)))):
            self.defectCounter = DefectCounter(int(self.__mainWindow.lineEdit_WeldLength_Length.text()),
                                               int(self.__mainWindow.lineEdit_BlowHole_Length.text()),
                                               int(self.__mainWindow.lineEdit_OneSidedWelding_Length.text()),
                                               int(self.__mainWindow.lineEdit_OffWeldingWithoutTab_Length.text()))
            slit_cursur = 0
            # FIXME : self.divide_val 에 대하여.. 꼭 360인가? 꼭 360으로 나눠야하는 이유는?
            step = round((self.slit_roi_detected[i][3] - self.slit_roi_detected[i][1]) / self.divide_val)
            while slit_cursur < (self.divide_val - 1):
                tic = time.time()
                if slit_cursur >= (self.divide_val - 1):
                    break
                slit_cursur += 1
                defect_class = self.detector.run(
                    self.srcImage_ori[self.slit_roi_detected[i][1] + int(slit_cursur * step),
                    self.slit_roi_detected[i][0]:self.slit_roi_detected[i][2]])

                self.defectCounter.run(defect_class)

                if not defect_class == "WeldLength":
                    heightData_df = pd.DataFrame(
                        self.srcImage_ori[self.slit_roi_detected[i][1] + int(slit_cursur * step),
                        self.slit_roi_detected[i][0]:self.slit_roi_detected[i][2]])
                    heightData_df.to_csv(
                        self.saveFilePath + "/{}/{}".format(i, int(slit_cursur * step)) + ".csv", header=False,
                        index=False)
                    #TODO : self.__paint_cursur, 색상을 더해서 반투명하게 만들기

                    self.__paint_cursur_accumulate(self.slit_roi_detected[i], int(slit_cursur * step), color=self.paint_color[defect_class])

            self.__update_slit_result(i)
        self.__update_defectList(0)
        self.show_image0(self.srcImage_roi)
        self.srcImage_roi_to_show = copy.deepcopy(self.srcImage_roi)
        self.event_clicked_pushButton_slit1()

    def __update_defectList(self, slit):
        self.current_selected_file_path = os.path.join(self.saveFilePath, "{}".format(slit))
        print("self.current_selected_file_path : ", self.current_selected_file_path)
        files = os.listdir(self.current_selected_file_path)
        files = [int(x.split('.')[0]) for x in files]
        files = [str(x) for x in sorted(files)]
        self.__mainWindow.listWidget.clear()

        count = 0
        for file in files:
            self.__mainWindow.listWidget.addItem(file)
            count += 1

    def select_changed_listwidget(self):
        if self.__mainWindow.listWidget.currentItem():
            self.current_selected_file = os.path.join(self.saveFilePath, "{}".format(self.slitPushed), self.__mainWindow.listWidget.currentItem().text() + ".csv")
            doc = pd.read_csv(self.current_selected_file, encoding='utf-8-sig', on_bad_lines='skip') # TODO : error_bad_lines에서 on_bad_lines 로 바뀜 
            defect_class = self.detector.run(np.squeeze(np.array(doc), axis=1))

            self.slit_cursur = [0, 0, 0, 0]
            self.__update_profile_result(int(self.__mainWindow.listWidget.currentItem().text()), defect_class)
            self.__paint_cursur(self.slit_roi_detected[self.slitPushed], int(self.__mainWindow.listWidget.currentItem().text()))
            self.show_image1(np.squeeze(np.array(doc), axis=1), self.detector.roi_idx)
            print(int(self.__mainWindow.listWidget.currentItem().text()), self.slitPushed)

    def event_clicked_pushButton_open_image(self):
        self.init_data()
        self.srcImage = self.show_fileDialog()
        if self.srcImage is None:
            return
        # TODO : 무조건 3차원으로 만듦
        # TODO : 만약 오른쪽 Roi 부분이 모두 값이 있다고하면, __get_roi 안들리고 바로 파일만 열어주면 되는건가?
        if len(self.srcImage.shape) == 2:
            self.srcImage_roi = cv2.cvtColor(self.srcImage, cv2.COLOR_GRAY2BGR)
        else:
            self.srcImage_roi = self.srcImage
        self.refImage_roi = copy.deepcopy(self.srcImage_roi)  # 순수하게 읽은 이미지를 ref로 저장
        for i in range(len(self.points_roi)): # points_roi_detected
            self.slit_roi_detected.append(self.__get_roi(copy.deepcopy(self.points_roi[i]))) # TODO : slit_roi_detected  정의 부분
        # self.event_clicked_pushButton_roi_next()
        self.srcImage_roi_to_show = copy.deepcopy(self.srcImage_roi)
        self.event_clicked_pushButton_slit1()
        self.show_image0(self.srcImage_roi)
        self.__mainWindow.listWidget.clear()

    def show_fileDialog(self):
        fileName = QtWidgets.QFileDialog.getOpenFileName(self.__mainWindow, 'Open file', 'D:/sample/busbar', "Image files (*.jpg *.png *.csv *.tif)")
        print(fileName)
        if fileName[0] == "":
            return None
        if fileName[0].split('.')[1] == "csv":
            srcImage = np.loadtxt(fileName[0], delimiter=',')
            self.saveFilePath = fileName[0].split('.')[0]
            if not os.path.exists(self.saveFilePath):
                os.makedirs(self.saveFilePath + r"\0")
                os.makedirs(self.saveFilePath + r"\1")
                os.makedirs(self.saveFilePath + r"\2")
                os.makedirs(self.saveFilePath + r"\3")
        else:
            srcImage = cv2.imread(fileName[0], cv2.IMREAD_UNCHANGED) # FIXME : imread_unchanged : alpha chennel 까지 포함하여 읽음..

        if srcImage is None:
            print("There is no data")
            return

        self.srcImage_ori = srcImage.copy()
        print(srcImage.shape)

        if srcImage.dtype == "float64":
            srcImage = np.where(srcImage < -10, 0, srcImage)
            #TODO : normalization 제거
            srcImage = (((srcImage - np.min(srcImage)) / (np.max(srcImage) - np.min(srcImage))) * 255).astype(np.uint8)
            # srcImage = (((srcImage + 10) / (10 - (-3))) * 255).astype(np.uint8)
        return srcImage

    def __get_roi(self, points_roi):
        divide_val = points_roi[3] - points_roi[1]
        points_roi_detected = copy.deepcopy(points_roi)
        # 상부 용접 시작 위치 획득
        index_roi = 0
        count_over_thr = 0
        while index_roi < (divide_val - 1):
            if index_roi >= (divide_val - 1): # while문 안에있는데 여기 들어갈수 있나?
                break
            index_roi += 1

            step = round((points_roi[3] - points_roi[1]) / divide_val) # FIXME : 당연히 1 인 식..?
            print(self.srcImage[points_roi[1] + (index_roi * step)])
            _ = self.detector.run(self.srcImage[points_roi[1] + (index_roi * step), points_roi[0]:points_roi[2]]) # 단면하나를 보냄(1차 배열)
            if self.detector.dist_b_max == -1:
                continue
            print(self.detector.dist_b_max, self.start_dist_thr)
            if self.detector.dist_b_max >= self.start_dist_thr:
                count_over_thr += 1
                if count_over_thr == 5:
                    top_y = points_roi[1] + (index_roi * step) + int(self.__mainWindow.lineEdit_ROI_offset_upper.text())
                    break
            else:
                count_over_thr = 0

        # 하부 용접 시작 위치 획득
        index_roi = divide_val - 1
        count_over_thr = 0
        while index_roi >= 0:
            tic = time.time()
            if index_roi < 0:
                break
            index_roi -= 1

            step = round((points_roi[3] - points_roi[1]) / divide_val)
            _ = self.detector.run(self.srcImage[points_roi[1] + (index_roi * step), points_roi[0]:points_roi[2]])
            if self.detector.dist_b_max == -1:
                continue
            if self.detector.dist_b_max >= self.start_dist_thr:
                count_over_thr += 1
                if count_over_thr == 5:
                    bottom_y = points_roi[1] + (index_roi * step) - int(self.__mainWindow.lineEdit_ROI_offset_lower.text())
                    break
            else:
                count_over_thr = 0

        points_roi_detected[1] = top_y
        points_roi_detected[3] = bottom_y
        # TODO : 새로운 ROI를 쏴주는 부분
        self.__paint_roi(points_roi, color=(100, 0, 100), depth=7) # 보라색
        self.__paint_roi(points_roi_detected, color=(0, 255, 0), depth=7) # 초록색
        return points_roi_detected

    def __paint_roi(self, slit_roi, color, depth=7):
        cv2.rectangle(self.srcImage_roi, (slit_roi[0], slit_roi[1]),
                      (slit_roi[2], slit_roi[3]), color, depth)
    
    def __paint_click_roi(self, slit_roi, color, depth=7):
        self.srcImage_roi = copy.deepcopy(self.srcImage_roi_to_show)
        cv2.rectangle(self.srcImage_roi, (slit_roi[0], slit_roi[1]),
                      (slit_roi[2], slit_roi[3]), color, depth)


    def show_image0(self, srcImage_roi):
        image = QImage(srcImage_roi, srcImage_roi.shape[1], srcImage_roi.shape[0], srcImage_roi.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(image).scaled(QSize(self.__mainWindow.label_image0.width(), self.__mainWindow.label_image0.height()), aspectMode=Qt.KeepAspectRatio)
        self.__mainWindow.label_image0.setPixmap(pixmap)

    def show_image1(self, heightData, roi_idx):
        heightDataTemp = np.where(heightData < -9.5, 0, heightData)
        min_heightData = np.min(heightDataTemp)
        heightData_ = heightDataTemp + abs(min_heightData)
        heightData_ = np.round(abs(heightData_), 2) * 300 + 100

        # zeroImage = np.ones((1100, len(heightData_), 3), np.uint8) * 255
        zeroImage = np.ones((3200, len(heightData_), 3), np.uint8) * 255
        for index in range(len(heightData_)):
            height = int(heightData_[index])
            zeroImage[height - 4:height, index] = (255 * roi_idx[index], 100, 100 * roi_idx[index])
        # zeroImage[int(heightData_[max_idx])-6:int(heightData_[max_idx])+2, max_idx-1:max_idx+2] = (0, 150, 0)
        # zeroImage[int(heightData_[min_idx])-6:int(heightData_[min_idx])+2, min_idx-1:min_idx+2] = (0, 150, 0)
        zeroImage_flip = cv2.flip(zeroImage, 0)

        image = QImage(zeroImage_flip, zeroImage_flip.shape[1], zeroImage_flip.shape[0], zeroImage_flip.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(image).scaled(
            QSize(self.__mainWindow.label_image1.width(), self.__mainWindow.label_image1.height()),
            aspectMode=Qt.IgnoreAspectRatio)
        self.__mainWindow.label_image1.setPixmap(pixmap)

    def set_slit(self):
        # TODO : 입력을 받는 위치 lineEdit_slit1_ROI_leftTopY
        self.points_roi = [[int(self.__mainWindow.lineEdit_slit1_ROI_leftTopX.text()),
                                   int(self.__mainWindow.lineEdit_slit1_ROI_leftTopY.text()),
                                   int(self.__mainWindow.lineEdit_slit1_ROI_rightBottomX.text()),
                                   int(self.__mainWindow.lineEdit_slit1_ROI_rightBottomY.text())],
                                   [int(self.__mainWindow.lineEdit_slit2_ROI_leftTopX.text()),
                                   int(self.__mainWindow.lineEdit_slit2_ROI_leftTopY.text()),
                                   int(self.__mainWindow.lineEdit_slit2_ROI_rightBottomX.text()),
                                   int(self.__mainWindow.lineEdit_slit2_ROI_rightBottomY.text())],
                                   [int(self.__mainWindow.lineEdit_slit3_ROI_leftTopX.text()),
                                   int(self.__mainWindow.lineEdit_slit3_ROI_leftTopY.text()),
                                   int(self.__mainWindow.lineEdit_slit3_ROI_rightBottomX.text()),
                                   int(self.__mainWindow.lineEdit_slit3_ROI_rightBottomY.text())],
                                   [int(self.__mainWindow.lineEdit_slit4_ROI_leftTopX.text()),
                                   int(self.__mainWindow.lineEdit_slit4_ROI_leftTopY.text()),
                                   int(self.__mainWindow.lineEdit_slit4_ROI_rightBottomX.text()),
                                   int(self.__mainWindow.lineEdit_slit4_ROI_rightBottomY.text())]]

        # self.points_roi와 같은 사이즈 선언
        self.slit_roi_detected = []
        self.slit_cursur = [0, 0, 0, 0]

    def __update_slit_result(self, idx):
        color = """
        color: rgb(50, 50, 50);
        selection-background-color: rgb(255, 255, 255);
        background-color: rgb(255, 255, 100);
        font: 10pt Arial;
        border: 1px solid rgb(100, 100, 100);
        border-radius: 1px;"""

        if idx == 0:
            self.__mainWindow.lineEdit_Slit1_WeldLength_Result.setText(str(self.defectCounter.count_weldLength))
            self.__mainWindow.lineEdit_Slit1_BlowHole_Result.setText(str(self.defectCounter.count_blowHole))
            self.__mainWindow.lineEdit_Slit1_OneSidedWelding_Result.setText(str(self.defectCounter.count_oneSidedWelding))
            self.__mainWindow.lineEdit_Slit1_OffWeldingWithoutTab_Result.setText(str(self.defectCounter.count_offWeldingWithoutTab))
            if self.count_weldLength_thr <= self.defectCounter.count_weldLength:
                self.__mainWindow.lineEdit_Slit1_WeldLength_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit1_WeldLength_Result.setStyleSheet("")
            if self.count_BlowHole_thr <= self.defectCounter.count_blowHole:
                self.__mainWindow.lineEdit_Slit1_BlowHole_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit1_BlowHole_Result.setStyleSheet("")
            if self.count_OneSidedWelding_thr <= self.defectCounter.count_oneSidedWelding:
                self.__mainWindow.lineEdit_Slit1_OneSidedWelding_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit1_OneSidedWelding_Result.setStyleSheet("")
            if self.count_OffWeldingWithoutTab_thr <= self.defectCounter.count_offWeldingWithoutTab:
                self.__mainWindow.lineEdit_Slit1_OffWeldingWithoutTab_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit1_OffWeldingWithoutTab_Result.setStyleSheet("")
        elif idx == 1:
            self.__mainWindow.lineEdit_Slit2_WeldLength_Result.setText(str(self.defectCounter.count_weldLength))
            self.__mainWindow.lineEdit_Slit2_BlowHole_Result.setText(str(self.defectCounter.count_blowHole))
            self.__mainWindow.lineEdit_Slit2_OneSidedWelding_Result.setText(str(self.defectCounter.count_oneSidedWelding))
            self.__mainWindow.lineEdit_Slit2_OffWeldingWithoutTab_Result.setText(str(self.defectCounter.count_offWeldingWithoutTab))
            if self.count_weldLength_thr <= self.defectCounter.count_weldLength:
                self.__mainWindow.lineEdit_Slit2_WeldLength_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit2_WeldLength_Result.setStyleSheet("")
            if self.count_BlowHole_thr <= self.defectCounter.count_blowHole:
                self.__mainWindow.lineEdit_Slit2_BlowHole_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit2_BlowHole_Result.setStyleSheet("")
            if self.count_OneSidedWelding_thr <= self.defectCounter.count_oneSidedWelding:
                self.__mainWindow.lineEdit_Slit2_OneSidedWelding_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit2_OneSidedWelding_Result.setStyleSheet("")
            if self.count_OffWeldingWithoutTab_thr <= self.defectCounter.count_offWeldingWithoutTab:
                self.__mainWindow.lineEdit_Slit2_OffWeldingWithoutTab_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit2_OffWeldingWithoutTab_Result.setStyleSheet("")
        elif idx == 2:
            self.__mainWindow.lineEdit_Slit3_WeldLength_Result.setText(str(self.defectCounter.count_weldLength))
            self.__mainWindow.lineEdit_Slit3_BlowHole_Result.setText(str(self.defectCounter.count_blowHole))
            self.__mainWindow.lineEdit_Slit3_OneSidedWelding_Result.setText(str(self.defectCounter.count_oneSidedWelding))
            self.__mainWindow.lineEdit_Slit3_OffWeldingWithoutTab_Result.setText(str(self.defectCounter.count_offWeldingWithoutTab))
            if self.count_weldLength_thr <= self.defectCounter.count_weldLength:
                self.__mainWindow.lineEdit_Slit3_WeldLength_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit3_WeldLength_Result.setStyleSheet("")
            if self.count_BlowHole_thr <= self.defectCounter.count_blowHole:
                self.__mainWindow.lineEdit_Slit3_BlowHole_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit3_BlowHole_Result.setStyleSheet("")
            if self.count_OneSidedWelding_thr <= self.defectCounter.count_oneSidedWelding:
                self.__mainWindow.lineEdit_Slit3_OneSidedWelding_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit3_OneSidedWelding_Result.setStyleSheet("")
            if self.count_OffWeldingWithoutTab_thr <= self.defectCounter.count_offWeldingWithoutTab:
                self.__mainWindow.lineEdit_Slit3_OffWeldingWithoutTab_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit3_OffWeldingWithoutTab_Result.setStyleSheet("")
        else:
            self.__mainWindow.lineEdit_Slit4_WeldLength_Result.setText(str(self.defectCounter.count_weldLength))
            self.__mainWindow.lineEdit_Slit4_BlowHole_Result.setText(str(self.defectCounter.count_blowHole))
            self.__mainWindow.lineEdit_Slit4_OneSidedWelding_Result.setText(str(self.defectCounter.count_oneSidedWelding))
            self.__mainWindow.lineEdit_Slit4_OffWeldingWithoutTab_Result.setText(str(self.defectCounter.count_offWeldingWithoutTab))
            if self.count_weldLength_thr <= self.defectCounter.count_weldLength:
                self.__mainWindow.lineEdit_Slit4_WeldLength_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit4_WeldLength_Result.setStyleSheet("")
            if self.count_BlowHole_thr <= self.defectCounter.count_blowHole:
                self.__mainWindow.lineEdit_Slit4_BlowHole_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit4_BlowHole_Result.setStyleSheet("")
            if self.count_OneSidedWelding_thr <= self.defectCounter.count_oneSidedWelding:
                self.__mainWindow.lineEdit_Slit4_OneSidedWelding_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit4_OneSidedWelding_Result.setStyleSheet("")
            if self.count_OffWeldingWithoutTab_thr <= self.defectCounter.count_offWeldingWithoutTab:
                self.__mainWindow.lineEdit_Slit4_OffWeldingWithoutTab_Result.setStyleSheet(color)
            else:
                self.__mainWindow.lineEdit_Slit4_OffWeldingWithoutTab_Result.setStyleSheet("")




    def plot(self, heightData):
        # self.graphWidget.setBackground('w')
        self.__mainWindow.graphWidget.plot([i for i in range(len(heightData))], heightData)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)

    ui_file_name = "Resources/mainWindow.ui"
    ui_file = QFile(ui_file_name)
    if not ui_file.open(QIODevice.ReadOnly):
        print(f"Cannot open {ui_file_name}: {ui_file.errorString()}")
        sys.exit(-1)
    loader = QUiLoader()
    window = loader.load(ui_file)
    ui_file.close()
    if not window:
        print(loader.errorString())
        sys.exit(-1)

    weldingTestApp = WeldingTestApp(window)
    weldingTestApp.show()

    sys.exit(app.exec())