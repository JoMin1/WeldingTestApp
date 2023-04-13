import sys
import os
import time
import cv2
import numpy as np
import pandas as pd

import pyqtgraph as pg
from PySide6.QtGui import Qt
from PySide6.QtGui import QImage, QPixmap, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication, QStyleFactory, QFileDialog
from PySide6.QtCore import QFile, QIODevice, Signal, QEvent, QObject, QSize

from busbar_classifier import detect



class WeldingTestApp:

    def __init__(self, window):

        self.__mainWindow = window
        self.__mainWindow.setWindowFlags(Qt.WindowTitleHint | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)

        self.init_data()
        self.init_connect_event()

        super().__init__()

    def show(self):

        QApplication.setStyle(QStyleFactory.create('Fusion'))
        self.__mainWindow.show()

    def init_data(self):

        self.__srcImage = None
        self.set__points_roi()
        self.__index_roi = 0
        self.__divide_val = 360

        self.count_defect = 0
        self.count_defect_thr = 15
        self.defect_profile_roi = []

        self.start_height_thr = 0.15

    def init_connect_event(self):

        self.__mainWindow.pushButton_open_image.clicked.connect(self.event_clicked_pushButton_open_image)
        self.__mainWindow.pushButton_roi_prev.clicked.connect(self.event_clicked_pushButton_roi_prev)
        self.__mainWindow.pushButton_roi_next.clicked.connect(self.evnet_clicked_pushButton_roi_next)
        self.__mainWindow.pushButton_roi_next_play.clicked.connect(self.evnet_clicked_pushButton_roi_next_play)
        self.__mainWindow.listWidget.currentItemChanged.connect(self.select_changed_listwidget)

    def event_clicked_pushButton_roi_prev(self):
        if self.__index_roi <= 0:
            return

        self.__index_roi -= 1
        self.__mainWindow.label_index_roi.setText(str(self.__points_roi[1] + self.__index_roi))

        step = round((self.__points_roi[3] - self.__points_roi[1]) / self.__divide_val)
        # zeroImage, isDefect = self.detect(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
        #                         self.__points_roi[0]:self.__points_roi[2]],
        #                         self.__points_roi[1] + (self.__index_roi * step))
        roi_idx, dist_b_max, dist_b_min, max_idx, min_idx = detect(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                                                     self.__points_roi[0]:self.__points_roi[2]])

        self.show_image0(self.srcImage, self.__index_roi * step)
        self.show_image1(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                         self.__points_roi[0]:self.__points_roi[2]], roi_idx, max_idx, min_idx)

    def evnet_clicked_pushButton_roi_next(self):
        if self.__index_roi >= (self.__divide_val - 1):
            return

        self.__index_roi += 1
        self.__mainWindow.label_index_roi.setText(str(self.__points_roi[1] + self.__index_roi))

        step = round((self.__points_roi[3] - self.__points_roi[1]) / self.__divide_val)
        roi_idx, dist_b_max, dist_b_min, max_idx, min_idx = detect(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                                                     self.__points_roi[0]:self.__points_roi[2]])

        self.show_image0(self.srcImage, self.__index_roi * step)
        self.show_image1(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                         self.__points_roi[0]:self.__points_roi[2]], roi_idx, max_idx, min_idx)

    def evnet_clicked_pushButton_roi_next_play(self):
        self.__index_roi = 0
        while self.__index_roi < (self.__divide_val - 1):
            tic = time.time()
            if self.__index_roi >= (self.__divide_val - 1):
                break
            self.__index_roi += 1

            step = round((self.__points_roi[3] - self.__points_roi[1]) / self.__divide_val)
            roi_idx, dist_b_max, dist_b_min, max_idx, min_idx = detect(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                                                      self.__points_roi[0]:self.__points_roi[2]])

            if dist_b_min < 0.5:
                heightData_df = pd.DataFrame(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                                             self.__points_roi[0]:self.__points_roi[2]])
                heightData_df.to_csv(self.saveFilePath + "/{}".format(
                    self.__points_roi[1] + (self.__index_roi * step)) + ".csv", header = False, index = False)
                isDefect = True
            else:
                isDefect = False

            if isDefect:
                self.count_defect += 1

            else:
                if self.count_defect >= self.count_defect_thr:
                    self.defect_profile_roi.append(
                        [self.__points_roi[0], self.__points_roi[1] + (self.__index_roi * step) - self.count_defect,
                         self.__points_roi[2], self.__points_roi[1] + (self.__index_roi * step)])
                self.count_defect = 0
            toc = time.time()
            print("total time : ", toc - tic)

        if self.count_defect >= self.count_defect_thr:
            self.defect_profile_roi.append(
                [self.__points_roi[0], self.__points_roi[3] - self.count_defect,
                 self.__points_roi[2], self.__points_roi[3]])
        self.count_defect = 0

        self.__update_defectList()
        self.show_image0(self.srcImage, 0)
        self.show_image1(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                         self.__points_roi[0]:self.__points_roi[2]], roi_idx, max_idx, min_idx)

    def __update_defectList(self):
        self.current_selected_file_path = os.path.join(self.saveFilePath)
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
            self.current_selected_file = os.path.join(self.saveFilePath, self.__mainWindow.listWidget.currentItem().text() + ".csv")
            doc = pd.read_csv(self.current_selected_file, encoding='utf-8-sig', error_bad_lines=False)
            roi_idx, dist_b_max, dist_b_min, max_idx, min_idx = detect(np.squeeze(np.array(doc), axis=1))

            self.show_image0(self.srcImage, int(self.__mainWindow.listWidget.currentItem().text()) - self.__points_roi[1])
            self.show_image1(np.squeeze(np.array(doc), axis=1), roi_idx, max_idx, min_idx)
            self.__index_roi = int(self.__mainWindow.listWidget.currentItem().text()) - self.__points_roi[1]

    def set__points_roi(self):
        self.__points_roi = [int(self.__mainWindow.lineEdit_ROI_leftTopX_2.text()),
                             int(self.__mainWindow.lineEdit_ROI_leftTopY_3.text()),
                             int(self.__mainWindow.lineEdit_ROI_rightBottomX_2.text()),
                             int(self.__mainWindow.lineEdit_ROI_rightBottomY_2.text())]

    def event_clicked_pushButton_open_image(self):
        self.init_data()
        self.srcImage = self.show_fileDialog()

        if self.srcImage is None:
            return

        self.__divide_val = self.__points_roi[3] - self.__points_roi[1]
        self.get_roi()
        self.__divide_val = self.__points_roi[3] - self.__points_roi[1]
        self.evnet_clicked_pushButton_roi_next()
        self.__mainWindow.listWidget.clear()

    def show_fileDialog(self):

        fileName = QFileDialog.getOpenFileName(self.__mainWindow, 'Open file', 'C:/Users/mjlee/Downloads/BUSBAR 3D DATA (1)', "Image files (*.jpg *.png *.csv)")
        print(fileName)

        if fileName[0] == "":
            return None

        if fileName[0].split('.')[1] == "csv":
            srcImage = np.loadtxt(fileName[0], delimiter=',')
            self.saveFilePath = fileName[0].split('.')[0]
            if not os.path.exists(self.saveFilePath):
                os.makedirs(self.saveFilePath)

        else:
            srcImage = cv2.imread(fileName[0], cv2.IMREAD_UNCHANGED)

        if srcImage is None:
            return

        self.__srcImage = srcImage.copy()
        self.__index_roi = 0

        if srcImage.dtype == "float64":
            srcImage = np.where(srcImage < -3, 0, srcImage)
            #TODO : normalization 제거
            # srcImage = (((srcImage - np.min(srcImage)) / (np.max(srcImage) - np.min(srcImage))) * 255).astype(np.uint8)
            srcImage = (((srcImage + 3) / (3 + 3)) * 255).astype(np.uint8)

        return srcImage

    def show_image0(self, srcImage, profile_posi=0):
        if len(srcImage.shape) == 2:
            tempImage = cv2.cvtColor(srcImage, cv2.COLOR_GRAY2BGR)
        else:
            tempImage = srcImage
        tempImage = cv2.rectangle(tempImage, (self.__points_roi_ori[0], self.__points_roi_ori[1]),
                                  (self.__points_roi_ori[2], self.__points_roi_ori[3]), (100, 0, 100), 7)
        tempImage = cv2.rectangle(tempImage, (self.__points_roi[0], self.__points_roi[1]),
                      (self.__points_roi[2], self.__points_roi[3]), (0, 255, 0), 7)

        for p in self.defect_profile_roi:
            tempImage = cv2.rectangle(tempImage, (p[0], p[1]), (p[2], p[3]), (255, 200, 100), 6)

        tempImage = cv2.line(tempImage, (self.__points_roi[0], self.__points_roi[1] + profile_posi),
                             (self.__points_roi[2], self.__points_roi[1] + profile_posi), (0, 0, 255), 6)

        image = QImage(tempImage, tempImage.shape[1], tempImage.shape[0], tempImage.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(image).scaled(QSize(self.__mainWindow.label_image0.width(), self.__mainWindow.label_image0.height()), aspectMode=Qt.KeepAspectRatio)
        self.__mainWindow.label_image0.setPixmap(pixmap)

    def show_image1(self, heightData, roi_idx, max_idx, min_idx):
        heightDataTemp = np.where(heightData < -9.5, 0, heightData)
        min_heightData = np.min(heightDataTemp)
        heightData_ = heightDataTemp + abs(min_heightData)
        heightData_ = np.round(abs(heightData_), 2) * 300 + 100

        zeroImage = np.zeros((1100, len(heightData_), 3), np.uint8)
        for index in range(len(heightData_)):
            height = int(heightData_[index])
            zeroImage[height - 4:height, index] = (255, 255 * roi_idx[index], 255 * roi_idx[index])
        print("max_idx, min_idx : ", max_idx, min_idx)
        zeroImage[int(heightData_[max_idx])-6:int(heightData_[max_idx])+2, max_idx-1:max_idx+2] = (0, 150, 0)
        zeroImage[int(heightData_[min_idx])-6:int(heightData_[min_idx])+2, min_idx-1:min_idx+2] = (0, 150, 0)
        zeroImage_flip = cv2.flip(zeroImage, 0)

        image = QImage(zeroImage_flip, zeroImage_flip.shape[1], zeroImage_flip.shape[0], zeroImage_flip.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(image).scaled(
            QSize(self.__mainWindow.label_image1.width(), self.__mainWindow.label_image1.height()),
            aspectMode=Qt.IgnoreAspectRatio)
        self.__mainWindow.label_image1.setPixmap(pixmap)

    def get_roi(self):
        self.__points_roi_tmp = self.__points_roi.copy()
        self.__points_roi_ori = self.__points_roi.copy()
        self.__index_roi = 0
        count_over_thr = 0
        while self.__index_roi < (self.__divide_val - 1):
            if self.__index_roi >= (self.__divide_val - 1):
                break
            self.__index_roi += 1

            step = round((self.__points_roi[3] - self.__points_roi[1]) / self.__divide_val)
            roi_idx, dist_b_max, dist_b_min, max_idx, min_idx = detect(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                                                     self.__points_roi[0]:self.__points_roi[2]])
            if dist_b_max >= self.start_height_thr:
                count_over_thr += 1
                if count_over_thr == 5:
                    self.__points_roi_tmp[1] = (self.__points_roi[1] + (self.__index_roi * step) + 400)
                    break
            else:
                count_over_thr = 0

        self.__index_roi = self.__divide_val - 1
        count_over_thr = 0
        while self.__index_roi >= 0:
            tic = time.time()
            if self.__index_roi < 0:
                break
            self.__index_roi -= 1

            step = round((self.__points_roi[3] - self.__points_roi[1]) / self.__divide_val)
            roi_idx, dist_b_max, dist_b_min, max_idx, min_idx = detect(self.__srcImage[self.__points_roi[1] + (self.__index_roi * step),
                                                     self.__points_roi[0]:self.__points_roi[2]])
            if dist_b_max >= self.start_height_thr:
                count_over_thr += 1
                if count_over_thr == 5:
                    self.__points_roi_tmp[3] = self.__points_roi[1] + (self.__index_roi * step) - 300
                    break
            else:
                count_over_thr = 0

        self.__points_roi = self.__points_roi_tmp.copy()
        self.__index_roi = 0


if __name__ == '__main__':

    app = QApplication(sys.argv)

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