import sys
import os
import time
import cv2
import numpy as np
import pandas as pd
import copy
import math

import pyqtgraph as pg
from PySide6.QtGui import Qt
from PySide6.QtGui import QImage, QPixmap, Qt
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtWidgets
from PySide6.QtCore import QFile, QIODevice, Signal, QEvent, QObject, QSize

from busbar_classifier import Detector

class DefectCounter():
    def __init__(self, length_weldLength_thr, length_blowHole_thr, length_oneSidedWelding_thr, length_offWeldingWithoutTab_thr):
        self.count_weldLength = 0
        self.length_weldLength = 0
        self.length_weldLength_thr = length_weldLength_thr

        self.count_blowHole = 0
        self.length_blowHole = 0
        self.length_blowHole_thr = length_blowHole_thr

        self.count_oneSidedWelding = 0
        self.length_oneSidedWelding = 0
        self.length_oneSidedWelding_thr = length_oneSidedWelding_thr

        self.count_offWeldingWithoutTab = 0
        self.length_offWeldingWithoutTab = 0
        self.length_offWeldingWithoutTab_thr = length_offWeldingWithoutTab_thr

    def run(self, defect_class):
        if defect_class == 'WeldLength':
            self.length_weldLength += 1
            self.length_blowHole = 0
            self.length_oneSidedWelding = 0
            self.length_offWeldingWithoutTab = 0
            if self.length_weldLength == self.length_weldLength_thr:
                self.count_weldLength += 1
                self.length_weldLength = 0

        elif defect_class == 'BlowHole':
            self.length_weldLength = 0
            self.length_blowHole += 1
            self.length_oneSidedWelding = 0
            self.length_offWeldingWithoutTab = 0
            if self.length_blowHole == self.length_blowHole_thr:
                self.count_blowHole += 1
                self.length_blowHole = 0

        elif defect_class == 'OneSidedWelding':
            self.length_weldLength = 0
            self.length_blowHole = 0
            self.length_oneSidedWelding += 1
            self.length_offWeldingWithoutTab = 0
            if self.length_oneSidedWelding == self.length_oneSidedWelding_thr:
                self.count_oneSidedWelding += 1
                self.length_oneSidedWelding = 0

        else:
            self.length_weldLength = 0
            self.length_blowHole = 0
            self.length_oneSidedWelding = 0
            self.length_offWeldingWithoutTab += 1
            if self.length_offWeldingWithoutTab == self.length_offWeldingWithoutTab_thr:
                self.count_offWeldingWithoutTab += 1
                self.length_offWeldingWithoutTab = 0