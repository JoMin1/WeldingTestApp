import os
import glob
import cv2


class Fileloader:
    def __init__(self, fileRoot, fileExt, list=True):
        self.fileRoot = fileRoot
        self.fileExt = fileExt
        self.list = list        # load 1 or all

        self.load_filePaths()


    def load_filePaths(self):
        self.filePaths = glob.glob(r"{}\*.{}".format(self.fileRoot, self.fileExt))

    def load(self, filePath):
        print(filePath)
        img = cv2.imread(filePath)
        return img

    def use_resizeAndcrop(self, img, resize, w, h, x=0, y=0):
        img = cv2.resize(img, (resize, resize))
        print(w, h)
        return img

    def use_templateMatching(self, img, refPath=""):
        refPath = r"D:\sample\Anomaly_Datasets\borg_ori\test\good\1 (47).bmp"
        ref = cv2.imread(refPath)
        ref = ref[530:560+2628, 673:673+2775]

        res = cv2.matchTemplate(img, ref, cv2.TM_CCOEFF_NORMED)
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
        x1, y1 = maxLoc
        h1, w1, _ = ref.shape
        img = img[y1:y1 + h1, x1:x1 + w1]


        return img

    def show(self, filePath, img):
        cv2.imshow('{}'.format(filePath), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()