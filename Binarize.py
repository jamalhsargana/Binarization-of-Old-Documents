import cv2
import numpy as np
from PIL import Image
from PIL import ImageFilter
import os
from matplotlib import pyplot as plt

root = os.getcwd()
os.chdir('Dataset')
dataset = [f for f in os.listdir('.') if os.path.isfile(f)]

def appendToFilename(filename, appendStr):
    return filename[:filename.find('.')] + ' [' + appendStr + ']' + filename[filename.find('.'):]

def saveImageToDir(img, filename, dir):
    oldDir = os.getcwd()
    os.chdir(os.path.join(root,dir))
    img.save(filename)
    os.chdir(oldDir)

def getWindowMeanAndSD(img, sum, sqsum, rowOffset, colOffset, winWidth, winHeight):
    A = sum[rowOffset, colOffset]
    B = sum[rowOffset, colOffset + winWidth]
    C = sum[rowOffset + winHeight, colOffset]
    D = sum[rowOffset + winHeight, colOffset + winWidth]
    totalSum = A + D - B - C

    A = sqsum[rowOffset, colOffset]
    B = sqsum[rowOffset, colOffset + winWidth]
    C = sqsum[rowOffset + winHeight, colOffset]
    D = sqsum[rowOffset + winHeight, colOffset + winWidth]
    totalSqSum = A + D - B - C

    winArea = winWidth * winHeight
    mean = totalSum / winArea
    sd = ((totalSqSum - mean * totalSum) / winArea) ** 0.5
    return (mean, sd)

class Wolf:
    def __init__(self, img, winWidth, winHeight, sum, sqsum):
        self.minGray = img.min()
        self.maxSD = 9999
        self.winMean = []
        self.winSD = []
        rows, cols = img.shape
        self.winWidth = winWidth
        self.winHeight = winHeight
        i = 0
        j = 0

        while i + self.winHeight < rows:
            self.winMean.append([])
            self.winSD.append([])
            j = 0
            while j + self.winWidth < cols:
                mean, sd = getWindowMeanAndSD(img, sum, sqsum, i, j, self.winWidth, self.winHeight)
                if sd > self.maxSD:
                    self.maxSD = sd
                self.winMean[i].append(mean)
                self.winSD[i].append(sd)
                j += 1
            i += 1

    def getLocalThreshold(self, rowOffset, colOffset, k=0.36):
        # thresh = mean+(k*(((totalSqSum - mean**2)/winArea)**0.5))
        mean = self.winMean[rowOffset][colOffset]
        sd = self.winSD[rowOffset][colOffset]
        # thresh = ((1-k)*mean) + (k*self.minGray) + (k*sd*(mean-self.minGray)/self.maxSD)
        thresh = mean + k * (sd / self.maxSD - 1) * (mean - self.minGray)
        return thresh

class NICK:

    """
    The value of Niblack factor k can vary from -0.1 to -0.2 depending upon the application requirement. k close to -0.2
    make sure that noise is all but eliminated but characters can break a little bit, while with values close to -0.1, some noise
    pixels can be left but the text will be extracted crisply and unbroken.
    """
    @classmethod
    def getLocalThreshold(cls, sum, sqsum, rowOffset, colOffset, winWidth, winHeight, k):
        A = sum[rowOffset, colOffset]
        B = sum[rowOffset, colOffset + winWidth]
        C = sum[rowOffset + winHeight, colOffset]
        D = sum[rowOffset + winHeight, colOffset + winWidth]
        totalSum = A + D - B - C

        A = sqsum[rowOffset, colOffset]
        B = sqsum[rowOffset, colOffset + winWidth]
        C = sqsum[rowOffset + winHeight, colOffset]
        D = sqsum[rowOffset + winHeight, colOffset + winWidth]
        totalSqSum = A + D - B - C

        winArea = winWidth*winHeight
        mean = totalSum/winArea
        thresh = mean+(k*(((totalSqSum - mean**2)/winArea)**0.5))
        # thresh = ((1-k)*mean) + (k*minGray) + (k*sd/
        return thresh

def binarize(img, winWidth=19, winHeight=19):
    rows, cols = img.shape
    i = 0
    j = 0
    sum, sqsum = cv2.integral2(img)
    w = Wolf(img, winWidth, winHeight, sum, sqsum)
    while i + winHeight < rows:
        j = 0
        while j + winWidth < cols:
            thresh = w.getLocalThreshold(i, j, k=0.36)
            # thresh = NICK.getLocalThreshold(sum, sqsum, i, j, winWidth, winHeight, -0.1)
            if img[i,j] > thresh:
                img[i,j] = 255
            else:
                img[i,j] = 0
            # retVal, window = cv2.threshold(img[i:i+winHeight, j:j+winWidth], thresh, 255, cv2.THRESH_BINARY)
            # img[i,j] = retVal
            j+=1
        i+=1
    return Image.fromarray(img)

for file in dataset:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = binarize(img)
    saveImageToDir(img, file, "Binarized Images")
    print file
    # integral = NICK.binarize(cvImg)
    # cv2.imshow('image', integral)
    # cv2.waitKey(0)
