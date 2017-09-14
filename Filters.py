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

class CVMedianFilter:
    @classmethod
    def apply(cls, img, kernelSize):
        return Image.fromarray(cv2.medianBlur(img, kernelSize))

class CVBilateralFilter:
    @classmethod
    def apply(cls, img, diameter=9, sigmaColor=75, sigmaSpace=75):
        return Image.fromarray(cv2.bilateralFilter(img,diameter,sigmaColor, sigmaSpace))

class CVFastNLMeansFilter:
    @classmethod
    def apply(cls, img, hLum=10, photoRender=10, searchWindow=7, blockSize=21):
        return Image.fromarray(cv2.fastNlMeansDenoisingColored(img, None, hLum, photoRender, searchWindow, blockSize))

class PILMinFilter:
    @classmethod
    def apply(cls, img, kernelSize=3):
        return img.filter(ImageFilter.MinFilter(kernelSize))

class PILMaxFilter:
    @classmethod
    def apply(cls, img, kernelSize=3):
        return img.filter(ImageFilter.MaxFilter(kernelSize))

class PILModeFilter:
    @classmethod
    def apply(cls, img, kernelSize=3):
        return img.filter(ImageFilter.ModeFilter(kernelSize))

class PILGaussianBlurFilter:
    @classmethod
    def apply(cls, img, radius=2):
        return img.filter(ImageFilter.GaussianBlur(radius))

class PILUnsharpMaskFilter:
    @classmethod
    def apply(cls, img, radius=2, percent=150, threshold=3):
        return img.filter(ImageFilter.UnsharpMask(radius, percent, threshold))


for file in dataset:
    cvImg = cv2.imread(file)
    pilImg = Image.fromarray(cvImg)

    # im = CVMedianFilter.apply(cvImg, 3)
    # newFile = appendToFilename(file, '3x3')
    # saveImageToDir(im, newFile, 'Filtered Images/Median')

    # im = CVMedianFilter.apply(cvImg, 5)
    # newFile = appendToFilename(file, '5x5')
    # saveImageToDir(im, newFile, 'Filtered Images/Median')

    # im = PILMinFilter.apply(pilImg, 3)
    # newFile = appendToFilename(file, '3x3')
    # saveImageToDir(im, newFile, 'Filtered Images/Min')

    # im = PILMinFilter.apply(pilImg, 5)
    # newFile = appendToFilename(file, '5x5')
    # saveImageToDir(im, newFile, 'Filtered Images/Min')

    # im = PILMaxFilter.apply(pilImg, 3)
    # newFile = appendToFilename(file, '3x3')
    # saveImageToDir(im, newFile, 'Filtered Images/Max')

    # im = PILMaxFilter.apply(pilImg, 5)
    # newFile = appendToFilename(file, '5x5')
    # saveImageToDir(im, newFile, 'Filtered Images/Max')

    # im = PILModeFilter.apply(pilImg, 3)
    # newFile = appendToFilename(file, '3x3')
    # saveImageToDir(im, newFile, 'Filtered Images/Mode')

    # im = PILModeFilter.apply(pilImg, 5)
    # newFile = appendToFilename(file, '5x5')
    # saveImageToDir(im, newFile, 'Filtered Images/Mode')

    # im = PILGaussianBlurFilter.apply(pilImg)
    # newFile = appendToFilename(file, 'rad2')
    # saveImageToDir(im, newFile, 'Filtered Images/Gaussian')

    # im = PILUnsharpMaskFilter.apply(pilImg)
    # newFile = appendToFilename(file, 'r2-p150-th3')
    # saveImageToDir(im, newFile, 'Filtered Images/Unsharp')

    # im = CVBilateralFilter.apply(cvImg)
    # newFile = appendToFilename(file, 'd9-sc75-sp75')
    # saveImageToDir(im, newFile, 'Filtered Images/Bilateral')

    # im = CVBilateralFilter.apply(cvImg, diameter=18)
    # newFile = appendToFilename(file, 'd18-sc75-sp75')
    # saveImageToDir(im, newFile, 'Filtered Images/Bilateral')

    # im = CVBilateralFilter.apply(cvImg, diameter=27)
    # newFile = appendToFilename(file, 'd27-sc75-sp75')
    # saveImageToDir(im, newFile, 'Filtered Images/Bilateral')

    im = CVFastNLMeansFilter.apply(cvImg)
    newFile = appendToFilename(file, 'h10-pr10-sw7-bs21')
    saveImageToDir(im, newFile, 'Filtered Images/Fast NL Means')
