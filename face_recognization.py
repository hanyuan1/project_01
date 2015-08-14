# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 13:12:04 2015

@author: hanyuanzhuang
"""

from PIL import Image, ImageDraw
import numpy
import cv
import os
import sys

IMAGE_SIZE = (40,40)
def createDatabase(path, number):
    
    imageMatrix = []
    for i in range(1,number+1):
        image = Image.open(path+'/'+str(i)+'.jpg')
        image = image.resize(IMAGE_SIZE) 
        grayImage = image.convert('L')
        imageArray = list(grayImage.getdata())
        imageMatrix.append(imageArray)        
        
    imageMatrix = numpy.array(imageMatrix) 

    
    return imageMatrix

def eigenfaceCore(Matrix):

    trainNumber, perTotal = numpy.shape(Matrix) 
    
 
    meanArray = Matrix.mean(0)

 
    diffMatrix = Matrix - meanArray
    

   
    diffMatrix = numpy.mat(diffMatrix)
    L = diffMatrix * diffMatrix.T 
    eigenvalues, eigenvectors = numpy.linalg.eig(L)
    
 
    eigenvectors = list(eigenvectors.T) 
    
    for i in range(0,trainNumber):
        if eigenvalues[i] < 1:
            eigenvectors.pop(i)
            
    eigenvectors = numpy.array(eigenvectors) 
    eigenvectors = numpy.mat(eigenvectors).T        
     
    
 
    eigenfaces = diffMatrix.T * eigenvectors
    return eigenfaces   

def recognize(testIamge, Matrix, eigenface):
   
    meanArray = Matrix.mean(0) 
    
   
    diffMatrix = Matrix - meanArray
    
  
    perTotal, trainNumber = numpy.shape(eigenface)
    
    
    projectedImage = eigenface.T * diffMatrix.T
    
  
    testimage = Image.open(testIamge)
    testimage = testimage.resize(IMAGE_SIZE)
    grayTestImage = testimage.convert('L')
    testImageArray = list(grayTestImage.getdata())
    testImageArray = numpy.array(testImageArray)
    
    differenceTestImage = testImageArray - meanArray
   
    differenceTestImage = numpy.array(differenceTestImage)
    differenceTestImage = numpy.mat(differenceTestImage)
    
    projectedTestImage = eigenface.T * differenceTestImage.T
  

    
    distance = []
    for i in range(0, trainNumber):
        q = projectedImage[:,i]
        temp = numpy.linalg.norm(projectedTestImage - q) 
        distance.append(temp)
  
    minDistance = min(distance)
    index = distance.index(minDistance)  
    im = Image.open(testIamge)
    im = im.resize((300,300), Image.ANTIALIAS)
    im.show()
    
    with open('/Users/hanyuanzhuang/desktop/project_01/trainData.txt') as f:
        for i, line in enumerate(f, 1):
            if i== index + 1:
                break
    print ('predicted')
    print line
    print ('true')
    
    for i in range(1,31):
        if i == index + 1:
            im = Image.open('/Users/hanyuanzhuang/desktop/project_01/trainData'+'/'+str(i)+'.jpg' )
            im = im.resize((300,300), Image.ANTIALIAS)
            im.show()
    q=''
    with open('/Users/hanyuanzhuang/desktop/project_01/testData2.txt') as f:
        q = f.read()
    s = str(1)
    e = s+'           '+q
   
    print e
    return index+1, q  
    
if __name__ == "__main__":
    TrainNumber = 31
    Matrix = createDatabase('/Users/hanyuanzhuang/desktop/project_01/trainData', TrainNumber)
    eigenface = eigenfaceCore(Matrix)
    unkown = 1
    testimage = '/Users/hanyuanzhuang/desktop/project_01/testData/' + str(unkown) +'.jpg'    
    recognize(testimage, Matrix, eigenface)
    
    
        