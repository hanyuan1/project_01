'''-For each face, an image file is generated
    -the images are strictly of the faces
'''

import cv2, os, sys, argparse
parser = argparse.ArgumentParser(description='walks through a given folder, detects faces on images inside, draws a rectangle around the faces and saves the image to the outdir with the filename x_filename, where x is the number of recognized faces')
parser.add_argument('-i','--indir', required=True,
	help='folder with images to analyze')
args = parser.parse_args()

def facechop(image):  
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)

    img = cv2.imread(image)

    #minisize = (img.shape[1],img.shape[0])
    #miniframe = cv2.resize(img, minisize)

    #faces = cascade.detectMultiScale(miniframe)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray,
		scaleFactor = 1.2,
		minNeighbors = 5,
		minSize = (20, 20),
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE)

    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

        sub_face = img[y:y+h, x:x+w]
        imgname = "_" + str(y) + image.split('/')[-1]
        face_file_name = "/Users/hanyuanzhuang/Desktop/project_01/test" + imgname
        cv2.imwrite(face_file_name, sub_face)

    cv2.imshow(image, img)

    return


indir = args.indir
if indir[-1] != '/':
	indir += '/'
 
count = 0
for filename in os.listdir(indir):
	if filename[-3:] != 'jpg':
		continue

	path = indir + filename
	savedto = facechop(path)
	print "saved image {0} to test".format(count, savedto)
	count += 1
