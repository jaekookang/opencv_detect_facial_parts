'''
Detect facial parts from a image

2018-06-27 Jaekoo

Note:
- Many parts of this code are referenced from www.pyimagesearch.com

References:
- https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
'''
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import os

# Check if shape model exists
if not os.path.exists('model/shape_predictor_68_face_landmarks.dat'):
	raise Exception('Please download shape model first; SEE model/download_model.sh')

# Input arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-m', '--shape-predictor-model', required=True,
                    help='path to the shape predictor model e.g., shape_predictor_68_face_landmarks.dat')
parser.add_argument('-i', '--image', required=True,
                    help='path to the source image')
parser.add_argument('-p', '--facial-parts', default='mouth', type=str,
                    choices=['mouth', 'right_eyebrow', 'left_eyebrow',
                             'right_eye', 'left_eye', 'nose', 'jaw'],
                    help='facial parts: mouth, right_eyebrow, left_eyebrow, right_eye, left_eye, nose, jaw')
parser.add_argument('-f', '--all-facial-parts', action='store_true', help='Show all facial parts')
parser.add_argument('-s', '--save', action='store_true', help='path to save the overlayed output image')
args = parser.parse_args()

# Initialize dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor_model)

# Load image, resize, and convert to grayscale
img = cv2.imread(args.image)
img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('image loaded')

# Detect faces in the grayscale image
# -> multiple faces can be detected
rects = detector(gray, 1)

for (i, rect) in enumerate(rects):
    # Determine a facial region from the gray image
    shape = predictor(gray, rect)
    # covert xy coordinates into numpy array
    shape = face_utils.shape_to_np(shape)

    assert args.facial_parts in face_utils.FACIAL_LANDMARKS_IDXS.keys()

    # Get x,y coordinates
    #  ("mouth", (48, 68))
    #  ("right_eyebrow", (17, 22))
    #  ("left_eyebrow", (22, 27))
    #  ("right_eye", (36, 42))
    #  ("left_eye", (42, 48))
    #  ("nose", (27, 35))
    #  ("jaw", (0, 17))
    (i, j) = face_utils.FACIAL_LANDMARKS_IDXS[args.facial_parts]

    # Copy the image to overlay predicted points
    clone = img.copy()
    cv2.putText(clone, args.facial_parts, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Loop over to draw facial parts
    for (x, y) in shape[i:j]:
        cv2.circle(clone, (x, y), 1, (0, 0, 225), -1)

    # Extract ROI of the face region separately
    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
    roi = img[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

    # Show the overlayed image as well as ROI
    print('Press any key on the image to close')
    cv2.imshow('ROI', roi)
    cv2.imshow('Image', clone)
    cv2.waitKey(0)

    if args.all_facial_parts:
    	# show all facial landmarks
    	all_face = face_utils.visualize_facial_landmarks(img, shape)
    	cv2.imshow("Image", all_face)
    	cv2.waitKey(0)


    if args.save:
    	# save the specific part
    	_, img_file = os.path.split(args.image)
    	fid, ext = img_file.split('.')
    	cv2.imwrite('result/{}_{}.{}'.format(fid, args.facial_parts, ext), clone)

    	# save all facial parts
    	if args.all_facial_parts:
    		cv2.imwrite('result/{}_all.{}'.format(fid, ext), all_face)






