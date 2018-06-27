# dlib model files: http://dlib.net/face_landmark_detection.py.html
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
	wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
	bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
else
	echo "model already downloaded"
fi
