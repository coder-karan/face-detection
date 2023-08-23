# importing library of python- OpenCV
# required to capture and processes the video frame by frame in realtime

import cv2

# loading the required trained model stored in XML file
# the file contains the trained Haar Cascade algorithm in it
# which will be used to detect faces in the image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# VideoCapture() is a in-built OpenCV function to capture video from a camera
# from webcam of my laptop in my case

capture = cv2.VideoCapture(0)

# we initialize a infinite while loop to read the frames of the video continuously
while 1:

	# read function allows us to read frames from a camera
	_return, frame = capture.read()

	# converting original frame to a gray frame
	grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detecting faces of different sizes in the input frame
	facesfound_list = face_cascade.detectMultiScale(grayscale_frame, 1.3,5)


	for (x,y,w,h) in facesfound_list:

		# The cv2.rectangle() function draws a rectangle on the frame
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		
	# Displaying the output in form of image read by our model in a window
	cv2.imshow('image',frame)

	# This condition will break the loop if user presses the escape key
	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break

# Close the window
capture.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()

