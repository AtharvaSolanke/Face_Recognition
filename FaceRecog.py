import cv2
import csv
import numpy
import face_recognition
from datetime import datetime

# capture video
capture_video_webcam = cv2.VideoCapture(0) # 0 for 1st camera available i.e. webcam

# Loading Image and encoding

image_Bill = face_recognition.load_image_file(r"Faces\Bill.jpeg")

imgEncoding_Bill = face_recognition.face_encodings(image_Bill)[0] # 0 as we need 1 face

image_Elon = face_recognition.load_image_file(r"Faces\Elon.jpg")

imgEncoding_Elon = face_recognition.face_encodings(image_Elon)[0]

# linking encoding and names in array

faces_encoding = [imgEncoding_Bill, imgEncoding_Elon]
faces_names = ["Bill", "Elon"]

faces = faces_names.copy()

# location of face in image

face_location_img = []
face_encoding_img = []

# Extracting Date and time

currentT = datetime.now()
current_date = currentT.strftime("%d-%m-%y")  # formating DT in required form

# make file to store Attendence
f = open(f"{current_date}.csv", "w+", newline="")

writer_csv = csv.writer(f) # write csv file

while True: # infinte loops which stops on command
    _, frame = capture_video_webcam.read()   # _ for vdo capture successful or not
    resize_frame = cv2.resize(frame, (0,0), fx = 0.50, fy= 0.50)
    rgb_conv = cv2.cvtColor(resize_frame, cv2.COLOR_BGR2RGB)

    # Recognise faces
    face_location_img = face_recognition.face_locations(rgb_conv) 
    
    # Convert to encoding of faces from web cam
    face_encoding_img = face_recognition.face_encodings(rgb_conv, face_location_img)

    for face_encoding in face_encoding_img: # check for similarities one by one
        match_faces = face_recognition.compare_faces(faces_encoding, face_encoding) 
        # compare faces encoding with known faces, has true false values

        # similarity
        face_distance = face_recognition.face_distance(faces_encoding, face_encoding)

        # distance min = more similar
        best_matching_i = numpy.argmin(face_distance)

        # if face matches then
         
        if(match_faces[best_matching_i]):
            name = faces_names[best_matching_i]
            # get the index of best matching face and get name through index
        name = faces_names[best_matching_i]
        # Add text to video
        if name in faces_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner = (150,50)
            font_scale = 1.25
            font_color = (0,0,255)
            thickness = 3
            line_type = 2
            cv2.putText(frame, name+ " Present", bottom_left_corner, font,font_scale,font_color, thickness, line_type)

            pt1 = (170, 75)
            pt2 = (475, 400)
            color = (0, 255, 0)
            thickness = 3
            lineType = cv2.LINE_4
 
            #rectangle
            img_rect = cv2.rectangle(frame, pt1, pt2, color, thickness, lineType)

            # get current time and write in csv file
            if name in faces:
                faces.remove(name)
                current_time = currentT.strftime    ("%H:%M:%S")
                writer_csv.writerow([name , current_time])

    cv2.imshow("Attendence", frame)   # show frame

    # quit the process
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break # end matching

# Free the camera
capture_video_webcam.release()

# destroy all windows
cv2.destroyAllWindows()
f.close()
