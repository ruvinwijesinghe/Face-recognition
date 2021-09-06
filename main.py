import cv2
import face_recognition
import numpy as np

imageRuvin = face_recognition.load_image_file('Image_records/Ruvin1.jpg')
imageRuvin = cv2.cvtColor(imageRuvin,cv2.COLOR_BGR2RGB)
imageRuvint_test = face_recognition.load_image_file('raw_images/r.jpg')
imageRuvint_test = cv2.cvtColor(imageRuvint_test,cv2.COLOR_BGR2RGB)

#finding faces of our image
face_Location = face_recognition.face_locations(imageRuvin)[0]
encode_Ruvin = face_recognition.face_encodings(imageRuvin)[0]
cv2.rectangle(imageRuvin,(face_Location[3],face_Location[0]),(face_Location[1],face_Location[2]),(255,0,255),2)

face_LocationTest = face_recognition.face_locations(imageRuvint_test)[0]
encode_Ruvin_test = face_recognition.face_encodings(imageRuvint_test)[0]
cv2.rectangle(imageRuvint_test,(face_LocationTest[3],face_LocationTest[0]),(face_LocationTest[1],face_LocationTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encode_Ruvin],encode_Ruvin_test)
print(results)
#finding the best mach
faceDis = face_recognition.face_distance([encode_Ruvin],encode_Ruvin_test)
print(results,faceDis)
#write result on the image
cv2.putText(imageRuvint_test,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('Ruvin wijesinghe',imageRuvin)
cv2.imshow('Ruvin wijesinghe Test',imageRuvint_test)
cv2.waitKey(0)