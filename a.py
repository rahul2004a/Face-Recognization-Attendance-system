# importing librarys
import cv2
import numpy as npy
import face_recognition as face_rec
# function



def resize(img, size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0] * size)
    dimension = (width, height)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)


# img declaration
rahul = face_rec.load_image_file('rahul.jpg',)
rahul = cv2.cvtColor(rahul, cv2.COLOR_BGR2RGB)
rahul = resize(rahul, 0.50)
rahul_test = face_rec.load_image_file('rahul_test.jpg')
rahul_test = resize(rahul_test, 0.50)
rahul_test = cv2.cvtColor(rahul_test, cv2.COLOR_BGR2RGB)

# finding face location

faceLocation_rahul = face_rec.face_locations(rahul)[0]
encode_rahul = face_rec.face_encodings(rahul)[0]
cv2.rectangle(rahul, (faceLocation_rahul[3], faceLocation_rahul[0]), (faceLocation_rahul[1], faceLocation_rahul[2]), (255, 0, 255), 3)


faceLocation_rahultest = face_rec.face_locations(rahul_test)[0]
encode_rahultest = face_rec.face_encodings(rahul_test)[0]
cv2.rectangle(rahul_test, (faceLocation_rahul[3], faceLocation_rahul[0]), (faceLocation_rahul[1], faceLocation_rahul[2]), (255, 0, 255), 3)

print("Normal image Measurement")
print(encode_rahul)
print("Test image Measurement")
print(encode_rahultest)

results = face_rec.compare_faces([encode_rahul], encode_rahultest)
print(results)
cv2.putText(rahul_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main_img', rahul)
cv2.imshow('test_img', rahul_test)
cv2.waitKey(0)
cv2.destroyAllWindows()