import cv2
import numpy as np

# 정면 얼굴 인식에 필요한 xml 파일을 가져옵니다.
face_classifier = cv2.CascadeClassifier('xml.File') 

def face_extractor(img):
    # 이미지를 Gray화 시킨 다음 scalefactor = 1.3 and minNeighbors = 5로 지정합니다.
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None
    # 얼굴이 감지되었을 때, 이미지에서 잘라낼 부분의 비율을 정합니다.
    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

capture = cv2.VideoCapture(0)
count = 0

# 웹캠이 켜졌을 때, count가 0에서 1씩 증가하며, 사이즈를(200,200)으로 조정합니다.
# 사이즈 조정 후 GrayScale로 .jpg형식으로 지정된 경로에 사진들을 저장합니다.
# 만약 얼굴이 검출되지 않을 경우 "Face not Found" 라는 문구가 출력되어 그냥 넘어갑니다.
while True:
    ret, frame = capture.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = 'File_Path'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass
# "Enter"키를 누르거나 count 가 100이 되었을 경우 카메라를 강제로 종료합니다.
    if cv2.waitKey(1) == 13 or count == 100:
        break

capture.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!')

# 코드 자체의 오류는 전혀 없었다.
# 코드 실행 결과 사진은 문제없이 촬영ㄷ외 되며, 지정된 경로로 저장은 잘 된다.
# 촬영시 얼굴을 똑바로 고정시켜야 잘 찍히며, 얼굴을 흔들거나 촬영각도를 이상하게 할 경우 'Face not Found'라는 문자를 출력하도록 작업했다.
# 일정 거리를 맞추어야 카메라가 인식을 해서 얼굴을 촬영 후 지정 된 경로에 저장해준다.
