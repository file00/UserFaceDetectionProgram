import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'File_Path'
#faces폴더에 있는 파일 리스트 얻기
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]
#데이터와 매칭될 라벨 변수
Training_Data, Labels = [], []
#파일 개수 만큼 루프

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #이미지 파일이 아니거나 못 읽어 왔다면 무시
    if images is None:
        continue
    #Training_Data 리스트에 이미지를 바이트 배열로 추가
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    #Labels 리스트엔 카운트 번호 추가
    Labels.append(i)
# 훈련 할 Data가 없는 경우 종료.
if len(Labels) == 0:
    print("There is no data to train.")
    exit()

#Labels를 32비트 정수로 변환
Labels = np.asarray(Labels, dtype=np.int32)
#모델 생성
model = cv2.face.LBPHFaceRecognizer_create(radius=4,neighbors=4,grid_x=4,grid_y=4,threshold=1000)
#학습 시작
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier('xml_File')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is ():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))
    return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라내어 전달

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # 위에서 학습한 모델로 예측을 시도함.
        result = model.predict(face)

        if result[1] < 500: # result[1] == 신뢰도이며, 0에 가까울수록 자신과 같다는 뜻임.
            confidence = int(100*(1-(result[1])/300))
            # 유사도를 화면에 표시
            display_string = str(confidence)+'% Confidence it is user'
        cv2.putText(image,display_string,(100,50), cv2.FONT_HERSHEY_COMPLEX,1,(250,0,255),2)

        if confidence > 75:
            cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
        else:
            cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
    except:
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if (cv2.waitKey(25) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# 정면에서 얼굴을 인식하기 때문에 정면을 가리게 되면 얼굴을 인식하지 못한다. (Ex) 물을 마신다고 텀블러로 본의 아니게 얼굴을 가렸는데, 인식을 못했다.)
# AttributeError: module 'cv2.face' has no attribute 'LBPHFaceRecognizer_create' Solution: pip uninstall opencv_contrib_python -> pip install opencv_contrib_python'
