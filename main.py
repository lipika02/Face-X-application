import streamlit as st
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

face_classifier=cv2.CascadeClassifier('C://Users//lipik//haarcascade_frontalface_default.xml')

#########################################################################################################
def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None
    
    for(x,y,w,h) in faces:
        cropped_faces=img[y:y+h,x:x+w]

    return cropped_faces

#########################################################################################################
def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
        return img,[]
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y+h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

#########################################################################################################
def image_collect():
    cap=cv2.VideoCapture(0)
    count=0
    frame_placeholder = st.empty()

    st.empty().write("Taking pictures...")

    while True:
        ret,frame = cap.read()
        if face_extractor(frame) is not None:
            count+=1
            face=cv2.resize(face_extractor(frame),(200,200))
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)

            file_name_path='C:/Users/lipik/Desktop/face reco/'+str(count)+'.jpg'

            cv2.imwrite(file_name_path,face)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,0,(0,255,0),2)
            frame_placeholder.image(face, channels="RGB")
            #cv2.imshow('Face Cropper',face)
        else:
            print("Face not found")
            pass  # do nothing !!
        
        if cv2.waitKey(1)==13 or count==300:  # 13 = Enter
            break
            
    cap.release()
    cv2.destroyAllWindows()

    print("Collecting Samples Complete!!")

    with st.empty():
        st.success("Done!!!")

#########################################################################################################
def face_detection_recognition():
    data_path='C:/Users/lipik/Desktop/face reco/'

    onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path,f))]

    Training_data, Labels = [],[]

    for i, files in enumerate(onlyfiles):
        image_path=data_path + onlyfiles[i]
        images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        Training_data.append(np.asarray(images,dtype=np.uint8,))
        Labels.append(i)

    Labels=np.asarray(Labels,dtype=np.int32)

    model=cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_data),np.asarray(Labels))
    print("Model Training Complete")

    ################################################################################################################################

    face_classifier = cv2.CascadeClassifier('C://Users//lipik//haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    frame_placeholder2 = st.empty()

    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                display_string = str(confidence) +'% Confidence it is user'
                
            cv2.putText(image, display_string, (100,120), cv2.FONT_HERSHEY_COMPLEX, 1, (250,120,255), 2)
            
            if confidence > 85:
                cv2.putText(image, "UNLOCKED", (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                frame_placeholder2.image(image, channels="RGB")
                #cv2.imshow('Face Cropper', image)
                        
            else:
                cv2.putText(image, "LOCKED", (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                frame_placeholder2.image(image, channels="RGB")
                #cv2.imshow('Face Cropper', image)
                
        except:
            cv2.putText(image, "Face Not Found", (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
            #cv2.imshow('Face Cropper', image)
            frame_placeholder2.image(image, channels="RGB")
            pass

        if cv2.waitKey(1) == 13: 
            break

    cap.release()
    cv2.destroyAllWindows()

#########################################################################################################
def main():
    st.markdown("<h1 style='text-align: center; font-size: 5rem;'>FACE X</h1>", unsafe_allow_html=True)

    # CSS
    page_bg_img = """
        <style>
        *{
        overflow: hidden;
        }

        [data-testid="stAppViewContainer"] {
        background: #2b2b2b;
        background-image: url("https://c4.wallpaperflare.com/wallpaper/670/883/685/plexus-atoms-neutrons-electrons-wallpaper-preview.jpg"); 
        background-size: cover;
        }

        #face-x {
        color: white;}

        [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
        }

        </style>
        """
   
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    # buttons
    x = st.button("Take Images")
    y = st.button("Face Detection and Recognition")

    try:
        if (x):
            image_collect()
            print("done with function 1")

        if (y):
            print("I am in function 2.")
            face_detection_recognition()

    except:
        pass

#########################################################################################################
# driver code
main()