import cv2
import mediapipe as mp
from mediapipe.tasks import python as py
from mediapipe.tasks.python import vision

base_settings = py.BaseOptions(model_asset_path='hand_landmarker.task')
ai_settings = vision.HandLandmarkerOptions(base_options=base_settings,num_hands=2)
ai_eye = vision.HandLandmarker.create_from_options(ai_settings)

cap = cv2.VideoCapture(0)
while True:
    ok,frame = cap.read()
    if not ok: break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_vid = mp.Image(mp.ImageFormat.SRGB,data=rgb)
    result = ai_eye.detect(mp_vid)
    if result.hand_landmarks:
        for mkono in result.hand_landmarks:
            for lm in mkono:
                h,w,_ = frame.shape
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame,(x,y),10,(0,0,255),-1)
                
    cv2.imshow("mkono umeonekana",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()