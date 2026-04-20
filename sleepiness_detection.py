import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)


class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.frame_count = 0
        self.EYE_CLOSED_FRAMES = 15

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (480, 360))

        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return img

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        drowsy = False

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 4)

            if len(eyes) >= 2:
                self.counter = 0
                for (ex, ey, ew, eh) in eyes[:2]:
                    cv2.rectangle(
                        face_color,
                        (ex, ey),
                        (ex+ew, ey+eh),
                        (0, 255, 0),
                        2
                    )
            else:
                self.counter += 1

            if self.counter >= self.EYE_CLOSED_FRAMES:
                drowsy = True

        if drowsy:
            cv2.putText(
                img,
                "DROWSY!",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        return img


st.title("🚗 Driver Sleepiness Detection")

# Start / Stop control
run = st.toggle("📷 Start Camera")

if run:
    webrtc_streamer(
        key="fast",
        video_transformer_factory=DrowsinessDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )
else:
    st.info("Camera Stopped")