import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import time


# Load Haar Cascades

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


# Page UI

st.set_page_config(page_title="Driver Sleepiness Detection", layout="centered")
st.title("🚗 Sleepiness Detection For Driver monitoring")

st.write("Measures:")
st.write(" FPS")
st.write(" Total Frames")
st.write(" Drowsy Frames")
st.write(" Detection Rate")


# Video Class

class DrowsinessDetector(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.frame_count = 0
        self.drowsy_frames = 0
        self.EYE_CLOSED_FRAMES = 15

        self.prev_time = time.time()
        self.fps = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (480, 360))

        # ---------------- FPS ----------------
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        self.frame_count += 1

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        drowsy = False

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

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

        # ---------------- Alert ----------------
        if drowsy:
            self.drowsy_frames += 1

            cv2.putText(
                img,
                "DROWSY!",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

        # ---------------- Metrics ----------------
        detection_rate = 0
        if self.frame_count > 0:
            detection_rate = (
                self.drowsy_frames / self.frame_count
            ) * 100

        cv2.putText(
            img,
            f"FPS: {int(self.fps)}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Frames: {self.frame_count}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Drowsy Frames: {self.drowsy_frames}",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            img,
            f"Rate: {detection_rate:.1f}%",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        return img



# Start / Stop Camera

run = st.toggle("📷 Start Camera")

if run:
   webrtc_streamer(
    key="sleep",
    video_transformer_factory=DrowsinessDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)
else:
    st.info("Camera Stopped")