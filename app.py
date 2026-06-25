import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2

# ---------------- SETUP ----------------
st.set_page_config(page_title="YOLOv8 Detection & Tracking", page_icon="🎯")

st.title("🎯 Object Detection + Tracking (YOLOv8)")

model = YOLO("yolov8n.pt")

# ---------------- UI ----------------
option = st.radio("Choose Mode", ["Image Detection", "Video Tracking", "Live Webcam Tracking"])

# ---------------- IMAGE ----------------
if option == "Image Detection":

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if file:
        image = Image.open(file)
        st.image(image, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)

        results = model(tmp.name)
        annotated = results[0].plot()

        st.image(annotated, use_container_width=True)

# ---------------- VIDEO ----------------
elif option == "Video Tracking":

    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if file:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(file.read())
            video_path = tmp.name

        st.info("Processing video... ⏳")

        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            frame = results[0].plot()

            st.image(frame, channels="BGR")

        cap.release()

        st.success("Done 🎯")

# ---------------- LIVE WEBCAM ----------------
elif option == "Live Webcam Tracking":

    st.warning("Press STOP in terminal to exit webcam")

    run = st.button("Start Webcam")

    if run:

        cap = cv2.VideoCapture(0)

        stframe = st.empty()

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, tracker="bytetrack.yaml")
            frame = results[0].plot()

            stframe.image(frame, channels="BGR")

        cap.release()