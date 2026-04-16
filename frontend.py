import streamlit as st
import threading
import cv2
import numpy as np
from queue import Queue
from PIL import Image
import isl_detection

st.set_page_config(page_title="HathSeyBaath", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #ffe4e1;
    }
    .title {
        font-size: 64px;
        color: #c71585;
        text-align: center;
        font-family: 'Comic Sans MS', cursive, sans-serif;
        text-shadow: 2px 2px 4px #000000;
        padding: 20px;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 30px;
    }
    .stButton>button {
        background-color: #ffb6c1;
        color: white;
        font-size: 20px;
        padding: 10px 30px;
        border-radius: 12px;
    }
    </style>
    <div class="title">HathSeyBaath</div>
    """,
    unsafe_allow_html=True
)

frame_queue = Queue()
letter_queue = Queue()
camera_running = st.session_state.get("camera_running", False)

placeholder = st.empty()
word_placeholder = st.empty()
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Camera"):
        if not camera_running:
            st.session_state.camera_running = True
            threading.Thread(target=isl_detection.run_isl, args=(frame_queue, letter_queue), daemon=True).start()

with col2:
    if st.button("Stop Camera"):
        st.session_state.camera_running = False

final_string = ""

while st.session_state.get("camera_running", False):
    if not frame_queue.empty():
        frame = frame_queue.get()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        placeholder.image(img, channels="RGB")

    if not letter_queue.empty():
        letter = letter_queue.get()
        final_string += letter
        word_placeholder.markdown(f"<h2 style='text-align: center; color: #c71585;'>Word: {final_string}</h2>", unsafe_allow_html=True)

    if not st.session_state.get("camera_running", False):
        break
