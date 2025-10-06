import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from PIL import Image

st.set_page_config(page_title='Live OMR Scanner', layout='wide')
st.title("Live OMR Scanner — 50 Questions")

# Create results folder
if not os.path.exists('results'):
    os.makedirs('results')

# Sidebar for teacher and subject
with st.sidebar:
    st.header("Settings")
    teacher_name = st.text_input("Teacher Name", value="")
    subject = st.selectbox("Subject", ["maths", "english", "hindi", "evs"])
    st.markdown("---")
    
    # Upload answer key
    uploaded_key = st.file_uploader("Upload answer_key.json", type=["json"])

# Only show camera input after answer key is uploaded
if uploaded_key is not None:
    try:
        answer_key = json.load(uploaded_key)
        st.success("Answer key loaded successfully!")

        # Bubble grid for 50 questions
        bubble_grid = {}
        for q in range(1, 51):
            bubble_grid[str(q)] = {
                "A": {"x":0.15, "y":0.05 + 0.018*q, "r":0.015},
                "B": {"x":0.30, "y":0.05 + 0.018*q, "r":0.015},
                "C": {"x":0.45, "y":0.05 + 0.018*q, "r":0.015},
                "D": {"x":0.60, "y":0.05 + 0.018*q, "r":0.015}
            }

        # Helper functions
        def pct_to_px(coord, w, h):
            return int(coord['x']*w), int(coord['y']*h), int(coord['r']*min(w,h))

        def read_image_from_bytes(file_bytes):
            data = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img

        # Camera input section
        st.subheader("Scan OMR Sheet (Tap 'Take Photo' to open rear camera)")
        uploaded_file = st.file_uploader(
            "Take a photo of the OMR sheet",
            type=["png", "jpg", "jpeg"],
            key="omr_cam"
        )

        if uploaded_file is not None:
            img = read_image_from_bytes(uploaded_file.read())
            h, w = img.shape[:2]

            student_id = f"{teacher_name}_{datetime.now().strftime('%H%M%S')}"
            results = {}
            correct_count = 0

            # Process 50 questions with improved detection
            for q, options in bubble_grid.items():
                marked = None
                for opt, coord in options.items():
                    x, y, r = pct_to_px(coord, w, h)
                    crop = img[max(0,y-r):y+r, max(0,x-r):x+r]
                    if crop.size == 0:
                        continue
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    filled_ratio = 1 - (cv2.countNonZero(thresh) / thresh.size)

                    # Only consider significant filled area as marked
                    if filled_ratio > 0.7 and cv2.countNonZero(thresh) > 50:
                        marked = opt
                        break

                results[q] = marked
                if marked == answer_key.get(q):
                    correct_count += 1

            # Show results
            st.subheader("Results")
            st.write(f"Student ID: {student_id}")
            st.write(f"Total Correct: {correct_count} / 50")
            st.dataframe(pd.DataFrame([results]))

            # Save to Excel
            df = pd.DataFrame([{**{"Student": student_id}, **results, "Total Correct": correct_count}])
            filename = f"results/{subject}_{teacher_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            df.to_excel(filename, index=False)
            st.success(f"Results saved: {filename}")
            st.download_button("Download Results", data=open(filename, "rb").read(), file_name=filename)

    except Exception:
        st.error("Invalid JSON file")
        st.stop()
else:
    st.warning("Please upload your answer_key.json first")
