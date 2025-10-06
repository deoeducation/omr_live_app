import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from streamlit.components.v1 import html

st.set_page_config(page_title='Live OMR Scanner', layout='wide')
st.title("Live OMR Scanner — 50 Questions (Rear Camera)")

# Create results folder
if not os.path.exists('results'):
    os.makedirs('results')

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    teacher_name = st.text_input("Teacher Name", value="")
    subject = st.selectbox("Subject", ["maths", "english", "hindi", "evs"])
    sensitivity = st.slider("Bubble detection sensitivity", 0.5, 0.9, 0.6)
    st.markdown("---")
    uploaded_key = st.file_uploader("Upload answer_key.json", type=["json"])

# Only proceed if answer key is uploaded
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

        def pct_to_px(coord, w, h):
            return int(coord['x']*w), int(coord['y']*h), int(coord['r']*min(w,h))

        def read_image_from_bytes(file_bytes):
            data = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return img

        # Session state for continuous scanning
        if "scan_next" not in st.session_state:
            st.session_state.scan_next = True

        if st.session_state.scan_next:
            st.subheader("Scan OMR Sheet (Rear Camera - Hold phone in landscape)")

            # HTML input to open rear camera
            html_code = """
            <input type="file" accept="image/*" capture="environment" id="rearCam">
            <script>
            document.getElementById('rearCam').addEventListener('change', function(event){
                const file = event.target.files[0];
                if(file){
                    const reader = new FileReader();
                    reader.onload = function(e){
                        const data_url = e.target.result;
                        const streamlitEvent = new CustomEvent("onFileSelect", {detail: data_url});
                        window.parent.document.dispatchEvent(streamlitEvent);
                    }
                    reader.readAsDataURL(file);
                }
            });
            </script>
            """
            html(html_code, height=50)

            uploaded_file = st.file_uploader("Take a photo of the OMR sheet", type=["png", "jpg", "jpeg"], key="omr_cam")

            if uploaded_file is not None:
                img = read_image_from_bytes(uploaded_file.read())
                h, w = img.shape[:2]

                student_id = f"{teacher_name}_{datetime.now().strftime('%H%M%S')}"
                results = {}
                correct_count = 0

                # Process 50 questions
                for q, options in bubble_grid.items():
                    marked = None
                    for opt, coord in options.items():
                        x, y, r = pct_to_px(coord, w, h)
                        crop = img[max(0,y-r):y+r, max(0,x-r):x+r]
                        if crop.size == 0:
                            continue
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        thresh = cv2.adaptiveThreshold(
                            gray, 255,
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV,
                            11, 2
                        )
                        filled_ratio = cv2.countNonZero(thresh) / thresh.size
                        if filled_ratio > sensitivity:
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

                # Overlay for debugging
                show_overlay = st.checkbox("Show detected bubbles overlay")
                if show_overlay:
                    overlay_img = img.copy()
                    for q, opt_coord in bubble_grid.items():
                        for opt, coord in opt_coord.items():
                            x, y, r = pct_to_px(coord, w, h)
                            color = (0,255,0) if results[q]==opt else (255,0,0)
                            cv2.circle(overlay_img, (x, y), r, color, 2)
                    st.image(overlay_img, channels="BGR", caption="Bubble Detection Overlay")

                # Next scan
                if st.button("Scan Next OMR"):
                    st.session_state.scan_next = True
                    st.experimental_rerun()

                st.session_state.scan_next = False

    except Exception as e:
        st.error(f"Invalid JSON file: {e}")
        st.stop()
else:
    st.warning("Please upload your answer_key.json first")
