import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title='Live OMR Scanner', layout='wide')
st.title("Live OMR Scanner — 50 Questions")

# --- Helper Functions ---
def create_results_folder():
    """Creates a 'results' directory if it doesn't exist."""
    if not os.path.exists('results'):
        os.makedirs('results')

def get_bubble_grid(num_questions=50):
    """Generates the dictionary of bubble coordinates as percentages."""
    bubble_grid = {}
    # These values will likely need to be adjusted for YOUR OMR sheet
    y_start = 0.165  # % from top to first bubble center
    y_step = 0.0153 # % distance between each bubble vertically
    x_coords = {"A": 0.29, "B": 0.44, "C": 0.59, "D": 0.74} # % from left
    radius = 0.009    # % of min(width, height)
    
    for q in range(1, num_questions + 1):
        bubble_grid[str(q)] = {
            opt: {"x": x, "y": y_start + (q - 1) * y_step, "r": radius}
            for opt, x in x_coords.items()
        }
    return bubble_grid

def process_omr_sheet(image_bytes, answer_key, sensitivity, bubble_grid):
    """Reads the image, detects marks, scores, and returns results and an overlay image."""
    # Decode the image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    overlay_img = img.copy()
    h, w = img.shape[:2]

    results = {}
    correct_count = 0

    # Iterate through each question in the grid
    for q_num, options in bubble_grid.items():
        marked_option = "Not Answered"
        
        # Check each option (A, B, C, D)
        for option, coords in options.items():
            # Convert percentage coordinates to pixel coordinates
            cx = int(coords['x'] * w)
            cy = int(coords['y'] * h)
            r = int(coords['r'] * min(w, h))

            # Crop a square region around the bubble
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = cx + r, cy + r
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Process the cropped image
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate the filled ratio
            filled_ratio = cv2.countNonZero(thresh) / (thresh.shape[0] * thresh.shape[1])

            # Check if this bubble is filled
            if filled_ratio > sensitivity:
                marked_option = option
                # Draw green circle on the detected answer
                cv2.circle(overlay_img, (cx, cy), r, (0, 255, 0), 2)
            else:
                # Draw red circle on unchecked options
                cv2.circle(overlay_img, (cx, cy), r, (0, 0, 255), 2)

        results[q_num] = marked_option
        if marked_option == answer_key.get(q_num):
            correct_count += 1
            
    return results, correct_count, overlay_img


# --- Main Application ---
create_results_folder()

# --- Sidebar for Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    teacher_name = st.text_input("Teacher Name", placeholder="Enter your name")
    subject = st.selectbox("Subject", ["Maths", "English", "Hindi", "EVS", "Science"])
    sensitivity = st.slider("Bubble Detection Sensitivity", 0.1, 0.9, 0.4, 0.05)
    st.markdown("---")
    uploaded_key = st.file_uploader("Upload Answer Key (answer_key.json)", type=["json"])

# --- Main Page Logic ---
if uploaded_key is None:
    st.warning("👈 Please upload your `answer_key.json` file to begin.")
    st.stop()

try:
    answer_key = json.load(uploaded_key)
except Exception as e:
    st.error(f"Error loading JSON file: {e}. Please ensure it's a valid JSON.")
    st.stop()

st.info("✅ Answer key loaded. You can now scan an OMR sheet using the camera.")

# Use Streamlit's built-in camera input
img_file_buffer = st.camera_input("Hold the OMR sheet steady and take a picture")

if img_file_buffer:
    # Get the bubble grid
    bubble_grid = get_bubble_grid()

    # Process the image
    student_answers, score, overlay_image = process_omr_sheet(
        img_file_buffer.getvalue(),
        answer_key,
        sensitivity,
        bubble_grid
    )
    
    student_id = f"{teacher_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # --- Display Results ---
    st.header("📊 Scan Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Student ID", value=student_id)
        st.dataframe(pd.DataFrame.from_dict(student_answers, orient='index', columns=['Answer']))
        
    with col2:
        st.metric("Final Score", value=f"{score} / 50")
        st.image(overlay_image, caption="Detected Bubbles (Green=Marked, Red=Unmarked)")

    # --- Save to Excel ---
    df = pd.DataFrame([{"Student": student_id, **student_answers, "Total Correct": score}])
    filename = f"results/{subject}_{student_id}.xlsx"
    df.to_excel(filename, index=False)
    
    st.success(f"Results saved to `{filename}`")
    with open(filename, "rb") as file:
        st.download_button(
            label="📥 Download Results as Excel",
            data=file,
            file_name=filename,
            mime="application/vnd.ms-excel"
        )
