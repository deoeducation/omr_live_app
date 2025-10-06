# Save this code as app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from operator import itemgetter

# --- UTILITY FUNCTIONS FOR IMAGE PROCESSING ---

def process_omr_sheet(image_bytes, answer_key, sensitivity, questions=50, choices=4):
    """
    Main function to process the OMR sheet image from bytes.
    Finds the sheet, warps it, finds bubbles, and grades them.
    """
    # 1. Read Image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not decode image.", None

    # 2. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 3. Find Document Contour
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    doc_contour = None
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

    if doc_contour is None:
        return None, "Could not find the 4 corners of the OMR sheet. Please retake the photo with the full sheet in view.", None

    # 4. Apply Perspective Warp
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = doc_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # 5. Find Bubbles and Grade
    # Apply adaptive thresholding to get a clean binary image
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    question_contours = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        # Filter for contours that are roughly circular/square and of a reasonable size
        if w >= 20 and h >= 20 and 0.8 <= aspect_ratio <= 1.2:
            question_contours.append(c)

    if len(question_contours) < (questions * choices):
         return None, f"Could not find enough bubbles. Found {len(question_contours)}, expected {questions * choices}. Try adjusting sensitivity or retaking the photo.", warped

    # Sort contours from top to bottom
    question_contours = sorted(question_contours, key=lambda c: cv2.boundingRect(c)[1])

    correct_count = 0
    results = {}
    overlay_img = warped.copy()

    # Group bubbles by question (row)
    for q in range(questions):
        # Get the contours for the current question (row)
        start_index = q * choices
        end_index = (q + 1) * choices
        
        # Sort the bubbles in the current row by their x-coordinate (left to right)
        row_contours = sorted(question_contours[start_index:end_index], key=lambda c: cv2.boundingRect(c)[0])
        
        marked_index = -1
        max_filled = 0

        # Find the most filled bubble in the row
        for i, c in enumerate(row_contours):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            
            if total > max_filled:
                max_filled = total
                marked_index = i
        
        # Check if the marking is above sensitivity threshold
        marked_answer = None
        # This sensitivity logic is a bit different now; it's a ratio to the area of the bubble
        # A simple pixel count is often sufficient after the initial filter
        if max_filled > (cv2.contourArea(row_contours[marked_index]) * sensitivity):
             marked_answer = "ABCD"[marked_index]

        question_num_str = str(q + 1)
        correct_answer = answer_key.get(question_num_str)
        
        results[question_num_str] = marked_answer
        
        if marked_answer == correct_answer:
            correct_count += 1
            # Draw green on the correct, marked answer
            if marked_index != -1:
                 cv2.drawContours(overlay_img, [row_contours[marked_index]], -1, (0, 255, 0), 3)
        else:
            # Draw red on the wrong, marked answer
            if marked_index != -1:
                cv2.drawContours(overlay_img, [row_contours[marked_index]], -1, (0, 0, 255), 3)


    return {"results": results, "score": correct_count}, "Success", overlay_img


# --- STREAMLIT APP ---

st.set_page_config(page_title='Live OMR Scanner', layout='wide')
st.title("Live OMR Scanner — 50 Questions")

# Create results folder
if not os.path.exists('results'):
    os.makedirs('results')

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    teacher_name = st.text_input("Teacher Name", value="Teacher")
    subject = st.selectbox("Subject", ["Maths", "English", "Science", "History"])
    sensitivity = st.slider("Bubble detection sensitivity", 0.1, 0.9, 0.3, 0.05, help="Lower value means a lighter mark can be detected.")
    st.markdown("---")
    uploaded_key = st.file_uploader("Upload answer_key.json", type=["json"])

if uploaded_key is None:
    st.warning("Please upload your answer_key.json file to begin.")
    st.info("""
    Your `answer_key.json` should look like this:
    ```json
    {
      "1": "A",
      "2": "C",
      "3": "D",
      "4": "B",
      "...": "...",
      "50": "A"
    }
    ```
    """)
    st.stop()

try:
    answer_key = json.load(uploaded_key)
except Exception as e:
    st.error(f"Error reading the answer key: {e}")
    st.stop()

st.success("Answer key loaded successfully!")
st.subheader("Scan OMR Sheet")

# Use st.camera_input for a more reliable camera experience
img_file_buffer = st.camera_input("Take a picture of the OMR sheet (ensure all 4 corners are visible)")

if img_file_buffer is not None:
    with st.spinner("Processing OMR sheet... 🧠"):
        # Get image bytes
        img_bytes = img_file_buffer.getvalue()

        # Process the image
        data, message, overlay_img = process_omr_sheet(img_bytes, answer_key, sensitivity)

    if data:
        st.success("OMR Sheet Processed Successfully! ✅")
        results = data['results']
        correct_count = data['score']

        student_id = f"{teacher_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        st.subheader("Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Student ID:** `{student_id}`")
            st.metric(label="**Final Score**", value=f"{correct_count} / 50")
        
        # Save to Excel
        df_data = {
            "Student": student_id,
            **results, # Unpack the results dictionary
            "Total Correct": correct_count
        }
        df = pd.DataFrame([df_data])
        filename = f"results/{subject}_{teacher_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        
        with col2:
            st.write("📄 **Results saved to Excel**")
            with open(filename, "rb") as file:
                st.download_button(
                    label="Download Results (.xlsx)",
                    data=file,
                    file_name=os.path.basename(filename),
                    mime="application/vnd.ms-excel"
                )

        st.subheader("Answer Details")
        st.dataframe(pd.DataFrame([results]))

        st.subheader("Debugging Overlay")
        st.image(overlay_img, caption="Green = Correctly Marked, Red = Incorrectly Marked", channels="BGR")

    else:
        st.error(f"Processing Failed: {message}")
        if overlay_img is not None:
            st.image(overlay_img, caption="Warped image for debugging", channels="BGR")
