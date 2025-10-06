# Save this code as app.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

# --- UTILITY FUNCTION FOR IMAGE PROCESSING (TAILORED FOR YOUR OMR SHEET) ---
# This function does not need to be changed.
def process_omr_sheet(image_bytes, answer_key, sensitivity, questions=50, choices=4):
    """
    This function is tailored to the specific 5-column layout of your OMR sheet.
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
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
    # 5. Crop to the 'Student Response' area
    h, w = warped.shape[:2]
    y_start = int(h * 0.58)
    answer_block = warped[y_start:, :]
    
    # 6. Process Bubbles within the Cropped Area
    answer_block_gray = cv2.cvtColor(answer_block, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(answer_block_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    question_bubbles = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        if w >= 15 and h >= 15 and 0.8 <= aspect_ratio <= 1.2:
            question_bubbles.append(c)
    
    total_bubbles_expected = questions * choices
    if len(question_bubbles) != total_bubbles_expected:
         return None, f"Error: Found {len(question_bubbles)} bubbles, but expected {total_bubbles_expected}. The image might be blurry or poorly lit.", answer_block

    # 7. Sort bubbles according to the 5-column layout
    bubbles_sorted_by_y = sorted(question_bubbles, key=lambda c: cv2.boundingRect(c)[1])
    
    correct_count = 0
    results = {}
    overlay_img = answer_block.copy()

    for i in range(10): # For each of the 10 rows
        start_idx = i * 20
        row_bubbles = bubbles_sorted_by_y[start_idx : start_idx + 20]
        row_bubbles_sorted_by_x = sorted(row_bubbles, key=lambda c: cv2.boundingRect(c)[0])

        for j in range(5): # For each of the 5 question columns
            question_num = (i + 1) + (j * 10)
            question_choices = row_bubbles_sorted_by_x[j*4 : (j+1)*4]

            marked_index = -1
            max_filled = 0
            for k, c in enumerate(question_choices):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total_pixels = cv2.countNonZero(mask)

                if total_pixels > max_filled:
                    max_filled = total_pixels
                    marked_index = k
            
            marked_answer = None
            if marked_index != -1 and max_filled > (cv2.contourArea(question_choices[marked_index]) * sensitivity):
                marked_answer = "ABCD"[marked_index]

            question_num_str = str(question_num)
            correct_answer = answer_key.get(question_num_str)
            results[question_num_str] = marked_answer

            target_contour = question_choices[marked_index]
            if marked_answer == correct_answer:
                correct_count += 1
                cv2.drawContours(overlay_img, [target_contour], -1, (0, 255, 0), 3) # Green
            elif marked_answer is not None:
                cv2.drawContours(overlay_img, [target_contour], -1, (0, 0, 255), 3) # Red

    return {"results": results, "score": correct_count}, "Success", overlay_img

# --- STREAMLIT APP ---
st.set_page_config(page_title='Live OMR Scanner', layout='wide')
st.title("OMR Sheet Scanner")

if not os.path.exists('results'):
    os.makedirs('results')

with st.sidebar:
    st.header("Settings")
    teacher_name = st.text_input("Teacher Name", value="Teacher")
    subject = st.selectbox("Subject", ["Maths", "English", "Science", "History"])
    sensitivity = st.slider("Bubble detection sensitivity", 0.1, 0.9, 0.3, 0.05, help="Lower value means a lighter mark can be detected.")
    st.markdown("---")
    uploaded_key = st.file_uploader("Upload answer_key.json", type=["json"])

if uploaded_key is None:
    st.warning("Please upload your answer_key.json file to begin.")
    st.stop()

try:
    answer_key = json.load(uploaded_key)
except Exception as e:
    st.error(f"Error reading the answer key: {e}")
    st.stop()

st.success("Answer key loaded successfully!")
st.subheader("Scan OMR Sheet 📸")

# *** NEW: Using st.file_uploader for reliable camera access on mobile ***
# This replaces the entire streamlit-webrtc component.
uploaded_file = st.file_uploader(
    "Take Photo of OMR Sheet",
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=False,
    label_visibility="collapsed"
)

if uploaded_file is not None:
    with st.spinner("Processing OMR sheet... 🧠"):
        img_bytes = uploaded_file.getvalue()
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
        
        # Ensure columns are sorted correctly from 1 to 50 for the DataFrame
        sorted_results = {str(k): results.get(str(k)) for k in range(1, 51)}
        df_data_sorted = {"Student": student_id, **sorted_results, "Total Correct": correct_count}
        df = pd.DataFrame([df_data_sorted])
        filename = f"results/{subject}_{teacher_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(filename, index=False)
        
        with col2:
            st.write("📄 **Results saved to Excel**")
            with open(filename, "rb") as file:
                st.download_button(label="Download Results (.xlsx)", data=file, file_name=os.path.basename(filename))

        st.dataframe(df)

        st.subheader("Debugging Overlay")
        st.image(overlay_img, caption="Green = Correctly Marked, Red = Incorrectly Marked", channels="BGR")

    else:
        st.error(f"Processing Failed: {message}")
        if overlay_img is not None:
            st.image(overlay_img, caption="Debugging Image", channels="BGR")

---

### ## Important: Update `requirements.txt`

Since we are no longer using `streamlit-webrtc`, you should remove it from your `requirements.txt` file on GitHub. This keeps your app clean and efficient.

Your `requirements.txt` should now look like this:
```text
streamlit
opencv-python-headless
numpy
pandas
openpyxl
