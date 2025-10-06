import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

# --- CORE IMAGE PROCESSING ENGINE ---
def process_omr_sheet(image_bytes, answer_key, sensitivity, questions=50, choices=4):
    """
    A robust OMR scanning function built from scratch for the specific sheet layout.
    It now finds the answer block by locating all answer bubbles first.
    """
    try:
        # 1. Decode Image from Bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Error: Could not decode the image file.", None

        # 2. Pre-processing for Contour Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        # 3. Find the Main Page Contour (the entire sheet)
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
            return None, "Fatal Error: Could not find the four corners of the OMR sheet. Please ensure the full sheet is visible on a dark background.", None

        # 4. Apply Perspective Warp to get a top-down view of the page
        pts = doc_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] # Top-left
        rect[2] = pts[np.argmax(s)] # Bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] # Top-right
        rect[3] = pts[np.argmax(diff)] # Bottom-left
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
        
        # 5. Find the 'STUDENT RESPONSE' block by finding all bubbles first
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        all_contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        question_bubbles = []
        for c in all_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            # Filter for bubble-like contours in the bottom 60% of the page
            if 15 <= w <= 40 and 15 <= h <= 40 and 0.8 <= aspect_ratio <= 1.2 and y > maxHeight * 0.4:
                question_bubbles.append(c)

        total_bubbles_expected = questions * choices
        if len(question_bubbles) < total_bubbles_expected * 0.9: # Allow for a few missed bubbles initially
            msg = f"Error: Could not find enough answer bubbles ({len(question_bubbles)} found, 200 expected). The image may be too dark, blurry, or have shadows."
            return None, msg, warped

        # 6. Create a bounding box around all found bubbles to define the answer block
        all_bubble_points = np.concatenate(question_bubbles)
        x, y, w, h = cv2.boundingRect(all_bubble_points)

        # Add some padding to the crop for better processing
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(warped.shape[1] - x, w + padding * 2)
        h = min(warped.shape[0] - y, h + padding * 2)

        answer_block = warped[y:y+h, x:x+w]
        answer_block_thresh = thresh[y:y+h, x:x+w]

        # 7. Re-find bubbles within the now accurately cropped Answer Block for grading
        bubble_contours, _ = cv2.findContours(answer_block_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_question_bubbles = []
        for c in bubble_contours:
            (cx, cy, cw, ch) = cv2.boundingRect(c)
            aspect_ratio = cw / float(ch)
            if cw >= 15 and ch >= 15 and 0.8 <= aspect_ratio <= 1.2:
                final_question_bubbles.append(c)
        
        if len(final_question_bubbles) != total_bubbles_expected:
            msg = f"Error: Found {len(final_question_bubbles)} bubbles in the isolated answer block, but expected {total_bubbles_expected}. Please retake the photo."
            return None, msg, answer_block

        # 8. Sort Bubbles based on the 5-column layout
        bubbles_sorted_by_y = sorted(final_question_bubbles, key=lambda c: cv2.boundingRect(c)[1])
        
        correct_count = 0
        results = {}
        overlay_img = answer_block.copy()

        for i in range(10): # For each of the 10 visual rows
            start_idx = i * 20 # 5 questions * 4 choices
            row_bubbles = bubbles_sorted_by_y[start_idx : start_idx + 20]
            row_bubbles_sorted_by_x = sorted(row_bubbles, key=lambda c: cv2.boundingRect(c)[0])

            for j in range(5): # For each of the 5 question columns
                question_num = (i + 1) + (j * 10)
                question_choices = row_bubbles_sorted_by_x[j*4 : (j+1)*4]

                marked_index = -1
                max_filled = 0
                for k, c in enumerate(question_choices):
                    mask = np.zeros(answer_block_thresh.shape, dtype="uint8")
                    cv2.drawContours(mask, [c], -1, 255, -1)
                    mask = cv2.bitwise_and(answer_block_thresh, answer_block_thresh, mask=mask)
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

                if marked_index != -1:
                    target_contour = question_choices[marked_index]
                    if marked_answer == correct_answer:
                        correct_count += 1
                        cv2.drawContours(overlay_img, [target_contour], -1, (0, 255, 0), 3) # Green
                    else:
                        cv2.drawContours(overlay_img, [target_contour], -1, (0, 0, 255), 3) # Red

        return {"results": results, "score": correct_count}, "Scan Successful!", overlay_img
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}", None


# --- STREAMLIT USER INTERFACE ---
def main():
    st.set_page_config(page_title="OMR Scanner Pro", layout="wide", page_icon="📄")

    # --- UI STYLING ---
    st.markdown("""
        <style>
            .main { background-color: #F0F2F6; }
            .st-emotion-cache-16txtl3 { padding: 2rem 3rem 1rem; }
            .st-emotion-cache-1y4p8pa { max-width: 100%; }
            .st-emotion-cache-uf99v8 { font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📄 OMR Scanner Pro")
    st.markdown("A modern OMR sheet scanner with high accuracy and a user-friendly interface.")

    if not os.path.exists('results'):
        os.makedirs('results')

    # --- SIDEBAR FOR SETTINGS ---
    with st.sidebar:
        st.header("⚙️ Settings")
        teacher_name = st.text_input("Teacher's Name", value="Teacher")
        subject = st.selectbox("Subject", ["Maths", "English", "Science", "History"])
        sensitivity = st.slider("Bubble Detection Sensitivity", 0.1, 0.9, 0.25, 0.05, 
                                help="Lower value detects lighter marks. Default is 0.25.")
        st.markdown("---")
        st.header("🔑 Answer Key")
        uploaded_key = st.file_uploader("Upload 'answer_key.json'", type=["json"])
        st.info("Ensure your JSON key is a simple map, e.g., `{\"1\": \"A\", \"2\": \"C\", ...}`")

    # --- MAIN WORKFLOW ---
    if uploaded_key is None:
        st.warning("Please upload the `answer_key.json` file via the sidebar to begin.")
        return

    try:
        answer_key = json.load(uploaded_key)
    except Exception as e:
        st.error(f"Error reading the answer key file: {e}")
        return

    st.success("Answer key loaded successfully. You can now scan an OMR sheet.")
    
    uploaded_file = st.file_uploader(
        "**Upload or Take a Photo of the OMR Sheet**",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_file:
        with st.spinner("Analyzing image... This may take a moment. 🧠"):
            img_bytes = uploaded_file.getvalue()
            data, message, overlay_img = process_omr_sheet(img_bytes, answer_key, sensitivity)

        if data:
            st.success(f"✅ {message}")
            results = data['results']
            correct_count = data['score']
            student_id = f"{teacher_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            st.header("📊 Scan Results")
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric(label="**Final Score**", value=f"{correct_count} / 50")
                st.write(f"**Student ID:** `{student_id}`")
                
                # Create and save Excel file
                sorted_results = {str(k): results.get(str(k)) for k in range(1, 51)}
                df_data = {"Student": student_id, **sorted_results, "Total Correct": correct_count}
                df = pd.DataFrame([df_data])
                filename = f"results/{subject}_{student_id}.xlsx"
                df.to_excel(filename, index=False)
                
                with open(filename, "rb") as file:
                    st.download_button(
                        label="📥 Download Results (.xlsx)",
                        data=file,
                        file_name=os.path.basename(filename)
                    )

            with col2:
                st.image(overlay_img, caption="Graded Answer Sheet (Green=Correct, Red=Incorrect)", use_column_width=True)
            
            with st.expander("View Detailed Answer Data"):
                st.dataframe(df)

        else:
            st.error(f"❌ {message}")
            if overlay_img is not None:
                st.image(overlay_img, caption="Debugging Image: This is the area the scanner tried to analyze.", use_column_width=True)

if __name__ == "__main__":
    main()

