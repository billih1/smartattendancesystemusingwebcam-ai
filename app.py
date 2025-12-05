import streamlit as st
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import firebase_admin
from firebase_admin import credentials, firestore
import datetime
import time
from cvzone.FaceMeshModule import FaceMeshDetector

# --- 1. CONFIG & SETUP ---
st.set_page_config(layout="wide", page_title="Smart Attendance", page_icon="⚡")

@st.cache_resource
def init_resources():
    # 1. Firebase
    if not firebase_admin._apps:
        try:
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
        except Exception as e: return None
    
    # 2. Blink Detector (Pre-load)
    detector = FaceMeshDetector(maxFaces=1)
    return firestore.client(), detector

db, mesh_detector = init_resources()

# --- 2. CORE LOGIC ---
def load_db():
    docs = db.collection("students").stream()
    embs, names = [], []
    for doc in docs:
        d = doc.to_dict()
        if "embedding" in d:
            embs.append(d["embedding"])
            names.append(d["name"])
    return embs, names

def mark_firebase(name):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    doc_id = f"{name}_{today}"
    doc_ref = db.collection("attendance").document(doc_id)
    
    if not doc_ref.get().exists:
        doc_ref.set({
            "name": name,
            "date": today,
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        return True # Newly Marked
    return False # Already Exists

# --- 3. UI LAYOUT ---
st.sidebar.title("⚡ Flash Attendance")
menu = st.sidebar.radio("Menu", ["Live Scanner", "Register", "Dashboard"])

# --- TAB: LIVE SCANNER (THE MOBILE EXPERIENCE) ---
if menu == "Live Scanner":
    st.title("📷 Smart Face Scanner")
    st.caption("⚡ Green = Instant | 🟡 Yellow = Blink Required")
    
    run = st.toggle("ACTIVATE CAMERA", value=False)
    frame_window = st.empty()
    status_msg = st.empty()
    
    if run:
        known_embs, known_names = load_db()
        cap = cv2.VideoCapture(0)
        
        # Blink Variables
        blink_counter = 0
        eye_closed = False
        
        while run:
            ret, frame = cap.read()
            if not ret: break
            
            # 1. Resize for Speed (Mobile feel)
            small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            
            # 2. Face Recognition (ArcFace)
            try:
                # Get Face & Embedding
                objs = DeepFace.represent(
                    img_path=small_frame,
                    model_name="ArcFace",
                    detector_backend="opencv", # Fast
                    enforce_detection=False
                )
                
                # We assume 1 face for attendance
                obj = objs[0]
                curr_emb = obj["embedding"]
                area = obj["facial_area"]
                
                # Scale coordinates back up
                x, y, w, h = area['x']*2, area['y']*2, area['w']*2, area['h']*2
                
                # Find Match
                best_score = 100
                best_name = "Unknown"
                
                for i, target in enumerate(known_embs):
                    a, b = np.array(curr_emb), np.array(target)
                    dist = 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    if dist < best_score:
                        best_score = dist
                        best_name = known_names[i]

                # --- HYBRID LOGIC START ---
                
                # CASE A: CRYSTAL CLEAR MATCH (Instant)
                if best_score < 0.40:
                    color = (0, 255, 0) # Green
                    text = f"CONFIRMED: {best_name}"
                    
                    # Mark Immediately
                    if mark_firebase(best_name):
                        status_msg.success(f"✅ Marked: {best_name}")
                    else:
                        status_msg.info(f"ℹ️ Already Done: {best_name}")

                # CASE B: SUSPICIOUS / FISHY (Blink Check)
                elif 0.40 <= best_score < 0.60:
                    color = (0, 255, 255) # Yellow
                    text = "VERIFYING: PLEASE BLINK..."
                    
                    # Run Blink Detection on the FULL frame
                    frame, faces = mesh_detector.findFaceMesh(frame, draw=False)
                    if faces:
                        face = faces[0]
                        leftUp, leftDown = face[159], face[145]
                        leftLeft, leftRight = face[130], face[243]
                        v_len, _ = mesh_detector.findDistance(leftUp, leftDown)
                        h_len, _ = mesh_detector.findDistance(leftLeft, leftRight)
                        ratio = (v_len / h_len) * 100
                        
                        # Blink Logic
                        if ratio < 35: eye_closed = True
                        if ratio > 40 and eye_closed:
                            blink_counter += 1
                            eye_closed = False
                            
                        # If Blinked -> Confirm Identity
                        if blink_counter >= 1:
                            color = (0, 255, 0)
                            text = f"VERIFIED: {best_name}"
                            if mark_firebase(best_name):
                                status_msg.success(f"✅ Verified & Marked: {best_name}")
                            else:
                                status_msg.info(f"ℹ️ Verified: {best_name}")
                            blink_counter = 0 # Reset
                    
                # CASE C: UNKNOWN
                else:
                    color = (0, 0, 255) # Red
                    text = "UNKNOWN"

                # Draw Clean Box (No Percentages)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 4)
                # Draw Name Tag Background
                cv2.rectangle(frame, (x, y-40), (x+w, y), color, -1)
                cv2.putText(frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            except:
                pass # No face found
                
            # Show Feed
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame)

        cap.release()

# --- TAB: REGISTER ---
elif menu == "Register":
    st.title("👤 Quick Register")
    c1, c2 = st.columns([1,2])
    name = c1.text_input("Name")
    img = c2.camera_input("Photo")
    
    if st.button("Save Profile") and img and name:
        bytes_data = img.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        try:
            emb = DeepFace.represent(cv2_img, model_name="ArcFace", detector_backend="opencv")[0]["embedding"]
            doc_id = name.lower().replace(" ", "_")
            db.collection("students").document(doc_id).set({"name": name, "id": doc_id, "embedding": emb})
            st.success(f"Saved {name}!")
        except: st.error("Face not clear.")

# --- TAB: DASHBOARD ---
elif menu == "Dashboard":
    st.title("📊 Logs")
    if st.button("Refresh"): st.rerun()
    docs = db.collection("attendance").stream()
    data = [{"Name": d.to_dict()['name'], "Time": d.to_dict()['time']} for d in docs]
    if data: st.dataframe(pd.DataFrame(data))