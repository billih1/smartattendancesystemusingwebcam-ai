# Smart Attendance System

A real-time, contactless biometric attendance system designed to automate tracking and prevent fraud. This project leverages Computer Vision (OpenCV), Deep Learning (ArcFace), and Cloud Database (Firebase) to mark attendance instantly while implementing liveness detection to prevent spoofing.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Completed-green)

## Project Overview

The system addresses the limitations of manual attendance by providing a secure, automated alternative. It features a role-based dashboard for administrators and students, real-time face tracking, and cloud synchronization for data persistence.

## Features

### Core Functionality
* **Face Recognition:** Utilizes the ArcFace model for high-accuracy identity verification (99%+ precision).
* **Real-Time Detection:** Implements MediaPipe and Haar Cascades for robust face tracking in varying lighting conditions.
* **Liveness Detection:** Prevents proxy attendance via photo/video spoofing using motion and blink analysis.
* **Anti-Duplicate Logic:** Automatically filters multiple detections to ensure a single attendance entry per session.

### Dashboard & Reporting
* **Admin Panel:** Complete control over student registration, attendance monitoring, and database management.
* **Student Portal:** Secure login for students to view personal attendance history and analytics.
* **Data Export:** Automated generation of CSV reports for record-keeping.

## Tech Stack

* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **AI/ML:** DeepFace, TensorFlow, Keras
* **Computer Vision:** OpenCV, MediaPipe, CVZone
* **Database:** Google Firebase Firestore (NoSQL)
* **Data Processing:** Pandas, NumPy

## Installation

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/smart-attendance-system.git](https://github.com/your-username/smart-attendance-system.git)
    cd smart-attendance-system
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration**
    * Create a project on the Google Firebase Console.
    * Generate a Service Account Key (JSON).
    * Rename the file to `serviceAccountKey.json` and place it in the root directory.

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## Project Structure

* `app.py`: Main application entry point containing Dashboard, Logic, and UI.
* `best.pt`: Custom trained YOLO model for enhanced detection (optional integration).
* `requirements.txt`: List of Python dependencies.
* `serviceAccountKey.json`: Firebase authentication credentials (not included in repo).

## Usage

1.  **Initialization:** Run the application. The system will automatically connect to the Firebase instance.
2.  **Registration:** Navigate to the "Register" tab to enroll new users. The system captures facial embeddings and stores them in the cloud.
3.  **Attendance Mode:** Switch to the "Live Scanner". The system uses a color-coded feedback loop:
    * **Green:** Confirmed match.
    * **Yellow:** Liveness check required (blink verification).
    * **Red:** Unknown or unauthorized user.
4.  **Reports:** Access the "Reports" tab to view logs or download the daily attendance sheet.

Developed by Tayyab AI & Computer Vision Enthusiast