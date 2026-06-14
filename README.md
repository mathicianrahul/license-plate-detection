# 🚗 License Plate Detection & Recognition System (ANPR)

An Advanced Automatic Number Plate Recognition (ANPR) system built using Python, YOLO, OpenCV, and OCR technologies. The project detects vehicle license plates, extracts registration numbers, tracks vehicles across frames, and improves recognition accuracy using image enhancement and confidence-based filtering.

---

## 📌 Overview

This project focuses on intelligent vehicle identification through Computer Vision and OCR techniques.

Traditional license plate recognition systems often suffer from:

* Low image quality
* Motion blur
* Poor lighting conditions
* Missed detections
* Incorrect OCR results

To overcome these challenges, this project introduces image enhancement, confidence scoring, tracking mechanisms, and detection memory to improve recognition reliability.

---

## ✨ Key Features

### 🚘 Vehicle & License Plate Detection

* Detects vehicles and license plates from images/videos.
* Generates bounding boxes for detected vehicles and plates.
* Supports real-time processing.

### 🔍 OCR-Based Number Recognition

* Extracts plate text using OCR.
* Supports EasyOCR / Tesseract OCR.
* Filters unreadable outputs.

### 📈 Image Enhancement

Implemented preprocessing techniques:

* Image Sharpening
* Contrast Enhancement (CLAHE)
* Noise Reduction
* Super Resolution Support (ESRGAN)

These techniques significantly improve OCR accuracy.

### 🎯 Confidence-Based Detection

Stores:

* Detection Confidence (YOLO)
* OCR Confidence

This makes the predictions more explainable and reliable.

### 🔄 Vehicle Tracking

Tracks vehicles across frames using:

* Frame ID
* Track ID
* Plate Number
* Confidence Score

This maintains consistency even when temporary detection failures occur.

### 🧠 Detection Memory

If a license plate is detected successfully once:

* The result is stored
* Reused in subsequent frames
* Reduces missing detections

### 🚫 False Detection Filtering

Automatically removes:

* Low-confidence detections
* Invalid OCR outputs
* Unreadable plate strings

---

## 🛠️ Technologies Used

| Category          | Tools                   |
| ----------------- | ----------------------- |
| Language          | Python                  |
| Computer Vision   | OpenCV, YOLO            |
| OCR               | EasyOCR / Tesseract OCR |
| Data Processing   | NumPy, Pandas           |
| Image Enhancement | CLAHE, Sharpening       |
| Deep Learning     | PyTorch                 |

---

## 🔄 System Workflow

1. Input Image/Video
2. Vehicle Detection
3. License Plate Localization
4. Image Enhancement
5. OCR Recognition
6. Confidence Calculation
7. Detection Filtering
8. Vehicle Tracking
9. Detection Memory Recovery
10. Final Plate Output

---

## 📊 Stored Detection Information

For every detected vehicle:

| Field                | Description              |
| -------------------- | ------------------------ |
| frame_id             | Current frame number     |
| track_id             | Vehicle tracking ID      |
| plate_text           | Recognized license plate |
| detection_confidence | YOLO confidence score    |
| ocr_confidence       | OCR confidence score     |
| vehicle_bbox         | Vehicle bounding box     |
| plate_bbox           | Plate bounding box       |

---

## 📸 Sample Result

Detected Vehicle → License Plate Localization → OCR Recognition → Final Plate Number

Example Output:

Plate Number: MH12AB1234

Detection Confidence: 96%

OCR Confidence: 93%

---

## 🚀 Applications

* Smart Traffic Monitoring
* Automated Toll Collection
* Parking Management Systems
* Vehicle Access Control
* Security & Surveillance
* Smart City Infrastructure
* Law Enforcement Systems

---

## 🎯 Skills Demonstrated

* Computer Vision
* Object Detection
* OCR Systems
* Vehicle Tracking
* Image Processing
* Deep Learning
* Python Development
* AI System Optimization
* Real-Time Video Analytics
* Data Engineering

---

## 👨‍💻 Author

### Rahul Girase

B.Tech Artificial Intelligence & Machine Learning
ISB&M College of Engineering, Pune

🔗 LinkedIn: https://www.linkedin.com/in/rahulgirase

🔗 GitHub: https://github.com/mathicianrahul

---

⭐ If you found this project useful, consider giving it a Star.
