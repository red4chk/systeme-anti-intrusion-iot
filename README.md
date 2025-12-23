

#  Intelligent Anti-Intrusion System (IAIS)

> **University Project – Computer Vision & Artificial Intelligence**
> 100% Software • No Hardware • Python • Vibe Coding Approach

---

##  Project Overview

This project consists of the design and implementation of an **intelligent anti-intrusion system** entirely simulated using software, without any physical hardware (no cameras, no sensors).

It combines:

* **Simulated IoT data**
* **Computer Vision**
* **Machine Learning**
* **Decision fusion logic**

The goal is to validate the **AI concepts used in real smart-home security systems** through a fully executable software prototype.

---

##  Project Objectives

###  General Objective

Develop a system capable of:

* Detecting intrusions
* Identifying suspicious behaviors
* Detecting anomalies
* Generating software-based alerts

###  Specific Objectives

* Simulate IoT sensors using CSV data
* Apply ML algorithms for intrusion classification
* Detect people in surveillance videos using YOLO
* Track individuals across video frames
* Fuse IoT and video decisions
* Visualize and log results

---

##  Project Constraints

| Element     | Constraint         |
| ----------- | ------------------ |
| Hardware    | ❌ None             |
| Sensors     | ❌ None             |
| Camera      | ❌ None             |
| Environment | ✅ PC only          |
| Language    | Python             |
| IDE         | VS Code            |
| Data        | Simulated / Public |

---

##  Functional Architecture

Simulated IoT Data ──► Machine Learning ──┐
├──► Decision System ──► Alert
Pre-recorded Video ──► Computer Vision ──┘

---

##  PART A — IoT DATA (SIMULATION)

### A.1 IoT Data Generation

**Objective**
Create a CSV file simulating smart-home sensors.

**Variables**

* motion (0 / 1)
* sound_level (0–100)
* vibration (0 / 1)
* temperature (°C)
* hour (0–23)
* label (0 = normal, 1 = intrusion)

**Steps**

1. Create a Python script
2. Generate coherent random data
3. Save data to CSV
4. Manually verify data integrity

 **Deliverable:** `iot_data.csv`

---

### A.2 Data Preprocessing

**Steps**

* Load CSV file
* Handle missing values
* Normalize features
* Split data into train / test sets

 **Deliverable:** Clean dataset ready for ML

---

### A.3 Machine Learning Classification

**Algorithms**

* Random Forest
* SVM (optional)
* XGBoost (bonus)

**Steps**

* Train the model
* Test on unseen data
* Compute:

  * Accuracy
  * Precision
  * Recall
  * Confusion Matrix

 **Deliverable:** ML results + evaluation graphs

---

##  PART B — ANOMALY DETECTION

### B.1 Objective

Detect unusual behavior without relying on labels.

**Methods**

* Isolation Forest
* Autoencoder (bonus)

**Steps**

* Train on normal data only
* Compute anomaly scores
* Visualize detected anomalies

 **Deliverable:** Anomaly detection plots

---

##  PART C — COMPUTER VISION (VIDEO)

### C.1 Video Data

**Steps**

* Download a surveillance-style video
* Place it in `data/videos/`
* Test video loading with OpenCV

---

### C.2 Person Detection (YOLOv8)

**Steps**

* Install YOLOv8
* Load the pretrained model
* Detect persons in each frame
* Draw bounding boxes

 **Deliverable:** Annotated video

---

### C.3 Tracking (DeepSORT)

**Steps**

* Assign unique IDs to detected persons
* Track movement across frames
* Display IDs on the video

---

### C.4 Video Suspicion Logic

**Rules**

* Human presence detected
* Long presence duration
* Entry into restricted zones

 **Deliverable:** “Video Intrusion Detected” message

---

##  PART D — DECISION FUSION

### D.1 Objective

Combine IoT and video decisions.

**Logic**
if intrusion_iot or intrusion_video:
  alert = True

 **Deliverable:** Final intrusion decision

---

##  PART E — OUTPUTS

**System Outputs**

* Console messages
* Saved annotated videos
* ML graphs
* Alert messages

---

##  Project Structure

anti_intrusion_project/
├── data/
│   ├── iot/
│   │   └── iot_data.csv
│   └── videos/
│       └── surveillance.mp4
│
├── src/
│   ├── iot/
│   │   ├── generate_data.py
│   │   ├── preprocess.py
│   │   ├── train_classifier.py
│   │   └── anomaly_detection.py
│   │
│   ├── vision/
│   │   ├── yolo_detect.py
│   │   ├── tracking.py
│   │   └── zone_logic.py
│   │
│   ├── fusion/
│   │   └── decision_system.py
│   │
│   └── main.py
│
├── results/
│   ├── figures/
│   ├── videos/
│   └── logs/
│
├── requirements.txt
└── README.md

---

##  Description of Key Files

| File                 | Description                    |
| -------------------- | ------------------------------ |
| generate_data.py     | Generates simulated IoT data   |
| preprocess.py        | Cleans and normalizes data     |
| train_classifier.py  | Trains and evaluates ML models |
| anomaly_detection.py | Detects anomalies              |
| yolo_detect.py       | Person detection using YOLO    |
| tracking.py          | Person tracking with DeepSORT  |
| decision_system.py   | Decision fusion logic          |
| main.py              | Runs the full system           |

---

##  Development Methodology — Vibe Coding

The system is built **module by module**:

1. IoT simulation → OK
2. Machine Learning → OK
3. Video processing → OK
4. Decision fusion → OK

 Never develop everything at once.

---

##  Key Sentence for the Jury

“This project simulates a complete intelligent anti-intrusion system by independently validating each AI component, without any hardware dependency.”

---

##  Next Steps

* Real-time video input
* Real IoT sensors
* Web dashboard
* Alert notifications (email / SMS)

---

**Author:** Chawki Mohamed Reda
**Module:** Computer Vision / Artificial Intelligence

