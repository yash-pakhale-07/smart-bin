♻️ Smart Bin AI — Intelligent Waste Segregation System

Smart Bin AI is a real-time computer vision-based system that detects waste items, classifies them into categories, reads plastic resin codes using OCR, and gamifies the disposal process to encourage sustainable behavior.

📊 a. Datasets Used & Preprocessing
📁 Datasets Used

This project uses multiple datasets from Kaggle:

TrashNet Dataset (garythung/trashnet)
Garbage Classification Dataset (asdasdasasdas/garbage-classification)
Waste Classification Dataset (techsash/waste-classification-data)

🔁 Preprocessing Steps

The datasets were cleaned and standardized using the following pipeline:

Downloaded datasets using Kaggle API
Mapped original classes into 4 categories:

Plastic
Paper
Metal
Glass

Removed irrelevant categories such as organic waste
Merged all datasets into a unified dataset

Split data:
80% Training
20% Validation

Resized all images to 224 × 224 pixels
Applied data augmentation:
Horizontal flip
Rotation (±15°)
Brightness variation (±20%)

📂 Final Dataset Structure
data/
 ├── train/
 │    ├── Plastic/
 │    ├── Paper/
 │    ├── Metal/
 │    └── Glass/
 └── val/
      ├── Plastic/
      ├── Paper/
      ├── Metal/
      └── Glass/
🤖 b. Model Used & Performance Metrics
🧠 Model Details

The model was trained using
Google Teachable Machine

Model Type: TensorFlow Lite (TFLite)
Input Size: 224 × 224 × 3
Classes: Plastic, Paper, Metal, Glass
⚙️ Training Configuration
Epochs: 80
Batch Size: 32
Learning Rate: 0.001
📈 Performance Metrics
Training Accuracy: ~90%
Validation Accuracy: ~85–92%
Inference Speed: < 100 ms per frame (CPU)

✔ The model is optimized for real-time webcam-based detection.

⭐ c. Key Features
🎥 1. Real-Time Waste Detection
Detects waste using webcam input
Classifies into:
Plastic
Paper
Metal
Glass

🔍 2. Plastic Resin Code Detection (OCR)
Uses pytesseract
Extracts resin codes (1–7)
Provides recycling instructions

♻️ 3. Smart Disposal Guidance
Suggests correct bins:
Blue → Plastic
Green → Paper
Grey → Metal
Brown → Glass

🎮 4. Gamification System
Carbon points system
Streak tracking

Badge unlocking:
Recycling Rookie
Eco Warrior
Carbon Saver
Planet Protector

📊 5. Feedback System
Instant feedback:
✅ Correct disposal
❌ Wrong disposal
Disposal history tracking

⚡ 6. Optimized Performance
Lightweight TFLite model
Frame skipping for efficiency
Fully offline system

🚀 4. Optional: Additional Details
🧠 Model Architecture
Based on transfer learning using a lightweight CNN architecture
Optimized for mobile and edge devices
Converted to TensorFlow Lite for faster inference

⚙️ System Architecture
Webcam Input → Image Preprocessing → TFLite Model → Prediction
        ↓
Plastic → OCR → Resin Code Detection
        ↓
Gamification + UI Feedback (Streamlit)

💡 Future Improvements
Add more waste categories
Improve OCR accuracy
Deploy on IoT smart bins
Add cloud dashboard for analytics