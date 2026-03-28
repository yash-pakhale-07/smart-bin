# ♻️ Smart Bin AI --- Intelligent Waste Segregation System

Smart Bin AI is a real-time computer vision-based system that detects
waste items, classifies them, reads plastic resin codes using OCR, and
gamifies disposal.

------------------------------------------------------------------------

## 📊 Datasets Used & Preprocessing

Datasets from Kaggle: - TrashNet - Garbage Classification - Waste
Classification

### Preprocessing:

-   Mapped to 4 classes: Plastic, Paper, Metal, Glass
-   Removed irrelevant classes
-   Merged datasets
-   80/20 train-validation split
-   Resized to 224x224
-   Applied augmentation (flip, brightness, rotation)

------------------------------------------------------------------------

## 🤖 Model & Performance

-   Model: TensorFlow Lite (Teachable Machine)
-   Input: 224x224
-   Epochs: 60
-   Batch Size: 32
-   Learning Rate: 0.001

### Performance:

-   Training Accuracy: \~90%
-   Validation Accuracy: \~85--92%
-   Inference Time: \<100ms

------------------------------------------------------------------------

## ⭐ Key Features

-   Real-time webcam detection
-   OCR for plastic resin codes
-   Smart bin suggestions
-   Gamification (points, streaks, badges)
-   Instant feedback system
-   Offline lightweight model

------------------------------------------------------------------------

## 🚀 Additional Details

### Architecture:

Webcam → Preprocessing → Model → Prediction → OCR → UI

### Future Improvements:

-   More categories
-   Better OCR
-   IoT integration
-   Cloud analytics

------------------------------------------------------------------------

## ▶️ Demo
https://drive.google.com/file/d/1hS0pMIB2rVX4hmCxdDudaaxxsLQRZhJx/view?usp=drivesdk
