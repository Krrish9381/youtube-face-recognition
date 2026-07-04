# 🎥 YouTube Face Recognition using YOLOv8n and ArcFace

A web application that recognizes known people in any YouTube video using **YOLOv8n-Face** for face detection and **ArcFace** for face recognition.

Simply paste a YouTube video link, and the application processes the video, detects faces, compares them against a database of known faces, and labels recognized individuals in the output video.

---

## 🚀 Features

- 🔗 Accepts any public YouTube video URL
- 🎯 Face detection using YOLOv8n-Face
- 🧠 Face recognition using ArcFace embeddings
- 👤 Identifies known people from a custom face database
- 📦 FastAPI backend
- 🌐 Simple frontend for video URL submission
- 🎥 Processes complete videos frame by frame
- 📍 Draws bounding boxes and names on detected faces

---

## 🛠 Tech Stack

### Backend
- Python
- FastAPI
- Uvicorn

### Frontend
- HTML
- CSS
- JavaScript

### AI & Computer Vision
- YOLOv8n-Face
- ArcFace (InsightFace)
- OpenCV
- NumPy
- ONNX Runtime

---

## 📂 Project Structure

```text
youtube-face-recognition/
│
├── app.py                   # FastAPI backend
├── frontend.py              # Frontend UI
├── video-recognition.py     # Video processing pipeline
├── yolov8n-face.pt          # YOLOv8 Face model
├── known_faces/             # Images of known people
├── face_db.pkl              # Face embeddings database
├── outputs/                 # Processed videos
├── requirements.txt
└── README.md
```

---

## ⚙️ Workflow

```
YouTube URL
      │
      ▼
Download Video
      │
      ▼
Extract Frames
      │
      ▼
YOLOv8n Face Detection
      │
      ▼
ArcFace Embedding Extraction
      │
      ▼
Compare with Known Faces
      │
      ▼
Draw Bounding Boxes + Names
      │
      ▼
Generate Output Video
```

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/your-username/youtube-face-recognition.git
```

Move into the project folder

```bash
cd youtube-face-recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the FastAPI server

```bash
uvicorn app:app --reload
```

Run the frontend

```bash
python frontend.py
```

Open the application in your browser and paste a YouTube video URL.

---

## 📸 How It Works

1. Enter a YouTube video URL.
2. The video is downloaded.
3. Each frame is processed.
4. YOLOv8n detects every face.
5. ArcFace generates embeddings.
6. Embeddings are matched with stored known faces.
7. Recognized people are labeled.
8. The processed video is returned.

---

## 📋 Requirements

- Python 3.10+
- FastAPI
- OpenCV
- Ultralytics
- InsightFace
- ONNX Runtime
- NumPy
- Pillow

Install everything using

```bash
pip install -r requirements.txt
```

---

## 🎯 Future Improvements

- Real-time webcam recognition
- Live YouTube stream support
- Face registration from the web interface
- GPU acceleration
- Face tracking (DeepSORT/ByteTrack)
- Multi-threaded video processing
- Docker deployment

---

## 📷 Demo

Add screenshots or a GIF here.

Example:

- Home Page
- Enter YouTube URL
- Face Detection Output
- Final Processed Video

---

## 👨‍💻 Author

**Krrish**

Computer Science Engineering Student

Interested in Artificial Intelligence, Computer Vision, Machine Learning, and Full Stack Development.

---

## ⭐ Support

If you found this project useful, please consider giving it a **Star ⭐** on GitHub.
