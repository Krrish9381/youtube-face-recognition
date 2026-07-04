import os
import cv2
import pickle
import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import yt_dlp

app = FastAPI(title="YouTube Face Recognition API")

# -------------------------------------------------
# Setup & Database Load
# -------------------------------------------------
os.environ["INSIGHTFACE_HOME"] = r"D:\insightface"

with open("face_db.pkl", "rb") as f:
    known_embeddings, known_names = pickle.load(f)

# Normalize database embeddings upfront for cleaner cosine similarity comparisons
known_embeddings = [emb / np.linalg.norm(emb) for emb in known_embeddings]

# Initialize ArcFace with CUDA Acceleration
face_app = FaceAnalysis(
    name="buffalo_l",
    root=r"D:\insightface",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
# Using a larger internal det_size helps detect smaller faces without distorting raw embeddings
face_app.prepare(ctx_id=0, det_size=(640, 640))

def get_youtube_stream_url(youtube_url: str) -> str:
    """Extracts the raw direct stream URL from a standard YouTube link."""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def generate_youtube_frames(raw_stream_url: str):
    cap = cv2.VideoCapture(raw_stream_url)
    
    tracked_faces = {}
    face_id_counter = 0
    frame_counter = 0
    FRAME_SKIP = 3  # Process heavy AI model every 3rd frame
    
    # Cosine similarity threshold for a confident match
    MATCH_THRESHOLD = 0.55 

    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame_counter += 1
        
        # 1. RUN DETECTOR AND RECOGNITION EVERY N-TH FRAME
        if frame_counter % FRAME_SKIP == 0 or not tracked_faces:
            # Native resolution passed directly to prevent INTER_CUBIC interpolation artifacts
            faces = face_app.get(frame)
            updated_tracked_faces = {}

            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                
                # Normalize the newly extracted embedding
                embedding = face.embedding / np.linalg.norm(face.embedding)

                best_score = -1
                best_name = "Unknown"

                # Database lookup
                for db_emb, db_name in zip(known_embeddings, known_names):
                    score = cosine_similarity([embedding], [db_emb])[0][0]
                    if score > best_score:
                        best_score = score
                        best_name = db_name

                if best_score < MATCH_THRESHOLD:
                    best_name = "Unknown"

                # 2. TRACKING VIA CENTROID PROXIMITY (Tightened for group/crowded shots)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                matched_id = None
                assigned_name = best_name
                
                for fid, data in tracked_faces.items():
                    bx1, by1, bx2, by2 = data["box"]
                    # Tightened the overlap window from 40 to 15 to stop neighboring players from swapping IDs
                    if bx1 - 15 <= cx <= bx2 + 15 and by1 - 15 <= cy <= by2 + 15:
                        # Only inherit historical track name if the current frame match is uncertain
                        if best_name == "Unknown" and data["name"] != "Unknown":
                            assigned_name = data["name"]
                            best_score = data.get("score", best_score)
                        matched_id = fid
                        break

                if matched_id is None:
                    matched_id = face_id_counter
                    face_id_counter += 1

                updated_tracked_faces[matched_id] = {
                    "box": (x1, y1, x2, y2), 
                    "name": assigned_name, 
                    "score": best_score
                }
            
            tracked_faces = updated_tracked_faces

        # 3. RENDER THE BOUNDING BOXES ON EVERY FRAME
        for fid, data in tracked_faces.items():
            x1, y1, x2, y2 = data["box"]
            assigned_name = data["name"]
            best_score = data.get("score", -1)

            color = (0, 255, 0) if assigned_name != "Unknown" else (0, 0, 255)
            
            if assigned_name != "Unknown" and best_score > 0:
                label_text = f"{assigned_name} ({best_score:.2f})"
            else:
                label_text = assigned_name

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, 
                label_text, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

        # Encode frame to JPEG bytes
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.get("/video_feed")
async def video_feed(url: str = Query(..., description="The YouTube URL to analyze")):
    try:
        raw_stream = get_youtube_stream_url(url)
        return StreamingResponse(
            generate_youtube_frames(raw_stream), 
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        return {"error": f"Failed to load stream: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)