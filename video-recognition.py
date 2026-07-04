import os
import cv2
import pickle
import numpy as np
import onnxruntime as ort
import torch

from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis

# -------------------------------------------------
# InsightFace Model Directory
# -------------------------------------------------
os.environ["INSIGHTFACE_HOME"] = r"D:\insightface"

print("=" * 60)
print("Torch Version :", torch.__version__)
print("CUDA Version  :", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))

print("\nONNX Runtime :", ort.__version__)
print("Available Providers :", ort.get_available_providers())
print("=" * 60)

# -------------------------------------------------
# Load Face Database
# -------------------------------------------------
with open("face_db.pkl", "rb") as f:
    known_embeddings, known_names = pickle.load(f)

known_embeddings = [
    emb / np.linalg.norm(emb)
    for emb in known_embeddings
]

print("Database Loaded")
print("Total Faces :", len(known_names))
print("=" * 60)

# -------------------------------------------------
# Initialize ArcFace
# -------------------------------------------------
app = FaceAnalysis(
    name="buffalo_l",
    root=r"D:\insightface",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("\nInsightFace Models")

for name, model in app.models.items():
    print("----------------------------------")
    print("Model :", name)
    print("Provider :", model.session.get_providers())

print("=" * 60)
print("ArcFace Ready")
print("=" * 60)

# -------------------------------------------------
# Webcam
# -------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to Quit")

# -------------------------------------------------
# Recognition Loop
# -------------------------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Face Detection + Embedding
    faces = app.get(frame)

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)

        best_score = -1
        best_name = "Unknown"

        for db_emb, db_name in zip(
            known_embeddings,
            known_names
        ):

            score = cosine_similarity(
                [embedding],
                [db_emb]
            )[0][0]

            if score > best_score:
                best_score = score
                best_name = db_name

        if best_score < 0.40:
            best_name = "Unknown"

        color = (0, 255, 0)

        if best_name == "Unknown":
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            f"{best_name} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()