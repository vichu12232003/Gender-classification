import cv2
import math
import argparse
import logging
from datetime import datetime
import time
import os
import csv
import numpy as np
from collections import deque
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def highlightFace(net, frame, conf_threshold=0.5):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


parser=argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--rtsp', help='RTSP stream URL')
parser.add_argument('--scale', type=float, default=2, help='Frame upscaling factor (default: 2)')
parser.add_argument('--conf', type=float, default=0.5, help='Face detection confidence threshold (default: 0.5)')
parser.add_argument('--padding', type=int, default=20, help='Padding around face for cropping (default: 20)')
args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Male','Female']

try:
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 3  # seconds

def open_video_stream():
    if args.rtsp:
        return cv2.VideoCapture(args.rtsp)
    elif args.image:
        return cv2.VideoCapture(args.image)
    else:
        return cv2.VideoCapture(0)

video = open_video_stream()
if not video.isOpened():
    print("Error: Could not open video stream.")
    exit()
padding=args.padding

# Set up logging
logging.basicConfig(filename='gender_age_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

GENDER_CONFIDENCE_THRESHOLD = 0.8  # Set your desired threshold

SCALE_FACTOR = args.scale  # Upscale the frame to help detect smaller/farther faces

# Setup for JSON export
JSON_FILE = 'predictions.json'
if not os.path.exists(JSON_FILE):
    with open(JSON_FILE, 'w') as f:
        json.dump([], f)

last_time = time.time()
fps = 0
male_count = 0
female_count = 0
uncertain_count = 0

# Face memory for cooldown (1 minute)
FACE_MEMORY = []  # Each entry: (x, y, w, h, last_seen_timestamp)
COOLDOWN_SECONDS = 60  # 1 minute
IOU_THRESHOLD = 0.5  # Intersection-over-union threshold for matching faces

# Add a flag to track no face detected state
no_face_shown = False

# Add user ID tracking
detected_users = []  # List of (x1, y1, x2, y2, last_seen, user_id)
user_counter = 1

def get_user_label(user_id):
    return f"user {user_id}"

def iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

# --- QDRANT CLOUD CONFIGURATION ---
QDRANT_URL = os.getenv("QDRANT_URL", "https://f4f7c042-cfd1-4cc9-a3d6-4633a9984411.us-east4-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.kHbOM8FIMG9ix58eSBB7znvNMI7n-X5LSrMdXBuABFs")

# Initialize Qdrant client
try:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    print("Connected to Qdrant cloud database")
except Exception as e:
    print(f"Failed to connect to Qdrant: {e}")
    qdrant_client = None

# Create collection for gender data
COLLECTION_NAME = "gender_detections"
try:
    if qdrant_client:
        # Check if collection exists, if not create it
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=4, distance=Distance.COSINE)
            )
            print(f"Created collection: {COLLECTION_NAME}")
        else:
            print(f"Collection {COLLECTION_NAME} already exists")
except Exception as e:
    print(f"Error with Qdrant collection: {e}")

while True:
    hasFrame, frame = video.read()
    if not hasFrame:
        print("Error: Lost connection to the video stream or end of stream.")
        reconnect_count = 0
        while reconnect_count < RECONNECT_ATTEMPTS:
            print(f"Attempting to reconnect... ({reconnect_count + 1}/{RECONNECT_ATTEMPTS})")
            time.sleep(RECONNECT_DELAY)
            video.release()
            video = open_video_stream()
            if video.isOpened():
                hasFrame, frame = video.read()
                if hasFrame:
                    print("Reconnected to the video stream.")
                    break
            reconnect_count += 1
        else:
            print("Failed to reconnect after multiple attempts. Exiting.")
            break
        if not hasFrame:
            continue

    # Upscale the frame
    frame_upscaled = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_LINEAR)
    resultImg_up, faceBoxes_up = highlightFace(faceNet, frame_upscaled, conf_threshold=args.conf)
    # Scale face boxes back to original frame size
    faceBoxes = [[int(x/SCALE_FACTOR) for x in box] for box in faceBoxes_up]
    # Draw rectangles on the original frame
    resultImg = frame.copy()
    for box in faceBoxes:
        cv2.rectangle(resultImg, (box[0], box[1]), (box[2], box[3]), (0,255,0), int(round(frame.shape[0]/150)), 8)

    if not faceBoxes:
        if not no_face_shown:
            print("No face detected")
            no_face_shown = True
    else:
        no_face_shown = False

    male_count = 0
    female_count = 0
    uncertain_count = 0

    now = time.time()
    # Clean up old memory
    FACE_MEMORY[:] = [entry for entry in FACE_MEMORY if now - entry[4] < COOLDOWN_SECONDS]
    detected_users[:] = [entry for entry in detected_users if now - entry[4] < COOLDOWN_SECONDS]

    for idx, faceBox in enumerate(faceBoxes):
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        # Skip if face crop is empty
        if face.size == 0:
            continue

        # Check if this face is already in memory (recently seen)
        matched = False
        user_id = None
        for i, (mx1, my1, mx2, my2, mtime) in enumerate(FACE_MEMORY):
            if iou(faceBox, [mx1, my1, mx2, my2]) > IOU_THRESHOLD:
                if now - mtime < COOLDOWN_SECONDS:
                    matched = True
                    # Find user_id for this face
                    for ux1, uy1, ux2, uy2, utime, uid in detected_users:
                        if iou(faceBox, [ux1, uy1, ux2, uy2]) > IOU_THRESHOLD:
                            user_id = uid
                            break
                    break
        if matched:
            continue  # Skip prediction/logging for this face
        # Assign new user_id
        user_id = user_counter
        user_counter += 1
        detected_users.append((faceBox[0], faceBox[1], faceBox[2], faceBox[3], now, user_id))
        FACE_MEMORY.append((faceBox[0], faceBox[1], faceBox[2], faceBox[3], now))

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender_confidence = genderPreds[0].max()
        gender=genderList[genderPreds[0].argmax()]
        if gender_confidence > GENDER_CONFIDENCE_THRESHOLD:
            display_gender = gender
            if gender == 'Male':
                color = (255, 0, 0)  # Blue
                male_count += 1
            else:
                color = (203, 192, 255)  # Pink
                female_count += 1
        else:
            display_gender = 'Uncertain'
            color = (128, 128, 128)  # Gray
            uncertain_count += 1
        display_text = f'{display_gender} ({gender_confidence:.2f})'
        print(display_text)

        # Log the result with timestamp and confidence
        logging.info(f'Gender: {display_gender} (Confidence: {gender_confidence:.2f})')

        # Write to JSON
        detection = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
            'user': get_user_label(user_id),
            'gender': display_gender,
            'confidence': float(f'{gender_confidence:.2f}')
        }
        # Append to JSON file
        with open(JSON_FILE, 'r+') as f:
            data = json.load(f)
            data.append(detection)
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
        
        # Store in Qdrant database
        if qdrant_client:
            try:
                # Create a simple vector representation (you can modify this based on your needs)
                # Using [user_id, confidence, gender_encoded, timestamp_encoded]
                gender_encoded = 1 if display_gender == 'Male' else (0 if display_gender == 'Female' else 0.5)
                timestamp_encoded = float(detection['timestamp'].split('_')[2]) / 1000000  # Normalize timestamp
                
                vector = [float(user_id), gender_confidence, gender_encoded, timestamp_encoded]
                
                point = PointStruct(
                    id=int(time.time() * 1000000),  # Use timestamp as unique ID
                    vector=vector,
                    payload={
                        'timestamp': detection['timestamp'],
                        'user': detection['user'],
                        'gender': detection['gender'],
                        'confidence': detection['confidence']
                    }
                )
                
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[point]
                )
                print(f"Stored detection for {detection['user']} in Qdrant")
            except Exception as e:
                print(f"Error storing in Qdrant: {e}")

        # Draw rectangle and text
        cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), color, int(round(frame.shape[0]/150)), 8)
        cv2.putText(resultImg, display_text, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    # FPS calculation
    new_time = time.time()
    fps = 1 / (new_time - last_time)
    last_time = new_time
    cv2.putText(resultImg, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    # Real-time stats
    stats_text = f'Male: {male_count}  Female: {female_count}  Uncertain: {uncertain_count}'
    cv2.putText(resultImg, stats_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Detecting age and gender", resultImg)
    key = cv2.waitKey(1) & 0xFF
    # Keyboard controls
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear log file
        open('gender_age_log.txt', 'w').close()
        print('Log file cleared.')
