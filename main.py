import cv2
import mediapipe as mp
import numpy as np
import signal, sys

# ==============================
# MediaPipe Setup
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==============================
# Camera Setup
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ==============================
# Landmark Indices
# ==============================
NOSE_TIP_IDX = 1
CHIN_IDX = 152
LEFT_EYE_CORNER_IDX = 33
RIGHT_EYE_CORNER_IDX = 263
LEFT_MOUTH_CORNER_IDX = 61
RIGHT_MOUTH_CORNER_IDX = 291

KEY_LANDMARK_INDICES = [
    LEFT_EYE_CORNER_IDX, RIGHT_EYE_CORNER_IDX,
    LEFT_MOUTH_CORNER_IDX, RIGHT_MOUTH_CORNER_IDX,
    NOSE_TIP_IDX, CHIN_IDX
]

# ==============================
# Graceful Exit
# ==============================
def signal_handler(sig, frame):
    print("Exiting gracefully...")
    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==============================
# Utility
# ==============================
def safe_div(a, b, eps=1e-6):
    return a / (b + eps)

class ExpSmoother:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None
    def update(self, new_val):
        if self.value is None:
            self.value = new_val
        else:
            self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value

yaw_smoother = ExpSmoother(0.2)
pitch_smoother = ExpSmoother(0.2)
asym_smoother = ExpSmoother(0.3)

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

# ==============================
# Head Pose Estimation
# ==============================
def get_head_pose(landmarks, frame_w, frame_h):
    image_points = np.array([
        (landmarks.landmark[NOSE_TIP_IDX].x * frame_w, landmarks.landmark[NOSE_TIP_IDX].y * frame_h),
        (landmarks.landmark[CHIN_IDX].x * frame_w, landmarks.landmark[CHIN_IDX].y * frame_h),
        (landmarks.landmark[LEFT_EYE_CORNER_IDX].x * frame_w, landmarks.landmark[LEFT_EYE_CORNER_IDX].y * frame_h),
        (landmarks.landmark[RIGHT_EYE_CORNER_IDX].x * frame_w, landmarks.landmark[RIGHT_EYE_CORNER_IDX].y * frame_h),
        (landmarks.landmark[LEFT_MOUTH_CORNER_IDX].x * frame_w, landmarks.landmark[LEFT_MOUTH_CORNER_IDX].y * frame_h),
        (landmarks.landmark[RIGHT_MOUTH_CORNER_IDX].x * frame_w, landmarks.landmark[RIGHT_MOUTH_CORNER_IDX].y * frame_h)
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),   # Right eye corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length,  0,              center[0]],
        [0,             focal_length,   center[1]],
        [0,             0,              1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    try:
        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )
        (rotation_matrix, _) = cv2.Rodrigues(rotation_vector)

        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = np.arctan2(-rotation_matrix[2, 0], sy)
        else:
            x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = np.arctan2(-rotation_matrix[2, 0], sy)

        pitch = np.degrees(x)
        yaw = np.degrees(y)
        return yaw, pitch, True
    except cv2.error:
        return 0, 0, False

# ==============================
# Main Loop
# ==============================
warning_counter = 0
PERSISTENCE_THRESHOLD = 15

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Frame capture failed. Retrying...")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    img_h, img_w, _ = frame.shape
    font, font_scale = cv2.FONT_HERSHEY_SIMPLEX, 1.0
    color_warn, color_normal = (0, 0, 255), (0, 255, 0)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        yaw, pitch, pose_ok = get_head_pose(face_landmarks, img_w, img_h)
        if pose_ok:
            yaw = yaw_smoother.update(yaw)
            pitch = pitch_smoother.update(pitch)

            # Landmark distances
            nose = face_landmarks.landmark[NOSE_TIP_IDX]
            le, re = face_landmarks.landmark[LEFT_EYE_CORNER_IDX], face_landmarks.landmark[RIGHT_EYE_CORNER_IDX]
            lm, rm = face_landmarks.landmark[LEFT_MOUTH_CORNER_IDX], face_landmarks.landmark[RIGHT_MOUTH_CORNER_IDX]

            d_nose_le = calculate_distance(nose, le)
            d_nose_re = calculate_distance(nose, re)
            d_mouth = calculate_distance(lm, rm)
            mouth_y_diff = abs(lm.y - rm.y)

            eye_ratio = safe_div(d_nose_le, d_nose_re)
            mouth_metric = safe_div(mouth_y_diff, d_mouth)

            # Thresholds (baseline)
            EYE_THR = 1.15
            MOUTH_THR = 0.1

            # allow more tolerance if yaw is high
            yaw_factor = 1 + abs(yaw) * 0.005
            print(yaw)
            eye_thr = EYE_THR * yaw_factor
            mouth_thr = MOUTH_THR * yaw_factor

            asym = 0
            if eye_ratio > eye_thr or eye_ratio < 1 / eye_thr:
                asym += 1
            if mouth_metric > mouth_thr:
                asym += 1

            asym = asym_smoother.update(asym)

            if asym > 0.5:  # Smoothed decision
                warning_counter += 1
                if warning_counter > PERSISTENCE_THRESHOLD:
                    cv2.putText(frame, "Warning: Asymmetry Detected", (30, 50),
                                font, font_scale, color_warn, 2)
            else:
                warning_counter = 0
                cv2.putText(frame, "Symmetry OK", (30, 50),
                            font, font_scale, color_normal, 2)

            # Draw landmarks
            for idx in KEY_LANDMARK_INDICES:
                x = int(face_landmarks.landmark[idx].x * img_w)
                y = int(face_landmarks.landmark[idx].y * img_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        else:
            cv2.putText(frame, "Pose unstable", (30, 50),
                        font, font_scale, color_warn, 2)
    else:
        cv2.putText(frame, "No face detected", (30, 50),
                    font, font_scale, color_warn, 2)

    cv2.imshow("Facial Asymmetry Detection (Smoothed)", frame)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# ==============================
# Cleanup
# ==============================
cap.release()
cv2.destroyAllWindows()
face_mesh.close()
