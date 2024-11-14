import cv2
import mediapipe as mp

# Initialize Mediapipe Face Mesh and drawing utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

def is_looking_straight(landmarks):
    # Extract landmarks for left and right eyes
    left_eye = landmarks[362]  # Outer corner of left eye
    right_eye = landmarks[133]  # Inner corner of right eye
    
    # Calculate eye width
    eye_width = abs(left_eye.x - right_eye.x)
    
    # Check if looking straight by comparing distances
    # You can adjust threshold based on your position
    threshold = 0.075
    return eye_width < threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally and convert color space to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame and get facial landmarks
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Check if user is looking straight at the screen
            looking_straight = is_looking_straight(face_landmarks.landmark)
            
            # Draw eye landmarks and highlight eyes in red
            for idx in [362, 263, 373, 374, 380, 381, 382, 362,   # Left eye contour
                        133, 155, 154, 153, 145, 144, 163, 7]:   # Right eye contour
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
            # Print if user is looking at the screen
            if looking_straight:
                cv2.putText(frame, "Looking at the screen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not looking at the screen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the result
    cv2.imshow("Eye Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
