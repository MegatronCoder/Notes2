import cv2

# Load the pre-trained Haar Cascade face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Apply a blur to each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        face_region = frame[y:y+h, x:x+w]
        
        # Apply Gaussian blur to the face region
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        
        # Replace the original face region with the blurred version
        frame[y:y+h, x:x+w] = blurred_face
    
    # Display the frame with blurred faces
    cv2.imshow("Face Blur", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
