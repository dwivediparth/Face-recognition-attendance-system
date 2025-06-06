import face_recognition
import cv2
import numpy as np
import pandas as pd
import datetime

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Add images of known individuals and their names
def load_known_faces():
    global known_face_encodings, known_face_names
    # Example images (replace with your own images)
    images = ["images/person1.jpg", "images/person2.jpg", "images/person3.jpg"]  # Add paths to your images
    names = ["Rony", "Ritik", "R Sujay"]  # Corresponding names
    
    for image_path, name in zip(images, names):
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image, num_jitters=10)[0]  # Increase encoding accuracy
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Initialize attendance DataFrame
attendance_df = pd.DataFrame(columns=["Name", "Date", "Time"])

# Mark attendance
def mark_attendance(name):
    global attendance_df
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Check if attendance is already marked for this name on the same date
    if not attendance_df[(attendance_df["Name"] == name) & (attendance_df["Date"] == date)].empty:
        return False  # Attendance already marked

    attendance_df = pd.concat([attendance_df, pd.DataFrame({"Name": [name], "Date": [date], "Time": [time]})], ignore_index=True)
    attendance_df.to_csv("attendance.csv", index=False)
    return True  # Attendance marked successfully

# Main function for real-time face recognition
def run_attendance_system():
    load_known_faces()
    video_capture = cv2.VideoCapture(0)  # Use webcam

    while True:
        # Capture a single frame
        ret, frame = video_capture.read()

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

        # Find all face locations and face encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")  # Use HOG-based detection
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Check if face matches a known face
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)  # Lower tolerance for stricter matching
            name = "Unknown"

            # Use the known face with the smallest distance if any match
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                if mark_attendance(name):  # If attendance is successfully marked
                    print(f"Attendance marked for {name}.")
                video_capture.release()
                cv2.destroyAllWindows()
                return  # End the session
            else:
                # Unknown face detected
                print("Unknown face detected.")
                video_capture.release()
                cv2.destroyAllWindows()
                return  # End the session

            # Display the results for known faces
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green rectangle
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow("Face Recognition Attendance System", frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_attendance_system()

