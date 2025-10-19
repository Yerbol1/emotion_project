import cv2
import time
from deepface import DeepFace
from collections import Counter

def analyze_emotion(duration=10):
    cap = cv2.VideoCapture(0)
    emotions = []
    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            emotions.append(dominant_emotion)
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            pass

        cv2.imshow("Face Analyzer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if emotions:
        most_common = Counter(emotions).most_common(1)[0][0]
    else:
        most_common = "unknown"
    return most_common
