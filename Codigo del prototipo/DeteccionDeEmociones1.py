import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {
    0: "Angry",
    1: "Disgusted",
    2: "Fearful",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprised"
}

estres_dict = {
    "Angry": f"100% estresado",
    "Disgusted": f"30% estresado",
    "Fearful": f"80% estresado",
    "Happy": f"0% estresado",
    "Neutral": f"0% estresado",
    "Sad": f"30% estresado",
    "Surprised": f"60% estresado",
}

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (500, 500))

    if not ret:
        break
    face_detector = cv2.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    num_faces = face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        emotion_prediction = emotion_model.predict(cropped_img)
        predict = emotion_prediction
        with open("./results.txt", "a") as file:
            file.write(
                f"{emotion_prediction[0][0]},{emotion_prediction[0][1]}, {emotion_prediction[0][2]}, {emotion_prediction[0][3]}, {emotion_prediction[0][4]}, {emotion_prediction[0][5]}, {emotion_prediction[0][6]}\n")

        maxindex = int(np.argmax(emotion_prediction))

        cv2.putText(frame, estres_dict[emotion_dict[maxindex]], (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    t = cv2.waitKey(5)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
