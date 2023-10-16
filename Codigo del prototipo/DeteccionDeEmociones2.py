import cv2
import mediapipe as mp
from deepface import DeepFace

estres_dict = {
    "angry": f"100% estresado",
    "disgust": f"30% estresado",
    "fear": f"80% estresado",
    "happy": f"0% estresado",
    "neutral": f"0% estresado",
    "sad": f"30% estresado",
    "surprise": f"70% estresado",
}

detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence=0.8, model_selection=0)

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while True: 
  ret, frame = cap.read()
  
  img = cv2.resize(frame, (0,0), None, 0.18, 0.18)
  ani, ali, c = img.shape
  
  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
  resrostros = rostros.process(rgb)
  
  if resrostros.detections is not None:
    for rostro in resrostros.detections: 
      al, an, c = frame.shape
      box = rostro.location_data.relative_bounding_box
      xi, yi, w, h = int(box.xmin * an), int(box.ymin * al), int(box.width * an), int(box.height * al)
      xf, yf = xi + w, yi + h
      
      cv2.rectangle(frame, (xi,yi), (xf,yf), (255,255,0),1)
      frame[10:ani + 10, 10:ali + 10] = img
      
      info = DeepFace.analyze(rgb,actions=["emotion"], enforce_detection=False)
      print(info)
      
      emociones = info[0]["dominant_emotion"]
      cv2.putText(frame, str(estres_dict[emociones] ), (10,135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
      
      with open("./results.txt", "a") as file:
        file.write(
          f"{info[0]['emotion']['angry']},{info[0]['emotion']['disgust']}, {info[0]['emotion']['fear']}, {info[0]['emotion']['happy']}, {info[0]['emotion']['neutral']}, {info[0]['emotion']['sad']}, {info[0]['emotion']['surprise']}\n")
      
  cv2.imshow("Deteccion de estres", frame)
  
  t = cv2.waitKey(5)
  if t == 27:
    break
  
cv2.destroyAllWindows()
cap.release()
