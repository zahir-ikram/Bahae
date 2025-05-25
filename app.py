import streamlit as st
import numpy as np
import cv2

from simple_emotion_net import SimpleEmotionNet  # ton fichier python

IMG_SIZE = 48
INPUT_SIZE = IMG_SIZE * IMG_SIZE
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

st.title("🧠 Détection d'Émotions (TIPE)")
st.write("Clique sur 'Lancer la caméra' pour démarrer.")

# 📌 Créer un modèle non ré-entraîné (tu peux l’entraîner avant si tu veux)
model = SimpleEmotionNet()

# 📸 Capture webcam
if st.button("Lancer la caméra"):
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    if not cap.isOpened():
        st.error("❌ Erreur avec la webcam")
    else:
        st.success("✅ Caméra démarrée. Appuie sur 'Stop' pour arrêter.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Problème d'image")
            break

        # Prétraitement image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))
        x = face.flatten().reshape(INPUT_SIZE, 1) / 255.0

        # 🔮 Prédiction
        emotion_idx = model.predict(x)
        emotion = emotion_labels[emotion_idx]

        # Affichage
        cv2.putText(frame, f"Emotion: {emotion}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(frame, channels="BGR")

        # Quitter en appuyant sur q (utile pour débogage local)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
