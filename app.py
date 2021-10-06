import re
from flask import *
from librosa.core import audio
import tensorflow
import glob
import os
import soundfile
import librosa
import numpy as np

app = Flask(__name__, template_folder="template")

model = tensorflow.keras.models.load_model("model")

values = {"fearful": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/59-596556_fear-clipart-fear-emotion-cartoon-face-of-fear-removebg-preview.png?alt=media&token=c451d9cc-5fa7-47e1-bfa6-4be37e358ea8",
          "calm": "https://d29fhpw069ctt2.cloudfront.net/clipart/100203/preview/smiling_face_of_a_child_2_preview_9c89.png",
          "happy": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/565-5650281_happy-boy-clipart-can-do-it-png-transparent-removebg-preview.png?alt=media&token=5964f656-e102-4f85-bb5c-ad4209209e39",
          "sad": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/202-2022552_emotional-clipart-sad-dad-sad-clip-art-removebg.png?alt=media&token=3f1938a7-790e-4923-aea7-7f81ee2807b9",
          "angry": "https://firebasestorage.googleapis.com/v0/b/myproject-d9de9.appspot.com/o/clipart466731.png?alt=media&token=8dd82f61-b3ef-46f2-86c7-e1cd61f24ff3"
          }


def extract(i):
    with soundfile.SoundFile(i) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        stft = np.abs(librosa.stft(X))
        result = np.array([])
        mfccs = np.mean(librosa.feature.mfcc(
            y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        chroma = np.mean(librosa.feature.chroma_stft(
            S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chroma))
        mel = np.mean(librosa.feature.melspectrogram(
            X, sr=sample_rate).T, axis=0)
        result = np.hstack((result, mel))
    return result


@app.route("/", methods=["post", "get"])
def index():
    result = False
    r = ""
    img = ""
    if request.method == "POST":
        audio = request.files["audio"]
        feat = np.array(extract(audio))
        pred = model.predict(feat.reshape(-1, 180))
        result = True
        features_encoded = ['x0_angry', 'x0_calm',
                            'x0_fearful', 'x0_happy', 'x0_sad']
        r = features_encoded[np.argmax(pred)].replace("x0_", "")
        img = values[r]
    return render_template("index.html", result=result, pred=r, img=img)


if __name__ == "__main__":
    app.run()
