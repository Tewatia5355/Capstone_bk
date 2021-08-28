from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io
import os
import librosa
from pydub import AudioSegment
from mainn import preprocess
from flask import Flask, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
UPLOAD_FOLDER = "input"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    print("Function Called")
    new_file = request.files["audio_data"]
    file_name = "Temp.wav"
    new_file.save(os.path.join(app.config['UPLOAD_FOLDER'], file_name))
    return preprocess()


app.run(port=8081)
