from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io
import os
import pickle
import librosa
from pydub import AudioSegment
import pandas as pd
from sklearn.preprocessing import RobustScaler
from boruta import BorutaPy


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    trim_ms = 0  # ms
    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms


def preprocess():
    # Loading Data
    file = ".//input//Temp.wav"
    rate = 16000
    sound = AudioSegment.from_file(file)
    sound = sound.set_frame_rate(rate)

    # Detecting start and end silence clip
    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    # Trimming Silence and Extracting Numpy array from the AudioSegment object
    data = np.array((sound[start_trim:len(sound)-end_trim]
                     ).get_array_of_samples(), dtype=np.float64)
    noise_reduced = nr.reduce_noise(
        y=data, sr=rate, thresh_n_mult_nonstationary=2, stationary=False)
    y = librosa.util.normalize(noise_reduced)
    y_filt = librosa.effects.preemphasis(y)
    S_preemph = librosa.amplitude_to_db(
        np.abs(librosa.stft(y_filt)), ref=np.max)
    output_name = "Temp.wav"
    sf.write(output_name, y_filt, 16000)
    return detect_emotion()


def detect_emotion():
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    file_list = ['Temp.wav']
    audio_df = pd.DataFrame(file_list)
    audio_df.columns = ['Filename']
    mfcc_df = []

    for index, path in enumerate(audio_df.Filename):
        X, sample_rate = librosa.load(path)
        mfccs = librosa.feature.mfcc(y=X, n_mfcc=26, sr=sample_rate)
        delta_m = librosa.feature.delta(mfccs)
        delta2_m = librosa.feature.delta(mfccs, order=2)

        ls = []
        for i in range(mfccs.shape[0]):
            temp = mfccs[i, :]
            lfeatures = [np.mean(temp), np.var(
                temp), np.amax(temp), np.amin(temp)]
            temp2 = np.array(lfeatures)
            ls.append(temp2)

        ls2 = []
        for i in range(delta_m.shape[0]):
            dtemp = delta_m[i, :]
            dlfeatures = [np.mean(dtemp), np.var(
                dtemp), np.amax(dtemp), np.amin(dtemp)]
            dtemp2 = np.array(dlfeatures)
            ls2.append(dtemp2)

        ls3 = []
        for i in range(delta2_m.shape[0]):
            stemp = delta2_m[i, :]
            slfeatures = [np.mean(stemp), np.var(
                stemp), np.amax(stemp), np.amin(stemp)]
            stemp3 = np.array(slfeatures)
            ls3.append(stemp3)

        source = np.array(ls).flatten()
        source = np.append(source, np.array(ls2).flatten())
        source = np.append(source, np.array(ls3).flatten())
        mfcc_df.append(source)
    df = pd.DataFrame(mfcc_df).to_numpy()
    feat_sel = pickle.load(open('feat_sel.sav', 'rb'))

    np.random.seed(42)
    scal = RobustScaler()
    data = scal.fit_transform(df)
    x_train = feat_sel.transform(data)
    model = pickle.load(open('svm_model.sav', 'rb'))
    pred = model.predict(x_train)
    print(pred)
    return pred[0]
