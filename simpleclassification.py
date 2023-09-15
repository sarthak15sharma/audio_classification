import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import librosa
import librosa.display

from itertools import cycle

sns.set_theme(style="white")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

audio_files_happy = glob("./assignment/dataset/*/03-01-04-*.wav")
audio_files_sad = glob("./assignment/dataset/*/03-01-05-*.wav")


def extract_features(data, sample_rate):
    result = np.array([])
    #ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    #Chroma Shift
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))

    #MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    #Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    #MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))

    return result


def get_features(file):
    data, sample_rate = librosa.load(file, duration=2.5, offset=0.6)

    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    return result

X, Y = [], []

for audio in audio_files_happy:
    feature = get_features(audio)
    X.append(feature)
    Y.append('Happy')

for audio in audio_files_sad:
    feature = get_features(audio)
    X.append(feature)
    Y.append('Sad')

# Features = pd.DataFrame(X)
# Features['labels'] = Y
# Features.to_csv('./assignment/features.csv', index=False)

train_data = glob('./assignment/dataset/Actor_*/03-01-04*.wav') + glob('./assignment/dataset/Actor_0*/03-01-05*.wav')

test_data = glob('./assignment/dataset/Actor_1*/03-01-05*.wav')

train_x = []
train_y = []

for audio in train_data:
    feature = get_features(audio)
    train_x.append(feature)
    if audio[36:38] == '04':
        train_y.append('Happy')
    else:
        train_y.append('Sad')

test = []

for audio in test_data:
    feature = get_features(audio)
    test.append(feature)

# clf = RandomForestClassifier(random_state=0)
# clf.fit(train_x, train_y)

# print(clf.predict(test))

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(np.array(train_x), train_y)
print(neigh.predict(np.array(test)))
