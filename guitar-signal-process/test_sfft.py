import time
import sys
import numpy as np
import alsaaudio as aa
import matplotlib.pyplot as plt
from scipy import signal

timp = np.arange(8000)
fs = 8000

sinus = np.sin(2 * np.pi * 120 * timp/8000)

plt.plot(sinus[:500])
plt.show()

fr, t, X = signal.stft(sinus, fs, return_onesided=True)

plt.figure()
plt.plot(np.arange(len(X)),X)
plt.show()

freq = (fs/2) / (np.shape(X)[0]) *3

