import time
import sys
import os
import numpy as np
import alsaaudio as aa
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks_cwt
from scipy.io import wavfile

start_time = time.time()

# Guitar data base directory
db_wd = '/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/'

# This represents notes on a guitar with 24 frets.
# Highest frequency is 1318.51 and lowest(
guitar_notes = {
    'E2':  {82.41:   [[0, 6]]},
    'F2':  {87.31:   [[1, 6]]},
    'F#2': {92.50:   [[2, 6]]},
    'G2':  {98.00:   [[3, 6]]},
    'G#2': {103.83:  [[4, 6]]},
    'A2':  {110.00:  [[0, 5],  [5, 6]]},
    'A#2': {116.54:  [[1, 5],  [6, 6]]},
    'B2':  {123.47:  [[2, 5],  [7, 6]]},
    'C3':  {130.81:  [[3, 5],  [8, 6]]},
    'C#3': {138.59:  [[4, 5],  [9, 6]]},
    'D3':  {146.83:  [[0, 4],  [5, 5],  [10, 6]]},
    'D#3': {155.56:  [[1, 4],  [6, 5],  [11, 6]]},
    'E3':  {164.81:  [[2, 4],  [7, 5],  [12, 6]]},
    'F3':  {174.61:  [[3, 4],  [8, 5],  [13, 6]]},
    'F#3': {185.00:  [[4, 4],  [9, 5],  [14, 6]]},
    'G3':  {196.00:  [[0, 3],  [5, 4],  [10, 5], [15, 6]]},
    'G#3': {207.65:  [[1, 3],  [6, 4],  [11, 5], [16, 6]]},
    'A3':  {220.00:  [[2, 3],  [7, 4],  [12, 5], [17, 6]]},
    'A#3': {233.08:  [[3, 3],  [8, 4],  [13, 5], [18, 6]]},
    'B3':  {246.94:  [[0, 2],  [4, 3],  [9, 4],  [14, 5], [19, 6]]},
    'C4':  {261.63:  [[1, 2],  [5, 3],  [10, 4], [15, 5], [20, 6]]},
    'C#4': {277.18:  [[2, 2],  [6, 3],  [11, 4], [16, 5], [21, 6]]},
    'D4':  {293.66:  [[3, 2],  [7, 3],  [12, 4], [17, 5], [22, 6]]},
    'D#4': {311.13:  [[4, 2],  [8, 3],  [13, 4], [18, 5], [23, 6]]},
    'E4':  {329.63:  [[0, 1],  [5, 2],  [9, 3],  [14, 4], [19, 5], [24, 6]]},
    'F4':  {349.23:  [[1, 1],  [6, 2],  [10, 3], [15, 4], [20, 5]]},
    'F#4': {369.99:  [[2, 1],  [7, 2],  [11, 3], [16, 4], [21, 5]]},
    'G4':  {392.00:  [[3, 1],  [8, 2],  [12, 3], [17, 4], [22, 5]]},
    'G#4': {415.30:  [[4, 1],  [9, 2],  [13, 3], [18, 4], [23, 5]]},
    'A4':  {440.00:  [[5, 1],  [10, 2], [14, 3], [19, 4], [24, 5]]},
    'A#4': {466.16:  [[6, 1],  [11, 2], [15, 3], [20, 4]]},
    'B4':  {493.88:  [[7, 1],  [12, 2], [16, 3], [21, 4]]},
    'C5':  {523.25:  [[8, 1],  [13, 2], [17, 3], [22, 4]]},
    'C#5': {554.37:  [[9, 1],  [14, 2], [18, 3], [23, 4]]},
    'D5':  {587.33:  [[10, 1], [15, 2], [19, 3], [24, 4]]},
    'D#5': {622.25:  [[11, 1], [16, 2], [20, 3]]},
    'E5':  {659.25:  [[12, 1], [17, 2], [21, 3]]},
    'F5':  {698.46:  [[13, 1], [18, 2], [22, 3]]},
    'F#5': {739.99:  [[14, 1], [19, 2], [23, 3]]},
    'G5':  {783.99:  [[15, 1], [20, 2], [24, 3]]},
    'G#5': {830.61:  [[16, 1], [21, 2]]},
    'A5':  {880.00:  [[17, 1], [22, 2]]},
    'A#5': {932.33:  [[18, 1], [23, 2]]},
    'B5':  {987.77:  [[19, 1], [24, 2]]},
    'C6':  {1046.50: [[20, 1]]},
    'C#6': {1108.73: [[21, 1]]},
    'D6':  {1174.66: [[22, 1]]},
    'D#6': {1244.51: [[23, 1]]},
    'E6':  {1318.51: [[24, 1]]},
  }

guitar_notes_vector = np.zeros(len(guitar_notes.keys()))
i = 0
for key, value in guitar_notes.items():
    guitar_notes_vector[i] = (list(value.keys())[0])
    i = i + 1

"""
Formula to calculate frequency from sfft bin:
(sampling_rate / 2) * number_of_bins (that is bin size)
"""
sampling_rate = 44100
#sampling_rate = 22000
#number_of_bins = 2048
number_of_bins = 4096
n_frames = -1
# overlap is 78.5%. for window size = 2048, overlap is 10 ms.
# another way of calculating this is by hop size,
overlap = int(0.785 * number_of_bins)
hop_size = number_of_bins - overlap
onset_window = 0.03  # 30ms
onset_window = 0.100  # 100ms
#onset_window = 0.065  # 65ms
tol = 1e-14                                                      # threshold used to compute phase

# Frequency Threshold is used to eliminate frequencies(noise) that are below 82.41Hz (minimum value on guitar) and
# eliminate frequencies above 1350.0Hz
#
frequency_threshold_min = 75.0    # in Hz
frequency_threshold_max = 1350.0  # in Hz

def half_wave_rectifier(x):
    return (x + abs(x)) / 2

# TODO uncomment this for windowed processing
#def time_stamp(curr_bin, ith):
#    return curr_bin * window_in_sec / len(X[0]) + ith * window_in_sec

def time_stamp(curr_bin):
     return curr_bin * len(new_data) / (sampling_rate * len(X[0]))


def peak_detection(mX, t):
    thresh = np.where(np.greater(mX[1:-1],t), mX[1:-1], 0)  # locations above threshold
    next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
    prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
    return ploc


"""
Interpolate peak values using parabolic interpolation
mX, pX: magnitude and phase spectrum, ploc: locations of peaks
returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
"""


def peak_interp(mX,  ploc):
    val = mX[ploc]                                                    # magnitude of peak bin
    lval = mX[ploc - 1]                                               # magnitude of bin at left
    rval = mX[ploc + 1]                                               # magnitude of bin at right
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)      # center of parabola
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)               # magnitude of peaks
    #ipphase = np.interp(iploc, np.arange(0, pX.size), pX)             # phase of peaks by linear interpolation
    #return iploc, ipmag, ipphase
    return iploc, ipmag


# rate, new_data = wavfile.read(db_wd + 'dataset2/audio/AR_B_fret_0-20.wav')
# r, new_data = wavfile.read('/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset4/acoustic_pickup/fast/country_folk/audio/country_2_150BPM.wav')
#r, new_data = wavfile.read('/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset1/Ibanez Power Strat Clean Bridge HU Chords/audio/1-E1-Major 00.wav')
# r, new_data = wavfile.read('/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset4/Ibanez 2820/fast/classical/audio/classical_1_80BPM.wav')
# r, new_data = wavfile.read('/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset4/Ibanez 2820/fast/classical/audio/classical_8_120BPM.wav')
# r, new_data = wavfile.read('/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset2/audio16bit/FS_G_V_slide_1.wav')
r, new_data = wavfile.read(db_wd + "/dataset2/audio16bit/AR_B_fret_0-20_1.wav")
# r, new_data = wavfile.read("/home/andrei/f/licenta/guitar-signal-process/gama-do-major-fender-squre.wav")
#r, new_data = wavfile.read("/home/andrei/f/licenta/guitar-signal-process/gama-do-major-2ori-80bpm-behringer.wav")
#r, new_data = wavfile.read("/home/andrei/f/licenta/guitar-signal-process/mi-fa-behringer.wav")
#r, new_data = wavfile.read("/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/audio/G53-40100-1111-00001.wav")
#r, new_data = wavfile.read("/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/audio/G53-41101-1111-00002.wav")
#r, new_data = wavfile.read("/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/audio/G53-42102-1111-00003.wav")
#r, new_data = wavfile.read("/home/andrei/f/licenta/IDMT-SMT-GUITAR_V2/dataset1/Fender Strat Clean Neck SC/audio/G53-43103-1111-00004.wav")

file_path = db_wd + "/dataset1/Fender Strat Clean Neck SC Chords/audio/"
results_file_path = "multiple_notes_rez.out"

single_note_file = open(results_file_path, "a")

for filename in os.listdir(file_path):

    r, new_data = wavfile.read(file_path + filename)

    bins_check = 3    # 3 is max num of bins to check for note. 6 bins is equivalent to 0.156 ms and window size of 1 sec. len(X[0]) / window_size(msec) => 52 / 1000ms => 0.052ms
    window_in_sec = 1 # 1000ms
    window_threshold = 2756 # 2756/sampling_rate; for s_r = 441000  => 0.0625 sec => 62ms; smallest note
    sig_size = len(new_data)
    sig_time = sig_size / sampling_rate

    n_w = sig_time / window_in_sec
    window_in_samples = window_in_sec * sampling_rate
    remain = abs(int(n_w) * int(window_in_samples) - sig_size)
    t_global = np.zeros(0)
    notes_per_window = np.zeros(0)

    fr, t, X = signal.stft(new_data, sampling_rate, window='hamming', return_onesided=True, nperseg=number_of_bins, noverlap=overlap)

    spectral_flux = np.zeros([len(X[0]), 1])
    spectral_flux_partial_sum = 0

    #abs_start_time = time.time()
    #absX = abs(X)
    #absX[absX < np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    #magnitude_spectrum_log = 20 * np.log10(absX)

    # X = magnitude_spectrum_log
    # Calculate Spectral Flux
    for n in range(1, len(X[0])):
        for k in range(0, len(X), 6):
            spectral_flux_partial_sum += half_wave_rectifier(abs(X[k][n] - X[k][n - 1]))
        spectral_flux[n - 1] = spectral_flux_partial_sum
        spectral_flux_partial_sum = 0

    w = 3
    m = 3
    # delta = 10**5 + 10**3 + 10**2
    alpha = 0.80
    delta = 10**3 + 2*10**2 + 5*10

    #New values
    #alpha = 0.80
    #delta = 2*10**2
    # alpha = 0.85
    g_alpha = np.zeros(len(X[0]))
    onset_candidate = np.zeros(len(X[0]))
    # onset_candidate[onset_candidate == 0] = np.nan

    for n in range(0, len(X[0])):
        if n == 0:  # base case
            g_alpha[n] = max(spectral_flux[n], (1 - alpha) * spectral_flux[n])
        else:
            g_alpha[n] = max(spectral_flux[n], alpha * g_alpha[n - 1] + (1 - alpha) * spectral_flux[n])

    for n in range(0, len(X[0])):
        if n - w > 0 and n + w < len(spectral_flux):
            min = spectral_flux[n - w:n + w].min()
            if spectral_flux[n] >= min:  # first condition
                if n - m * w > 0:
                    sum = spectral_flux[n - m * w: n + w].sum()
                    if spectral_flux[n] >= sum / (m * w + w + 1) + delta:  # second condition
                        if n != 0:
                            if spectral_flux[n] >= g_alpha[n - 1]:  # third condition
                                onset_candidate[n] = spectral_flux[n]

    onset_bin = np.where(onset_candidate != 0)
    onset_bin = onset_bin[0]
    onset_bin = onset_bin
    #print("Number of notes detected ", len(onset_bin))
    onset_time_stamp = np.apply_along_axis(time_stamp, -1, onset_bin)
    #onset_time_stamp = np.apply_along_axis(time_stamp, -1, onset_bin, i)        # TODO uncomment this for windowed processing

    # Apply window for onset candidates
    # Do not take notes that are closer than onset_window (currently 65ms) from one another
    valid_onset = []
    valid_onset_idx = []
    k = 0
    while k < len(onset_time_stamp):
        j = 1
        if k + j < len(onset_time_stamp) and abs(onset_time_stamp[k] - onset_time_stamp[k + j]) < onset_window:
            j += 1
        # if k + j < len(onset_time_stamp) and abs(onset_time_stamp[k] - onset_time_stamp[k + j]) < onset_window:
        #     j += 1
        valid_onset.append(onset_time_stamp[k])
        valid_onset_idx.append(k)
        k += j

    if np.size(valid_onset) == 0:
        continue

    # Peak detection
    peak_threshold = 300

    frequency_axis = r * np.arange(number_of_bins / 2 + 1) / float(number_of_bins)

    #print("Valid onset bins: ", onset_bin[valid_onset_idx])
    #TODO implement mechanism that will ignore current note if previous was same, in the current bin range
    previous_note = 0.0
    for bin in onset_bin[valid_onset_idx]:
        notes_per_window = np.zeros(0)
        for k in range(0, bins_check): #  Check 3 bins (aprox. 150 ms) for note
            if k + bin >= len(X[0]):
                break

            # Find peaks
            peaks, _ = signal.find_peaks(abs(X[:, bin + k]), height=peak_threshold)
            if len(peaks) == 0:
                continue

            iploc, ipmag = peak_interp(abs(X[:, bin + k]), peaks)
            if ipmag.size != 0:
                notes_per_window = np.append(notes_per_window, r * iploc / float(number_of_bins))
                #print(notes_per_window)
                if notes_per_window.min() < frequency_threshold_min or notes_per_window.min() > frequency_threshold_max:        # Filter frequencies that are not on guitar, below 75 Hz and above 1350 Hz
                    #print("Ignore", notes_per_window.min())
                    continue
            else:
                break

        #TODO if frequency is not in threshold range(min and max) set a flag and do not enter here
        #Calculate index of the aproximation note

        if len(notes_per_window) != 0:
            index = (np.abs(guitar_notes_vector - notes_per_window.min())).argmin()
            if previous_note == guitar_notes_vector[index]:
                # print("Ignore note. Previous is the same")
                previous_note = 0.0 # Reset
            else:
                output = "File: " + filename + "Note " + str(list(guitar_notes.keys())[index]) + ": " + str(guitar_notes_vector[index]) + "\n"
                print(output)
                single_note_file.write(output)

                #print("Note ", list(guitar_notes.keys())[index], ": ", guitar_notes_vector[index], " : ", notes_per_window.min())
                previous_note = guitar_notes_vector[index]

                #TODO 1. create data structure
                #TODO 2. send
        else:
            continue

single_note_file.close()