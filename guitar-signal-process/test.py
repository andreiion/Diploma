import time
import sys
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


for i in range(0, 1):
#for i in range(0, int(n_w)):

    # Apply shot time fourier transform
    #  X - SFFT of x. each column represents the nth frame. each row represents kth frequency

    """
        if i == 0:
            fr, t, X = signal.stft(new_data[int(i * window_in_samples):int((i + 1) * window_in_samples)], sampling_rate,
                                   window='hamming', return_onesided=True, nperseg=number_of_bins, noverlap=overlap)
        elif i == int(n_w) - 1:
            fr, t, X = signal.stft(new_data[int(i * window_in_samples) + 1:int((i + 1) * window_in_samples) - 1], sampling_rate,
                                   window='hamming', return_onesided=True, nperseg=number_of_bins, noverlap=overlap)
        else:
            fr, t, X = signal.stft(new_data[int(i * window_in_samples) + 1:int((i + 1) * window_in_samples)],
                                  sampling_rate, window='hamming', return_onesided=True, nperseg=number_of_bins, noverlap=overlap)
        if i != 0:
            t = t + i * t[-1]
    """

    #fr, t, X = signal.stft(new_data, sampling_rate, window='hamming', return_onesided=True, nperseg=number_of_bins,
    #                   noverlap=overlap)

    f = plt.figure(123)
    plt.pcolormesh(t, fr, np.abs(X), vmin=0, vmax=128)
    plt.ylim(0, 4000)
    plt.title('SFFT Magnitude Spectrum')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
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

    """
        if onset_time_stamp.size != 0:
            print("Time stamp: ", onset_time_stamp, " Bin: ", onset_bin)
    """
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

            plt.figure(i + 1)
            plt.title("Candidate Frequencies and Peak interpolation.")
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power of signal')
            plt.plot(frequency_axis, abs(X[:, bin + k]))
            plt.plot(r * iploc / float(number_of_bins), ipmag, label='Peak Interpolation', marker='x', linestyle='')
            plt.show()

        #TODO if frequency is not in threshold range(min and max) set a flag and do not enter here
        #Calculate index of the aproximation note

        if len(notes_per_window) != 0:
            index = (np.abs(guitar_notes_vector - notes_per_window.min())).argmin()
            if previous_note == guitar_notes_vector[index]:
                # print("Ignore note. Previous is the same")
                previous_note = 0.0 # Reset
            else:
                print("Note ", list(guitar_notes.keys())[index], ": ", guitar_notes_vector[index], " : ", notes_per_window.min())
                previous_note = guitar_notes_vector[index]

                #TODO 1. create data structure
                #TODO 2. send
        else:
            continue

    plt.figure(1000)
    plt.title("Threshold function vs Onset detection")
    label_print = False
    for onset in valid_onset:
         if not label_print:
            plt.axvline(x=onset, color='r', linestyle='--', label='Onset Detection')
            label_print = True
         plt.axvline(x=onset, color='r', linestyle='--')
    plt.plot(t, g_alpha, label='Threshold')
    # plt.plot(t, mean_square_of_enegies / 10**4, label='Mean Square Energy')
    plt.plot(t, spectral_flux, label='Spectral Flux')
    # plt.plot(t, onset_candidate, label='Onset detection', marker='x', linestyle='')
    plt.xlabel('Time [sec]')
    plt.legend(loc=0)
    plt.show()

# last piece
# if abs(int(n_w) * window_in_samples - sig_size) > window_threshold:
#     fr, t, X = signal.stft(new_data[int(n_w) * int(window_in_samples) + 1:sig_size - 1], sampling_rate,
#                            window='hamming', return_onesided=True, nperseg=number_of_bins, noverlap=overlap)
#
#     t = t + (int(n_w) + 2) * t[-1]
#
#     spectral_flux = np.zeros([len(X[0]), 1])
#     spectral_flux_partial_sum = 0
#
#     absX = abs(X)
#     absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # if zeros add epsilon to handle log
#     magnitude_spectrum_log = 20 * np.log10(absX)
#
#     # X = magnitude_spectrum_log
#     # Calculate Spectral Flux
#     for n in range(1, len(X[0])):
#         for k in range(0, len(X)):
#             spectral_flux_partial_sum += half_wave_rectifier(abs(X[k][n] - X[k][n - 1]))
#         spectral_flux[n - 1] = spectral_flux_partial_sum
#         spectral_flux_partial_sum = 0
#
#     w = 3
#     m = 3
#     # delta = 10**5 + 10**3 + 10**2
#     alpha = 0.95
#     delta = 5 * 10 ** 3
#     # alpha = 0.85
#     g_alpha = np.zeros(len(X[0]))
#     onset_candidate = np.zeros(len(X[0]))
#     # onset_candidate[onset_candidate == 0] = np.nan
#
#     for n in range(0, len(X[0])):
#         if n == 0:  # base case
#             g_alpha[n] = max(spectral_flux[n], (1 - alpha) * spectral_flux[n])
#         else:
#             g_alpha[n] = max(spectral_flux[n], alpha * g_alpha[n - 1] + (1 - alpha) * spectral_flux[n])
#
#     for n in range(0, len(X[0])):
#         if n - w > 0 and n + w < len(spectral_flux):
#             min = spectral_flux[n - w:n + w].min()
#             if spectral_flux[n] >= min:  # first condition
#                 if n - m * w > 0:
#                     sum = spectral_flux[n - m * w: n + w].sum()
#                     if spectral_flux[n] >= sum / (m * w + w + 1) + delta:  # second condition
#                         if n != 0:
#                             if spectral_flux[n] >= g_alpha[n - 1]:  # third condition
#                                 onset_candidate[n] = spectral_flux[n]
#
#     onset_time_stamp = np.where(onset_candidate != 0)
#     onset_time_stamp = onset_time_stamp[0]
#     onset_bin = onset_time_stamp
#     onset_time_stamp = np.apply_along_axis(time_stamp, -1, onset_time_stamp, i)
#
#     if onset_time_stamp.size != 0:
#         print(onset_time_stamp)
#     # print(len(X[0]))
#
#     # Apply window for onset candidates
#     valid_onset = []
#     k = 0
#     while k < len(onset_time_stamp):
#         j = 1
#         while k + j < len(onset_time_stamp) and abs(onset_time_stamp[k] - onset_time_stamp[k + j]) < onset_window:
#             j += 1
#         valid_onset.append(onset_time_stamp[k])
#         k += j
#
#     # Peak detection
#
#     plt.figure(2)
#     plt.title("Threshold function vs Onset detection")
#     label_print = False
#     for onset in valid_onset:
#         if not label_print:
#             plt.axvline(x=onset, color='r', linestyle='--', label='Onset Detection')
#             label_print = True
#         plt.axvline(x=onset, color='r', linestyle='--')
#     plt.plot(t, g_alpha, label='Threshold')
#     # plt.plot(t, mean_square_of_enegies / 10**4, label='Mean Square Energy')
#     plt.plot(t, spectral_flux, label='Spectral Flux')
#     # plt.plot(t, onset_candidate, label='Onset detection', marker='x', linestyle='')
#     plt.xlabel('Time [sec]')
#     plt.legend(loc=0)
#     plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
# Remain process

"""
#f = open("/home/andrei/f/licenta/guitar-signal-process/gama.wav", 'rb')
#f = open(db_wd + "/dataset1/Fender Strat Clean Neck SC/audio/G53-40100-1111-00001.wav", 'rb')
#f = open(db_wd +  "dataset2/audio/AR_B_fret_0-20.wav", 'rb')
#f = open(db_wd + "/dataset2/audio/AR_Lick10_FN.wav", 'rb')
#f = open(db_wd + "dataset1/Ibanez Power Strat Clean Bridge HU Chords/audio/4-A2-Minor 08.wav", 'rb')
start_time = time.time()
data = f.read()
new_data = []
# data is a byte array. so we take 16bites(2Bytes) and convert them into integers
# By doing this we will have data into int values and can process them
new_data_size = 0
for i in range(0,int(data.__len__()), 4):
    two_bytes = int.from_bytes(bytearray(data[i:i + 4]), 'little', signed=True)
    new_data.append(two_bytes)
print("--- %s seconds ---" % (time.time() - start_time))
"""
""" Calculate phase 
filter = np.zeros([len(X[0]), 1])
onsets = np.zeros([len(X[0]), 1])
hole_signal_energy = 0

frames = len(X)
k_t = 0.12 * 10**(-3) * 30 # milsecondss
k_e = 100
# tau = int(n_frames / 5)  # 311 corresponding to 1.5 frames, of 5 ms hopsize with sample rate 44100Hz
# TODO integrate tau dependent by number of frames
#tau = int(n_frames / 40)  # 311 corresponding to 1.5 frames, of 5 ms hopsize with sample rate 44100Hz
#onset_candidate_interval_size = int(5 * sampling_rate / n_frames)
tau = int(len(X[0]) / 20)
#onset_candidate_interval_size = int(5 * sampling_rate / (len(X[0]) + n_frames))
onset_candidate_interval_size = int(tau / 2)
# I is the maximum of intervals taken into account before the onset candidate
I = int(2 * tau / onset_candidate_interval_size) + 1
I = 3

mean_square_of_enegies = np.zeros([len(X[0]), 1])
mean_square_partial_sum = 0

# fig = plt.figure()
# plt.plot(abs(np.fft.rfft(data, 2048)))
# for i in range(0, X.len)

for i in range(len(spectral_flux)):
    if i < len(spectral_flux) - 45:
        if (spectral_flux2[i + 45] -  spectral_flux2[i] ) / spectral_flux2[i + 45] > 0.2:
            onset[i] = spectral_flux[i]

def calc_mean_square_of_energyes():
    tau = 51
    # Calculate Mean Square of energies
    for n in range(0, len(spectral_flux)):
        mean_square_partial_sum = 0
        for i in range(-tau, tau):
            if (n + i > 0) and (n + i < len(spectral_flux)):
                mean_square_partial_sum += spectral_flux[n + i] ** 2
        mean_square_of_enegies[n] = mean_square_partial_sum / tau

plt.plot(onset)
plt.plot(spectral_flux)
plt.show()

pX = np.unwrap(np.angle(X[120]))
mX = 20 * np.log10(abs(X[120]))
#mX = abs(X[1])
t = 0
ploc = peak_detection(mX, t)

iploc, ipmag = peak_interp(mX, pX, ploc)

plt.plot(sampling_rate * iploc / float(number_of_bins), marker = 'x', linestyle='')
#plt.plot(sampling_rate * iploc / float(number_of_bins), ipmag)
plt.show()
"""
"""
plt.plot(t, mean_square_of_enegies)
plt.title('Mean Square of Energies')
plt.xlabel('Time [sec]')
plt.show()
"""
"""
# Calculate hole energy signal
hole_signal_energy = np.sum(np.power(spectral_flux, 2)) / len(spectral_flux)

# Choose a valid onset by following next conditions
min_poz = -1
i = 0
sample_index = np.zeros([len(mean_square_of_enegies), 1])
sample_index2 = np.zeros([len(mean_square_of_enegies), 1])
min_value_vec = np.zeros([len(mean_square_of_enegies), 1])
i_vec = np.zeros([len(mean_square_of_enegies), 1])
for n in range(0, len(mean_square_of_enegies)):
    if n + tau < len(mean_square_of_enegies):  # bounds check
        # TODO 50 = number of bins / 5 - make this dependend by number of bins
        for i in range(0, onset_candidate_interval_size, 5):
            if n - (i + 5) * I < 0:
                if n != 0:
                    min_value = mean_square_of_enegies[0:n].min()
                else:
                    min_value = mean_square_of_enegies[0]
            else:
                min_value = mean_square_of_enegies[(n - (i + 5) * I):(n - i * I)].min()

            if min_value > mean_square_of_enegies[n + tau]:
                i_vec[n] = 10 ** 12
                break
            min_value_vec[n] = min_value
            if hole_signal_energy < k_e * mean_square_of_enegies[n + tau]:
                sample_index[n] = n - i * I
                if n - i * I > (k_t * sampling_rate):
                    sample_index2[n] = n - i * I
                    onsets[n] = 1

for n in range(0, len(mean_square_of_enegies)):
    if onsets[n] == 1:
        filter[n] = mean_square_of_enegies[n]

# plt.plot(padded_mse, label='mean square energy')

plt.title("Onset detection over time")
plt.plot(t, onsets)
plt.xlabel('Time [sec]')
plt.show()

zeros_tau = np.zeros(tau)
zeros_tau = np.reshape(zeros_tau, (len(zeros_tau), 1))
np.concatenate([zeros_tau, mean_square_of_enegies])
padded_mse = np.concatenate([zeros_tau, mean_square_of_enegies])

plt.title("Mean Square of Energies and Onset Values")
#plt.plot(t, padded_mse, label='mean square energy')
plt.plot(t, mean_square_of_enegies, label='mean square energy')
plt.plot(t, filter, label='min values')
plt.legend(loc=5)
plt.xlabel('Time [sec]')
plt.show()

plt.axhline(hole_signal_energy, color='r')
plt.plot(t, k_e*mean_square_of_enegies)
plt.xlabel('Time [sec]')
plt.show()

plt.plot(t, sample_index)
# plt.plot(sample_index2, color = 'green')
plt.axhline(k_t * sampling_rate, color='r')
plt.show()

plt.title("Minimum Values and Mean Square Energy")
plt.plot(t, min_value_vec, color='green', label='min val')
plt.plot(t, mean_square_of_enegies, color='r', label='mse')
plt.plot(t, i_vec, color='b', label='mse')
plt.legend(loc=4)
plt.xlabel('Time [sec]')
plt.show()

size = 0
for i in range(len(onsets)):
    if onsets[i] == 1:
        size = size + 1
        freq = (sampling_rate / 2) / len(X) * i
        print(freq)
print(size)
p = 20*np.log10(np.abs(np.fft.fft(new_data, 2048)))
f = np.linspace(0, 8000/2.0, len(p))
plt.plot(p)
plt.xlabel("Frequency(Hz)")
plt.ylabel("Power(dB)")
plt.show()
"""
