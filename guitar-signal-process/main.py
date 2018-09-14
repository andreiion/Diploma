import numpy as np
import alsaaudio as aa
#import matplotlib.pyplot as plt
from scipy import signal
import socket
import sys, os
import time
import json
from threading import Thread, Lock
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import deque
#from multiprocessing import Process
#from joblib import Parallel, delayed
#import multiprocessing
from numba import jit

"""
"""""""""""""""""""""""""""""""" This is where all global variables are put """""""""""""""""""""""""""""""" 
"""
lock = Lock()

# Queue used to read from alsa and to send data to other thread do process signal.
# This is a producer - consumer type of application.
# q = queue.Queue()

queue = deque()
queue2 = deque()
# Queue containing guitar notes in json format. It will be send by HTTP Server at every GET Request
#
q_notes = deque()

## This is the new method used to pass data from one thread to another.
## With only one producer and one consumer these will work fine.
BUF_SIZE = 16384  # condition is that BUF_SIZE has to be even disible with sys.maxsize + 1
shared_buf_signal_data = np.empty(BUF_SIZE, dtype=bytearray)

producer_count = 0
consumer_count = 0

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

sampling_rate = 44100
# sfft_window_size_N = 2048
sfft_window_size_N = 4096
# sfft_window_size_N = 8192
n_frames = -1
# overlap is 78.5%. for window size = 2048, overlap is 10 ms.
# another way of calculating this is by hop size,
overlap = int(0.785 * sfft_window_size_N)
hop_size = sfft_window_size_N - overlap
onset_window = 0.03  # 30ms
onset_window = 0.100  # 100ms
# onset_window = 0.065  # 65ms
tol = 1e-14

bins_check = 3   # 3 is max num of bins to check for note. 6 bins is equivalent to 0.156 ms and window size of 1 sec. len(X[0]) / window_size(msec) => 52 / 1000ms => 0.052ms
window_in_sec = 1 # 500ms
window_in_samples = window_in_sec * sampling_rate
window_threshold = 2756  # 2756/sampling_rate; for s_r = 441000  => 0.0625 sec => 62ms; smallest note
t_global = np.zeros(0)
global_time_init = False
notes_per_window = np.zeros(0)

# Frequency Threshold is used to eliminate frequencies(noise) that are below 82.41Hz (minimum value on guitar) and
# eliminate frequencies above 1350.0Hz
#
frequency_threshold_min = 75.0    # in Hz
frequency_threshold_max = 1350.0  # in Hz

# These are the signal data variables. These cause trouble with scope and variables visibility
X = np.zeros(0)
# new_data = []

global_time = time.time()


# Alsa specific
alsa_window = 0.01 # seconds
period_size = sampling_rate * alsa_window

start_time = time.time()


"""
"""""""""""""""""""""""""""""""" This is where all functions are put """""""""""""""""""""""""""""""" 
"""

# HTTPRequestHandler class
class guitarHTTPServer_ReqHandler(BaseHTTPRequestHandler):

    # GET
    def do_GET(self):
        # Send response status code
        self.send_response(200)

        # Send headers
        #self.send_header('Content-type','text/html')
        #self.end_headers()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        # Send data to client
        while q_notes:
            try:
                message = q_notes.popleft()
                #print("message: ", message)
            except IndexError:
                print("message is bad. error")
                return
            print("message: ", json.dumps(message).encode())
            self.wfile.write(json.dumps(message).encode())
        return


def start_http_server():
    print("Start HTTP server...")

    # Server settings
    server_address = ('10.8.0.10', 1500)
    httpd = HTTPServer(server_address, guitarHTTPServer_ReqHandler)
    print("Running server...", socket.gethostname())
    httpd.serve_forever()


class SocketThread(Thread):
    def __init__(self, id):
        Thread.__init__(self)
        self.id = id
        self.kill_received = False

    def read_from_alsa(self): # producer
        try:
            capture_guitar_sound()
        finally:
            print("alsa capture error")
            os._exit(1)

    def process_signal(self):
        process_guitar_sound()

    def create_http_server(self):
        start_http_server()

    def run(self):
        if self.id == 1:
            self.read_from_alsa()
        elif self.id == 2:
            self.process_signal()
        else:
            self.create_http_server()


# Create vector containing only frequency of notes
guitar_notes_vector = np.zeros(len(guitar_notes.keys()))
guitar_position_vector = []
i = 0
for key, value in guitar_notes.items():
    guitar_notes_vector[i] = (list(value.keys())[0])
    guitar_position_vector.append(list(value.values())[0])
    i = i + 1


def half_wave_rectifier(x):
    return (x + abs(x)) / 2

@jit
def half_wave_rectifier_optimized(x1, x2):
    rez = abs(x1 - x2)
    return (rez + abs(rez)) / 2

def time_stamp(curr_bin, ith):
    global X
    return curr_bin * window_in_sec / len(X[0]) + ith * window_in_sec

# def time_stamp(curr_bin):
#    return curr_bin * len(new_data) /   (sampling_rate * len(X[0]))


def peak_detection(mX, t):
    thresh = np.where(np.greater(mX[1:-1],t), mX[1:-1], 0)  # locations above threshold
    next_minor = np.where(mX[1:-1]>mX[2:], mX[1:-1], 0)     # locations higher than the next one
    prev_minor = np.where(mX[1:-1]>mX[:-2], mX[1:-1], 0)    # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor                 # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1                            # add 1 to compensate for previous steps
    return ploc


def peak_interp(mX,  ploc):
    """
    Interpolate peak values using parabolic interpolation
    mX, pX: magnitude and phase spectrum, ploc: locations of peaks
    returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
    """
    val = mX[ploc]                                                    # magnitude of peak bin
    lval = mX[ploc - 1]                                               # magnitude of bin at left
    rval = mX[ploc + 1]                                               # magnitude of bin at right
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)      # center of parabola
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)               # magnitude of peaks
    # ipphase = np.interp(iploc, np.arange(0, pX.size), pX)             # phase of peaks by linear interpolation
    # return iploc, ipmag, ipphase
    return iploc, ipmag


# Crete a dictionary that will be converted in json data and return new  position of previous note
def create_note_data(note_index, time_stamp, prev_fret_pos):

    final_pos = []
    note_data = {}
    # This is first note, take first position of the note
    if prev_fret_pos == -1:
        final_pos = guitar_position_vector[note_index][0]
        prev_fret_pos = guitar_position_vector[note_index][0][0]
    else:
        # If we find a note who has distance maximum than 3 frets => that is our wanted fret
        for note_pos in guitar_position_vector[note_index]:
            if abs(note_pos[0] - prev_fret_pos) <= 3:
                final_pos = note_pos
                prev_fret_pos = note_pos[0]
        if len(final_pos) == 0:     #  No position was found => Take first
            final_pos = guitar_position_vector[note_index][0]
            prev_fret_pos = guitar_position_vector[note_index][0][0]

    note_data["note"] = list(guitar_notes.keys())[note_index]
    note_data["onset"] = time_stamp
    note_data["fret_pos"] = final_pos[0]
    note_data["string_pos"] = final_pos[1]


    return [note_data, prev_fret_pos]


# TODO this is a very bold try
def parallel_spectral_flux(n_idx):
    global X

    spectral_flux_partial_sum = 0
    for k in range(0, len(X)):
        spectral_flux_partial_sum += half_wave_rectifier(abs(X[k][n_idx]) - abs(X[k][n_idx - 1]))
    return spectral_flux_partial_sum

@jit
def calc_spectral_flux(X):
    spectral_flux = np.zeros(len(X[0]))
    spectral_flux_partial_sum = 0

    #for k in range(0, len(X), 6):
    for n in range(1, len(X[0])):
        for k in range(0, len(X), 6):
            spectral_flux_partial_sum += half_wave_rectifier_optimized(X[k][n], X[k][n - 1])
        spectral_flux[n - 1] = spectral_flux_partial_sum
        spectral_flux_partial_sum = 0

    spectral_flux = spectral_flux.T[:-1]
    return spectral_flux

def signal_process(signal_data):
    global X
    global t_global
    global global_time_init

    #start_time = time.time()
    X = np.zeros(0)

    # Apply stft on signal data. and with any remaining data, will check if we have samples above threshold
    fr, t, X = signal.stft(signal_data, sampling_rate,
                           window='hamming', return_onesided=True, nperseg=sfft_window_size_N, noverlap=overlap)

    #print(X[0])
    # Initialize global time at first call
    if global_time_init == False:
        t_global = np.resize(t_global, np.shape(t))
        global_time_init = True

    t_global = t_global[-1] + t

    #print("--- Spectral Flux: %s seconds ---" % round(time.time() - start_time, 3))

    """
        f = plt.figure()
        plt.pcolormesh(t, fr, np.abs(X), vmin=0, vmax=128)
        plt.ylim(0, 4000)
        plt.title('STFT Magnitude Spectrum')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
    """
    # absX = abs(X)
    # absX[absX < np.finfo(float).eps] = np.finfo(float).eps    # if zeros add epsilon to handle log
    # magnitude_spectrum_log = 20 * np.log10(absX)
    # X=magnitude_spectrum_log

    #start_time = time.time()
    spectral_flux = calc_spectral_flux(X)
    #print(spectral_flux)
    #print("--- Spectral Flux: %s seconds ---" % round(time.time() - start_time, 4))

    #print("--- Parallel time: %s seconds ---" % round(time.time() - start_time, 3))

    #print("X[0]", X[0])
    # plt.plot(t, spectral_flux, label='Spectral Flux')
    # plt.xlabel('Time [sec]')
    # plt.legend(loc=0)
    # plt.show()

    # delta = 10**5 + 10**3 + 10**2
    # alpha = 0.95
    # delta = 5*10**3
    w = 3
    m = 3

    # These are the band new values!
    alpha = 0.80
    delta = 2*10**2

    g_alpha = np.zeros(len(spectral_flux))
    onset_candidate = np.zeros(len(spectral_flux))
    # onset_candidate[onset_candidate == 0] = np.nan

    for n in range(0, len(spectral_flux)):
        if n == 0:  # base case
            g_alpha[n] = max(spectral_flux[n], (1 - alpha) * spectral_flux[n])
        else:
            g_alpha[n] = max(spectral_flux[n], alpha * g_alpha[n - 1] + (1 - alpha) * spectral_flux[n])

    for n in range(0, len(spectral_flux)):
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
    onset_time_stamp = np.apply_along_axis(time_stamp, -1, onset_bin, 0)
    # for global times
    # onset_time_stamp = np.apply_along_axis(np.sum, 0, t_global[-1])
    # if onset_time_stamp.size != 0:
    # print("Time stamp: ", onset_time_stamp, " Bin: ", onset_bin)
    # print(len(X[0]))

    # Apply window for onset candidates
    valid_onset = []
    valid_onset_idx = []
    k = 0
    while k < len(onset_time_stamp):
        j = 1
        if k + j < len(onset_time_stamp) and abs(onset_time_stamp[k] - onset_time_stamp[k + j]) < onset_window:
            j += 1
        valid_onset.append(onset_time_stamp[k])
        valid_onset_idx.append(k)
        k += j

    if np.size(valid_onset) == 0:
        return

    # Peak detection
    peak_threshold = 200

    frequency_axis = sampling_rate * np.arange(sfft_window_size_N / 2 + 1) / float(sfft_window_size_N)

    # print("Valid onset bins: ", onset_bin[valid_onset_idx])
    previous_note = 0.0 # Previous note is used as a mechanism to prevent duplicates in note detection
    previous_fret_pos = -1
    bin_note_detected = -1 # This is position bin + k where a note was detected
    for bin in onset_bin[valid_onset_idx]:
        notes_per_window = np.zeros(0)
        for k in range(0, bins_check):
            if k + bin >= len(spectral_flux):
                break

            peaks, _ = signal.find_peaks(abs(X[:, bin + k]), height=peak_threshold)
            # print("peaks: ", peaks)
            if len(peaks) == 0:
                continue

            iploc, ipmag = peak_interp(abs(X[:, bin + k]), peaks)
            # plt.figure(1)
            # plt.plot(frequency_axis, abs(X[:, bin + k]))
            # plt.plot(sampling_rate * iploc / float(sfft_window_size_N), ipmag, marker='x', linestyle='')
            # plt.show()
            if ipmag.size != 0:
                notes_per_window = np.append(notes_per_window, sampling_rate * iploc / float(sfft_window_size_N))
                bin_note_detected = bin + k
                # print("notes_per_window ", notes_per_window)
                # Take only notes that are above threshold_min and below threshold max
                notes_window_index_greater = np.where(np.greater(notes_per_window, frequency_threshold_min))
                # notes_window_index_less = np.where(np.less(notes_per_window, frequency_threshold_max))
                notes_per_window = notes_per_window[notes_window_index_greater[0]]
                # notes_per_window = notes_per_window[notes_window_index_less[0]]
                # print("notes_per_window ", notes_per_window)

                # TODO check if notes_window is empty
            else:
                break

        # Calculate index of the aproximation note
        if len(notes_per_window) != 0:
            index = (np.abs(guitar_notes_vector - notes_per_window.min())).argmin()
            # print("index ", index)
            if previous_note == guitar_notes_vector[index]:
                # print("Ignore note. Previous is the same")
                previous_note = 0.0  # Reset
            else:
                # print("Note ", list(guitar_notes.keys())[index], ": ", guitar_notes_vector[index], " : ",
                #      notes_per_window.min())
                previous_note = guitar_notes_vector[index]

                # TODO 1. create data structure
                # TODO 2. send
                [note_data, previous_fret_pos] = create_note_data(index, t_global[bin_note_detected], previous_fret_pos)

                # print(t_global)
                print("Note: ", note_data["note"], "Time: ", round(note_data["onset"], 3), "Poz: ", note_data["fret_pos"], note_data["string_pos"])
                json_note_data = json.dumps(note_data)

                q_notes.append(json_note_data)
                break

        else:
            return

    # TODO
    # 1. find fret based on previous discovered note

    """
        plt.figure(1000)
        plt.title("Threshold function vs Onset detection")
        label_print = False
        for onset in valid_onset:
            if not label_print:
                plt.axvline(x=onset, color='r', linestyle='--', label='Onset Detection')
                label_print = True
            plt.axvline(x=onset, color='r', linestyle='--')
        plt.plot(t_global, g_alpha, label='Threshold')
        # plt.plot(t, mean_square_of_enegies / 10**4, label='Mean Square Energy')
        plt.plot(t_global, spectral_flux, label='Spectral Flux')
        # plt.plot(t, onset_candidate, label='Onset detection', marker='x', linestyle='')
        plt.xlabel('Time [sec]')
        plt.legend(loc=0)
        plt.show()
    """


def process_guitar_sound():
    # start_time = time.time()

    # global new_data
    global shared_buf_signal_data
    global producer_count
    global consumer_count

    new_data = []

    # Open a file to test data integrity
    # f = open("/home/andrei/f/licenta/guitar-signal-process/test.wav", 'wb')
    while True:
        try:
            get_data = queue.popleft()
        except IndexError:
            continue
        # print("produced: ", producer_count, "consumed", consumer_count)

        # while producer_count - consumer_count == 0:
        #     continue
        # get_data = shared_buf_signal_data[consumer_count % BUF_SIZE]
        # consumer_count = consumer_count + 1

        # f.write(get_data)

        # because new_data is created from 4 bits, we need to multiply by 2 new_data to get correct timing
        #start_time = time.time()
        for i in range(0, int(get_data.__len__()), 2):
            two_bytes = int.from_bytes(bytearray(get_data[i:i + 2]), 'little', signed=True)
            new_data.append(two_bytes)

        if len(new_data) >= sampling_rate:

            # start_time = time.time()

            # p = Process(target=signal_process, args=(new_data,))
            # p.start()
            # p.join()

            signal_process(new_data)

            # print("--- Process time: %s seconds ---" % round(time.time() - start_time, 5))
            # Reset data Vector
            new_data = []

            #print("process data. size ", new_data.__len__())
            # Inspect how much time it took
            #print("--- Process time: %s seconds ---" % (time.time() - start_time))
            # start_time = time.time()


# Produce
def capture_guitar_sound():
    global shared_buf_signal_data
    global producer_count
    global consumer_count

    device = 'hw:1'
    # device = 'default'

    inp = aa.PCM(aa.PCM_CAPTURE, aa.PCM_NONBLOCK, device=device)

    inp.setchannels(1)
    inp.setrate(sampling_rate)
    inp.setformat(aa.PCM_FORMAT_S16_LE)

    # For our purposes, it is suficcient to know that reads from the device
    # will return this many frames. Each frame being 2 bytes long.
    # This means that the reads below will return either 320 bytes of data
    # or 0 bytes of data. The latter is possible because we are in nonblocking
    # mode.
    inp.setperiodsize(int(period_size))

    # Open a file to test data integrity
    # f = open("/home/andrei/f/licenta/guitar-signal-process/test.wav", 'wb')

    while 1:
        l, data = inp.read()

        # print("--- abs time %s seconds ---" % (time.time() - abs_start_time))

        if l > 0: # append data if any
            # f.write(data)
            queue.append(data)

            # print("size: ", l)

            # print("produced: ", producer_count, "consumed", consumer_count)

        elif l < 0:
            print("data lost!")

def main():

    thread1 = SocketThread(1)
    thread2 = SocketThread(2)
    thread3 = SocketThread(3)

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()


if __name__ == "__main__":
    main()


