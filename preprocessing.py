import random
import librosa
import numpy as np
import torchaudio

def preprocess(file_path, sample_rate=22100, duration_s=5, cqt_bins=120, cqt_bins_per_octave=24, max_decibel=80, data_augment=False):
    #load the file at the sample rate in input, default=22.1 kHz
    audio = librosa.core.load(file_path, sr=sample_rate)
    #split the file in array and sample rate
    audio_signal, sample_rate = audio
    #array lenght for resizing the audio file
    to_resize_lenght = sample_rate * duration_s

    #changing speed and pitch for data augmentation
    if(data_augment):
        librosa.effects.pitch_shift(audio_signal, sample_rate, n_steps=random.choice[1,2,3,4,5])
        return librosa.effects.time_stretch(audio_signal, random.choice([0.9, 1.1, 1.2, 1.3, 1.4, 1.5]))

    if len(audio_signal)>to_resize_lenght:    
        #audio is longer, then it is truncated at random offset
        max_offset = len(audio_signal)-to_resize_lenght        
        offset = np.random.randint(max_offset)        
        audio_signal = audio_signal[offset:(to_resize_lenght+offset)]
    else:
        #the audio is smaller or equal, then it is padded with 0s
        if to_resize_lenght > len(audio_signal):
            max_offset = to_resize_lenght - len(audio_signal)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        audio_signal = np.pad(audio_signal, (offset, to_resize_lenght - len(audio_signal) - offset), "constant")

    #Short-time Fourier transform spectrogram
    stft_spectrogram = librosa.stft(audio_signal)
    #Constant-Q transform spectrogram
    cqt_spectrogram = librosa.cqt(audio_signal, n_bins=cqt_bins, bins_per_octave=cqt_bins_per_octave)
    #Scale the spectograms
    stft_spectrogram = librosa.amplitude_to_db(abs(stft_spectrogram), top_db=max_decibel) #1025x216
    cqt_spectrogram = librosa.amplitude_to_db(abs(cqt_spectrogram), top_db=max_decibel) #120x216
    return stft_spectrogram, cqt_spectrogram #216x1025 & 216x120
   