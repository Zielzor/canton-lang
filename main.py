import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim

def extract_spectrogram(audio_path, n_fft=2048, hop_length=512):
    y, sr = librosa.load(audio_path)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db, sr

def plot_spectrogram(S_db, sr, hop_length, title='Spectrogram'):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

def compare_spectrograms(S_db_1, S_db_2):
    # Ensure the spectrograms have the same shape
    min_shape = min(S_db_1.shape, S_db_2.shape)
    S_db_1_resized = S_db_1[:min_shape[0], :min_shape[1]]
    S_db_2_resized = S_db_2[:min_shape[0], :min_shape[1]]

    # Compute data range
    data_range = S_db_1_resized.max() - S_db_1_resized.min()
    
    similarity, _ = ssim(S_db_1_resized, S_db_2_resized, data_range=data_range, full=True)
    return similarity

# Example usage
audio_path_1 = r'file1.wav'
audio_path_2 = r'file2.wav'

S_db_1, sr_1 = extract_spectrogram(audio_path_1)
S_db_2, sr_2 = extract_spectrogram(audio_path_2)

plot_spectrogram(S_db_1, sr_1, 512, title='Spectrogram 1')
plot_spectrogram(S_db_2, sr_2, 512, title='Spectrogram 2')

similarity = compare_spectrograms(S_db_1, S_db_2)
print(f"Spectrogram similarity: {similarity:.4f}")
