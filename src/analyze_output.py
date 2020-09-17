import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import argparse
from pathlib import Path
import h5py


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--generated', type=Path, required=True)
    args = parser.parse_args()
    output_dir = Path(args.input).parent
    output_name = args.generated.name[:-4]

    input_wave, isr = librosa.load(args.input)
    input_fft = librosa.amplitude_to_db(np.abs(librosa.stft(input_wave)))

    generated_wave, gsr = librosa.load(args.generated)
    generated_fft = librosa.amplitude_to_db(np.abs(librosa.stft(generated_wave)))

    base_wave = h5py.File('../flute/final/train/allemande_fifth_fragment_preston.h5', 'r')
    base_wave = np.array(base_wave.get('wav'))
    base_fft = librosa.amplitude_to_db(np.abs(librosa.stft(base_wave)))

    fig = plt.figure(figsize=[15,20])
    plt.subplot(3,1,1)
    librosa.display.specshow(input_fft, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'piano {args.input.name[:-4]}')
    plt.xlabel('time')

    plt.subplot(3,1,2)
    librosa.display.specshow(generated_fft, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'flute {args.generated.name[:-4]}')
    plt.xlabel('time')

    plt.subplot(3,1,3)
    librosa.display.specshow(base_fft, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('flute allemande_fifth_fragment_preston')
    plt.xlabel('time')
    
    fig.savefig(f'{output_dir}/{output_name}.png')
    



if __name__ == "__main__":
    main()