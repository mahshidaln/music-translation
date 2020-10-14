import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt 
import argparse
from pathlib import Path
import h5py
import sys
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True, nargs='*')
    parser.add_argument('--generated', type=Path, required=True, nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    
    if len(args.input) == 1 and args.input[0].is_dir():
        top = args.input[0]
        input_paths = list(top.glob('**/*.wav')) 
    else:
        input_paths = args.input

    if len(args.generated) == 1 and args.generated[0].is_dir():
        top = args.generated[0]
        gen_paths = list(top.glob('**/*.wav')) 
    else:
        gen_paths = args.generated
    
    output_dir = args.output
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for input_path in input_paths:
        input_wave, isr = librosa.load(input_path)
        input_fft = librosa.amplitude_to_db(np.abs(librosa.stft(input_wave)))

        input_ons = librosa.onset.onset_strength(input_wave, isr)
        input_onf = librosa.onset.onset_detect(input_wave, isr)
        input_ont = librosa.times_like(input_ons, isr)
        #print(input_path)
        for gen_path in gen_paths:
            #print(gen_path)
            if(gen_path.name == input_path.name):
                generated_wave, gsr = librosa.load(gen_path)
                generated_fft = librosa.amplitude_to_db(np.abs(librosa.stft(generated_wave)))

                gen_ons = librosa.onset.onset_strength(generated_wave, gsr)
                gen_onf = librosa.onset.onset_detect(generated_wave, gsr)
                gen_ont = librosa.times_like(gen_ons, gsr)

                fig = plt.figure(figsize=[15,15])
                ax1 = plt.subplot(2,1,1)
                librosa.display.specshow(input_fft, y_axis='linear', x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'{input_path.name[:-4]}')
                plt.xlabel('time')

                plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
                librosa.display.specshow(generated_fft, y_axis='linear', x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'{gen_path.name[:-4]} to flute')
                plt.xlabel('time')
                
                fig.savefig(f'{output_dir}/{str(gen_path.name)[:-4]}-spec.png')               

                
                fig = plt.figure(figsize=[15,15])
                ax2 = plt.subplot(2,1,1)
                plt.plot(input_ont, input_ons)
                plt.vlines(input_ont[input_onf], 0, input_ons.max(), color='r', alpha=0.9, linestyle='--')
                plt.title(f'{input_path.name[:-4]}')
                plt.xlabel('time')

                plt.subplot(2,1,2, sharex=ax2, sharey=ax2)
                plt.plot(gen_ont, gen_ons)
                plt.vlines(gen_ont[gen_onf], 0, gen_ons.max(), color='r', alpha=0.9, linestyle='--')
                plt.title(f'{gen_path.name[:-4]} to flute')
                plt.xlabel('time')

                fig.savefig(f'{output_dir}/{str(gen_path.name)[:-4]}-onset.png')
    

if __name__ == "__main__":
    main()