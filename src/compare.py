from pathlib import Path
import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', type=Path, required=True, help='Path to the pretrained model')
    parser.add_argument('--tuned-model', type=Path, required=True, help='Path to the finetuned models')
    args = parser.parse_args()
    base_encoder = torch.load(args.base_model)['encoder_state']
    tuned_encoder = torch.load(args.tuned_model)['encoder_state']
    base_decoder = torch.load(args.base_model)['decoder_state']
    tuned_decoder = torch.load(args.tuned_model)['decoder_state']

    print('------- encoders --------')

    for key in base_encoder.keys():
        if not torch.equal(base_encoder[key], tuned_encoder[key]):
            print(f'encoders {key}: different')
        #else:
          #  print(f'encoders {key}: similar')
    print('------- decoders ---------')

    for key in base_decoder.keys():
        if not torch.equal(base_decoder[key], tuned_decoder[key]):
            print(f'decoders {key}: different')
        #else:
         #   print(f'decoders {key}: similar')
    

if __name__ == '__main__':
    main()