from itertools import chain
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
import argparse
import librosa
import h5py
import sys
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import clip_grad_value_

import utils
import wavenet_models
from wavenet import WaveNet
from data import DatasetSet
from utils import save_audio, create_output_dir, LossMeter, wrap
from wavenet_models import cross_entropy_loss, Encoder, ZDiscriminator

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn', force=True)

parser = argparse.ArgumentParser(description='Finetuning the Universe')

parser.add_argument('--gpu', type=str, required=True, help='Specify which GPUs to use separated by a comma. Ex: 2,3')
# Env options:
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 92)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--expName', type=str, required=True,
                    help='Experiment name')
parser.add_argument('--data',
                    metavar='D', type=Path, help='Data path', nargs='+')
parser.add_argument('--checkpoint', required=True, type=Path, 
                    help='Checkpoint path')
#parser.add_argument('--load-optimizer', action='store_true')
parser.add_argument('--per-epoch', action='store_true',
                    help='Save model per epoch')

# Distributed
parser.add_argument('--dist-url', default='env://',
                    help='Distributed training parameters URL')
parser.add_argument('--dist-backend', default='nccl')
#parser.add_argument('--local_rank', type=int,
#                   help='Ignored during training.')

# Data options
parser.add_argument('--seq-len', type=int, default=16000,
                    help='Sequence length')
parser.add_argument('--epoch-len', type=int, default=10000,
                    help='Samples per epoch')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--num-workers', type=int, default=10,
                    help='DataLoader workers')
parser.add_argument('--data-aug', action='store_true',
                    help='Turns data aug on')
parser.add_argument('--magnitude', type=float, default=0.5,
                    help='Data augmentation magnitude.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr-decay', type=float, default=0.98,
                    help='new LR = old LR * decay')
parser.add_argument('--short', action='store_true',
                    help='Run only a few batches per epoch for testing')
parser.add_argument('--h5-dataset-name', type=str, default='wav',
                    help='Dataset name in .h5 file')

# Decoder options
parser.add_argument('--decoder', type=int, nargs='*', default=[],
                        help='Only output for the following decoder ID')
parser.add_argument('--decoder-update', type=int, default=1,
                    help='Number of decoder layers to be updated in the fine tuning process')


class Finetuner:
    def __init__(self, args):
        self.args = args
        self.args.n_datasets = len(args.data)
        self.modelPath = Path('checkpoints') / args.expName
        
        self.logger = create_output_dir(args, self.modelPath)
        self.data = [DatasetSet(d, args.seq_len, args) for d in args.data] 

        self.losses_recon = [LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.loss_total = LossMeter('total')

        self.evals_recon = [LossMeter(f'recon {i}') for i in range(self.args.n_datasets)]
        self.eval_total = LossMeter('eval total')

        self.start_epoch = 0

        #torch.manual_seed(args.seed)
        #torch.cuda.manual_seed(args.seed)

        #get the pretrained model checkpoints    
        checkpoint = args.checkpoint.parent.glob(args.checkpoint.name + '_*.pth')
        checkpoint = [c for c in checkpoint if extract_id(c) in args.decoder][0]

        model_args = torch.load(args.checkpoint.parent / 'args.pth')[0]
       
        self.encoder = Encoder(model_args)
        self.decoder = WaveNet(model_args) 
        
        self.encoder = Encoder(model_args)
        self.encoder.load_state_dict(torch.load(checkpoint)['encoder_state'])
        
        #encoder freeze
        for param in self.encoder.parameters():
            param.requires_grad = False
            #self.logger.debug(f'encoder at start: {param}')
        
        self.decoder = WaveNet(model_args)
        self.decoder.load_state_dict(torch.load(checkpoint)['decoder_state'])

        #decoder freeze
        for param in self.decoder.layers[:-args.decoder_update].parameters():
            param.requires_grad = False 
            #self.logger.debug(f'decoder at start: {param}')

        self.encoder = torch.nn.DataParallel(self.encoder).cuda()
        self.decoder = torch.nn.DataParallel(self.decoder).cuda()
        self.model_optimizer = optim.Adam(chain(self.encoder.parameters(),
                                                self.decoder.parameters()),
                                                lr=args.lr)

        self.lr_manager = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, args.lr_decay)
        self.lr_manager.step()

    def train_batch(self, x, x_aug, dset_num):
        'train batch without considering the discriminator'
        x = x.float()
        x_aug = x_aug.float()
        z = self.encoder(x_aug)
        y = self.decoder(x, z)

        recon_loss = cross_entropy_loss(y, x)
        self.losses_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())
        loss = recon_loss.mean()
        
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        self.loss_total.add(loss.data.item())

        return loss.data.item()

            
    def train_epoch(self, epoch):
        for meter in self.losses_recon:
            meter.reset()
        self.loss_total.reset()

        self.decoder.train()

        n_batches = self.args.epoch_len

        with tqdm(total=n_batches, desc='Train epoch %d' % epoch) as train_enum:
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 3:
                    break

                if self.args.distributed:
                    assert self.args.rank < self.args.n_datasets, "No. of workers must be equal to #dataset"
                    # dset_num = (batch_num + self.args.rank) % self.args.n_datasets
                    dset_num = self.args.rank
                else:
                    dset_num = batch_num % self.args.n_datasets

                x, x_aug = next(self.data[dset_num].train_iter)

                x = wrap(x)
                x_aug = wrap(x_aug)
                batch_loss = self.train_batch(x, x_aug, dset_num)

                train_enum.set_description(f'Train (loss: {batch_loss:.2f}) epoch {epoch}')
                train_enum.update()

    def eval_batch(self, x, x_aug, dset_num):
        x, x_aug = x.float(), x_aug.float()
        z = self.encoder(x)
        y = self.decoder(x, z)
        
        recon_loss = cross_entropy_loss(y, x)
        self.evals_recon[dset_num].add(recon_loss.data.cpu().numpy().mean())

        total_loss = recon_loss.mean().data.item()
        self.eval_total.add(total_loss)

        return total_loss


    def evaluate_epoch(self, epoch): 
        for meter in self.evals_recon:
            meter.reset()
        self.eval_total.reset()

        self.encoder.eval()
        self.decoder.eval()

        n_batches = int(np.ceil(self.args.epoch_len / 10))

        with tqdm(total=n_batches) as valid_enum, torch.no_grad():
            for batch_num in range(n_batches):
                if self.args.short and batch_num == 10:
                    break

                if self.args.distributed:
                    assert self.args.rank < self.args.n_datasets, "No. of workers must be equal to #dataset"
                    dset_num = self.args.rank
                else:
                    dset_num = batch_num % self.args.n_datasets

                x, x_aug = next(self.data[dset_num].valid_iter)

                x = wrap(x)
                x_aug = wrap(x_aug)
                batch_loss = self.eval_batch(x, x_aug, dset_num)

                valid_enum.set_description(f'Test (loss: {batch_loss:.2f}) epoch {epoch}')
                valid_enum.update()

    @staticmethod
    def format_losses(meters):
        losses = [meter.summarize_epoch() for meter in meters]
        return ', '.join('{:.4f}'.format(x) for x in losses)

    def train_losses(self):
        meters = [*self.losses_recon]
        return self.format_losses(meters)

    def eval_losses(self):
        meters = [*self.evals_recon]
        return self.format_losses(meters)

    def finetune(self):  
        best_eval = float('inf')

        for epoch in range(self.start_epoch, self.start_epoch + self.args.epochs):
            self.logger.info(f'Starting epoch, Rank {self.args.rank}, Dataset: {self.args.data[self.args.rank]}')
            self.train_epoch(epoch)
            self.evaluate_epoch(epoch)

            self.logger.info(f'Epoch %s Rank {self.args.rank} - Train loss: (%s), Test loss (%s)',
                             epoch, self.train_losses(), self.eval_losses())
            self.lr_manager.step()
            val_loss = self.eval_total.summarize_epoch()

            if val_loss < best_eval:
                self.save_model(f'bestmodel_{self.args.rank}.pth')
                best_eval = val_loss

            if not self.args.per_epoch:
                self.save_model(f'lastmodel_{self.args.rank}.pth')
            else:
                self.save_model(f'lastmodel_{epoch}_rank_{self.args.rank}.pth')

            if self.args.is_master:
                torch.save([self.args,
                            epoch],
                           '%s/args.pth' % self.modelPath)
            self.logger.debug('Ended epoch')


    def save_model(self, filename):
        save_path = self.modelPath / filename

        torch.save({'encoder_state': self.encoder.module.state_dict(),
                    'decoder_state': self.decoder.module.state_dict(),
                    'model_optimizer_state': self.model_optimizer.state_dict(),
                    'dataset': self.args.rank,
                    },
                   save_path)

        self.logger.debug(f'Saved model to {save_path}')
    
def extract_id(path):
    decoder_id = str(path)[:-4].split('_')[-1]
    return int(decoder_id)

def main():
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    torch.cuda.empty_cache()
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    
    if args.distributed:
        if int(os.environ['RANK']) == 0:
            args.is_master = True
        else:
            args.is_master = False
        args.rank = int(os.environ['RANK'])

        print('Before init_process_group')
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url)
    else:
        args.rank = 0
        args.is_master = True

    Finetuner(args).finetune()

if __name__ == '__main__':
    main()
