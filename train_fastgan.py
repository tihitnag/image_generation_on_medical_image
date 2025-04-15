import argparse
from fastgan.train import Train
import torch


def setup_argparse():
    parser = argparse.ArgumentParser(description='Knee KL Grade GAN')

    parser.add_argument('--path', type=str, default='../data/KLGradeGANs/10-shot_train.txt',
                        help='Path to text file containing training set images')
    parser.add_argument('--kl_grade', type=str, default=None, help="KL grade to train GAN for. Ex. KL0, KL1, KL2, ...")
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=2, help='mini batch number of images (reduced for CPU training)')
    parser.add_argument('--im_size', type=int, default=512, help='image resolution (reduced for CPU training)')
    parser.add_argument('--ckpt', type=str, help='checkpoint weight path if have one')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')

    args = parser.parse_args()
    
    # Set device to CPU
    args.cuda = -1  # Force CPU usage
    
    return args

if __name__ == '__main__':
    args = setup_argparse()
    trainer = Train(args)
    trainer.train_model()

