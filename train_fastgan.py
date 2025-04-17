import argparse
from fastgan.train import Train
import torch
import os
import sys

# Set environment variable to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_argparse():
    parser = argparse.ArgumentParser(description='Knee KL Grade GAN')

    parser.add_argument('--path', type=str, default='../data/KLGradeGANs/10-shot_train.txt',
                        help='Path to text file containing training set images')
    parser.add_argument('--kl_grade', type=str, default=None, help="KL grade to train GAN for. Ex. KL0, KL1, KL2, ...")
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=10000, help='number of iterations (reduced from 50000)')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=1, help='mini batch number of images (reduced for memory)')
    parser.add_argument('--im_size', type=int, default=256, help='image resolution (reduced for memory)')
    parser.add_argument('--ckpt', type=str, help='checkpoint weight path if have one')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for data loading')
    parser.add_argument('--save_interval', type=int, default=1000, help='how often to save checkpoints')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='directory to save models and images')

    args = parser.parse_args()
    
    # Create save directory with proper permissions
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Models will be saved to: {os.path.abspath(args.save_dir)}")
    except Exception as e:
        print(f"Error creating save directory: {str(e)}")
        print("Trying to save in current directory...")
        args.save_dir = '.'
    
    # Set device based on CUDA availability
    if torch.cuda.is_available():
        args.device = f'cuda:{args.cuda}'
        # Enable memory efficient settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        args.device = 'cpu'
        print("Warning: CUDA not available, falling back to CPU")
    
    return args

if __name__ == '__main__':
    try:
        args = setup_argparse()
        trainer = Train(args)
        trainer.train_model()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        sys.exit(1)

