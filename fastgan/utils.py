import torch
import torch.nn.functional as F
from torchvision import utils as vutils

import os
import shutil

import json

from fastgan.models import load_params, copy_G_params


def get_dir(args):
    task_name = 'train_results/' + args.name
    saved_model_folder = os.path.join(task_name, 'models')
    saved_image_folder = os.path.join(task_name, 'images')

    os.makedirs(saved_model_folder, exist_ok=True)
    os.makedirs(saved_image_folder, exist_ok=True)

    for f in os.listdir('./'):
        if '.py' in f:
            shutil.copy(f, task_name + '/' + f)

    with open(os.path.join(saved_model_folder, '../args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    return saved_model_folder, saved_image_folder

def parse_image_set(image_set):
    files = []
    with open(image_set, "r") as f:
        for image in f.readlines():
            image = image.strip()
            files.append(image)

    return files

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]


def save_iter_image(iteration, saved_image_folder, netG, avg_param_G, fixed_noise, real_image, rec_img_all, rec_img_small,
                    rec_img_part):
    backup_para = copy_G_params(netG)
    load_params(netG, avg_param_G)
    with torch.no_grad():
        vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder + '/%d.jpg' % iteration, nrow=4)
        vutils.save_image(torch.cat([
            F.interpolate(real_image, 128),
            rec_img_all, rec_img_small,
            rec_img_part]).add(1).mul(0.5), saved_image_folder + '/rec_%d.jpg' % iteration)
    load_params(netG, backup_para)


def save_model(iteration, saved_model_folder, netG, netD, avg_param_G, optimizerG, optimizerD):
    try:
        # Ensure the directory exists
        os.makedirs(saved_model_folder, exist_ok=True)
        
        # Save the model with error handling
        try:
            torch.save({
                'g': netG.state_dict(),
                'd': netD.state_dict(),
                'g_ema': avg_param_G,
                'opt_g': optimizerG.state_dict(),
                'opt_d': optimizerD.state_dict()
            }, os.path.join(saved_model_folder, f'all_{iteration}.pth'))
            print(f"Successfully saved model checkpoint at iteration {iteration}")
        except Exception as e:
            print(f"Error saving model checkpoint: {str(e)}")
            # Try saving to a different location if the first attempt fails
            backup_path = os.path.join(os.path.dirname(saved_model_folder), 'backup_models')
            os.makedirs(backup_path, exist_ok=True)
            torch.save({
                'g': netG.state_dict(),
                'd': netD.state_dict(),
                'g_ema': avg_param_G,
                'opt_g': optimizerG.state_dict(),
                'opt_d': optimizerD.state_dict()
            }, os.path.join(backup_path, f'all_{iteration}.pth'))
            print(f"Saved backup checkpoint to {backup_path}")
    except Exception as e:
        print(f"Critical error saving model: {str(e)}")
        print("Training will continue but checkpoints may not be saved")
