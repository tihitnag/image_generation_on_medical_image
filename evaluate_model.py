import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from PIL import Image
import os
from tqdm import tqdm
import lpips
from fastgan.data import ImageFolder
from fastgan.models import Generator

def calculate_fid(real_images, fake_images):
    """Calculate FID score between real and generated images"""
    # Convert images to numpy arrays
    real_images = real_images.numpy()
    fake_images = fake_images.numpy()
    
    # Calculate mean and covariance
    mu1, sigma1 = np.mean(real_images, axis=0), np.cov(real_images, rowvar=False)
    mu2, sigma2 = np.mean(fake_images, axis=0), np.cov(fake_images, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def evaluate_model(model_path, real_data_path, num_samples=1000, batch_size=8):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load generator
    generator = Generator(ngf=128, nz=512, im_size=256)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['g'])
    generator = generator.to(device)
    generator.eval()
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load real data
    real_dataset = ImageFolder(real_data_path, transform=transform)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)
    
    # Generate fake images
    fake_images = []
    real_images = []
    
    print("Generating and evaluating images...")
    with torch.no_grad():
        # Generate fake images
        for _ in tqdm(range(num_samples // batch_size)):
            noise = torch.randn(batch_size, 512, device=device)
            fake_batch = generator(noise)
            fake_images.append(fake_batch.cpu())
            
        # Get real images
        for batch in tqdm(real_loader):
            real_images.append(batch)
            
    # Calculate FID
    fake_images = torch.cat(fake_images, dim=0)
    real_images = torch.cat(real_images, dim=0)
    
    fid_score = calculate_fid(real_images, fake_images)
    print(f"FID Score: {fid_score:.2f}")
    
    # Calculate LPIPS (perceptual similarity)
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=torch.cuda.is_available())
    lpips_scores = []
    
    for i in tqdm(range(min(len(fake_images), len(real_images)))):
        score = percept(fake_images[i:i+1], real_images[i:i+1])
        lpips_scores.append(score.item())
    
    avg_lpips = np.mean(lpips_scores)
    print(f"Average LPIPS Score: {avg_lpips:.4f}")
    
    # Save some sample images
    os.makedirs('evaluation_samples', exist_ok=True)
    for i in range(min(10, len(fake_images))):
        img = fake_images[i].permute(1, 2, 0).numpy()
        img = ((img + 1) * 127.5).astype(np.uint8)
        Image.fromarray(img).save(f'evaluation_samples/sample_{i}.png')
    
    return {
        'fid_score': fid_score,
        'lpips_score': avg_lpips,
        'num_samples': num_samples
    }

if __name__ == '__main__':
    # Example usage
    model_path = 'train_results/normal_images/models/all_2000.pth'  # Update this path
    real_data_path = 'data/your_dataset/train.txt'  # Update this path
    
    results = evaluate_model(model_path, real_data_path)
    print("\nEvaluation Results:")
    print(f"FID Score: {results['fid_score']:.2f}")
    print(f"LPIPS Score: {results['lpips_score']:.4f}")
    print(f"Number of samples evaluated: {results['num_samples']}") 