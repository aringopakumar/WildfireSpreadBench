import os, sys, torch, h5py, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddpm.unet import Unet
from ddpm.diffusion_proc import Diffusion

class FixedVisualizeDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        test_path = Path(data_dir) / "2021"
        for folder in [f for f in test_path.iterdir() if f.is_dir()]:
            files = sorted(list(folder.glob("*.h5")))
            for i in range(len(files) - 1):
                self.samples.append((files[i], files[i+1]))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        input_path, target_path = self.samples[idx]
        with h5py.File(input_path, 'r') as f_in, h5py.File(target_path, 'r') as f_out:
            raw_target = np.nan_to_num(f_out['imagery'][22], nan=0.0)
            full_target = np.where(raw_target > 0, 1.0, 0.0).astype(np.float32)
            
            # VEGETATION ABLATION: I1, I2, M11, NDVI, EVI2 (0, 1, 2, 3, 4) and Fire (22)
            raw_cond = f_in['imagery'][[0, 1, 2, 3, 4, 22]]
            
            c_I1 = np.nan_to_num(raw_cond[0], nan=0.0)
            c_I2 = np.nan_to_num(raw_cond[1], nan=0.0)
            c_M11 = np.nan_to_num(raw_cond[2], nan=0.0)
            c_ndvi = np.nan_to_num(raw_cond[3], nan=0.0)
            c_evi2 = np.nan_to_num(raw_cond[4], nan=0.0)
            
            raw_c_fire = np.nan_to_num(raw_cond[5], nan=0.0)
            c_fire = np.where(raw_c_fire > 0, 1.0, 0.0)
            
            # Stack into exactly 6 channels
            full_cond = np.stack([c_fire, c_I1, c_I2, c_M11, c_ndvi, c_evi2], axis=0).astype(np.float32)
            
            h, w = full_target.shape
            new_h, new_w = (h // 32) * 32, (w // 32) * 32
            top, left = (h - new_h) // 2, (w - new_w) // 2
            
            cond_img = full_cond[:, top:top+new_h, left:left+new_w]
            target_img = full_target[top:top+new_h, left:left+new_w]

        return torch.from_numpy(target_img[np.newaxis, ...]), torch.from_numpy(cond_img)

def visualize_prediction(model, loader, diffusion, device, save_path="fire_prediction_plot.png"):
    model.eval()
    print("Hunting for a real fire...")
    for target, cond in loader:
        if cond[0, 0].sum() >= 50: 
            break

    target, cond = target.to(device), cond.to(device)
    print(f"Found a fire with {int(cond[0,0].sum().item())} pixels! Generating...")
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        prediction = diffusion.p_sample_loop(model, target.shape, cond)
        
    final_pred = prediction[-1][0, 0]
    
    # Bulletproof check to handle both numpy arrays and PyTorch tensors
    if torch.is_tensor(final_pred):
        final_pred = final_pred.cpu().numpy()
            
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cond[0, 0].cpu().numpy(), cmap='magma', vmin=0, vmax=1)
    axs[0].set_title("Input: Today's Fire")
    axs[0].axis('off')
    
    axs[1].imshow(target[0, 0].cpu().numpy(), cmap='magma', vmin=0, vmax=1)
    axs[1].set_title("Ground Truth: Tomorrow's Fire")
    axs[1].axis('off')
    
    axs[2].imshow(final_pred, cmap='magma', vmin=0, vmax=1)
    axs[2].set_title("Model Prediction")
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Success! Saved to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, required=True,
                        help="Root of HDF5 dataset (contains 2018/, 2019/, ...)")
    parser.add_argument("--ckpt_dir",  type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="fire_ddpm_last.pt",
                        help="Checkpoint filename to load")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Updated to cond_ch=6
    model = Unet(in_ch=1, cond_ch=6, output_ch=1, model_ch=128, channel_mult=(1, 2, 2, 4)).to(device)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    
    dataset = FixedVisualizeDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True) 
    diffusion = Diffusion(timesteps=1000, noise_schedule='sigmoid')
    visualize_prediction(model, loader, diffusion, device)