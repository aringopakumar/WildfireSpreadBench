import os, sys, argparse, torch, h5py, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ddpm.unet import Unet
from ddpm.diffusion_proc import Diffusion

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.unified_eval import evaluate_ddpm

class FireSpreadEvalDataset(Dataset):
    def __init__(self, data_dir, conditional_offset=1):
        self.data_dir = Path(data_dir)
        self.conditional_offset = conditional_offset
        self.samples = []
        self._prepare_test_sequences()

    def _prepare_test_sequences(self):
        test_path = self.data_dir / "2021"
        fire_folders = [f for f in test_path.iterdir() if f.is_dir()]
        for folder in fire_folders:
            files = sorted(list(folder.glob("*.h5")))
            if len(files) <= self.conditional_offset: continue
            for i in range(len(files) - self.conditional_offset):
                self.samples.append((files[i], files[i + self.conditional_offset]))
                
        import random
        random.seed(42)
        random.shuffle(self.samples)
        self.samples = self.samples[:200]

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

        return torch.from_numpy(cond_img), torch.from_numpy(target_img[np.newaxis, ...])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, required=True,
                        help="Root of HDF5 dataset (contains 2018/, 2019/, ...)")
    parser.add_argument("--ckpt_dir",  type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="fire_ddpm_last.pt",
                        help="Checkpoint filename to load")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(entity="ram-algoverse", project="wildfire-flow", config={
        "task": "evaluation",
        "model": "DDPM (Unet, cond_ch=6, model_ch=128, CFG)",
    })

    # Updated to cond_ch=6
    model = Unet(in_ch=1, cond_ch=6, output_ch=1, model_ch=128, channel_mult=(1, 2, 2, 4)).to(device)
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    
    dataset = FireSpreadEvalDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    diffusion = Diffusion(timesteps=1000, noise_schedule='sigmoid')

    results = evaluate_ddpm(model, diffusion, loader, device, epoch=None, wandb_log=True)
    wandb.finish()