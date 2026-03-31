# Use this

import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import random
import random
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as T
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
import torchvision.transforms.functional as TF


from tqdm import tqdm
import glob

from datetime import datetime
from pathlib import Path
import sys
import re
        
import wandb
WANDB_AVAILABLE = True
FIXED_PROJECT_NAME = "muxgel"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
data_root = PROJECT_ROOT / "data" / "calibration_data"

print(f"Project Root: {PROJECT_ROOT}")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)



class realFusedImgDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 split="train", 
                 train_ratio=0.9, 
                 fixed_sensor_id=None, 
                 seed=42,
                 img_h=240, 
                 img_w=320):
        
        self.data_root = Path(data_root)
        self.fixed_sensor_id = fixed_sensor_id
        self.img_h = img_h
        self.img_w = img_w
        self.to_tensor = T.ToTensor()
        self.samples = []

        self.tact_jitter_params = {
            'contrast': (0.9, 1.1), 
            'saturation': (0.9, 1.1),
            'hue': (-0.01, 0.01)   
        }
 
        self.pattern = re.compile(r"^(?P<prefix>.+)_(?P<id>\d+)_x_(?P<x>-?[\d\.]+)_y_(?P<y>-?[\d\.]+)_depth_(?P<d>[\d\.]+)\.jpg$")

        self._build_index(seed, split, train_ratio)

    def _build_index(self, seed, split, train_ratio):
        all_states = []
        
        for color_obj_dir in self.data_root.iterdir():
            if not color_obj_dir.is_dir(): continue
            
            sensor0_dir = color_obj_dir / "sensor_0"
            if not sensor0_dir.exists(): continue
            
            for img_path in sensor0_dir.glob("*.jpg"):
                if "ref" in img_path.name: continue
                
                match = self.pattern.match(img_path.name)
                if match:
                    info = match.groupdict()
                    base_state = {
                        "color_obj_dir": color_obj_dir.name,
                        "prefix": info["prefix"],
                        "x": info["x"],
                        "y": info["y"],
                        "d": info["d"]
                    }
                    
                    target_sensors = [2, 4, 8] if self.fixed_sensor_id is None else [self.fixed_sensor_id]
                    
                    for s_id in target_sensors:
                        check_path = self.data_root / color_obj_dir.name / f"sensor_{s_id}" / f"{info['prefix']}_{s_id}_x_{info['x']}_y_{info['y']}_depth_{info['d']}.jpg"
                        
                        if check_path.exists():
                            new_state = base_state.copy()
                            new_state["sensor_id"] = s_id 
                            all_states.append(new_state)
 
        rng = np.random.RandomState(seed)
        rng.shuffle(all_states)
        
        n_total = len(all_states)
        n_train = int(n_total * train_ratio)
        
        if split == "train":
            self.samples = all_states[:n_train]
        else:
            self.samples = all_states[n_train:]
            
        print(f"Loaded Real {split} Dataset: {len(self.samples)} samples.")

    def _load_img(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        # Resize to match network expectation
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        return img

    def __len__(self):
        return len(self.samples)
    
    def apply_physics_based_jitter(self, fused_cv, ref_cv):
        fused_float = fused_cv.astype(np.float32)
        ref_float = ref_cv.astype(np.float32)

        if random.random() > 0.5:
            ambient_light = random.uniform(0, 40.0) 
            fused_float += ambient_light

        if random.random() > 0.5:
            exposure_scale = random.uniform(0.9, 1.1)
            fused_float *= exposure_scale
            ref_float *= exposure_scale
        
        fused_out = np.clip(fused_float, 0, 255).astype(np.uint8)
        ref_out = np.clip(ref_float, 0, 255).astype(np.uint8)
        return fused_out, ref_out


    def __getitem__(self, idx):
        s = self.samples[idx]
        color_obj_dir = s["color_obj_dir"]
        prefix, x, y, d = s["prefix"], s["x"], s["y"], s["d"]
        
        grid_id = s["sensor_id"]

        def get_path(s_id, is_ref=False):
            folder = self.data_root / color_obj_dir / f"sensor_{s_id}"
            if is_ref:
                filename = f"{prefix}_{s_id}_ref.jpg"
            else:
                filename = f"{prefix}_{s_id}_x_{x}_y_{y}_depth_{d}.jpg"
            return folder / filename

        fused_img = self._load_img(get_path(grid_id))
        ref_img   = self._load_img(get_path(grid_id, is_ref=True))
        rgb_gt = self._load_img(get_path(0))
        tact_gt = self._load_img(get_path(1))
        tact_bg = self._load_img(get_path(1, is_ref=True))

        fused_img, ref_img = self.apply_physics_based_jitter(
            fused_img, ref_img
        )

        tact_diff = tact_gt.astype(np.float32) - tact_bg.astype(np.float32)
        tact_diff_norm = tact_diff / 255.0

        dummy_mask = np.zeros((self.img_h, self.img_w), dtype=np.float32)

        sample = {
            "fused_img": self.to_tensor(fused_img),     # (3, H, W)
            "ref_img": self.to_tensor(ref_img),         # (3, H, W)
            
            "rgb_gt": self.to_tensor(rgb_gt),           # (3, H, W)
            "tact_gt": self.to_tensor(tact_gt),      
            "tact_bg": self.to_tensor(tact_bg),  
            
            "tact_diff_gt": torch.from_numpy(tact_diff_norm).permute(2, 0, 1).float(), 
            
            "ck_msk": torch.from_numpy(dummy_mask).unsqueeze(0).float(),
            "index": idx,
        }

        return sample
    

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), 
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DualResNet34UNet(nn.Module):
    def __init__(self, n_channels=6, n_classes_tact=3, n_classes_vis=3):
        super(DualResNet34UNet, self).__init__()
        self.n_channels = n_channels
        
        # --- Shared Encoder (ResNet34) ---
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        if n_channels == 6:
            original_weights = resnet.conv1.weight.data.clone()
            new_conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv1.weight[:, :3, :, :] = original_weights
                new_conv1.weight[:, 3:, :, :] = original_weights
                new_conv1.weight *= 0.5
            resnet.conv1 = new_conv1

        self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        
        self.encoder1 = resnet.layer1 # 64
        self.encoder2 = resnet.layer2 # 128
        self.encoder3 = resnet.layer3 # 256
        self.encoder4 = resnet.layer4 # 512
        
        # --- Decoder 1: Tactile Stream ---
        self.tact_up1 = Up(512 + 256, 256)
        self.tact_up2 = Up(256 + 128, 128)
        self.tact_up3 = Up(128 + 64, 64)
        # ResNet layer1 output is 64ch, initial conv output is 64ch. 
        # We need to bridge the gap from layer1 back to original resolution
        self.tact_up4 = Up(64 + 64, 64) 
        self.tact_out = nn.Conv2d(64, n_classes_tact, kernel_size=1)

        # --- Decoder 2: Vision Stream ---
        self.vis_up1 = Up(512 + 256, 256)
        self.vis_up2 = Up(256 + 128, 128)
        self.vis_up3 = Up(128 + 64, 64)
        self.vis_up4 = Up(64 + 64, 64)
        self.vis_out = nn.Conv2d(64, n_classes_vis, kernel_size=1)
        
        self.tact_final_act = nn.Tanh() 
        self.vis_final_act = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (Batch, n_channels, H, W)
        # n_channels: 6
        # --- Encoding ---
        # ResNet Stem
        x = self.inc[0](x) # conv1
        x = self.inc[1](x) # bn1
        x0 = self.inc[2](x) # relu (64 ch, H/2, W/2) -> Skip 0
        x = self.inc[3](x0) # maxpool
        
        x1 = self.encoder1(x)  # (64 ch) -> Skip 1
        x2 = self.encoder2(x1) # (128 ch) -> Skip 2
        x3 = self.encoder3(x2) # (256 ch) -> Skip 3
        x4 = self.encoder4(x3) # (512 ch) -> Bottleneck

        # --- Tactile Decoding ---
        t = self.tact_up1(x4, x3)
        t = self.tact_up2(t, x2)
        t = self.tact_up3(t, x1)
        t = self.tact_up4(t, x0) 
        t = F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
        tact_pred = self.tact_out(t)
        tact_pred = self.tact_final_act(tact_pred)

        # --- Vision Decoding ---
        v = self.vis_up1(x4, x3)
        v = self.vis_up2(v, x2)
        v = self.vis_up3(v, x1)
        v = self.vis_up4(v, x0)
        v = F.interpolate(v, scale_factor=2, mode='bilinear', align_corners=False)
        vis_pred = self.vis_out(v)
        vis_pred = self.vis_final_act(vis_pred)

        return tact_pred, vis_pred
    
def gradient_loss(pred, target):
    kernel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(pred.device)
    kernel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).to(pred.device)
    
    channels = pred.shape[1]
    kernel_x = kernel_x.expand(channels, -1, -1, -1)
    kernel_y = kernel_y.expand(channels, -1, -1, -1)
    
    pred_grad_x = F.conv2d(pred, kernel_x, groups=channels, padding=1)
    pred_grad_y = F.conv2d(pred, kernel_y, groups=channels, padding=1)
    target_grad_x = F.conv2d(target, kernel_x, groups=channels, padding=1)
    target_grad_y = F.conv2d(target, kernel_y, groups=channels, padding=1)
    
    loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    return loss

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16]])
        for p in self.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, input, target):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0.0
        x, y = input, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

class DecouplingLoss(nn.Module):
    def __init__(self, device, w_tact=1.0, w_vis=1.0, w_grad=0.5, w_perceptual=0.1, w_ssim=5.0):
        super().__init__()
        self.w_tact = w_tact
        self.w_vis = w_vis
        self.w_grad = w_grad
        self.w_perceptual = w_perceptual
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss(device)
        self.w_ssim = w_ssim
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device) 
        
    def forward(self, tact_pred, tact_gt, vis_pred, vis_gt, tact_recon_pred=None, tact_recon_gt=None):
        # 1. Vision Loss (Background)
        loss_vis_l1 = self.l1(vis_pred, vis_gt)
        loss_vis_perc = self.perceptual(vis_pred, vis_gt)
        loss_vis = loss_vis_l1 + self.w_perceptual * loss_vis_perc
        
        # 2. Tactile Loss
        loss_tact_pixel = self.l1(tact_pred, tact_gt)
        loss_tact_grad = gradient_loss(tact_pred, tact_gt)
        

        tact_pred_norm = (tact_pred + 1.0) / 2.0
        tact_gt_norm = (tact_gt + 1.0) / 2.0
        
        loss_tact_perc = self.perceptual(tact_recon_pred, tact_recon_gt)
        loss_tact_ssim = 1.0 - self.ssim_metric(tact_pred, tact_gt)
        
        loss_tact = loss_tact_pixel + self.w_grad * loss_tact_grad + self.w_perceptual * loss_tact_perc + self.w_ssim * loss_tact_ssim

        total_loss = self.w_tact * loss_tact + self.w_vis * loss_vis

        return total_loss,\
               {"l_tact": loss_tact.item(), "l_vis": loss_vis.item(), "l_vis_l1": loss_vis_l1.item(), "l_vis_perc": loss_vis_perc.item(), "l_tact_ssim": loss_tact_ssim.item()}

class reconEvaluator:
    def __init__(self, device, save_dir='./checkpoints', metric_weights=None, run=None):

        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.run = run
        if metric_weights is None:
            self.metric_weights = {'tact_ssim': 1.0}
        else:
            self.metric_weights = metric_weights

        self.maximize_list = ['tact_ssim', 'vis_ssim', 'tact_psnr', 'vis_psnr']

        # Metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        self.mse_func = nn.MSELoss()

        self.best_score = -float('inf')
        self.reset()

    def reset(self):
        self.metrics_sum = {
            'tact_mse': 0.0, 'tact_psnr': 0.0, 'tact_ssim': 0.0, 'tact_lpips': 0.0,
            'vis_mse': 0.0,  'vis_psnr': 0.0,  'vis_ssim': 0.0,  'vis_lpips': 0.0
        }
        self.count = 0

    @torch.no_grad()
    def update(self, tact_pred, tact_gt, vis_pred, vis_gt):

        batch_size = tact_pred.size(0)
        self.count += batch_size

        tp, tg = torch.clamp(tact_pred, 0, 1), torch.clamp(tact_gt, 0, 1)
        vp, vg = torch.clamp(vis_pred, 0, 1),  torch.clamp(vis_gt, 0, 1)

        self.metrics_sum['tact_mse']   += self.mse_func(tp, tg).item() * batch_size
        self.metrics_sum['tact_psnr']  += self.psnr(tp, tg).item() * batch_size
        self.metrics_sum['tact_ssim']  += self.ssim(tp, tg).item() * batch_size
        self.metrics_sum['tact_lpips'] += self.lpips(tp, tg).item() * batch_size

        self.metrics_sum['vis_mse']   += self.mse_func(vp, vg).item() * batch_size
        self.metrics_sum['vis_psnr']  += self.psnr(vp, vg).item() * batch_size
        self.metrics_sum['vis_ssim']  += self.ssim(vp, vg).item() * batch_size
        self.metrics_sum['vis_lpips'] += self.lpips(vp, vg).item() * batch_size

    def compute(self):
        if self.count == 0: return self.metrics_sum
        avg_metrics = {k: v / self.count for k, v in self.metrics_sum.items()}
        return avg_metrics
    
    def _calculate_weighted_score(self, current_metrics):
        total_score = 0.0
        for name, weight in self.metric_weights.items():
            value = current_metrics[name]
            if name in self.maximize_list:
                total_score += weight * value
            else:
                total_score -= weight * value
        return total_score

    def save_if_best(self, model, epoch, current_metrics, global_step):
        
        current_score = self._calculate_weighted_score(current_metrics)
        is_best = False

        if current_score > self.best_score:
            self.best_score = current_score
            is_best = True

        torch.save(model.state_dict(), os.path.join(self.save_dir, "last_checkpoint.pth"))

        if is_best:
            save_path = os.path.join(self.save_dir, "best_checkpoint.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New Best Model at epoch {epoch+1}! (Score: {self.best_score:.4f}) Saved to {save_path}")

            if self.run is not None:
                art = wandb.Artifact(f"{FIXED_PROJECT_NAME}-best-model", type="model")
                art.add_file(save_path)
                self.run.log_artifact(art)
                self.run.log({"best_weighted_score": self.best_score}, step=global_step)
        return is_best



def train(args):
    def to_u8_hwc(x):
        x = x.detach().cpu().clamp(0,1)
        if x.size(0) == 1: 
            x = x.repeat(3,1,1)
        x = (x.permute(1,2,0).numpy()*255).astype(np.uint8)
        x = x[..., ::-1]
        return x
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.cuda.amp.GradScaler()
    set_seed(args.seed)

    run = None
    if WANDB_AVAILABLE and args.wandb:
        run_name = f"{args.run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if args.ckpt:
            run_name += "_resumed"
        run = wandb.init(project=args.project, entity=args.entity, config=vars(args), name=run_name, reinit=True)
    else:
        print("WANDB logging is disabled.")

    train_ds = realFusedImgDataset(data_root=data_root, split="train", train_ratio=0.9, seed=args.seed)
    val_ds = realFusedImgDataset(data_root=data_root, split="val", train_ratio=0.9, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = DualResNet34UNet(n_channels=6, n_classes_tact=3, n_classes_vis=3).to(device)

    criterion = DecouplingLoss(device, w_tact=args.w_tact, w_vis=args.w_vis, w_grad=args.w_grad, w_perceptual=args.w_perceptual, w_ssim=args.w_ssim)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    eval_w_tact_mse = args.eval_w_tact_mse
    eval_w_tact_psnr = args.eval_w_tact_psnr
    eval_w_tact_ssim = args.eval_w_tact_ssim
    eval_w_tact_lpips = args.eval_w_tact_lpips
    eval_w_vis_mse = args.eval_w_vis_mse
    eval_w_vis_psnr = args.eval_w_vis_psnr
    eval_w_vis_ssim = args.eval_w_vis_ssim
    eval_w_vis_lpips = args.eval_w_vis_lpips
    eval_metric_weights = {
            'tact_mse': eval_w_tact_mse, 'tact_psnr': eval_w_tact_psnr, 'tact_ssim': eval_w_tact_ssim, 
            'tact_lpips': eval_w_tact_lpips, 
            'vis_mse': eval_w_vis_mse,  'vis_psnr': eval_w_vis_psnr,  'vis_ssim': eval_w_vis_ssim,  
            'vis_lpips': eval_w_vis_lpips
        }
    
    start_epoch = 0

    checkpoint = torch.load(args.sim_ckpt)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    for param in model.inc.parameters():
        param.requires_grad = False
    # for param in model.encoder1.parameters():
    #     param.requires_grad = False

    if run:
        wandb.watch(model, log="all")

    ckpt_dir = os.path.join(args.outdir, args.project, args.run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    evaluator = reconEvaluator(device, save_dir=ckpt_dir, metric_weights = eval_metric_weights, run=run)
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        train_loss = 0.0
        for batch in pbar:
            global_step += 1
            fused_img = batch["fused_img"].to(device).float()
            ref_img = batch["ref_img"].to(device).float()
            rgb_gt = batch["rgb_gt"].to(device)
            tact_gt = batch["tact_gt"].to(device)
            tact_diff_gt = batch["tact_diff_gt"].to(device)
            ck_msk = batch["ck_msk"].to(device)
            tact_bg = batch["tact_bg"].to(device)

            input_tensor = torch.cat([fused_img, ref_img], dim=1)

            with torch.cuda.amp.autocast():
                tact_diff_pred, vis_pred = model(input_tensor)
                tact_recon_pred = tact_diff_pred + tact_bg
                tact_recon_pred = torch.clamp(tact_recon_pred, 0, 1)
                loss, loss_dict = criterion(tact_diff_pred, tact_diff_gt, vis_pred, rgb_gt, tact_recon_pred, tact_gt)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss = loss.item(), tact_l=loss_dict['l_tact'], vis_l=loss_dict['l_vis'])

            if run is not None and (pbar.n % args.log_interval == 0):
                with torch.no_grad():
                    vis_pred = torch.clamp(vis_pred, 0, 1)
                    tact_pred = tact_diff_pred + tact_bg
                    tact_pred = torch.clamp(tact_pred, 0, 1)
                    
                    b0 = 0
                    grid = np.concatenate([
                        to_u8_hwc(fused_img[b0]),
                        to_u8_hwc(ref_img[b0]),
                        to_u8_hwc(tact_gt[b0]),
                        to_u8_hwc(rgb_gt[b0]),
                        to_u8_hwc(tact_pred[b0]),
                        to_u8_hwc(vis_pred[b0]),
                    ], axis=1)
                    wandb.log({"preview": wandb.Image(grid, caption="fused | ref | tact_gt_recon | rgb_gt | tact_pred | vis_pred")}, step=global_step)
        
        scheduler.step()
        print(f"Epoch {epoch+1} finished. Avg Loss: {train_loss / len(train_loader):.4f}")

        model.eval()
        evaluator.reset() 

        with torch.no_grad():
            for batch in val_loader:           
                fused_img = batch["fused_img"].to(device).float()
                ref_img = batch["ref_img"].to(device).float()
                rgb_gt = batch["rgb_gt"].to(device)
                tact_gt = batch["tact_gt"].to(device)
                tact_diff_gt = batch["tact_diff_gt"].to(device)
                tact_bg = batch["tact_bg"].to(device)
                ck_msk = batch["ck_msk"].to(device)
                tact_bg = batch["tact_bg"].to(device)
                

                input_tensor = torch.cat([fused_img, ref_img], dim=1)
                tact_diff_pred, vis_pred = model(input_tensor)
                tact_pred = tact_diff_pred + tact_bg
                tact_pred = torch.clamp(tact_pred, 0, 1)
                evaluator.update(tact_pred, tact_gt, vis_pred, rgb_gt)
            
            val_metrics = evaluator.compute()
            is_best = evaluator.save_if_best(model, epoch, val_metrics, global_step)

            print(f"\nEpoch {epoch+1} Validation Results:")
            print(f"Tactile -> SSIM: {val_metrics['tact_ssim']:.4f} | PSNR: {val_metrics['tact_psnr']:.2f} | LPIPS: {val_metrics['tact_lpips']:.4f}")
            print(f"Vision  -> SSIM: {val_metrics['vis_ssim']:.4f}  | PSNR: {val_metrics['vis_psnr']:.2f}")

            if run is not None:
                # art = wandb.Artifact(f"{FIXED_PROJECT_NAME}-epoch-{epoch+1}", type="model")
                # art.add_file(ckpt_path)
                # run.log_artifact(art)
                run.log(val_metrics, step=global_step)

        ckpt_path = os.path.join(ckpt_dir, f"unet_epoch{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(), 
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args':vars(args)},ckpt_path)

    if run is not None:
        run.finish()

if __name__ == "__main__":
    print("start")
    parser = argparse.ArgumentParser()

    parser.add_argument('--project', type=str, default='muxgel')
    parser.add_argument('--entity', type=str, default=None, help='wandb team/user (optional)')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--w_tact', type=float, default=10.0, help='Weight for tactile loss')
    parser.add_argument('--w_grad', type=float, default=5.0, help='Weight for tactile loss')
    parser.add_argument('--w_vis', type=float, default=2.0, help='Weight for vision loss')
    parser.add_argument('--w_perceptual', type=float, default=0.1, help='Weight for perceptual loss')
    parser.add_argument('--w_ssim', type=float, default=5.0, help='Weight for SSIM tactile loss')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=2e-5)

    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=42)
    
    # parser.add_argument('--evaluate', action='store_true',
    #                     help='Run evaluation after training (not implemented yet)')
    parser.add_argument('--sim_ckpt', type=str, default='/home/zhixian/code/MuxGel/outputs/muxgel/sim_di_rest_run/best_checkpoint.pth')
    parser.add_argument('--ckpt', type=str, default='')
    # parser.add_argument('--eval_out', type=str, default='./recon_imgs')
    parser.add_argument('--run_name', type=str, default='real_di_rest_run')

    parser.add_argument('--eval_w_tact_mse', type=float, default=0.0)
    parser.add_argument('--eval_w_tact_psnr', type=float, default=0.0)
    parser.add_argument('--eval_w_tact_ssim', type=float, default=1.0)
    parser.add_argument('--eval_w_tact_lpips', type=float, default=0.8)
    parser.add_argument('--eval_w_vis_mse', type=float, default=0.0)
    parser.add_argument('--eval_w_vis_psnr', type=float, default=0.0)
    parser.add_argument('--eval_w_vis_ssim', type=float, default=0.5)
    parser.add_argument('--eval_w_vis_lpips', type=float, default=0.4)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--predefined', action='store_true')

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    args = parser.parse_args()
    train(args)