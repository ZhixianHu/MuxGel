import os
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
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


# from einops import rearrange
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.imgProcess_di_resT import imgFusionWithBg
from src.imgProcess_di_resT import tactImgUpdate
from src.imgProcess_di_resT import tactBgObtain
from src.imgProcess_di_resT import nonContactLightMapObtain
from src.imgProcess_di_resT import tactChange, tactImgUpdate
from src.imgProcess_di_resT import rgbBackgroundFillIn

OBJ_PATCH_DIR = str(PROJECT_ROOT / "data" / "mujoco_patch_output_320_240")

_PRESS_RE = re.compile(r"_pressDepth_(\d*\.?\d+)mm\.(jpg|png)$")
_CONTACT_NPZ_RE = re.compile(r"_contact_masks_(\d*\.?\d+)mm\.npz$")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def split_by_press(obj_patch_dir: str = OBJ_PATCH_DIR, train_ratio: float = 0.9, val_ratio: float = 0.05, seed: int = 42):
    if not os.path.exists(obj_patch_dir):
        # Fallback for debugging if dir doesn't exist
        print(f"Warning: Directory {obj_patch_dir} not found. Using dummy data.")
        return [], [], []
    
    objPath = sorted([os.path.join(obj_patch_dir, d) for d in os.listdir(obj_patch_dir)
               if os.path.isdir(os.path.join(obj_patch_dir, d))])
    
    if len(objPath) == 0:
        raise ValueError(f"No object directories found in {obj_patch_dir}")
    
    objPatchPair = [(p,i) for p in objPath for i in range(1,51)]
    
    rng = np.random.RandomState(seed)
    rng.shuffle(objPatchPair)

    total = len(objPatchPair)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    train_pairs = objPatchPair[:n_train]
    val_pairs = objPatchPair[n_train:n_train + n_val]
    test_pairs = objPatchPair[n_train + n_val:]

    return train_pairs, val_pairs, test_pairs

def load_images_to_ram(path_list, height=240, width=320):
    n_samples = len(path_list)
    data_block = np.zeros((n_samples, height, width, 3), dtype=np.uint8)
    
    valid_count = 0
    for i, path in enumerate(tqdm(path_list, desc="Loading images")):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Warning: Failed to read {path}")
            continue
        data_block[i] = img 
        valid_count += 1

    if valid_count < n_samples:
        print(f"Trimming empty slots: {valid_count}/{n_samples} valid.")
        data_block = data_block[:valid_count]
    return data_block


class simFusedImgDataset(Dataset):
    def __init__(self,
                 patch_pairs,
                 tacShadow: bool = True,
                 fixed_mask: bool = True,
                 fixed_cell_num: Optional[int] = None,
                 noise_sigma_range: Tuple[float, float] = (0.0, 25.0/255.0),
                #  transform=None,
                 downsample=False,
                 downsample_h=240,
                 downsample_w=320
                 ):
        self.patch_pairs = patch_pairs
        if len(self.patch_pairs) == 0:
            raise ValueError("No patch pairs provided to the dataset.")
        self.fixed_mask = fixed_mask
        self.fixed_cell_num = fixed_cell_num
        self.noise_sigma_range = noise_sigma_range
        # self.transform = transform
        self.tacShadow = tacShadow
        self.downsample = downsample
        self.downsample_h = downsample_h
        self.downsample_w = downsample_w

        # color jitter for data augmentation
        self.vis_jitter_params = {
            'contrast': (0.5, 1.5),
            'saturation': (0.5, 1.5),
            'hue': (-0.1, 0.1)
        }
        self.tact_jitter_params = {
            'contrast': (0.7, 1.3),
            'saturation': (0.7, 1.3),
            'hue': (-0.03, 0.03) 
        }
        
        self.ambient_range = (0.4, 1.6)
        self.led_range = (0.8, 1.2)
        self.leakage_factor = 0.3
        
    def read_patch(self, patchPair):
        def _read_npz(npz_path: Path) -> dict:
            with np.load(npz_path, allow_pickle=False) as data:
                return {k: data[k] for k in data.files}
            
        p = Path(patchPair[0])
        generateCount = int(patchPair[1])
        prefix = f"{generateCount:03d}_"
        patch_npz = p / f"{prefix}patch_masks.npz"
        patch = _read_npz(patch_npz)
        background_mask = patch["background_mask"].astype(np.uint8)
        valid_mask = patch["valid_mask"].astype(np.uint8)
        rgb = cv2.imread(str(p / f"{prefix}rgb.jpg"), cv2.IMREAD_COLOR)
        if rgb is None:
            raise IOError(f"Failed to read image: {p / f'{prefix}rgb.jpg'}")
        
        tact_map = {}
        tact_shadow_map = {} 
        contact_map = {} 

        for img_path in p.glob(f"{prefix}*pressDepth_*mm.jpg"):
            m = _PRESS_RE.search(img_path.name)
            if not m:
                continue
            depth = float(m.group(1))

            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                raise IOError(f"Failed to read image: {img_path}")

            if "_tact_shadow_" in img_path.name:
                tact_shadow_map[depth] = img
            elif "_tact_" in img_path.name:
                tact_map[depth] = img
        
        for npz_path in p.glob(f"{prefix}contact_masks_*mm.npz"):
            m = _CONTACT_NPZ_RE.search(npz_path.name)
            if not m:
                continue
            depth = float(m.group(1))
            d = _read_npz(npz_path)
            contact_map[depth] = (d["contact_mask"].astype(np.uint8),
                                d["org_contact_mask"].astype(np.uint8))

        depths = sorted(set(tact_map.keys()) & set(tact_shadow_map.keys()) & set(contact_map.keys()))
        if len(depths) == 0:
            raise FileNotFoundError(f"No complete press-depth set found for {prefix}")

        tact_list = [tact_map[d] for d in depths]
        tact_shadow_list = [tact_shadow_map[d] for d in depths]
        contact_mask_list = [contact_map[d][0] for d in depths]
        org_contact_mask_list = [contact_map[d][1] for d in depths]

        return {
            "press_depths_mm": depths,
            "tact": tact_list,
            "tact_shadow": tact_shadow_list,
            "contact_mask": contact_mask_list,
            "org_contact_mask": org_contact_mask_list,
            "background_mask": background_mask,
            "valid_mask": valid_mask,
            "rgb": rgb,
        }

    def __len__(self):
        return len(self.patch_pairs) * 5
    
    def adjust_image_cv(self, img, brightness, contrast, saturation, hue_shift):
        img = img.astype(np.float32)

        img = (img - 127.5) * contrast + 127.5
        img = img * brightness
        img = np.clip(img, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        hsv[:, :, 1] *= saturation
        hsv[:, :, 0] += hue_shift
        hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        img_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img_out
    
    def apply_correlated_jitter_cv(self, tactImgList, visImgList):
        outputTactList = []
        outputVisList = []
        global_ambient = np.random.uniform(*self.ambient_range)
        internal_led = np.random.uniform(*self.led_range)
        vis_brightness = global_ambient * np.random.uniform(0.95, 1.05)
        leakage_effect = (global_ambient - 1.0) * self.leakage_factor
        tact_brightness = internal_led + leakage_effect
        tact_brightness = max(0.2, tact_brightness)

        vis_con = np.random.uniform(*self.vis_jitter_params['contrast'])
        vis_sat = np.random.uniform(*self.vis_jitter_params['saturation'])
        vis_hue = np.random.uniform(*self.vis_jitter_params['hue'])

        tact_con = np.random.uniform(*self.tact_jitter_params['contrast'])
        tact_sat = np.random.uniform(*self.tact_jitter_params['saturation'])
        tact_hue = np.random.uniform(*self.tact_jitter_params['hue'])

        for tact in tactImgList:
            tact = self.adjust_image_cv(tact, tact_brightness, tact_con, tact_sat, tact_hue)
            outputTactList.append(tact)
        for vis in visImgList:
            vis = self.adjust_image_cv(vis, vis_brightness, vis_con, vis_sat, vis_hue)
            outputVisList.append(vis)
        return outputTactList, outputVisList
    
    def __getitem__(self, idx):
        pair_idx = idx // 5
        trial_idx = idx % 5
        patchPair = self.patch_pairs[pair_idx]

        patchSample = self.read_patch(patchPair)
        depth = patchSample["press_depths_mm"][trial_idx]
        tact_img = patchSample["tact"][trial_idx]
        tact_shadow_img = patchSample["tact_shadow"][trial_idx]
        contact_mask = patchSample["contact_mask"][trial_idx]
        org_contact_mask = patchSample["org_contact_mask"][trial_idx]
        background_mask = patchSample["background_mask"]
        valid_mask = patchSample["valid_mask"]
        rgb_img = patchSample["rgb"]

        
        # if self.transform is not None:
        #     rgb_img = self.transform(rgb_img)
        #     tact_img = self.transform(tact_img)
        #     tact_shadow_img = self.transform(tact_shadow_img)
        #     contact_mask = self.transform(contact_mask)
        #     org_contact_mask = self.transform(org_contact_mask)
        #     background_mask = self.transform(background_mask)
        #     valid_mask = self.transform(valid_mask)

        if self.tacShadow:
            tact = tact_shadow_img
        else:
            tact = tact_img
        
        rgb_img, nonContactMsk, contactMsk, backgroundMsk = rgbBackgroundFillIn(rgb_img, contact_mask, background_mask)
        
        if (self.fixed_mask) and (self.fixed_cell_num is not None):
            cell_num = self.fixed_cell_num
        else:
            cell_num = np.random.choice([2,3,4,5,6,7,8])
        
        nonContactLightMap = nonContactLightMapObtain()
        tactBg = tactBgObtain()
        # tact = tactImgUpdate(tact, tactBg)
        tactDiff = tactChange(tact)
        # color jittering
        outputTactJitter, outputVisJitter = self.apply_correlated_jitter_cv([tactBg], [rgb_img, nonContactLightMap]) 
        tactBg = outputTactJitter[0]
        rgb_img_jitter = outputVisJitter[0]
        nonContactLightMap = outputVisJitter[1]
        tact = tactImgUpdate(tactDiff, tactBg)
        
        fused_img, refImg, rgb_processed, ckMsk = imgFusionWithBg(rgb_img_jitter, tact, tactBg, nonContactLightMap,
                                                                              nonContactMsk, contactMsk, backgroundMsk, cell_num)
        
        tact_Diff_norm = (tactDiff.astype(np.float32) / 255.0)
        to_tensor = T.ToTensor()
            
        sample = {
            "fused_img": to_tensor(fused_img), # input one
            "ref_img": to_tensor(refImg), # input two
            "rgb_gt": to_tensor(rgb_processed),
            # "rgb_processed": to_tensor(rgb_processed),
            "rgb_jitter": to_tensor(rgb_img_jitter),
            # "rgb_gt": to_tensor(rgb_img),
            "tact_gt": to_tensor(tact),
            "tact_diff_gt": torch.from_numpy(tact_Diff_norm).permute(2, 0, 1).float(),
            "tact_bg": to_tensor(tactBg),
            "ck_msk": torch.from_numpy(ckMsk).unsqueeze(0).float(),
            "contact_mask": torch.from_numpy(contact_mask).unsqueeze(0).float(),
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

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(dim, dim, 3, dilation=dilation),
            nn.InstanceNorm2d(dim),
            nn.SiLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, dilation=1),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

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

class DecouplingLoss(nn.Module):
    def __init__(self, w_tact=1.0, w_vis=1.0, w_grad=0.5):
        super().__init__()
        self.w_tact = w_tact
        self.w_vis = w_vis
        self.w_grad = w_grad
        self.l1 = nn.L1Loss()
        
    def forward(self, tact_pred, tact_gt, vis_pred, vis_gt, mask=None):
        # 1. Vision Loss (Background)
        loss_vis = self.l1(vis_pred, vis_gt)
        
        # 2. Tactile Loss
        loss_tact_pixel = self.l1(tact_pred, tact_gt)
        loss_tact_grad = gradient_loss(tact_pred, tact_gt)
        
        loss_tact = loss_tact_pixel + self.w_grad * loss_tact_grad
        
        return self.w_tact * loss_tact + self.w_vis * loss_vis, \
               {"l_tact": loss_tact.item(), "l_vis": loss_vis.item()}

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
    train_pairs, val_pairs, _ = split_by_press(seed=args.seed)

    fixed_cell_num = args.fixed_cell_num if args.fixed_mask else None

    train_ds = simFusedImgDataset(
        patch_pairs=train_pairs,
        fixed_mask=args.fixed_mask,
        fixed_cell_num=fixed_cell_num,
        # transform = transform,
        downsample = args.downsample,
        downsample_h = args.downsample_h,
        downsample_w = args.downsample_w,
    )

    val_full_ds = simFusedImgDataset(
        patch_pairs=val_pairs,
        fixed_mask=args.fixed_mask,
        fixed_cell_num=fixed_cell_num,
        # transform = transform,
        downsample = args.downsample,
        downsample_h = args.downsample_h,
        downsample_w = args.downsample_w,
    )

    val_limit_samples = args.val_limit_samples

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_full_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = DualResNet34UNet(n_channels=6, n_classes_tact=3, n_classes_vis=3).to(device)

    if run:
        wandb.watch(model, log="all")

    criterion = DecouplingLoss(w_tact=args.w_tact, w_vis=args.w_vis, w_grad=args.w_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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

    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Resuming training from checkpoint: {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] 
            print(f"Jumping to epoch {start_epoch + 1}")
        else:
            print("Checkpoint didn't contain epoch info. Starting from epoch 0 but with loaded weights.")

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

                loss, loss_dict = criterion(tact_diff_pred, tact_diff_gt, vis_pred, rgb_gt, mask=ck_msk)

            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
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

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--w_tact', type=float, default=10.0, help='Weight for tactile loss')
    parser.add_argument('--w_grad', type=float, default=5.0, help='Weight for tactile loss')
    parser.add_argument('--w_vis', type=float, default=1, help='Weight for vision loss')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=2e-4)

    parser.add_argument('--fixed_mask', action='store_true')
    parser.add_argument('--fixed_cell_num', type=int, default=4,
                    help='Exact checkerboard cell number per col/row when --fixed_mask')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--seed', type=int, default=42)
    
    # parser.add_argument('--evaluate', action='store_true',
    #                     help='Run evaluation after training (not implemented yet)')
    parser.add_argument('--ckpt', type=str, default='')
    # parser.add_argument('--eval_out', type=str, default='./recon_imgs')
    parser.add_argument('--run_name', type=str, default='sim_di_rest_run')

    parser.add_argument('--eval_w_tact_mse', type=float, default=0.0)
    parser.add_argument('--eval_w_tact_psnr', type=float, default=0.0)
    parser.add_argument('--eval_w_tact_ssim', type=float, default=1.0)
    parser.add_argument('--eval_w_tact_lpips', type=float, default=0.8)
    parser.add_argument('--eval_w_vis_mse', type=float, default=0.0)
    parser.add_argument('--eval_w_vis_psnr', type=float, default=0.0)
    parser.add_argument('--eval_w_vis_ssim', type=float, default=0.5)
    parser.add_argument('--eval_w_vis_lpips', type=float, default=0.4)

    parser.add_argument('--val_limit_samples', type=int, default=2000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--downsample_h', type=int, default=240)
    parser.add_argument('--downsample_w', type=int, default=320)
    parser.add_argument('--predefined', action='store_true')

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    args = parser.parse_args()
    

    # if args.evaluate:
    #     if args.ckpt == '':
    #         args.ckpt = os.path.join(args.outdir, args.project, f"gan_best.pt")
    #     evaluate_model(args.ckpt, args.image_root, out_dir=args.eval_out, device='cuda' if torch.cuda.is_available() else 'cpu',
    #                    height=args.height, width=args.width,
    #                    fixed_mask=args.fixed_mask, fixed_cell_num=args.fixed_cell_num, fixed_duty=args.fixed_duty,
    #                    base=args.base)
    # else:
    #     train(args)

    train(args)