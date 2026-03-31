
from pathlib import Path
import cv2
import numpy as np
import random
import sys
import os
import math

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

scenePath = Path(str(PROJECT_ROOT / "data" / "indoorCVPRBlur_320_240"))
exts = [".jpg", ".png"]
scenePath = [p for p in scenePath.rglob("*") if p.suffix in exts]

orgLightMapPath = Path(str(PROJECT_ROOT / "assets" / "sensor" / "pureVision"))
orgLightMapPath = [p for p in orgLightMapPath.rglob("*") if p.suffix == ".jpg"]
nonContactLightMapList = []
for path in orgLightMapPath:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
    nonContactLightMapList.append(img)

# nonContactLightMap = random.choice(nonContactLightMapList)

br_proc_org = cv2.imread(str(PROJECT_ROOT / "assets/sensor/pureTactile/f0_initial.png"), cv2.IMREAD_COLOR)
br_proc_org = cv2.GaussianBlur(br_proc_org.astype(np.float32),(5,5),0)
br_proc_org = cv2.resize(br_proc_org, (320, 240), interpolation=cv2.INTER_LINEAR)


orgTactileImgPath = Path(str(PROJECT_ROOT / "assets/sensor/pureTactile"))
orgTactileImgPath = [p for p in orgTactileImgPath.rglob("*") if p.suffix == ".png" and "initial" not in p.name]
orgTactileImgList = []
for path in orgTactileImgPath:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.GaussianBlur(img.astype(np.float32),(5,5),0)
    img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_LINEAR)
    orgTactileImgList.append(img)

def simulateRelighting(img_obj, img_light, mask, led_strength = 1.0, ambient_strength=1.0):
    img_light = cv2.GaussianBlur(img_light, (3, 3), 0)
    obj_float = img_obj.astype(np.float32) / 255.0
    light_float = img_light.astype(np.float32) / 255.0
    led_effect = obj_float * light_float * led_strength
    ambient_effect = obj_float * ambient_strength
    result_float = led_effect + ambient_effect
    result_float = np.clip(result_float, 0.0, 1.0)
    result_img = (result_float * 255).astype(np.uint8)

    res = np.zeros_like(img_obj)
    res[mask] = result_img[mask]
    return res

def make_checkerboard_mask(h: int, w: int, cell_num: int = 4, duty: float = 0.5,
                           ) -> np.ndarray:
    """Return HxW binary mask with a checkerboard pattern.
    duty: fraction of black (masked) cells per direction (approximate)
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cell_h = h // cell_num
    cell_w = w // cell_num
    xx = xx // cell_w
    yy = yy // cell_h
    cb = (xx + yy) % 2
    # duty tweak: randomly flip some cells to hit ~duty
    if duty != 0.5:
        # compute per-cell mask then expand
        nx, ny = math.ceil(w / cell_w), math.ceil(h / cell_h)
        cells = (np.add.outer(np.arange(ny), np.arange(nx)) % 2).astype(np.float32)
        # flip proportion of cells
        desired_black = duty
        if desired_black < 0.5:
            p = (0.5 - desired_black) * 2
            flip = (np.random.rand(*cells.shape) < p)
            cells = np.where(flip, 1 - cells, cells)
        else:
            p = (desired_black - 0.5) * 2
            flip = (np.random.rand(*cells.shape) < p)
            cells = np.where(flip, 1 - cells, cells)
        cb = cells[yy.astype(int), xx.astype(int)]
    return cb.astype(np.uint8)

def make_randomWavy_checkerboard(cell_num, max_amp=5.0, h=480, w=640):
    def generate_boundary_lines(length_axis, length_cross_axis, number_lines):
        base_positions = np.linspace(0, length_cross_axis, number_lines + 2)[1:-1]
        lines = []
        t = np.arange(length_axis)
        for i in range(len(base_positions)):
            base_pos = base_positions[i]
            base_pos = np.random.randint(-5,6) + base_pos
            base_positions[i] = base_pos
            amp = np.random.uniform(0, max_amp)
            freq = np.random.uniform(0.02, 0.08)
            phase = np.random.uniform(0, 2 * np.pi)
            line = base_pos + amp * np.sin(freq * t + phase)
            lines.append(line)
        return np.array(lines), base_positions
    
    v_lines, v_bases = generate_boundary_lines(h, w, cell_num-1)
    h_lines, h_bases = generate_boundary_lines(w, h, cell_num-1)
    yy, xx = np.mgrid[0:h, 0:w]
    x_grid_idx = np.sum(xx[np.newaxis, :, :] > v_lines[:, :, np.newaxis], axis=0)
    y_grid_idx = np.sum(yy[np.newaxis, :, :] > h_lines[:, np.newaxis, :], axis=0)
    mask = (x_grid_idx + y_grid_idx) % 2

    x_grid_idx_straight = np.sum(xx[np.newaxis, :, :] > v_bases[:, np.newaxis, np.newaxis], axis=0)
    y_grid_idx_straight = np.sum(yy[np.newaxis, :, :] > h_bases[:, np.newaxis, np.newaxis], axis=0)
    mask_straight = (x_grid_idx_straight + y_grid_idx_straight) % 2

    if np.random.rand() > 0.5:
        mask = 1 - mask
        mask_straight = 1 - mask_straight
    return (mask > 0), (mask_straight > 0)

def tactBgObtain():
    tactBg = random.choice(orgTactileImgList)
    return tactBg

def tactChange(tactImg):
    tactDiff = tactImg.astype(np.float32) - br_proc_org.astype(np.float32)
    return tactDiff

def tactImgUpdate(tactDiff, tactBg):
    tactImg = tactDiff.astype(np.float32) + tactBg.astype(np.float32)
    tactImg = np.clip(tactImg, 0, 255).astype(np.uint8)
    return tactImg

def nonContactLightMapObtain():
    nonContactLightMap = random.choice(nonContactLightMapList)
    return nonContactLightMap

def rgbBackgroundFillIn(rgbImg, contactMask, backgroundMask):
    nonContactImg = np.zeros_like(rgbImg)
    rgb = rgbImg.copy()

    nonContactMsk = (contactMask == 0).astype(bool)
    contactMsk = (contactMask > 0).astype(bool)
    blackMsk = np.all(rgbImg <= 5, axis=2).astype(bool)
    backgroundMsk = nonContactMsk & (backgroundMask > 0) & blackMsk
    if np.sum(backgroundMsk) > 0:
        backgroundImg = cv2.imread(str(random.choice(scenePath)), cv2.IMREAD_COLOR)
        nonContactImg[backgroundMsk] = backgroundImg[backgroundMsk]
        rgb[backgroundMsk] = backgroundImg[backgroundMsk]
    return rgb, nonContactMsk, contactMsk, backgroundMsk

def imgFusion(rgbImg, contactMask, backgroundMask, tactImg, tactBg, nonContactLightMap, ckBdCellNum=4):
    # fusedImg: tac+Vis; rbgProcessed: Vis with relighting; rgb: original rgbImg with background fill-in
    h, w = rgbImg.shape[:2]
    nonContactImg = np.zeros_like(rgbImg)
    rgbProcessed = rgbImg.copy()
    rgb = rgbImg.copy()

    nonContactMsk = (contactMask == 0).astype(bool)
    contactMsk = (contactMask > 0).astype(bool)
    blackMsk = np.all(rgbImg <= 5, axis=2).astype(bool)
    backgroundMsk = nonContactMsk & (backgroundMask > 0) & blackMsk
    if np.sum(backgroundMsk) > 0:
        backgroundImg = cv2.imread(str(random.choice(scenePath)), cv2.IMREAD_COLOR)
        # backgroundImg = cv2.resize(backgroundImg, (w,h), interpolation=cv2.INTER_LINEAR)
        nonContactImg[backgroundMsk] = backgroundImg[backgroundMsk]
        rgb[backgroundMsk] = backgroundImg[backgroundMsk]
    
    nonContactLightMap = random.choice(nonContactLightMapList)
    nonContactNonBgMsk = nonContactMsk & ~backgroundMsk
    mask = np.ones_like(rgbImg, dtype=bool)
    nonContactImg[nonContactNonBgMsk] = rgb[nonContactNonBgMsk]
    nonContactImg = simulateRelighting(nonContactImg, nonContactLightMap, mask)

    contactImg = rgbImg.copy()
    contactLightMap = tactImg.copy()
    contactImg = simulateRelighting(contactImg, contactLightMap, contactMsk, led_strength=1.0)

    rgbProcessed[contactMsk] = contactImg[contactMsk]
    rgbProcessed[nonContactMsk] = nonContactImg[nonContactMsk]

    # checkerMsk = make_checkerboard_mask(rgbImg.shape[0], rgbImg.shape[1], cell_num=ckBdCellNum, duty=ckBdDuty)
    checkerMsk, checkerMskStraight = make_randomWavy_checkerboard(ckBdCellNum, h=h, w=w)

    refImg = nonContactLightMap.copy()
    refImg[checkerMskStraight] = tactBg[checkerMskStraight]

    fusedImg = rgbProcessed.copy()
    fusedImg[checkerMsk] = tactImg[checkerMsk]
    return fusedImg, refImg, rgbProcessed, rgb, checkerMsk

def imgFusionWithBg(rgbImgwithBg, tactImg, tactBg, nonContactLightMap, nonContactMsk, contactMsk, backgroundMsk, ckBdCellNum=4):
    # fusedImg: tac+Vis; rbgProcessed: Vis with relighting; rgb: original rgbImg with background fill-in
    
    h, w = rgbImgwithBg.shape[:2]
    rgbProcessed = rgbImgwithBg.copy()
    nonContactImg = np.zeros_like(rgbImgwithBg)
    # nonContactLightMap = random.choice(nonContactLightMapList)
    nonContactNonBgMsk = nonContactMsk & ~backgroundMsk
    mask = np.ones_like(rgbImgwithBg, dtype=bool)
    nonContactImg[nonContactNonBgMsk] = rgbImgwithBg[nonContactNonBgMsk]
    nonContactImg = simulateRelighting(nonContactImg, nonContactLightMap, mask)

    contactImg = rgbImgwithBg.copy()
    contactLightMap = tactImg.copy()
    contactImg = simulateRelighting(contactImg, contactLightMap, contactMsk, led_strength=1.0)

    rgbProcessed[contactMsk] = contactImg[contactMsk]
    rgbProcessed[nonContactMsk] = nonContactImg[nonContactMsk]

    # checkerMsk = make_checkerboard_mask(rgbImg.shape[0], rgbImg.shape[1], cell_num=ckBdCellNum, duty=ckBdDuty)
    checkerMsk, checkerMskStraight = make_randomWavy_checkerboard(ckBdCellNum, h=h, w=w)

    refImg = nonContactLightMap.copy()
    refImg[checkerMskStraight] = tactBg[checkerMskStraight]
    # checkerMskStraight = cv2.GaussianBlur(checkerMskStraight.astype(np.float32), (5,5),0)
    # refImg = tactBg.copy() * checkerMskStraight[..., np.newaxis] + nonContactLightMap.copy() * (1 - checkerMskStraight[..., np.newaxis])
    fusedImg = rgbProcessed.copy()
    fusedImg[checkerMsk] = tactImg[checkerMsk]
    return fusedImg, refImg, rgbProcessed, checkerMsk
