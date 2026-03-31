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

nonContactLightMap = random.choice(nonContactLightMapList)


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
        for base_pos in base_positions:
            amp = np.random.uniform(0, max_amp)
            freq = np.random.uniform(0.02, 0.08)
            phase = np.random.uniform(0, 2 * np.pi)
            line = base_pos + amp * np.sin(freq * t + phase)
            lines.append(line)
        return np.array(lines)
    
    v_lines = generate_boundary_lines(h, w, cell_num-1)
    h_lines = generate_boundary_lines(w, h, cell_num-1)
    yy, xx = np.mgrid[0:h, 0:w]
    x_grid_idx = np.sum(xx[np.newaxis, :, :] > v_lines[:, :, np.newaxis], axis=0)
    y_grid_idx = np.sum(yy[np.newaxis, :, :] > h_lines[:, np.newaxis, :], axis=0)
    mask = (x_grid_idx + y_grid_idx) % 2
    if np.random.rand() > 0.5:
        mask = 1 - mask 
    return (mask > 0)


def imgFusion(rgbImg, contactMask, backgroundMask, tactImg, ckBdCellNum=4):
    # fusedImg: tac+Vis; rbgProcessed: Vis with relighting; rgb: original rgbImg with background fill-in
    nonContactImg = np.zeros_like(rgbImg)
    h, w = rgbImg.shape[:2]
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
    checkerMsk = make_randomWavy_checkerboard(ckBdCellNum, h=h, w=w)
    fusedImg = rgbProcessed.copy()
    fusedImg[checkerMsk] = tactImg[checkerMsk]
    return fusedImg, rgbProcessed, rgb, checkerMsk


def imgFusionPredefined(rgbImg, contactMask, backgroundMask, tactImg, preloaded_bg, preloaded_light, ckBdCellNum=4, h=240, w=320):
    # fusedImg: tac+Vis; rbgProcessed: Vis with relighting; rgb: original rgbImg with background fill-in
    nonContactImg = np.zeros_like(rgbImg)
    # h, w = rgbImg.shape[:2]
    # backgroundImg = cv2.imread(str(random.choice(scenePath)), cv2.IMREAD_COLOR)
    # backgroundImg = cv2.resize(backgroundImg, (w,h), interpolation=cv2.INTER_LINEAR)
    backgroundImg = preloaded_bg 
    nonContactLightMap = preloaded_light

    rgbProcessed = rgbImg.copy()
    rgb = rgbImg.copy()

    nonContactMsk = (contactMask == 0).astype(bool)
    contactMsk = (contactMask > 0).astype(bool)
    blackMsk = np.all(rgbImg <= 5, axis=2).astype(bool)
    backgroundMsk = nonContactMsk & (backgroundMask > 0) & blackMsk
    if np.sum(backgroundMsk) > 0:
        nonContactImg[backgroundMsk] = backgroundImg[backgroundMsk]
        rgb[backgroundMsk] = backgroundImg[backgroundMsk]
    
    nonContactNonBgMsk = nonContactMsk & ~backgroundMsk
    mask = np.ones_like(rgbImg, dtype=bool)
    nonContactImg[nonContactNonBgMsk] = rgb[nonContactNonBgMsk]
    # nonContactLightMap = cv2.imread(str(random.choice(orgLightMapPath)), cv2.IMREAD_COLOR)
    # nonContactLightMap = cv2.resize(nonContactLightMap, (rgbImg.shape[1], rgbImg.shape[0]), interpolation=cv2.INTER_LINEAR)
    nonContactImg = simulateRelighting(nonContactImg, nonContactLightMap, mask)

    contactImg = rgbImg.copy()
    contactLightMap = tactImg.copy()
    contactImg = simulateRelighting(contactImg, contactLightMap, contactMsk, led_strength=1.0)

    rgbProcessed[contactMsk] = contactImg[contactMsk]
    rgbProcessed[nonContactMsk] = nonContactImg[nonContactMsk]

    # checkerMsk = make_checkerboard_mask(rgbImg.shape[0], rgbImg.shape[1], cell_num=ckBdCellNum, duty=ckBdDuty)
    checkerMsk = make_randomWavy_checkerboard(ckBdCellNum, h=h, w=w)
    fusedImg = rgbProcessed.copy()
    fusedImg[checkerMsk] = tactImg[checkerMsk]
    return fusedImg, rgbProcessed, rgb, checkerMsk