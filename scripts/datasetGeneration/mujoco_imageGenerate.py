"""
Generate local visual-tactile patch data from MuJoCo scanned objects.

For each object, this script samples random surface patches, renders local RGB and depth images,
simulates tactile responses at multiple pressing depths, and saves the resulting images and masks.
"""

import os
from os import path as osp
import math
import numpy as np
import xml.etree.ElementTree as ET
import trimesh
import mujoco
from imageio import imwrite
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import correlate
import scipy.ndimage as ndimage
from scipy import interpolate
import cv2
from pathlib import Path
import argparse

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2] 

sys.path.append(str(PROJECT_ROOT / "external" / "Taxim"))
DATA_FOLDER = PROJECT_ROOT / "external" / "Taxim" / "calibs"
GELPAD_MODEL_PATH = osp.join(DATA_FOLDER, 'gelmap_tacex.npy')

import Basics.params as pr
import Basics.sensorParams as psp

np.random.seed(42)

BASE_DIR = PROJECT_ROOT / "external" / "mujoco_scanned_objects" / "models"
OUT_DIR = PROJECT_ROOT / "data" / "mujoco_patch_output"

IMG_WIDTH = 640
IMG_HEIGHT = 480

PATCH_W_MM = 18.6
PATCH_H_MM = 14.3
PATCH_W = PATCH_W_MM / 1000.0  
PATCH_H = PATCH_H_MM / 1000.0

D = 0.014                    
CAM_NAME = "patch_cam"
NUM_PATCHES_PER_OBJ = 50  

RENDER_RATIO = 2 

os.makedirs(OUT_DIR, exist_ok=True)


class CalibData:
    """Holds calibration parameters for the tactile sensor (Taxim)."""
    def __init__(self, dataPath):
        self.dataPath = dataPath
        data = np.load(dataPath)

        self.numBins = data['bins']
        self.grad_r = data['grad_r']
        self.grad_g = data['grad_g']
        self.grad_b = data['grad_b']


class depthTactileSimulator(object):
    """
    Simulates optical tactile sensor responses (here, GelSight mini) based on depth maps.
    Calculates shading, shadows, and elastomer deformation.
    Note:
        Adapted from the Taxim framework (https://github.com/Robo-Touch/Taxim).
    """
    def __init__(self, data_folder, gelpad_model_path):
        # polytable
        calib_data = osp.join(data_folder, "polycalib_tacex.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack_tacex.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()

        #shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = np.load(osp.join(data_folder, "shadowTable_tacex.npz"),allow_pickle=True)
        self.direction = shadowData['shadowDirections']
        self.shadowTable = shadowData['shadowTable']

        self.gelpad_model_path = gelpad_model_path

        self.gel_map = np.load(self.gelpad_model_path)
        self.gel_map_blur = cv2.GaussianBlur(self.gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

    def processInitialFrame(self):
        """
        Smooth the initial frame
        """
        # gaussian filtering with square kernel with
        # filterSize : kscale*2+1
        # sigma      : kscale
        kscale = pr.kscale

        img_d = self.f0.astype('float')
        convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

        f0 = self.f0.copy()
        for ch in range(img_d.shape[2]):
            f0[:,:, ch] = convEachDim(img_d[:,:,ch])

        frame_ = img_d

        # Checking the difference between original and filtered image
        diff_threshold = pr.diffThreshold
        dI = np.mean(f0-frame_, axis=2)
        idx =  np.nonzero(dI<diff_threshold)

        # Mixing image based on the difference between original and filtered image
        frame_mixing_per = pr.frameMixingPercentage
        h,w,ch = f0.shape
        pixcount = h*w

        for ch in range(f0.shape[2]):
            f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]

        return f0
    
    def padding(self, img):
        return np.pad(img, ((1,1), (1,1)), 'symmetric')
    
    def simulating(self, heightMap, contact_mask, contact_height, shadow=False):
        """
        Simulate the tactile image from the height map
        heightMap: heightMap of the contact
        contact_mask: indicate the contact area
        contact_height: the height of each pix
        shadow: whether add the shadow

        return:
        sim_img: simulated tactile image w/o shadow
        shadow_sim_img: simluated tactile image w/ shadow
        """

        # generate gradients of the height map
        grad_mag, grad_dir = self.generate_normals(heightMap)

        # generate raw simulated image without background
        sim_img_r = np.zeros((psp.h,psp.w,3))
        bins = psp.numBins

        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T
        binm = bins - 1

        # discritize grids
        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]

        idx_x = np.floor(grad_mag/x_binr).astype('int')
        idx_y = np.floor((grad_dir+np.pi)/y_binr).astype('int')

        # look up polynomial table and assign intensity
        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = np.sum(A * params_r,axis = 1)
        est_g = np.sum(A * params_g,axis = 1)
        est_b = np.sum(A * params_b,axis = 1)

        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))

        # attach background to simulated image
        sim_img = sim_img_r + self.bg_proc

        if not shadow:
            return sim_img, sim_img

        # add shadow
        cx = psp.w//2
        cy = psp.h//2

        # find shadow attachment area
        kernel = np.ones((5, 5), np.uint8)
        dialate_mask = cv2.dilate(np.float32(contact_mask),kernel,iterations = 2)
        enlarged_mask = dialate_mask == 1
        boundary_contact_mask = 1*enlarged_mask - 1*contact_mask
        contact_mask = boundary_contact_mask == 1

        # (x,y) coordinates of all pixels to attach shadow
        x_coord = xx[contact_mask]
        y_coord = yy[contact_mask]

        # get normal index to shadow table
        normMap = grad_dir[contact_mask] + np.pi
        norm_idx = normMap // pr.discritize_precision
        # get height index to shadow table
        contact_map = contact_height[contact_mask]
        height_idx = (contact_map * psp.pixmm - self.shadow_depth[0]) // pr.height_precision
        if height_idx.size == 0:
            return None, None
        # height_idx_max = int(np.max(height_idx))
        total_height_idx = self.shadowTable.shape[2]

        shadowSim = np.zeros((psp.h,psp.w,3))

        # all 3 channels
        for c in range(3):
            frame = sim_img_r[:,:,c].copy()
            # frame_back = sim_img_r[:,:,c].copy()
            for i in range(len(x_coord)):
                # get the coordinates (x,y) of a certain pixel
                cy_origin = y_coord[i]
                cx_origin = x_coord[i]
                # get the normal of the pixel
                n = int(norm_idx[i])
                # get height of the pixel
                h = int(height_idx[i]) + 6
                if h < 0 or h >= total_height_idx:
                    continue
                # get the shadow list for the pixel
                v = self.shadowTable[c,n,h]

                # number of steps
                num_step = len(v)

                # get the shadow direction
                theta = self.direction[n]
                d_theta = theta
                ct = np.cos(d_theta)
                st = np.sin(d_theta)
                # use a fan of angles around the direction
                theta_list = np.arange(d_theta-pr.fan_angle, d_theta+pr.fan_angle, pr.fan_precision)
                ct_list = np.cos(theta_list)
                st_list = np.sin(theta_list)
                for theta_idx in range(len(theta_list)):
                    ct = ct_list[theta_idx]
                    st = st_list[theta_idx]

                    for s in range(1,num_step):
                        cur_x = int(cx_origin + pr.shadow_step * s * ct)
                        cur_y = int(cy_origin + pr.shadow_step * s * st)
                        # check boundary of the image and height's difference
                        if cur_x >= 0 and cur_x < psp.w and cur_y >= 0 and cur_y < psp.h and heightMap[cy_origin,cx_origin] > heightMap[cur_y,cur_x]:
                            frame[cur_y,cur_x] = np.minimum(frame[cur_y,cur_x],v[s])

            shadowSim[:,:,c] = frame
            shadowSim[:,:,c] = ndimage.gaussian_filter(shadowSim[:,:,c], sigma=(pr.sigma, pr.sigma), order=0)

        shadow_sim_img = shadowSim+ self.bg_proc
        shadow_sim_img = cv2.GaussianBlur(shadow_sim_img.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        return sim_img, shadow_sim_img
    
    def generateHeightMapFromHeightInput(self, objHeight, pressing_height_mm):
        """Calculates contact by intersecting object geometry with the gel surface."""
        gel_map = self.gel_map_blur.copy()

        heightMap = objHeight.copy()/psp.pixmm
        # print("heightMap min/max:", heightMap.min(), heightMap.max())
        # print("gel_map min/max:", gel_map.min(), gel_map.max())
 
        max_g = np.max(gel_map)
        # min_g = np.min(gel_map)
        max_o = np.max(heightMap)
        pressing_height_pix = pressing_height_mm/psp.pixmm
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)
        contact_mask = heightMap > gel_map

        zq = np.zeros((psp.h, psp.w))
        zq[contact_mask]  = heightMap[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]  
        return zq, gel_map, contact_mask    

    def deformApprox(self, pressing_height_mm, height_map, gel_map, contact_mask):
        """
        Approximates elastomer deformation using a multi-scale Gaussian pyramid.
        Simulates how the gel flows around the object during indentation.
        """
        zq = height_map.copy()
        zq_back = zq.copy()
        pressing_height_pix = pressing_height_mm/psp.pixmm
        # contact mask which is a little smaller than the real contact mask
        mask = (zq-(gel_map)) > pressing_height_pix * pr.contact_scale
        mask = mask & contact_mask

        # approximate soft body deformation with pyramid gaussian_filter
        for i in range(len(pr.pyramid_kernel_size)):
            zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.pyramid_kernel_size[i],pr.pyramid_kernel_size[i]),0)
            zq[mask] = zq_back[mask]
        zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

        contact_height = zq - gel_map

        return zq, mask, contact_height

    def interpolate(self,img):
        """
        fill the zero value holes with interpolation
        """
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        # mask invalid values
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        # get the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='linear', fill_value = 0) # cubic # nearest # linear
        return GD1

    def generate_normals(self, height_map):
        """
        get the gradient (magnitude & direction) map from the height map
        """
        [h,w] = height_map.shape
        top = height_map[0:h-2,1:w-1] # z(x-1,y)
        bot = height_map[2:h,1:w-1] # z(x+1,y)
        left = height_map[1:h-1,0:w-2] # z(x,y-1)
        right = height_map[1:h-1,2:w] # z(x,y+1)
        dzdx = (bot-top)/2.0
        dzdy = (right-left)/2.0

        mag_tan = np.sqrt(dzdx**2 + dzdy**2)
        grad_mag = np.arctan(mag_tan)
        invalid_mask = mag_tan == 0
        valid_mask = ~invalid_mask
        grad_dir = np.zeros((h-2,w-2))
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir
    
    def heightToTactileSimulating(self, objHeight, press_depth, resize_to_sensor=True):
        height = objHeight.copy()
        
        if resize_to_sensor and (height.shape[0] != psp.h or height.shape[1] != psp.w):
            height = cv2.resize(height, (psp.w, psp.h), interpolation=cv2.INTER_LINEAR)
        
        height_map, gel_map, org_contact_mask = self.generateHeightMapFromHeightInput(
            height, press_depth
        )
        heightMap, contact_mask, contact_height = self.deformApprox(
            press_depth, height_map, gel_map, org_contact_mask
        )

        ratio = contact_mask.mean()
        if ratio < 0.05:
            print("contact ratio too small:", ratio)
            return None, None, None, None
        sim_img, shadow_sim_img = self.simulating(heightMap,
                                                  contact_mask,
                                                  contact_height,
                                                  shadow=True)
        return sim_img, shadow_sim_img, contact_mask, org_contact_mask


# --- MuJoCo Scene Utilities ---
def ensure_inertial_for_all_joint_bodies(root,
                                         mass="0.1",
                                         diaginertia="1e-4 1e-4 1e-4", pos="0 0 0", quat="1 0 0 0"):
    for body in root.findall(".//body"):
        has_joint = body.find("joint") is not None
        if not has_joint:
            continue

        inertial = body.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(body, "inertial")

        inertial.set("pos", inertial.get("pos", "0 0 0"))
        inertial.set("quat", inertial.get("quat", "1 0 0 0"))
        inertial.set("mass", inertial.get("mass", mass))
        inertial.set("diaginertia", inertial.get("diaginertia", diaginertia))

def ensure_inertial_for_body(worldbody, body_name="model",
                             mass="0.1", diaginertia="1e-4 1e-4 1e-4", pos="0 0 0", quat="1 0 0 0"):
    body = worldbody.find(f".//body[@name='{body_name}']")
    if body is None:
        body = worldbody.find("body")
    if body is None:
        return

    inertial = body.find("inertial")
    if inertial is None:
        inertial = ET.SubElement(body, "inertial")
    inertial.set("pos", pos)
    inertial.set("quat", quat)
    inertial.set("mass", mass)
    inertial.set("diaginertia", diaginertia)

def make_collision_geoms_massless(root, mesh_prefix="model_collision_"):
    for geom in root.findall(".//geom"):
        mesh_name = geom.get("mesh")
        if mesh_name and mesh_name.startswith(mesh_prefix):
            geom.set("density", "0")
            geom.set("mass", "0")

def disable_inertia_from_geom(root):
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("inertiafromgeom", "false")

def add_camera_to_xml_string(xml_path, cam_name, init_pos="0 0 0.03", init_fovy="30"):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    base_dir = os.path.dirname(os.path.abspath(xml_path))
    # print(base_dir)

    for mesh in root.findall(".//model_collision_"):
        f = mesh.get("file")
        print(f)
        if f and not os.path.isabs(f):
            mesh.set("file", os.path.join(base_dir, f))
    for tex in root.findall(".//texture"):
        f = tex.get("file")
        if f and not os.path.isabs(f):
            tex.set("file", os.path.join(base_dir, f))

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"{xml_path} no <worldbody> tag found.")
    cam = ET.SubElement(worldbody, "camera")
    cam.set("name", cam_name)
    cam.set("pos", init_pos)
    cam.set("fovy", init_fovy)

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    map_tag = visual.find("map")
    if map_tag is None:
        map_tag = ET.SubElement(visual, "map")
    map_tag.set("znear", "0.0005")
    map_tag.set("zfar", "0.05")

    worldbody = root.find("worldbody")

    disable_inertia_from_geom(root)
    ensure_inertial_for_all_joint_bodies(root, mass="0.1", diaginertia="1e-4 1e-4 1e-4")

    make_collision_geoms_massless(root, mesh_prefix="model_collision_")
    xml_str = ET.tostring(root, encoding="unicode")
    return xml_str


def load_mesh(obj_path):
    mesh = trimesh.load(obj_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    return mesh

def sample_surface_point_and_normal(mesh, mesh_center):
    points, face_indices = trimesh.sample.sample_surface(mesh, 1)
    p = points[0]
    n = mesh.face_normals[face_indices[0]]
    v = p - mesh_center
    if np.dot(v, n) < 0:
        n = -n

    n = n / np.linalg.norm(n)
    return p, n

def compute_fovy_from_patch_height_and_distance(patch_h, d):
    fovy_rad = 2.0 * math.atan(patch_h / (2.0 * d))
    return math.degrees(fovy_rad)

def build_camera_rotation_from_normal(n):
    n = n / np.linalg.norm(n)
    z_axis = n

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(up, z_axis)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rot = np.stack([x_axis, y_axis, z_axis], axis=1)   # 3x3

    quat = np.zeros(4, dtype=np.float64)

    # rot_flat = rot.T.reshape(9).astype(np.float64)

    mujoco.mju_mat2Quat(quat, rot.reshape(9))
    return quat


def quick_obj_ratio(model, data, scene, con, opt, cam_view,
                    cam_id, cam_pos, cam_quat, fovy_deg,
                    ratio = 4):
    w = IMG_WIDTH // ratio
    h = IMG_HEIGHT // ratio
    model.cam_pos[cam_id] = cam_pos
    model.cam_quat[cam_id] = cam_quat
    model.cam_fovy[cam_id] = fovy_deg

    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(model, data, opt, None, cam_view,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth_raw = np.zeros((h, w), dtype=np.float32)

    viewport = mujoco.MjrRect(0, 0, w, h)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
    mujoco.mjr_render(viewport, scene, con)
    mujoco.mjr_readPixels(rgb, depth_raw, viewport, con)

    depth_raw = np.flipud(depth_raw)

    obj_mask = depth_raw < 0.9999
    return obj_mask.mean()



def render_patch_at_random_surface_point(model, data, scene, con, opt, cam_view,
                                         mesh, cam_id,
                                         img_width, img_height,
                                         patch_w, patch_h, d):
 
    mesh_center = mesh.centroid
    p_local, n_local = sample_surface_point_and_normal(mesh, mesh_center)

    p_world = p_local
    n_world = n_local

    cam_pos = p_world + d * n_world
    cam_quat = build_camera_rotation_from_normal(n_world)
    fovy_deg = compute_fovy_from_patch_height_and_distance(patch_h, d)

    obj_ratio = quick_obj_ratio(model, data, scene, con, opt, cam_view,
                                cam_id, cam_pos, cam_quat, fovy_deg)
    
    if obj_ratio < 0.6:
        # print("ratio:", ratio)
        return None, None, None, None, None, None, None, None, None

    model.cam_pos[cam_id] = cam_pos
    model.cam_quat[cam_id] = cam_quat
    model.cam_fovy[cam_id] = fovy_deg

    RENDER_H = img_height * RENDER_RATIO
    RENDER_W = img_width * RENDER_RATIO

    mujoco.mj_forward(model, data)
    mujoco.mjv_updateScene(model, data, opt, None, cam_view,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)

    rgb_hi = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
    depth_raw_hi = np.zeros((RENDER_H, RENDER_W), dtype=np.float32)


    viewport = mujoco.MjrRect(0, 0, RENDER_W, RENDER_H)
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, con)
    mujoco.mjr_render(viewport, scene, con)
    mujoco.mjr_readPixels(rgb_hi, depth_raw_hi, viewport, con)
    rgb_hi = np.flipud(rgb_hi)
    depth_raw_hi = np.flipud(depth_raw_hi)
    
    rgb = cv2.resize(rgb_hi, (img_width, img_height), interpolation=cv2.INTER_AREA)

    znear = model.vis.map.znear
    zfar = model.vis.map.zfar
    depth_linear_hi = (znear / (1.0 - depth_raw_hi * (1.0 - znear / zfar))) * 1000.0 
    depth_linear = depth_linear_hi.reshape(img_height, RENDER_RATIO, img_width, RENDER_RATIO).min(axis=(1,3))


    depth_valid_mask = (depth_linear > znear * 1000.0 + 0.5) & (depth_linear < zfar * 1000.0 - 0.5) 

    if np.any(depth_valid_mask):
        d0 = np.percentile(depth_linear[depth_valid_mask], 1.0)  
        depth_valid_mask = depth_valid_mask & (depth_linear <= d0 + 2.0)
        ratio = depth_valid_mask.mean()
    else:
        ratio = 0.0

    if ratio < 0.6:
        # print("ratio:", ratio)
        return None, None, None, None, None, None, None, None, None
    # print("rgb min/max/mean:", rgb.min(), rgb.max(), rgb.mean())
    if rgb.mean() <= 1e-1:
        return None, None, None, None, None, None, None, None, None
    backgroundMask = ~(depth_valid_mask) & (depth_linear > (depth_linear.min() + .0))
    return rgb, depth_linear, cam_pos, cam_quat, p_world, n_world, fovy_deg, depth_valid_mask, backgroundMask


def process_one_object(sim, obj_dir):
    
    obj_path = os.path.join(obj_dir, "model.obj")
    xml_path = os.path.join(obj_dir, "model.xml")

    if not (os.path.isfile(obj_path) and os.path.isfile(xml_path)):
        print(f"Skip {obj_dir}, missing model.obj or model.xml")
        return

    mesh = load_mesh(obj_path)

    xml_str = add_camera_to_xml_string(xml_path, CAM_NAME)

    tmp_xml_path = os.path.join(obj_dir, "model_with_cam.xml")
    with open(tmp_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    model = mujoco.MjModel.from_xml_path(tmp_xml_path)
    data = mujoco.MjData(model)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera '{CAM_NAME}' not found in model.")
    

    RENDER_H = IMG_HEIGHT * RENDER_RATIO
    RENDER_W = IMG_WIDTH * RENDER_RATIO

    model.vis.global_.offwidth = RENDER_W
    model.vis.global_.offheight = RENDER_H
    model.vis.quality.offsamples = 8
    model.vis.quality.numslices = 40
    model.vis.quality.numstacks = 24
    model.vis.quality.numquads = 8

    gl_ctx = mujoco.GLContext(RENDER_W, RENDER_H) 
    gl_ctx.make_current()

    scene = mujoco.MjvScene(model, maxgeom=10000)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    cam_view = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam_view)
    cam_view.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam_view.fixedcamid = cam_id

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    mujoco.mj_forward(model, data)
    name = os.path.basename(obj_dir.rstrip(os.sep))
    out_dir_obj = os.path.join(OUT_DIR, name)
    os.makedirs(out_dir_obj, exist_ok=True)

    generateCount = 0
   
    while generateCount < NUM_PATCHES_PER_OBJ:
        rgb = None 
        generateCount += 1
        count = 0
        
        while rgb is None:
            if count > 50:
                print(f"Cannot generate enough patches for {name}, skip")
                return
            if (count+1) % 10 == 0:
                print(f"[{name}] patch {generateCount}: {count+1} tries...")
            count += 1
            rgb, depth_linear, cam_pos, cam_quat, p_world, n_world, fovy_deg, valid, backgroundMask = \
                render_patch_at_random_surface_point(
                    model, data, scene, con, opt, cam_view,
                    mesh, cam_id,
                    IMG_WIDTH, IMG_HEIGHT,
                    PATCH_W, PATCH_H, D
                )

        depth = depth_linear.copy() # mm
        depth_offset = depth[valid].min()
        depth = depth - depth_offset
        height_base = depth[valid].max()
        depth[~valid] = height_base
        objHeight = (height_base - depth) 
        # print(objHeight.min(), objHeight.max())
        

        print(f"[{name}] patch {generateCount}: fovy={fovy_deg:.2f} deg, "
              f"cam_pos={cam_pos}, center={p_world}")
        
        N_PRESS = 5
        MAX_TRY_PER_PRESS = 10

        allOK = True

        sim_vis_list = [None]*5
        shadow_sim_vis_list = [None]*5
        contact_mask_list = [None]*5
        org_contact_mask_list = [None]*5
        pressing_height_mm_list = [None]*5

        for press_idx in range(N_PRESS):
            sim_img = None
            shadow_sim_img = None
            contact_mask = None
            org_contact_mask = None

            for attempt in range(MAX_TRY_PER_PRESS):
                pressing_height_mm = round(np.random.uniform(0.01, 1.5), 3)

                while pressing_height_mm in pressing_height_mm_list:
                    pressing_height_mm = round(np.random.uniform(0.01, 1.5), 3)
                sim_img, shadow_sim_img, contact_mask, org_contact_mask = sim.heightToTactileSimulating(
                    objHeight,
                    pressing_height_mm,
                    resize_to_sensor=True
                )

                if sim_img is not None:
                    break 
            
            if sim_img is None:
                allOK = False
                break

            sim_vis = sim_img.astype(np.uint8)
            # sim_vis = cv2.cvtColor(sim_vis, cv2.COLOR_RGB2BGR)
            shadow_vis = shadow_sim_img.astype(np.uint8)
            # shadow_vis = cv2.cvtColor(shadow_vis, cv2.COLOR_RGB2BGR)
            sim_vis_list[press_idx] = sim_vis
            shadow_sim_vis_list[press_idx] = shadow_vis
            contact_mask_list[press_idx] = contact_mask
            org_contact_mask_list[press_idx] = org_contact_mask
            pressing_height_mm_list[press_idx] = pressing_height_mm

        if not allOK:
            generateCount -= 1
            continue

        rgb_path = os.path.join(out_dir_obj, f"{generateCount:03d}_rgb.jpg")
        imwrite(rgb_path, rgb)
    

        for press_idx in range(N_PRESS):
            pressing_height_mm = pressing_height_mm_list[press_idx]
            sim_vis = sim_vis_list[press_idx]
            shadow_vis = shadow_sim_vis_list[press_idx]
            tact_path = os.path.join(out_dir_obj, f"{generateCount:03d}_tact_pressDepth_{pressing_height_mm}mm.jpg")
            tact_shadow_path = os.path.join(out_dir_obj, f"{generateCount:03d}_tact_shadow_pressDepth_{pressing_height_mm}mm.jpg")
            cv2.imwrite(tact_path, sim_vis.astype(np.uint8))
            cv2.imwrite(tact_shadow_path, shadow_vis.astype(np.uint8))

            contact_mask = contact_mask_list[press_idx]
            org_contact_mask = org_contact_mask_list[press_idx]
            # contact_mask_path = os.path.join(out_dir_obj, f"{generateCount:03d}_contact_mask_pressDepth_{pressing_height_mm}mm.png")

            # org_contact_mask_path = os.path.join(out_dir_obj, f"{generateCount:03d}_org_contact_mask_pressDepth_{pressing_height_mm}mm.png")
            contact_npz_path = os.path.join(out_dir_obj, f"{generateCount:03d}_contact_masks_{pressing_height_mm}mm.npz")
            np.savez_compressed(
                contact_npz_path,
                contact_mask=contact_mask.astype(np.uint8),
                org_contact_mask=org_contact_mask.astype(np.uint8),
            )

        # backgroundMask_path = os.path.join(out_dir_obj, f"{generateCount:03d}_background_mask.png")
        background_mask = np.zeros_like(contact_mask, dtype=np.uint8)
        background_mask[backgroundMask] = 1
        
        # valid_mask_path = os.path.join(out_dir_obj, f"{generateCount:03d}_valid_mask.png")
        valid_mask = np.zeros_like(contact_mask, dtype=np.uint8)
        valid_mask[valid] = 1

        patchMask_npz_path = os.path.join(out_dir_obj, f"{generateCount:03d}_patch_masks.npz")
        np.savez_compressed(
            patchMask_npz_path,
            background_mask=background_mask,
            valid_mask=valid_mask,
        )


def main():
    sim = depthTactileSimulator(DATA_FOLDER, GELPAD_MODEL_PATH)
    flag = 0
    for entry in sorted(os.listdir(BASE_DIR)):
        flag += 1
        obj_dir = os.path.join(BASE_DIR, entry)
        if not os.path.isdir(obj_dir):
            continue
        print(f"Processing {obj_dir}")        
        process_one_object(sim, obj_dir)

        
if __name__ == "__main__":
    main()
