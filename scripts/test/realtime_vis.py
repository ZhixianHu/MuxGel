'''
python scripts/test/realtime_resNet34Unet_dualInput.py \
  --ckpt outputs/sim-tacvis-inpaint/6Ch_CorrelatedJitter_difference_recon/best_checkpoint.pth \
  --tact_ref_path path/to/sensor_1_ref.jpg


python scripts/test/realtime_resNet34Unet_dualInput.py \
  --ckpt outputs/real-tacvis-inpaint/6Ch_CorrelatedJitter_difference_recon/best_checkpoint.pth \
  --tact_ref_path calibration/calibration_data/black_ball/sensor_1/black_ball_1_ref.jpg \
  --cam_ref_path calibration/calibration_data/black_ball/sensor_8/black_ball_8_ref.jpg 
'''


import os
import time
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.train_real_di_rest import DualResNet34UNet
from external.gsmini import gs3drecon

def capture_frame(cap, target_size=(320, 240)):
    ret, frame = cap.read()
    count = 0
    while not ret and count < 5:
        print("Failed to capture frame. Retrying...")
        time.sleep(0.5)
        ret, frame = cap.read()
        count += 1
        
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return False, None
    frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    return True, frame_resized

def tensor_to_cv2(tensor):
    img = tensor.detach().cpu().clamp(0, 1).squeeze(0) 
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img

def diff_to_cv2(tensor):
    img = tensor.detach().cpu().clamp(-1, 1).squeeze(0)
    img = (img + 1.0) / 2.0  # 映射到 [0, 1]
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img

def main(args):
    nn = gs3drecon.Reconstruction3D(320, 240)
    maskMarkersFlag = False
    netPath = "external/gsmini/nnmini.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Loading model...")
    model = DualResNet34UNet(n_channels=6, n_classes_tact=3, n_classes_vis=3).to(device)
    
    _ = nn.load_nn(netPath, device)
    vis3d = gs3drecon.Visualize3D(320, 240, "")

    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found at {args.ckpt}")
        
    checkpoint = torch.load(args.ckpt, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("Model loaded successfully!")


    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera ID {args.cam_id}")

    target_size = (args.width, args.height)
    to_tensor = T.ToTensor()

    if not os.path.exists(args.tact_ref_path):
        raise FileNotFoundError(f"Pure Tactile Reference not found: {args.tact_ref_path}")
    print(f"Loading Pure Tactile Reference from: {args.tact_ref_path}")

    tact_ref_cv = cv2.imread(args.tact_ref_path, cv2.IMREAD_COLOR)
    tact_ref_cv = cv2.resize(tact_ref_cv, target_size, interpolation=cv2.INTER_LINEAR)
    tact_ref_tensor = to_tensor(tact_ref_cv).unsqueeze(0).to(device).float()

    cam_ref_cv = None
    if args.cam_ref_path and os.path.exists(args.cam_ref_path):
        print(f"Loading Camera Grid Reference from: {args.cam_ref_path}")
        cam_ref_cv = cv2.imread(args.cam_ref_path, cv2.IMREAD_COLOR)
        cam_ref_cv = cv2.resize(cam_ref_cv, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        print("\n" + "="*50)
        print(">> No camera reference path provided.")
        print(">> Starting preview. Please ensure the grid sensor is UNTOUCHED in dark/ambient environment.")
        print("="*50 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            preview = cv2.resize(frame, (640, 480))
            cv2.imshow("Preview - Press 'Enter' in terminal to capture CAM REF", preview)
            cv2.waitKey(1)
            user_input = input(">> Press [ENTER] to capture grid reference, or 'q' to quit: ")
            
            if user_input.lower() == 'q':
                cap.release()
                cv2.destroyAllWindows()
                return
            
            ret, cam_ref_cv = capture_frame(cap, target_size=target_size)
            if ret:
                print("Camera Grid Reference captured successfully!")
                break

    cv2.destroyAllWindows()
    cam_ref_tensor = to_tensor(cam_ref_cv).unsqueeze(0).to(device).float()

    print("\nStarting real-time inference. Press 'ESC' or 'q' on the image window to stop.")
    
    with torch.no_grad():
        dummy_input = torch.cat([cam_ref_tensor, cam_ref_tensor], dim=1)
        model(dummy_input)

    fps_list = []

    bg_frames = []
    bg_model_gray = None
    CALIB_FRAMES = 30 
    align_status_text = "[Sys] Calibrating BG..."
    align_color = (0, 0, 0)

    while True:
        start_time = time.time()

        ret, live_cv = capture_frame(cap, target_size=target_size)
        if not ret: break
        
        live_tensor = to_tensor(live_cv).unsqueeze(0).to(device).float()

        input_tensor = torch.cat([live_tensor, cam_ref_tensor], dim=1)

        with torch.no_grad(), torch.cuda.amp.autocast():
            tact_diff_pred, vis_pred = model(input_tensor)

            tact_recon_pred = torch.clamp(tact_diff_pred + tact_ref_tensor, 0.0, 1.0)

        vis_pred_cv = tensor_to_cv2(vis_pred)
        tact_recon_cv = tensor_to_cv2(tact_recon_pred)
        tact_diff_cv = diff_to_cv2(tact_diff_pred)
        
        if nn.dm_zero_counter < 50:
            dm = nn.get_depthmap(tact_recon_cv, maskMarkersFlag)
            continue
        # print("Done initialization of gelsight depth")
        dm = nn.get_depthmap(tact_recon_cv, maskMarkersFlag)
        vis3d.update(dm)

        print(dm.min(), dm.max(), dm.mean())



        disp_live = cv2.rotate(live_cv, cv2.ROTATE_90_CLOCKWISE)
        disp_vis_pred = cv2.rotate(vis_pred_cv, cv2.ROTATE_90_CLOCKWISE)
        disp_tact_pred = cv2.rotate(tact_recon_cv, cv2.ROTATE_90_CLOCKWISE)

        disp_vis_gray = cv2.cvtColor(disp_vis_pred, cv2.COLOR_BGR2GRAY)
        img_h, img_w = disp_vis_pred.shape[:2] # h=320, w=240
        center_y = img_h // 2
        deadzone = 10

        if len(bg_frames) < CALIB_FRAMES:
            bg_frames.append(disp_vis_gray)
            if len(bg_frames) == CALIB_FRAMES:
                bg_model_gray = np.median(bg_frames, axis=0).astype(np.uint8)
                align_status_text = "[Track] Ready"
        else:
            diff = cv2.absdiff(bg_model_gray, disp_vis_gray)
            _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY) # 35为差分阈值，可调
            
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(largest_contour) > 400:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # cv2.drawContours(disp_vis_pred, [largest_contour], -1, (0, 255, 0), 2)
                        # cv2.circle(disp_vis_pred, (cX, cY), 5, (0, 0, 255), -1)
                        
                        if cY < center_y - deadzone:
                            align_status_text = "[Align] Object Above ^"
                            align_color = (0, 0, 255) 
                        elif cY > center_y + deadzone:
                            align_status_text = "[Align] Object Below v"
                            align_color = (0, 0, 255) 
                        else:
                            align_status_text = "[Align] Target Centered OK"
                            align_color = (0, 150, 0) 
                else:
                    align_status_text = "[Track] No Object"
                    align_color = (100, 100, 100)
            else:
                align_status_text = "[Track] No Object"
                align_color = (100, 100, 100)



        display_grid = np.hstack((disp_live, disp_vis_pred, disp_tact_pred))
        # W_final, H_final = 1440, 640
        W_final = int(1440 * 0.8)
        H_final = int(640 * 0.8)
        W_panel = int(240 * 2 * 0.8)
        
        display_grid = cv2.resize(display_grid, (W_final, H_final), interpolation=cv2.INTER_NEAREST)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        y_offset = 40

        color_black = (0, 0, 0)
        color_white = (255, 255, 255)

        # cv2.putText(display_grid, "Live Input", (10, y_offset), font, font_scale, color_white, thickness)
        # fps = 1.0 / (time.time() - start_time)
        # cv2.putText(display_grid, f"FPS: {fps:.1f}", (10, y_offset + 40), font, 0.8, color_white, 2)

        # cv2.putText(display_grid, "Vision Reconstruction", (W_panel + 10, y_offset), font, font_scale, color_white, thickness)
        # cv2.putText(display_grid, align_status_text, (W_panel + 10, y_offset + 40), font, 0.8, align_color, 2)
        # cv2.putText(display_grid, "Tactile Reconstruction", (W_panel * 2 + 10, y_offset), font, font_scale, color_white, thickness)

        cv2.imshow("Real-Time Visuo-Tactile Inference", display_grid)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Visuo-Tactile Inference")
    parser.add_argument('--ckpt', type=str, default="outputs/muxgel/real_di_rest_run/best_checkpoint.pth")
    parser.add_argument('--cam_id', type=int, default=1, help="Camera ID")
    

    parser.add_argument('--tact_ref_path', type=str, default="data/calibration_data/black_ball/sensor_1/black_ball_1_ref.jpg", help="Path to pure tactile ref (sensor_1_ref.jpg)")
    parser.add_argument('--cam_ref_path', type=str, default="data/calibration_data/black_ball/sensor_4/black_ball_4_ref.jpg", help="Path to grid camera ref (Optional, will prompt to capture if empty)")
    # parser.add_argument('--cam_ref_path', type=str, default="assets/sensor/tacRef/ref4.jpg", help="Path to grid camera ref (Optional, will prompt to capture if empty)")
    
    # parser.add_argument('--cam_ref_path', type=str)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)

    args = parser.parse_args()
    main(args)
