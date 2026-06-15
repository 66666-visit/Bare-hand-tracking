import os
import argparse
import torch
import numpy as np
import cv2
import math
from ultralytics import YOLO 
from models.resnet import resnet50
from utils.common_utils import *
from hand_data_iter.datasets import draw_bd_handpose

# ==========================================
# 【全新核心算子】1 Euro Filter (一欧元滤波器)
# 专为动态人机交互设计的低延迟、自适应平滑器
# ==========================================
import math
import numpy as np

# ==========================================
# 【核心算子】1 Euro Filter (一欧元滤波器)
# 物理本质：一个基于瞬时速度进行自适应截止频率调节的一阶低通滤波器
# 核心优势：纯 O(1) 标量运算，完美解决“低速抗抖”与“高速跟手”的矛盾
# ==========================================
class OneEuroFilter:
    def __init__(self, mincutoff=0.05, beta=0.1, dcutoff=1.0, freq=30):
        """
        初始化滤波器参数
        :param mincutoff: 最小截止频率 (Hz)。决定了系统在“静止/悬停”时的极低通过率，值越小，滤除高频微颤的能力越强。
        :param beta: 速度增益系数。决定了系统在“高速运动”时截止频率的爬升率，值越大，消除滞后感（跟手性）越好。
        :param dcutoff: 速度滤波器的固定截止频率 (Hz)。用于对“速度”本身进行一次基础平滑，防止速度信号因底噪而突变。
        :param freq: 传感器的采样频率 (fps)。例如摄像头是 30帧/秒。
        """
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        # 状态记忆变量
        self.x_prev = None  # 上一帧的平滑位置 (对应状态更新方程)
        self.dx_prev = None # 上一帧的平滑速度

    def alpha(self, cutoff):
        """
        核心映射：将物理世界的“截止频率 (Hz)”转化为离散 EMA 公式的“平滑权重 alpha (0~1)”
        数学推导：利用了一阶 RC 低通滤波器的差分方程离散化
        """
        te = 1.0 / self.freq                     # Te: 采样周期 (每一帧经过的时间，如 0.033s)
        tau = 1.0 / (2 * math.pi * cutoff)       # Tau: 时间常数 (与截止频率成反比，频率越小，系统越迟钝)
        return 1.0 / (1.0 + tau / te)            # 返回平滑权重 a。cutoff 越小，a 越趋近于 0（极其平滑）

    def __call__(self, x):
        """
        执行单帧滤波更新
        :param x: 当前帧传入的原始带噪坐标 (Numpy Array，如 [100, 200])
        :return: 当前帧平滑后的最优坐标
        """
        x = np.array(x, dtype=np.float32)
        
        # 1. 初始帧判定：如果没有历史记忆，直接信任当前观测值
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            return x

        # ==========================================
        # 第一阶段：提取并净化“速度信号”
        # ==========================================
        # 计算原始一阶差分速度 (距离 / 时间)
        dx = (x - self.x_prev) * self.freq
        
        # 对速度信号本身进行一次静态 EMA 低通滤波 (防止 YOLO 高频底噪导致速度误判)
        edx = self.alpha(self.dcutoff) * dx + (1.0 - self.alpha(self.dcutoff)) * self.dx_prev
        self.dx_prev = edx

        # ==========================================
        # 第二阶段：自适应魔法 (速度驱动截止频率)
        # ==========================================
        # 计算空间 2 维向量的 L2 范数 (即提取标量物理速率)，保证各向同性，防止轨迹变形
        speed = np.linalg.norm(edx) 
        
        # 线性映射模型：fc = f_min + beta * |v|
        # 速度越快，截止频率越高，大门敞开；速度越慢，频率越低，死死锁住
        cutoff = self.mincutoff + self.beta * speed

        # ==========================================
        # 第三阶段：执行最终位置平滑
        # ==========================================
        # 根据动态截止频率，计算出这一帧专属的平滑权重 a
        a = self.alpha(cutoff)
        
        # 执行标准的 EMA 更新：最优位置 = a * 观测值 + (1-a) * 历史预测
        x_hat = a * x + (1.0 - a) * self.x_prev
        
        # 更新记忆留给下一帧
        self.x_prev = x_hat
        
        return x_hat
    
    def reset(self):
        """
        状态机重置接口：当目标脱离视野或追踪丢失时调用。
        清空历史记忆，防止下一次目标重新出现时产生“跨越全屏幕的橡胶拉扯飞线”。
        """
        self.x_prev = None
        self.dx_prev = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default = './resnet_50-size-256-loss-0.0642.pth')
    parser.add_argument('--num_classes', type=int , default = 42) 
    parser.add_argument('--GPUS', type=str, default = '0')
    ops = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- 状态机配置 ---
    active_counter = 0
    prev_active_counter = 0  
    STABLE_FRAMES_THRESHOLD = 15  
    MOVE_THRESHOLD = 8  # 【修复1】将 50 改为 8，让系统对“重新移动重新取景”极其敏感
          
    # 实例化 1 Euro 滤波器
    # 调参指南：如果觉得悬停时还在抖，把 mincutoff 调小(如 0.01)；如果拖拽感觉慢半拍，把 beta 调大(如 0.5)
    tip1_filter = OneEuroFilter(mincutoff=0.01, beta=0.1)   
    tip2_filter = OneEuroFilter(mincutoff=0.01, beta=0.1)   
    box_filter = OneEuroFilter(mincutoff=0.05, beta=0.1)    
    
    tracker = None     
    tracking_active = False  
    show_clahe_vision = False   
    
    model_pose = resnet50(num_classes = ops.num_classes, img_size=256).to(device)
    model_pose.eval()
    if os.access(ops.model_path, os.F_OK):
        model_pose.load_state_dict(torch.load(ops.model_path, map_location=device))
    
    detector_hand = YOLO('hand_yolov8n.pt').to('cuda') 
    
    cap = cv2.VideoCapture(0)
    # 建议使用 1080P 或 720P
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    with torch.no_grad():
        while True:
            ret, img = cap.read()
            if not ret: break
            
            img = cv2.flip(img, 1)  
            h, w, _ = img.shape 
            
            need_enhancement = tracking_active or show_clahe_vision or (active_counter >= STABLE_FRAMES_THRESHOLD - 1)
            
            if need_enhancement:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)   
                l_enhanced = clahe.apply(l)
                track_img = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
            else:
                track_img = img 
                
            display_img = track_img.copy() if show_clahe_vision else img.copy()

            if show_clahe_vision:
                cv2.putText(display_img, "AI MICROSCOPE VISION: ON", (20, 40), 
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

            raw_tips = []

            # ==========================================
            # 模块 1：后台物理追踪器 (CSRT)
            # ==========================================
            if tracking_active and tracker is not None:
                success, bbox = tracker.update(track_img)    
                if success:
                    tx, ty, tw, th = [int(v) for v in bbox]
                    cv2.rectangle(display_img, (tx, ty), (tx+tw, ty+th), (0, 255, 255), 3)   
                    cv2.rectangle(display_img, (tx, ty-30), (tx+120, ty), (0, 255, 255), -1)
                    cv2.putText(display_img, "LOCKED", (tx+10, ty-8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
                else:
                    cv2.putText(display_img, "TRACKING LOST", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
                    tracking_active = False  
                    tracker = None           
                    active_counter = 0       

            # ==========================================
            # 模块 2：双手 YOLO 检测与姿态估计
            # ==========================================
            results_hand = detector_hand(img, verbose=False, conf=0.45, imgsz=640)
            
            for r in results_hand:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    p = 25
                    x1_p, y1_p, x2_p, y2_p = max(0, x1-p), max(0, y1-p), min(w, x2+p), min(h, y2+p)
                    
                    img_crop = img[y1_p:y2_p, x1_p:x2_p]
                    if img_crop.size == 0: continue
                    ch, cw = img_crop.shape[:2]
                    
                    img_resize = cv2.resize(img_crop, (256, 256))
                    img_input = (img_resize.astype(np.float32) - 128.) / 256.
                    img_input = img_input.transpose(2, 0, 1)
                    img_tensor = torch.from_numpy(img_input).unsqueeze_(0).to(device)

                    output = model_pose(img_tensor.float()).cpu().detach().numpy().squeeze()
                    
                    pts_hand = {}
                    for i in range(21):
                        px, py = (output[i*2] * float(cw)) + x1_p, (output[i*2+1] * float(ch)) + y1_p
                        pts_hand[str(i)] = {"x": px, "y": py}
                    
                    draw_bd_handpose(display_img, pts_hand, 0, 0)
                    index_x = pts_hand['8']['x']
                    index_y = pts_hand['8']['y']
                    
                    cv2.circle(display_img, (int(index_x), int(index_y)), 6, (255, 0, 255), -1)
                    cv2.circle(display_img, (int(index_x), int(index_y)), 2, (255, 255, 255), -1)
                    
                    raw_tips.append([index_x, index_y])

            # ==========================================
            # 模块 3：1 Euro 滤波平滑与状态机逻辑
            # ==========================================
            if len(raw_tips) >= 2: 
                raw_tips = sorted(raw_tips, key=lambda tip: tip[0]) 

                # 【修复2】截断保护：无论背景有多乱，强行只认最左边的两只手，彻底消灭维度报错崩溃
                raw_tips = raw_tips[:2]

                # 调用 1 Euro Filter 平滑指尖坐标
                smoothed_tip1 = tip1_filter(raw_tips[0])
                smoothed_tip2 = tip2_filter(raw_tips[1])
                current_smoothed_tips = [smoothed_tip1, smoothed_tip2]

                is_stable = False
                if prev_smoothed_tips is not None:
                    d1 = math.sqrt((current_smoothed_tips[0][0]-prev_smoothed_tips[0][0])**2 + (current_smoothed_tips[0][1]-prev_smoothed_tips[0][1])**2)
                    d2 = math.sqrt((current_smoothed_tips[1][0]-prev_smoothed_tips[1][0])**2 + (current_smoothed_tips[1][1]-prev_smoothed_tips[1][1])**2)
                    
                    if d1 < MOVE_THRESHOLD and d2 < MOVE_THRESHOLD:
                        is_stable = True

                prev_active_counter = active_counter 
                
                if is_stable:
                    active_counter += 1
                else:
                    # 【修复3】手势重定向逻辑：只要手重新剧烈滑动，立即扣减蓄力并杀掉旧追踪器
                    active_counter = max(0, active_counter - 2) 
                    
                    if active_counter < STABLE_FRAMES_THRESHOLD and tracking_active:
                        print("🛑 检测到手势重新移动，主动释放旧目标，准备重新取景...")
                        tracking_active = False
                        tracker = None
                
                prev_smoothed_tips = current_smoothed_tips
                p1, p2 = current_smoothed_tips[0], current_smoothed_tips[1]
                mid_point = (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2))

                # --- 充能态 ---
                if active_counter < STABLE_FRAMES_THRESHOLD:
                    angle = int((active_counter / STABLE_FRAMES_THRESHOLD) * 360)  
                    cv2.circle(display_img, mid_point, 30, (100, 100, 100), 2)
                    cv2.ellipse(display_img, mid_point, (30, 30), -90, 0, angle, (255, 255, 0), 4)
                    cv2.putText(display_img, "FOCUSING", (mid_point[0]-35, mid_point[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # --- 激活态 (绘制旋转框) ---
                else:
                    active_counter = STABLE_FRAMES_THRESHOLD
                    
                    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                    width = math.sqrt(dx**2 + dy**2)
                    rot_angle = math.degrees(math.atan2(dy, dx))
                    
                    rect = ((mid_point[0], mid_point[1]), (width, width*0.7), rot_angle)
                    raw_box_pts = np.float32(cv2.boxPoints(rect))
                    
                    # 使用 1 Euro 平滑 8 个顶点组成的矩阵
                    smoothed_box_pts = box_filter(raw_box_pts)
                    box_pts = np.int64(smoothed_box_pts)

                    cv2.drawContours(display_img, [box_pts], 0, (0, 255, 0), 2)
                    
                    if prev_active_counter < STABLE_FRAMES_THRESHOLD:
                        rx, ry, rw, rh = cv2.boundingRect(box_pts)
                        rx, ry = max(0, rx), max(0, ry)
                        rw, rh = min(w-rx, rw), min(h-ry, rh)
                        
                        if rw > 10 and rh > 10: 
                            print("🔥 目标已锁定！提取高对比度物理指纹中...")
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(track_img, (rx, ry, rw, rh))
                            tracking_active = True
                                                
            else:
                active_counter = 0
                prev_active_counter = 0
                prev_smoothed_tips = None
                
                # 手退出屏幕时，清零平滑器记忆
                tip1_filter.reset()
                tip2_filter.reset()
                box_filter.reset()

            cv2.imshow('Zero-shot Visual Anchor', display_img)
            
            key = cv2.waitKey(1)
            if key == 27: # ESC 退出
                break
            elif key == ord('c') or key == ord('C'): 
                show_clahe_vision = not show_clahe_vision

    cap.release()
    cv2.destroyAllWindows()