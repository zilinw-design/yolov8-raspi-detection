import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np
import random
import os
import json
import math
from shapely.geometry import Polygon as ShapelyPolygon

# ==================== 配置区 ====================
OUTPUT_DIR = "dataset_nuedc_2025"
DPI = 300
A4_WIDTH = 8.27   # A4 纸宽度 (英寸) -> 210mm
A4_HEIGHT = 11.69 # A4 纸高度 (英寸) -> 297mm
BORDER_CM = 2.0   # 边缘黑色线宽 2cm
BORDER_INCH = BORDER_CM / 2.54  # 转换为英寸 (约 0.787 英寸)

NUM_SAMPLES_PER_CLASS = 10  # 每类生成的样本数，可调

dataset_annotations = []

def create_a4_canvas():
    """创建一个带有 2cm 黑边和底部标记点的 A4 纸画布"""
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), dpi=DPI, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect='equal')
    ax.set_xlim(0, A4_WIDTH)
    ax.set_ylim(0, A4_HEIGHT)
    ax.axis('off')

    # 1. 绘制 2cm 黑色边框
    # 下边框
    ax.add_patch(patches.Rectangle((0, 0), A4_WIDTH, BORDER_INCH, facecolor='black'))
    # 上边框
    ax.add_patch(patches.Rectangle((0, A4_HEIGHT - BORDER_INCH), A4_WIDTH, BORDER_INCH, facecolor='black'))
    # 左边框
    ax.add_patch(patches.Rectangle((0, 0), BORDER_INCH, A4_HEIGHT, facecolor='black'))
    # 右边框
    ax.add_patch(patches.Rectangle((A4_WIDTH - BORDER_INCH, 0), BORDER_INCH, A4_HEIGHT, facecolor='black'))

    # 2. 绘制底部中点的标记点 (在黑边正中画一个白色高亮圆点，便于识别)
    marker_x = A4_WIDTH / 2.0
    marker_y = BORDER_INCH / 2.0
    ax.add_patch(patches.Circle((marker_x, marker_y), BORDER_INCH * 0.3, facecolor='white'))

    return fig, ax

def get_safe_random_pos(size):
    """获取安全的随机坐标，防止图形压到 2cm 黑边"""
    safe_margin = BORDER_INCH + (size / 2.0) + 0.2
    cx = random.uniform(safe_margin, A4_WIDTH - safe_margin)
    cy = random.uniform(safe_margin, A4_HEIGHT - safe_margin)
    return cx, cy

def draw_shape(ax, shape_type, cx, cy, size, angle, number=None):
    """通用形状绘制函数"""
    if shape_type == 'square':
        rect = patches.Rectangle(
            (-size/2, -size/2), size, size,
            facecolor='black', edgecolor='black', linewidth=1.5
        )
        t = mtransforms.Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        
        # 如果需要印制白色阿拉伯数字
        if number is not None:
            # 字体大小根据正方形大小自适应（1英寸=72磅）
            fontsize = size * 72 * 0.4 
            ax.text(cx, cy, str(number), color='white', fontsize=fontsize, 
                    ha='center', va='center', weight='bold', rotation=angle)

    elif shape_type == 'circle':
        circle = patches.Circle((cx, cy), size/2, facecolor='black', edgecolor='black')
        ax.add_patch(circle)
    elif shape_type == 'triangle':
        # 等边三角形计算
        h = size * np.sqrt(3) / 2
        pts = np.array([[0, 2/3*h], [-size/2, -1/3*h], [size/2, -1/3*h]])
        rad = np.radians(angle)
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pts_rot = pts @ rot.T + np.array([cx, cy])
        tri = patches.Polygon(pts_rot, closed=True, facecolor='black', edgecolor='black')
        ax.add_patch(tri)

# ==================== 1. 基本目标(绝对居中) + 单个正方形 ====================
def generate_basic_and_single(num_samples):
    folder = os.path.join(OUTPUT_DIR, "1_single")
    os.makedirs(folder, exist_ok=True)
    # 包含了电赛的三个基本目标（居中），以及发挥目标的第一个（单个随机正方形）
    modes = ['center_circle', 'center_triangle', 'center_square', 'random_single_square']
    
    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        mode = random.choice(modes)
        size = random.uniform(2.0, 3.5)
        
        if mode.startswith('center'):
            cx, cy = A4_WIDTH / 2.0, A4_HEIGHT / 2.0
            angle = 0  # 基本目标一般不倾斜
            shape = mode.split('_')[1]
            draw_shape(ax, shape, cx, cy, size, angle)
            desc = f"Basic target: Centered {shape}"
        else:
            cx, cy = get_safe_random_pos(size)
            angle = random.uniform(0, 90)
            draw_shape(ax, 'square', cx, cy, size, angle)
            desc = "Advanced target: Single random square"
        
        filename = f"target_{i+1:03d}.png"
        plt.savefig(os.path.join(folder, filename), dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("1_single", filename),
            "category": "basic_or_single",
            "description": desc
        })
    print(f"✅ [1/4] 已生成 基本目标(居中) 及 单个正方形")

# ==================== 2. 彼此分离、面积不等的正方形组合 ====================
def generate_separated_squares(num_samples):
    folder = os.path.join(OUTPUT_DIR, "2_multi")
    os.makedirs(folder, exist_ok=True)
    
    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        num_shapes = random.randint(3, 5)
        placed_items = []
        
        for _ in range(num_shapes):
            size = random.uniform(1.0, 2.5) # 保证面积不等
            angle = random.uniform(0, 90)
            
            # 碰撞检测（确保分离）
            for attempt in range(50):
                cx, cy = get_safe_random_pos(size)
                conflict = False
                for px, py, psize in placed_items:
                    min_dist = (size + psize) / 2.0 + 0.3  # 0.3 为分离间隔
                    if math.hypot(cx - px, cy - py) < min_dist:
                        conflict = True
                        break
                if not conflict:
                    placed_items.append((cx, cy, size))
                    draw_shape(ax, 'square', cx, cy, size, angle)
                    break
                    
        filename = f"separated_{i+1:03d}.png"
        plt.savefig(os.path.join(folder, filename), dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("2_multi", filename),
            "category": "separated_squares",
            "description": "Multiple non-overlapping squares of unequal areas."
        })
    print(f"✅ [2/4] 已生成 彼此分离、面积不等的正方形组合")

# ==================== 3. 带有白色阿拉伯数字编号的正方形组合 ====================
def generate_numbered_squares(num_samples):
    folder = os.path.join(OUTPUT_DIR, "3_tilted") # 沿用原文件夹命名框架
    os.makedirs(folder, exist_ok=True)
    
    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        num_shapes = random.randint(2, 4)
        placed_items = []
        
        # 挑选不重复的 1 位数字编号
        numbers = random.sample(range(0, 10), num_shapes)
        
        for idx in range(num_shapes):
            size = random.uniform(1.5, 3.0)
            angle = random.uniform(0, 45) # 允许轻微倾斜
            
            for attempt in range(50):
                cx, cy = get_safe_random_pos(size)
                conflict = False
                for px, py, psize in placed_items:
                    if math.hypot(cx - px, cy - py) < (size + psize) / 2.0 + 0.2:
                        conflict = True
                        break
                if not conflict:
                    placed_items.append((cx, cy, size))
                    # 绘制带数字的正方形
                    draw_shape(ax, 'square', cx, cy, size, angle, number=numbers[idx])
                    break

        filename = f"numbered_{i+1:03d}.png"
        plt.savefig(os.path.join(folder, filename), dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("3_tilted", filename),
            "category": "numbered_squares",
            "description": "Squares with single-digit white Arabic numbers."
        })
    print(f"✅ [3/4] 已生成 带有 1 位白色阿拉伯数字编号的正方形")

# ==================== 4. 局部重叠、面积不等的正方形 ====================
def generate_overlapping_squares(num_samples):
    folder = os.path.join(OUTPUT_DIR, "4_overlapping")
    os.makedirs(folder, exist_ok=True)
    
    for count in range(num_samples):
        fig, ax = create_a4_canvas()
        num_squares = random.randint(3, 5)
        placed_polys = []
        
        for i in range(num_squares):
            for attempt in range(200):
                size = random.uniform(1.8, 3.5) # 面积不等
                cx, cy = get_safe_random_pos(size)
                angle = random.uniform(0, 90)
                
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                half = size / 2.0
                
                raw_corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
                rotated_corners = [(x*cos_a - y*sin_a + cx, x*sin_a + y*cos_a + cy) for x, y in raw_corners]
                poly = ShapelyPolygon(rotated_corners)
                
                if not placed_polys:
                    placed_polys.append(poly)
                    break
                else:
                    valid_candidate = True
                    overlaps_at_least_one = False
                    
                    for p in placed_polys:
                        inter_area = poly.intersection(p).area
                        if inter_area > 0:
                            if inter_area > (0.05 * poly.area):
                                overlaps_at_least_one = True
                            if inter_area > (0.50 * poly.area) or inter_area > (0.50 * p.area):
                                valid_candidate = False
                                break
                    
                    if valid_candidate and overlaps_at_least_one:
                        placed_polys.append(poly)
                        break

        # 沿用之前的“纯黑填充 + 白边”方案，这是解决重叠图形特征识别的最优解
        for p in placed_polys:
            coords = list(p.exterior.coords)
            patch = patches.Polygon(
                coords, closed=True, 
                facecolor='black', edgecolor='white', linewidth=2.0
            )
            ax.add_patch(patch)

        filename = f"overlap_{count+1:03d}.png"
        plt.savefig(os.path.join(folder, filename), dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("4_overlapping", filename),
            "category": "overlapping_squares",
            "description": "Overlapping squares of unequal areas with white cut-lines."
        })
    print(f"✅ [4/4] 已生成 局部重叠、面积不等的正方形组合")

# ==================== 主入口 ====================
if __name__ == "__main__":
    print(f"🚀 开始生成 2025电赛 专用目标物数据集 (每类 {NUM_SAMPLES_PER_CLASS} 张)...")
    
    generate_basic_and_single(NUM_SAMPLES_PER_CLASS)
    generate_separated_squares(NUM_SAMPLES_PER_CLASS)
    generate_numbered_squares(NUM_SAMPLES_PER_CLASS)
    generate_overlapping_squares(NUM_SAMPLES_PER_CLASS)
    
    json_path = os.path.join(OUTPUT_DIR, "labels.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_annotations, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 全部生成完毕！")
    print(f"📁 数据集目录: {os.path.abspath(OUTPUT_DIR)}")