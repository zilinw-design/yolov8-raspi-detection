import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np
import random
import os
import json
import math
import cv2  # 新增：用于生成真实的 3D 投影倾斜视角
from shapely.geometry import Polygon as ShapelyPolygon

# ==================== 配置区 ====================
OUTPUT_DIR = "dataset_nuedc_2025_v2"
DPI = 300
A4_WIDTH = 8.27   # 210mm (英寸)
A4_HEIGHT = 11.69 # 297mm (英寸)
BORDER_CM = 2.0   # 边缘黑色线宽 2cm
BORDER_INCH = BORDER_CM / 2.54  

NUM_SAMPLES_PER_CLASS = 10  # 每类生成样本数，建议训练时调到 100-500

dataset_annotations = []

def create_a4_canvas():
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), dpi=DPI, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect='equal')
    ax.set_xlim(0, A4_WIDTH)
    ax.set_ylim(0, A4_HEIGHT)
    ax.axis('off')

    # 绘制 2cm 黑色边框（4填实Rect + 4角点方块确保连通）
    ax.add_patch(patches.Rectangle((0, 0), A4_WIDTH, BORDER_INCH, facecolor='black'))
    ax.add_patch(patches.Rectangle((0, A4_HEIGHT - BORDER_INCH), A4_WIDTH, BORDER_INCH, facecolor='black'))
    ax.add_patch(patches.Rectangle((0, 0), BORDER_INCH, A4_HEIGHT, facecolor='black'))
    ax.add_patch(patches.Rectangle((A4_WIDTH - BORDER_INCH, 0), BORDER_INCH, A4_HEIGHT, facecolor='black'))
    # 四个角加方块填补拼缝
    for cx, cy in [(0, 0), (A4_WIDTH - BORDER_INCH, 0),
                   (0, A4_HEIGHT - BORDER_INCH), (A4_WIDTH - BORDER_INCH, A4_HEIGHT - BORDER_INCH)]:
        ax.add_patch(patches.Rectangle((cx, cy), BORDER_INCH, BORDER_INCH, facecolor='black'))

    # 绘制底部中点的白色标记点
    ax.add_patch(patches.Circle((A4_WIDTH / 2.0, BORDER_INCH / 2.0), BORDER_INCH * 0.3, facecolor='white'))
    return fig, ax

def get_safe_random_pos(size):
    """安全随机坐标，防止 10-16cm 的大图压到黑边"""
    safe_margin = BORDER_INCH + (size / 2.0) + 0.1
    # 限制 x 最小最大范围避免报错
    min_x = min(safe_margin, A4_WIDTH / 2.0)
    max_x = max(A4_WIDTH - safe_margin, A4_WIDTH / 2.0)
    cx = random.uniform(min_x, max_x)
    cy = random.uniform(safe_margin, A4_HEIGHT - safe_margin)
    return cx, cy

def draw_shape(ax, shape_type, cx, cy, size, angle, number=None):
    if shape_type == 'square':
        rect = patches.Rectangle((-size/2, -size/2), size, size, facecolor='black', edgecolor='black', linewidth=1.5)
        t = mtransforms.Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
        if number is not None:
            fontsize = size * 72 * 0.4 
            ax.text(cx, cy, str(number), color='white', fontsize=fontsize, 
                    ha='center', va='center', weight='bold', rotation=angle)
    elif shape_type == 'circle':
        ax.add_patch(patches.Circle((cx, cy), size/2, facecolor='black', edgecolor='black'))
    elif shape_type == 'triangle':
        h = size * np.sqrt(3) / 2
        pts = np.array([[0, 2/3*h], [-size/2, -1/3*h], [size/2, -1/3*h]])
        rad = np.radians(angle)
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pts_rot = pts @ rot.T + np.array([cx, cy])
        ax.add_patch(patches.Polygon(pts_rot, closed=True, facecolor='black', edgecolor='black'))
        
    # 返回标准化的图形数据字典，用于 labels.json
    return {
        "type": shape_type,
        "center": [round(cx, 2), round(cy, 2)],
        "size_inch": round(size, 2),
        "angle": round(angle, 2),
        "number": number
    }

# ==================== 1. 基本目标(居中) + 单个正方形 ====================
def generate_basic_and_single(num_samples):
    folder = os.path.join(OUTPUT_DIR, "1_single")
    os.makedirs(folder, exist_ok=True)
    modes = ['center_circle', 'center_triangle', 'center_square', 'random_single_square']
    # 保证每种至少2个，剩余随机
    assigned = modes * 2 + random.choices(modes, k=num_samples - 8)
    random.shuffle(assigned)

    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        mode = assigned[i]
        # 修复 1：基本图形尺寸调整为 4.0 ~ 6.3 英寸 (约 10-16cm)
        size = random.uniform(4.0, 6.3) 
        shapes_info = []
        
        if mode.startswith('center'):
            cx, cy = A4_WIDTH / 2.0, A4_HEIGHT / 2.0
            shape = mode.split('_')[1]
            shape_data = draw_shape(ax, shape, cx, cy, size, 0)
            shapes_info.append(shape_data)
        else:
            cx, cy = get_safe_random_pos(size)
            angle = random.uniform(0, 15)  # 比赛目标轻微角度
            shape_data = draw_shape(ax, 'square', cx, cy, size, angle)
            shapes_info.append(shape_data)
        
        filename = f"target_{i+1:03d}.png"
        plt.savefig(os.path.join(folder, filename), dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("1_single", filename),
            "category": "basic_or_single",
            "shapes": shapes_info
        })
    print(f"✅ [1/5] 已生成大尺寸 (10-16cm) 的基本图形及单正方形")

# ==================== 2. 彼此分离、面积不等的正方形组合 ====================
def generate_separated_squares(num_samples):
    folder = os.path.join(OUTPUT_DIR, "2_multi")
    os.makedirs(folder, exist_ok=True)
    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        num_shapes = random.randint(3, 5)
        placed_items, shapes_info = [], []
        
        for _ in range(num_shapes):
            size = random.uniform(1.5, 3.5)
            angle = random.uniform(0, 10)  # 分离正方形基本不旋
            for attempt in range(50):
                cx, cy = get_safe_random_pos(size)
                conflict = False
                for px, py, psize in placed_items:
                    if math.hypot(cx - px, cy - py) < (size + psize)/2.0 + 0.3:
                        conflict = True
                        break
                if not conflict:
                    placed_items.append((cx, cy, size))
                    shape_data = draw_shape(ax, 'square', cx, cy, size, angle)
                    shapes_info.append(shape_data)
                    break
                    
        filename = f"separated_{i+1:03d}.png"
        plt.savefig(os.path.join(folder, filename), dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("2_multi", filename),
            "category": "separated_squares",
            "shapes": shapes_info
        })
    print(f"✅ [2/5] 已生成彼此分离的正方形组合")

# ==================== 3. 带有白色阿拉伯数字编号的正方形 (重命名) ====================
def generate_numbered_squares(num_samples):
    folder = os.path.join(OUTPUT_DIR, "3_numbered")  # 修复 3：重命名文件夹
    os.makedirs(folder, exist_ok=True)
    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        num_shapes = random.randint(2, 4)
        placed_items, shapes_info = [], []
        numbers = random.sample(range(0, 10), num_shapes)
        
        for idx in range(num_shapes):
            size = random.uniform(2.0, 4.0)
            angle = random.uniform(0, 10)  # 数字需正对
            for attempt in range(50):
                cx, cy = get_safe_random_pos(size)
                conflict = False
                for px, py, psize in placed_items:
                    if math.hypot(cx - px, cy - py) < (size + psize)/2.0 + 0.2:
                        conflict = True
                        break
                if not conflict:
                    placed_items.append((cx, cy, size))
                    shape_data = draw_shape(ax, 'square', cx, cy, size, angle, number=numbers[idx])
                    shapes_info.append(shape_data)
                    break

        filename = f"numbered_{i+1:03d}.png"
        plt.savefig(os.path.join(folder, filename), dpi=DPI)
        plt.close(fig)
        dataset_annotations.append({
            "filename": os.path.join("3_numbered", filename),
            "category": "numbered_squares",
            "shapes": shapes_info
        })
    print(f"✅ [3/5] 已生成带有白色数字编号的正方形 (3_numbered)")

# ==================== 4. 局部重叠正方形 ====================
def generate_overlapping_squares(num_samples):
    folder = os.path.join(OUTPUT_DIR, "4_overlapping")
    os.makedirs(folder, exist_ok=True)
    for count in range(num_samples):
        fig, ax = create_a4_canvas()
        num_squares = random.randint(3, 5)
        placed_polys, shapes_info = [], []
        
        for i in range(num_squares):
            for attempt in range(200):
                size = random.uniform(2.0, 4.5)
                cx, cy = get_safe_random_pos(size)
                angle = random.uniform(0, 20)  # 轻微角度产生重叠
                rad = np.radians(angle)
                cos_a, sin_a = np.cos(rad), np.sin(rad)
                half = size / 2.0
                raw_corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
                rotated_corners = [(x*cos_a - y*sin_a + cx, x*sin_a + y*cos_a + cy) for x, y in raw_corners]
                poly = ShapelyPolygon(rotated_corners)
                
                if not placed_polys:
                    placed_polys.append(poly)
                    shapes_info.append({"type": "square", "center": [round(cx,2), round(cy,2)], "size_inch": round(size,2), "angle": round(angle,2)})
                    break
                else:
                    valid = True
                    overlaps = False
                    for p in placed_polys:
                        inter = poly.intersection(p).area
                        if inter > 0:
                            if inter > (0.05 * poly.area): overlaps = True
                            if inter > (0.5 * poly.area) or inter > (0.5 * p.area):
                                valid = False; break
                    if valid and overlaps:
                        placed_polys.append(poly)
                        shapes_info.append({"type": "square", "center": [round(cx,2), round(cy,2)], "size_inch": round(size,2), "angle": round(angle,2)})
                        break

        for p in placed_polys:
            coords = list(p.exterior.coords)
            ax.add_patch(patches.Polygon(coords, closed=True, facecolor='black', edgecolor='white', linewidth=2.0))

        filename = f"overlap_{count+1:03d}.png"
        plt.savefig(os.path.join(folder, filename), dpi=DPI)
        plt.close(fig)
        dataset_annotations.append({
            "filename": os.path.join("4_overlapping", filename),
            "category": "overlapping_squares",
            "shapes": shapes_info
        })
    print(f"✅ [4/5] 已生成重叠、面积不等的正方形组合")

# ==================== 5. 真实视角投影倾斜测试 (新增) ====================
def generate_tilted_samples(num_samples):
    folder = os.path.join(OUTPUT_DIR, "5_tilted")
    os.makedirs(folder, exist_ok=True)
    
    for i in range(num_samples):
        # 复用 `1_single` 和 `3_numbered` 的逻辑生成底图
        fig, ax = create_a4_canvas()
        size = random.uniform(3.0, 5.0)
        cx, cy = A4_WIDTH / 2.0, A4_HEIGHT / 2.0
        
        # 随机选择一种形态
        shapes_info = []
        if random.random() > 0.5:
            shapes_info.append(draw_shape(ax, 'square', cx, cy, size, random.uniform(0,45), number=random.randint(0,9)))
        else:
            shapes_info.append(draw_shape(ax, 'square', cx, cy, size, 0))
            
        # 存入内存/临时文件进行变换
        temp_path = os.path.join(folder, "temp.png")
        plt.savefig(temp_path, dpi=DPI)
        plt.close(fig)
        
        # 使用 OpenCV 进行 30-60度的真实空间透视倾斜模拟
        img = cv2.imread(temp_path)
        h, w = img.shape[:2]
        tilt_angle = random.uniform(30, 60)
        
        # 透视形变逻辑: 模拟镜头从斜上方俯拍，上方变窄，整体高度被压缩
        compression = math.cos(math.radians(tilt_angle)) # 高度压缩率
        scale_top = 1.0 - 0.4 * (tilt_angle / 60.0)      # 顶部收缩率 (最远端)
        
        new_h = int(h * compression)
        top_w = int(w * scale_top)
        offset_x = (w - top_w) // 2
        
        # 源矩阵与目标投影矩阵
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        pts2 = np.float32([[offset_x, h - new_h], [offset_x + top_w, h - new_h], [0, h], [w, h]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # 投影变换，留白区域填补为纯白(255,255,255)
        warped_img = cv2.warpPerspective(img, matrix, (w, h), borderValue=(255, 255, 255))
        
        filename = f"tilted_3D_{tilt_angle:.1f}deg_{i+1:03d}.png"
        cv2.imwrite(os.path.join(folder, filename), warped_img)
        os.remove(temp_path)  # 清理临时文件
        
        dataset_annotations.append({
            "filename": os.path.join("5_tilted", filename),
            "category": "tilted_perspective",
            "tilt_angle_deg": round(tilt_angle, 1),
            "original_shapes": shapes_info  # 记录变换前的物理位置坐标
        })
        
    print(f"✅ [5/5] 已生成真实空间 30°-60° 透视倾斜模拟数据 (5_tilted)")

# ==================== 主入口 ====================
if __name__ == "__main__":
    print(f"🚀 开始生成 2025电赛 优化版数据集 (包含 {OUTPUT_DIR})...")
    
    generate_basic_and_single(NUM_SAMPLES_PER_CLASS)
    generate_separated_squares(NUM_SAMPLES_PER_CLASS)
    generate_numbered_squares(NUM_SAMPLES_PER_CLASS)
    generate_overlapping_squares(NUM_SAMPLES_PER_CLASS)
    generate_tilted_samples(NUM_SAMPLES_PER_CLASS)  # 新增修复 2
    
    json_path = os.path.join(OUTPUT_DIR, "labels.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        # 修复 4：丰富 JSON 的层级信息
        json.dump(dataset_annotations, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 全部生成完毕！")
    print(f"📁 详细标注已存入: {os.path.abspath(json_path)}")