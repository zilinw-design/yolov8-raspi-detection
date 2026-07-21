import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtransforms
import numpy as np
import random
import os
import json
import math
from shapely.geometry import Polygon as ShapelyPolygon, LineString, MultiLineString

# ==================== 配置区 ====================
OUTPUT_DIR = "dataset_a4"
DPI = 300
A4_WIDTH = 8.27   # 英寸
A4_HEIGHT = 11.69  # 英寸
NUM_SAMPLES_PER_CLASS = 5  # 👈 您可以在这里修改生成的图片数量（比如改成 100）

# 全局变量，用于收集所有生成图片的标注信息
dataset_annotations = []

def create_a4_canvas():
    """创建一个模拟 A4 纸纯白画布"""
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), dpi=DPI, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect='equal')
    ax.set_xlim(0, A4_WIDTH)
    ax.set_ylim(0, A4_HEIGHT)
    ax.axis('off')
    return fig, ax

def draw_shape(ax, shape_type, cx, cy, size, angle):
    """辅助函数：在坐标轴上绘制单个图形"""
    if shape_type == 'square':
        rect = patches.Rectangle(
            (-size/2, -size/2), size, size,
            facecolor='black', edgecolor='black', linewidth=1.5
        )
        t = mtransforms.Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)
    elif shape_type == 'circle':
        circle = patches.Circle((cx, cy), size/2, facecolor='black', edgecolor='black')
        ax.add_patch(circle)
    elif shape_type == 'triangle':
        h = size * np.sqrt(3) / 2
        pts = np.array([[0, 2/3*h], [-size/2, -1/3*h], [size/2, -1/3*h]])
        rad = np.radians(angle)
        rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        pts_rot = pts @ rot.T + np.array([cx, cy])
        tri = patches.Polygon(pts_rot, closed=True, facecolor='black', edgecolor='black')
        ax.add_patch(tri)

# ==================== 1. 单个基本图形 ====================
def generate_single_shapes(num_samples):
    folder = os.path.join(OUTPUT_DIR, "1_single")
    os.makedirs(folder, exist_ok=True)
    
    shapes = ['square', 'circle', 'triangle']
    
    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        
        # 随机选择一个图形
        shape = random.choice(shapes)
        cx = random.uniform(2.0, A4_WIDTH - 2.0)
        cy = random.uniform(2.0, A4_HEIGHT - 2.0)
        size = random.uniform(1.5, 3.0)
        angle = random.uniform(0, 360)
        
        draw_shape(ax, shape, cx, cy, size, angle)
        
        filename = f"single_{i+1:03d}.png"
        path = os.path.join(folder, filename)
        plt.savefig(path, dpi=DPI)
        plt.close(fig)
        
        # 记录标注信息
        dataset_annotations.append({
            "filename": os.path.join("1_single", filename),
            "category": "single_shape",
            "contains_shapes": [shape],
            "description": f"A single {shape} with size {size:.2f} and rotation {angle:.1f} degrees."
        })
    print(f"✅ 生成 {num_samples} 张单图形")

# ==================== 2. 组合图形（随机组合，不重叠） ====================
def generate_multi_shapes(num_samples):
    folder = os.path.join(OUTPUT_DIR, "2_multi")
    os.makedirs(folder, exist_ok=True)
    all_shapes = ['square', 'circle', 'triangle']
    
    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        
        # 随机选择 2 到 3 个不重复的图形进行组合
        num_shapes_to_pick = random.randint(2, 3)
        chosen_shapes = random.sample(all_shapes, num_shapes_to_pick)
        
        placed_items = [] # 记录已放置的图形 (cx, cy, size)，用于防重叠
        
        for shape in chosen_shapes:
            size = random.uniform(1.2, 2.0)
            angle = random.uniform(0, 360)
            
            # 碰撞检测算法：尝试 50 次寻找不重叠的位置
            cx, cy = 0, 0
            for _ in range(50):
                cx = random.uniform(2.0, A4_WIDTH - 2.0)
                cy = random.uniform(2.0, A4_HEIGHT - 2.0)
                
                conflict = False
                for px, py, psize in placed_items:
                    # 计算两个图形圆心距离。加上 0.5 的安全边距保证绝对不粘连
                    min_dist = (size + psize) / 2.0 + 0.5 
                    if math.hypot(cx - px, cy - py) < min_dist:
                        conflict = True
                        break
                
                if not conflict:
                    break # 找到合适位置，跳出循环
            
            placed_items.append((cx, cy, size))
            draw_shape(ax, shape, cx, cy, size, angle)
            
        filename = f"multi_{i+1:03d}.png"
        path = os.path.join(folder, filename)
        plt.savefig(path, dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("2_multi", filename),
            "category": "multi_shapes",
            "contains_shapes": chosen_shapes,
            "description": f"Multiple non-overlapping shapes: {', '.join(chosen_shapes)}."
        })
    print(f"✅ 生成 {num_samples} 张组合图形(无重叠)")

# ==================== 3. 倾斜图形（记录角度） ====================
def generate_tilted_shapes(num_samples):
    folder = os.path.join(OUTPUT_DIR, "3_tilted")
    os.makedirs(folder, exist_ok=True)
    
    for i in range(num_samples):
        fig, ax = create_a4_canvas()
        
        tilt_type = random.choice(['horizontal', 'vertical'])
        angle_deg = random.uniform(30.0, 60.0)
        rad = np.radians(angle_deg)
        
        if tilt_type == 'horizontal':
            sx, sy = np.cos(rad), 1.0
        else:
            sx, sy = 1.0, np.cos(rad)

        # 稍微调整中心点以适应随机画布
        centers = [(2.5, 8.0), (5.8, 8.0), (4.1, 4.0)]
        sizes = [2.0, 2.0, 2.2]

        # 渲染正方形、圆形、三角形的倾斜版本...
        # 1. 正方形
        cx, cy = centers[0]
        s = sizes[0]
        sq_verts = np.array([[-s/2, -s/2], [s/2, -s/2], [s/2, s/2], [-s/2, s/2]])
        sq_tilted = sq_verts * [sx, sy] + [cx, cy]
        ax.add_patch(patches.Polygon(sq_tilted, closed=True, facecolor='black', edgecolor='black'))

        # 2. 圆形 (椭圆)
        cx, cy = centers[1]
        rx, ry = (s/2) * sx, (s/2) * sy
        ellipse = patches.Ellipse((cx, cy), width=rx*2, height=ry*2, facecolor='black', edgecolor='black')
        ax.add_patch(ellipse)

        # 3. 三角形
        cx, cy = centers[2]
        h = s * np.sqrt(3) / 2
        tri_verts = np.array([[0, 2/3*h], [-s/2, -1/3*h], [s/2, -1/3*h]])
        tri_tilted = tri_verts * [sx, sy] + [cx, cy]
        ax.add_patch(patches.Polygon(tri_tilted, closed=True, facecolor='black', edgecolor='black'))

        filename = f"tilted_{tilt_type}_{angle_deg:.1f}deg_{i+1:03d}.png"
        path = os.path.join(folder, filename)
        plt.savefig(path, dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("3_tilted", filename),
            "category": "tilted_shapes",
            "contains_shapes": ["square", "circle", "triangle"],
            "tilt_info": {
                "axis": tilt_type,
                "angle": round(angle_deg, 2)
            },
            "description": f"Shapes tilted {tilt_type}ly by {angle_deg:.1f} degrees."
        })
    print(f"✅ 生成 {num_samples} 张倾斜图形")

# ==================== 4. 重叠图形（虚线处理） ====================
def generate_overlapping_squares(num_samples):
    folder = os.path.join(OUTPUT_DIR, "4_overlapping")
    os.makedirs(folder, exist_ok=True)
    
    for count in range(num_samples):
        fig, ax = create_a4_canvas()
        num_squares = random.randint(3, 5)
        square_polys = []
        
        for i in range(num_squares):
            cx = random.uniform(3.0, A4_WIDTH - 3.0)
            cy = random.uniform(4.0, A4_HEIGHT - 4.0)
            size = random.uniform(2.0, 3.5)
            angle = random.uniform(0, 90)
            
            rad = np.radians(angle)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            half = size / 2.0
            
            raw_corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
            rotated_corners = [(x*cos_a - y*sin_a + cx, x*sin_a + y*cos_a + cy) for x, y in raw_corners]
            square_polys.append(ShapelyPolygon(rotated_corners))
        
        for i in range(len(square_polys)):
            current_poly = square_polys[i]
            coords = list(current_poly.exterior.coords)
            edges = [LineString([coords[j], coords[j+1]]) for j in range(len(coords)-1)]
            
            higher_polys = square_polys[i+1:]
            if higher_polys:
                blocker_union = higher_polys[0]
                for p in higher_polys[1:]:
                    blocker_union = blocker_union.union(p)
            else:
                blocker_union = None
                
            for edge in edges:
                if blocker_union is None or not edge.intersects(blocker_union):
                    x, y = edge.xy
                    ax.plot(x, y, color='black', lw=2, linestyle='-')
                else:
                    visible_part = edge.difference(blocker_union)
                    hidden_part = edge.intersection(blocker_union)
                    
                    if not visible_part.is_empty:
                        if isinstance(visible_part, LineString):
                            ax.plot(*visible_part.xy, color='black', lw=2, linestyle='-')
                        elif isinstance(visible_part, MultiLineString):
                            for line in visible_part.geoms:
                                ax.plot(*line.xy, color='black', lw=2, linestyle='-')
                                
                    if not hidden_part.is_empty:
                        if isinstance(hidden_part, LineString):
                            ax.plot(*hidden_part.xy, color='black', lw=1.5, linestyle='--')
                        elif isinstance(hidden_part, MultiLineString):
                            for line in hidden_part.geoms:
                                ax.plot(*line.xy, color='black', lw=1.5, linestyle='--')

        filename = f"overlap_{count+1:03d}.png"
        path = os.path.join(folder, filename)
        plt.savefig(path, dpi=DPI)
        plt.close(fig)
        
        dataset_annotations.append({
            "filename": os.path.join("4_overlapping", filename),
            "category": "overlapping_squares",
            "contains_shapes": ["square"],
            "square_count": num_squares,
            "description": f"Overlapping {num_squares} squares, hidden edges dashed."
        })
    print(f"✅ 生成 {num_samples} 张重叠虚线图形")

# ==================== 主入口 ====================
if __name__ == "__main__":
    print(f"开始生成 A4 数据集 (每类 {NUM_SAMPLES_PER_CLASS} 张)...")
    
    generate_single_shapes(NUM_SAMPLES_PER_CLASS)
    generate_multi_shapes(NUM_SAMPLES_PER_CLASS)
    generate_tilted_shapes(NUM_SAMPLES_PER_CLASS)
    generate_overlapping_squares(NUM_SAMPLES_PER_CLASS)
    
    # 将所有的标注信息导出为 JSON 文件，用于 AI 训练
    json_path = os.path.join(OUTPUT_DIR, "labels.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_annotations, f, indent=4, ensure_ascii=False)
        
    print(f"\n🎉 全部生成完毕！")
    print(f"📁 数据集已保存到: {OUTPUT_DIR}")
    print(f"🏷️ 标注文件已保存到: {json_path}")