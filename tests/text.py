def create_a4_canvas():
    """创建一个带有 100% 绝对连通 2cm 黑边和底部标记点的 A4 纸画布"""
    fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT), dpi=DPI, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect='equal')
    ax.set_xlim(0, A4_WIDTH)
    ax.set_ylim(0, A4_HEIGHT)
    ax.axis('off')

    # 1. 先绘制覆盖整张 A4 纸的黑色大矩形（作为一体化连通底色）
    ax.add_patch(patches.Rectangle((0, 0), A4_WIDTH, A4_HEIGHT, facecolor='black'))

    # 2. 在内部叠一个白色矩形（四边各缩进 2cm），“掏空”出无缝连通的 2cm 黑色外框
    ax.add_patch(patches.Rectangle(
        (BORDER_INCH, BORDER_INCH), 
        A4_WIDTH - 2 * BORDER_INCH, 
        A4_HEIGHT - 2 * BORDER_INCH, 
        facecolor='white'
    ))

    # 3. 绘制底部中点的白色标记点（位于底部 2cm 黑色边框的正中心）
    marker_x = A4_WIDTH / 2.0
    marker_y = BORDER_INCH / 2.0
    ax.add_patch(patches.Circle((marker_x, marker_y), BORDER_INCH * 0.3, facecolor='white'))

    return fig, ax