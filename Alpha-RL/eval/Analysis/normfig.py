import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterSciNotation, PercentFormatter

# ---------------- 数据 ----------------
labels  = ['DIST', 'SFT', 'Dr.GRPO','PPO', 'RLOO', 'GRPO', 'DAPO']
# labels  = ['', '', '','', '', '', '']
values  = [20.75, 10.50, 0.2383,0.11, 0.048, 0.041,  0.037]  # 左轴：norm（对数轴）
percent = [12.60,  7.76, 18.36,25.59, 19.64, 18.75,  19.34]   # 右轴：百分比（线图）
percent2 = [29.30,  27.73, 37.50, 49.02, 39.45, 37.3,  36.52]   # 右轴：百分比（线图）

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 14,
    "axes.linewidth": 1,
})
# 统一的马卡龙配色
PALETTE = {
    "DIST": "#6A82CB",
    "SFT":  "#F6B8B4",
    "PPO":  "#BFA2DB",
    "RLOO": "#F7D7B0",
    "GRPO": "#8DD2C5",
    "Dr.GRPO": "#B4E197",
    "DAPO": "#87CEEB",
    "DAPO*":"#B0E0E6"
}
colors = [PALETTE[l] for l in labels]

# ---------------- 作图 ----------------
fig, ax = plt.subplots(figsize=(12, 8))

# 柱状图（左轴：对数刻度）
bars = ax.bar(labels, values, color=colors, width=0.6, alpha=0.65,log=True, edgecolor="white", linewidth=0.8)
ax.yaxis.set_major_formatter(LogFormatterSciNotation())
# ax.set_ylabel('Update Norm', fontsize=16)
ax.set_ylim(0.03, 22)   # 对数轴不能从0开始，设个小于最小值的下限
# 网格（只对左轴 y）
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

# 在柱顶添加数值标注（对数轴上用比例上移）
# for bar in bars:
#     h = bar.get_height()
#     ax.text(
#         bar.get_x() + bar.get_width()/2,
#         h * 1.10,  # 对数轴上用乘法抬高位置
#         f'{h:.4f}',
#         ha='center', va='bottom', fontsize=14
#     )

# 右侧 y 轴：百分比折线
ax2 = ax.twinx()
x = np.arange(len(labels))
line1, = ax2.plot(
    x, percent, 
    marker='D', markersize=7, linewidth=2, 
    color="gray", linestyle="-",   # 灰色实线
)

line2, = ax2.plot(
    x, percent2, 
    marker='D', markersize=7, linewidth=2, 
    color="gray", linestyle="--",  # 灰色虚线
)


# ax2.set_ylabel('Percentage (%)', fontsize=15)
#ax2.set_ylim(0, 50)
ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

# # 折线上添加数值标注
# for xi, p in zip(x, percent):
#     ax2.annotate(f'{p:.2f}%', (xi, p), textcoords="offset points", xytext=(0, 10),
#                  ha='center', fontsize=14, color="black")

# for xi, p in zip(x, percent2):
#     ax2.annotate(f'{p:.2f}%', (xi, p), textcoords="offset points", xytext=(0, -15),
#                  ha='center', fontsize=14, color="black")

# x 轴与标题
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=15)
#ax.set_title('Comparison of Update Norms (log) and Percentages', fontsize=18)

# 图例（右轴线图）
ax2.legend(loc='upper right', frameon=False, fontsize=26)

ax.tick_params(axis='x', which='both', labelbottom=False)
# ax.tick_params(axis='y', labelsize=12)
# ax2.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='y', which='both', labelleft=False)
ax2.tick_params(axis='y', which='both', labelleft=False, labelright=False)

# ---------------- 只去掉最上面的黑框 ----------------
ax.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('norm.svg', dpi=300, bbox_inches='tight')
plt.show()
