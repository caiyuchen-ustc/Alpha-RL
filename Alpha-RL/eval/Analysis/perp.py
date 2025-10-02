import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
# 数据
# Greedy = [1-0.059, 1-0.0231, 1-0.0151, 1-0.0149, 1-0.0081]
# Non_Greedy = [0.059,  0.0231, 0.0151,0.0149, 0.0081]


Base_CCot = [1.0447,1.0496,1.0705, 1.0496, 1.0496, 1.0447, 1.0496]
RL_Ccot = [1.8521,  1.4243,1.0967,  1.0501, 1.0487,1.0458, 1.0483]
# 方法和颜色
colors_greedy ={
    "DIST": "#6A82CB",
    "SFT":  "#F6B8B4",
    "PPO":  "#BFA2DB",
    "RLOO": "#F7D7B0",
    "GRPO": "#8DD2C5",
    "Dr.GRPO": "#B4E197",
    "DAPO": "#87CEEB",
}


# 获取标签列表
labels = list(colors_greedy.keys())

# 转换数据（这里未乘以100，保持原始比例）
base_vals = np.array(Base_CCot)
rl_vals = np.array(RL_Ccot)

# 设置x轴位置
x = np.arange(len(labels))

# 创建图形和轴，调整比例更协调
fig, ax = plt.subplots(figsize=(10, 6))

# Base CCot 柱子
bars1 = ax.bar(x - 0.2, base_vals, width=0.4, edgecolor="gray", 
               color=[colors_greedy[l] for l in labels], alpha=0.9)

# RL CCot 柱子（带灰色斜线）
bars2 = ax.bar(x + 0.2, rl_vals, width=0.4, 
               color=[colors_greedy[l] for l in labels], edgecolor="gray", 
               hatch="///", alpha=0.6)

# 添加数值标签，更清晰地展示数据
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.4f}", 
            ha="center", va="bottom", fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f"{height:.4f}", 
            ha="center", va="bottom", fontsize=10)

# 创建自定义图例，明确标识两种数据
legend_elements = [
    mpatches.Patch(facecolor='none', edgecolor='gray', linewidth=1.5, label='                 '),
    mpatches.Patch(facecolor='none', edgecolor='gray', linewidth=1.5, 
                  hatch="///", label='                 ')
]

# 设置图例，位置更合理
ax.legend(handles=legend_elements, fontsize=18, loc='upper left', 
          frameon=False, ncol=2)

# 设置x轴刻度和标签，增强可读性
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10, rotation=0)  # 可根据需要调整rotation旋转标签

# 设置y轴范围，使数据展示更合理
ax.set_ylim(0.9, 2.0)  # 略微扩大范围，避免标签超出



ax.set_xticks(x)
ax.set_xticklabels([''] * len(labels))  # 空字符串列表，去除标签


# # 去掉y轴刻度标签但保留刻度线
# ax.set_yticklabels([''] * len(ax.get_yticks()))

# 添加网格线，便于读数
ax.grid(axis="y", linestyle="--", alpha=0.6)

# 美化图表：去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加图表标题
# ax.set_title("不同模型中Base CCot与RL CCot的对比", fontweight="bold", fontsize=14, pad=20)

# 调整布局，避免元素重叠
plt.tight_layout()

# # 保存图表
# plt.savefig("perp.png", dpi=300)   # PNG格式
plt.savefig("perp.svg")            # SVG矢量图

# 显示图表
plt.show()