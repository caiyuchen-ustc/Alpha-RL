import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
# 数据
# Greedy = [1-0.059, 1-0.0231, 1-0.0151, 1-0.0149, 1-0.0081]
# Non_Greedy = [0.059,  0.0231, 0.0151,0.0149, 0.0081]

plt.rcParams.update({
    "font.family": "Helvetica",  # 设置为 Helvetica
    "axes.unicode_minus": False,
})


Greedy = [1-0.1731,1-0.1078,1-0.059, 1-0.0231, 1-0.0151, 1-0.0149, 1-0.0081]
Non_Greedy = [0.1731,0.1078,0.059,  0.0231, 0.0151,0.0149, 0.0081]
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


# 方法和对应的颜色
colors_greedy = {
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

# 转换为百分比
greedy_vals = np.array(Greedy) * 100
nongreedy_vals = np.array(Non_Greedy) * 100

# 设置x轴位置
x = np.arange(len(labels))

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 6))

# Greedy 柱子 - 移除原label参数
bars1 = ax.bar(x - 0.2, greedy_vals, width=0.4, edgecolor="gray", 
               color=[colors_greedy[l] for l in labels], alpha=0.9)

# Non-Greedy 柱子 - 移除原label参数
bars2 = ax.bar(x + 0.2, nongreedy_vals, width=0.4, 
               color=[colors_greedy[l] for l in labels], edgecolor="gray", 
               hatch="///", alpha=0.6)

# 添加百分比标签
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.2f}%", 
            ha="center", va="bottom", fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{height:.2f}%", 
            ha="center", va="bottom", fontsize=10)

# 创建自定义图例，无填充色
legend_elements = [
    mpatches.Patch(facecolor='none', edgecolor='gray', linewidth=1.5, label='      '),
    mpatches.Patch(facecolor='none', edgecolor='gray', linewidth=1.5, hatch="///", label='      ')
]

# 设置图例：单行显示在左上角
ax.legend(handles=legend_elements, fontsize=18, loc='upper left', ncol=2, frameon=False)


# # 设置图表标题和轴标签
# ax.set_ylabel("百分比 (%)", fontsize=12)
# ax.set_title("不同模型中Greedy和Non-Greedy解码的比例", fontweight="bold", fontsize=14)

# 设置x轴刻度和标签
ax.set_xticks(x)
ax.set_xticklabels([''] * len(labels))  # 空字符串列表，去除标签

# 设置y轴范围
ax.set_ylim(0, 110)

# 去掉y轴刻度标签但保留刻度线
ax.set_yticklabels([''] * len(ax.get_yticks()))


# 设置y轴范围
ax.set_ylim(0, 110)

# 添加网格线
ax.grid(axis="y", linestyle="--", alpha=0.6)

# 美化图表：去除顶部和右侧边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整布局
plt.tight_layout()


# plt.savefig("greedy.png", dpi=300)   # PNG
plt.savefig("greedy.svg")            # SVG 矢量图
plt.show()