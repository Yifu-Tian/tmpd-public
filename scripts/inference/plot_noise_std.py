import matplotlib.pyplot as plt

# ==========================================
# 1. 终极版全套真实数据
# ==========================================
noise_stds = [0.5, 0.7, 0.8, 1.0, 1.8, 2.0, 3.0]
tfr_rates = [0.89, 0.93, 0.95, 0.94, 0.92, 0.91, 0.83] 
smoothness_costs = [56.29, 54.74, 54.66, 54.14, 42.97, 32.78, 24.47]

# ==========================================
# 2. 学术风画图设置
# ==========================================
plt.rcParams.update({
    "font.family": "serif", 
    "font.size": 12,
    "axes.linewidth": 1.5,
})

fig, ax1 = plt.subplots(figsize=(8, 5))

# X轴范围
ax1.set_xlim(0.3, 3.2) 

# --- 画左轴 (TFR - 拓扑成功率) ---
color1 = 'tab:blue'
ax1.set_xlabel(r'Langevin Noise Variance ($\sigma_{extra}$)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Tangle-Free Rate (TFR)', color=color1, fontsize=14, fontweight='bold')

# 绘制 TFR 曲线
line1 = ax1.plot(noise_stds, tfr_rates, marker='o', markersize=9, color=color1, 
                 linewidth=3.0, label='TFR (Topological Safety)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0.80, 1.0) # 把底端放低到0.80，凸显3.0的暴跌
ax1.grid(True, linestyle='--', alpha=0.5)

# --- 画右轴 (Smoothness Cost - 运动学代价) ---
ax2 = ax1.twinx()  
color2 = 'tab:red'
ax2.set_ylabel(r'Smoothness Cost ($\times 10^3$)', color=color2, fontsize=14, fontweight='bold')

# 绘制 Smoothness 曲线
line2 = ax2.plot(noise_stds, smoothness_costs, marker='s', markersize=9, color=color2, 
                 linewidth=3.0, linestyle='--', label='Smoothness Cost (Kinematic Quality)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(20, 60) # 适配最新的平滑度范围

# --- 图例与排版 ---
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center', framealpha=0.9, fontsize=11)

plt.title("Trade-off between Topology and Kinematics in TMPD", fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()

# 保存
plt.savefig("noise_tradeoff_final.pdf", format='pdf', dpi=300)
plt.show()