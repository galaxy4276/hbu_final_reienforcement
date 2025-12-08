
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Data
# ID, Mean, Min, Max, Time(h), Label
data = [
    ('35463', 1488.0, 1379.1, 1607.5, 37.75, 'Initial (35463)'),
    ('d9bbc', 1186.0, 1052.1, 1270.5, 3.97, 'WideNet (d9bbc)'),
    ('f04ec', 2703.1, 628.2, 3249.9, 3.45, 'Blue (f04ec)'),
    ('0b810', 3073.3, 2012.0, 3365.7, 8.48, 'Orange (0b810)'),
]

ids = [x[0] for x in data]
means = [x[1] for x in data]
mins = [x[2] for x in data]
maxs = [x[3] for x in data]
times = [x[4] for x in data]
labels = [x[5] for x in data]

# 1. Efficiency Frontier (Time vs Score)
plt.figure(figsize=(10, 6))
# 35463=Gray, d9bbc=Gray, f04ec=Navy, 0b810=Cyan
plt.scatter(times, means, s=200, c=['gray', 'gray', 'navy', 'cyan'], alpha=0.8, edgecolors='black')

for i, txt in enumerate(labels):
    plt.annotate(txt, (times[i], means[i]), xytext=(5, 5), textcoords='offset points', fontsize=11, weight='bold')

plt.title('Exp Efficiency Frontier: Training Time vs. Final Performance', fontsize=14, weight='bold')
plt.xlabel('Total Training Time (Hours)', fontsize=12)
plt.ylabel('Peak Mean Score', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Annotate arrow from 35463 to 0b810 (Cyan)
plt.arrow(37.75, 1488, (8.48-37.75)*0.9, (3073-1488)*0.9, color='cyan', width=0.1, head_width=100, length_includes_head=True, alpha=0.5)
plt.text(23, 2200, "Efficiency Breakthrough\n(4x Faster, 2x Score)", color='darkcyan', fontsize=12, ha='center', weight='bold')

save_path1 = "/home/kimjihun/hbu_final_reienforcement/docs/archive/efficiency_frontier.png"
plt.savefig(save_path1, dpi=300, bbox_inches='tight')
plt.close()

# 2. Stability Analysis
plt.figure(figsize=(10, 6))
x_pos = np.arange(len(ids))
errors = [np.array(means) - np.array(mins), np.array(maxs) - np.array(means)]

bars = plt.bar(x_pos, means, yerr=errors, align='center', alpha=0.7, color=['gray', 'gray', 'navy', 'cyan'], capsize=10)
plt.xticks(x_pos, labels, fontsize=11, rotation=15)
plt.ylabel('Peak Performance (Score)', fontsize=12)
plt.title('Stability Analysis: Performance Range (Max - Min)', fontsize=14, weight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{int(height)}', ha='center', va='bottom', fontsize=12, weight='bold')

# Annotate huge error bar on f04ec
# (Done purely by visual)

save_path2 = "/home/kimjihun/hbu_final_reienforcement/docs/archive/stability_comparison.png"
plt.savefig(save_path2, dpi=300, bbox_inches='tight')
plt.close()

print("Plots generated.")
