'''
Accuracy analysis of all participant results in experiment 1 and plot as bar chart for visualization
'''

import matplotlib.pyplot as plt
import numpy as np

# Data
participants = np.arange(1, 12)
stage1 = [86.7, 86.7, 100, 100, 66.7, 73.3, 100, 86.7, np.nan, 100, 60]
stage2 = [46.7, 80, 80, 100, 100, 86.7, 73.3, 66.7, 100, 93.3, 66.7]
stage3 = [93.3, 100, 100, 100, 86.7, 80, 100, 53.3, 100, 100, 100]
stage4 = [66.7, 46.7, 73.3, 87, 66.7, 93.3, 53.3, 86.7, 86.7, 93.3, 66.7]

# Plotting
bar_width = 0.2
index = np.arange(len(participants))

fig, ax = plt.subplots(figsize=(7, 5))
ax.grid(axis='y', zorder=0)
bars1 = ax.bar(index - bar_width, stage1, bar_width, label='Stage 1', zorder=3)
bars2 = ax.bar(index, stage2, bar_width, label='Stage 2', zorder=3)
bars3 = ax.bar(index + bar_width, stage3, bar_width, label='Stage 3', zorder=3)
bars4 = ax.bar(index + 2 * bar_width, stage4, bar_width, label='Stage 4', zorder=3)

# Add labels, tick labels, and legend
ax.set_xlabel('Participants', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xticks(index)
ax.set_xticklabels(participants)
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# Display the plot
plt.tight_layout()
plt.show()
