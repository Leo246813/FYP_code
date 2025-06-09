'''
Experiment 2 distraction task: delay time between response and actual finished time of sequence
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# 5 participants, 10 trials each
# Delay data (response times in seconds)
delay_data = [
    [5.2, 0, 3, 3.6, 1.5, 0, 2.5, 0, 0, 0],             # Participant 1
    [2.2, 2.8, 0, 2.4, 3.1, 2.3, 2.6, 3.3, 2.4, 2.6],   # Participant 2
    [0, 3.1, 9, 0, 4.1, 2.8, 2.3, 0, 0, 0],             # Participant 3
    [2.1, 1.8, 2.9, 3.2, 2, 0, 2.2, 1.9, 3.9, 1.7],     # Participant 4
    [6.1, 3.8, 3.2, 2.5, 3.8, 5.6, 3.7, 4.8, 3.9, 2.2]  # Participant 5
]

# Accuracy data (1 = correct, 0 = incorrect)
accuracy_data = [
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1],  # Participant 1
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Participant 2
    [0, 0, 1, 0, 0, 1, 1, 1, 1, 0],  # Participant 3
    [1, 1, 1, 1, 1, 1, 0, 1, 0, 1],  # Participant 4
    [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]   # Participant 5
]

print("Response Time vs Accuracy Correlation Analysis")
print("=" * 60)
print("Data Structure: 5 participants × 10 trials each")
print()

# Flatten the data and filter out delay = 0 pairs
filtered_delays = []
filtered_accuracy = []
excluded_count = 0
total_count = 0

for participant in range(5):
    for trial in range(10):
        total_count += 1
        delay = delay_data[participant][trial]
        acc = accuracy_data[participant][trial]
        
        if delay != 0:  # Include only non-zero delays
            filtered_delays.append(delay)
            filtered_accuracy.append(acc)
        else:
            excluded_count += 1

print(f"Total data points: {total_count}")
print(f"Valid data points (delay ≠ 0): {len(filtered_delays)}")
print(f"Excluded data points (delay = 0): {excluded_count}")
print()

# Calculate correlation
correlation_coeff, p_value = stats.pearsonr(filtered_delays, filtered_accuracy)
r_squared = correlation_coeff ** 2

print("Correlation Results:")
print(f"Pearson correlation coefficient (r): {correlation_coeff:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"R-squared (r²): {r_squared:.4f}")
print()

# Interpret correlation strength
def interpret_correlation(r):
    abs_r = abs(r)
    if abs_r < 0.1:
        return "negligible"
    elif abs_r < 0.3:
        return "weak"
    elif abs_r < 0.5:
        return "moderate"
    elif abs_r < 0.7:
        return "strong"
    else:
        return "very strong"

correlation_strength = interpret_correlation(correlation_coeff)
direction = "positive" if correlation_coeff > 0 else "negative"

print("Interpretation:")
print(f"The correlation is {correlation_strength} and {direction}.")
if correlation_coeff > 0:
    print("This suggests that longer response times are associated with higher accuracy.")
else:
    print("This suggests that longer response times are associated with lower accuracy.")

if p_value < 0.05:
    print(f"The correlation is statistically significant (p < 0.05).")
else:
    print(f"The correlation is not statistically significant (p ≥ 0.05).")
print()

# Descriptive statistics
print("Descriptive Statistics:")
print(f"Response Time - Mean: {np.mean(filtered_delays):.2f}s, Std: {np.std(filtered_delays):.2f}s")
print(f"Response Time - Range: {min(filtered_delays):.1f}s to {max(filtered_delays):.1f}s")
print(f"Accuracy - Mean: {np.mean(filtered_accuracy):.2f}")
print(f"Correct answers: {sum(filtered_accuracy)}/{len(filtered_accuracy)} ({100*np.mean(filtered_accuracy):.1f}%)")
print()

# Group analysis
correct_times = [delay for delay, acc in zip(filtered_delays, filtered_accuracy) if acc == 1]
incorrect_times = [delay for delay, acc in zip(filtered_delays, filtered_accuracy) if acc == 0]

print("Group Comparison:")
print(f"Correct answers - Mean time: {np.mean(correct_times):.2f}s (n={len(correct_times)})")
if incorrect_times:
    print(f"Incorrect answers - Mean time: {np.mean(incorrect_times):.2f}s (n={len(incorrect_times)})")
    
    # T-test to compare means
    t_stat, t_p_value = stats.ttest_ind(correct_times, incorrect_times)
    print(f"T-test for difference in means: t = {t_stat:.3f}, p = {t_p_value:.4f}")
    
    if t_p_value < 0.05:
        print("The difference in response times is statistically significant.")
    else:
        print("The difference in response times is not statistically significant.")
else:
    print("No incorrect answers in filtered data")
print()

# Participant-level analysis
print("Participant-Level Analysis:")
print("-" * 40)
for i in range(5):
    participant_delays = [delay_data[i][j] for j in range(10) if delay_data[i][j] != 0]
    participant_accuracy = [accuracy_data[i][j] for j in range(10) if delay_data[i][j] != 0]
    
    if len(participant_delays) > 1:
        p_corr, p_p_val = stats.pearsonr(participant_delays, participant_accuracy)
        print(f"Participant {i+1}: r = {p_corr:.3f}, valid trials = {len(participant_delays)}/10")
    else:
        print(f"Participant {i+1}: insufficient data (only {len(participant_delays)} valid trials)")

print()

# Create visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Scatter plot
colors = ['red' if acc == 0 else 'blue' for acc in filtered_accuracy]
ax1.scatter(filtered_delays, filtered_accuracy, c=colors, alpha=0.6, s=50)
ax1.set_xlabel('Response Time (seconds)')
ax1.set_ylabel('Accuracy (0=Wrong, 1=Correct)')
ax1.set_title(f'Response Time vs Accuracy\n(r = {correlation_coeff:.3f}, p = {p_value:.3f})')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# Add trend line
if len(filtered_delays) > 1:
    z = np.polyfit(filtered_delays, filtered_accuracy, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(filtered_delays), max(filtered_delays), 100)
    ax1.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

# Box plot comparison
if incorrect_times:
    ax2.boxplot([correct_times, incorrect_times], labels=['Correct\n(1)', 'Incorrect\n(0)'])
    ax2.set_ylabel('Response Time (seconds)')
    ax2.set_title('Response Time Distribution by Accuracy')
    ax2.grid(True, alpha=0.3)
else:
    ax2.hist(correct_times, bins=8, alpha=0.7, edgecolor='black', color='blue')
    ax2.set_xlabel('Response Time (seconds)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Response Time Distribution\n(Correct Answers Only)')
    ax2.grid(True, alpha=0.3)

# Participant means
participant_means_delay = []
participant_means_accuracy = []
for i in range(5):
    valid_delays = [delay_data[i][j] for j in range(10) if delay_data[i][j] != 0]
    valid_accuracy = [accuracy_data[i][j] for j in range(10) if delay_data[i][j] != 0]
    if valid_delays:
        participant_means_delay.append(np.mean(valid_delays))
        participant_means_accuracy.append(np.mean(valid_accuracy))

ax3.bar(range(1, len(participant_means_delay)+1), participant_means_delay, alpha=0.7, color='skyblue')
ax3.set_xlabel('Participant')
ax3.set_ylabel('Mean Response Time (seconds)')
ax3.set_title('Mean Response Time by Participant')
ax3.set_xticks(range(1, len(participant_means_delay)+1))
ax3.grid(True, alpha=0.3)

ax4.bar(range(1, len(participant_means_accuracy)+1), participant_means_accuracy, alpha=0.7, color='lightgreen')
ax4.set_xlabel('Participant')
ax4.set_ylabel('Mean Accuracy')
ax4.set_title('Mean Accuracy by Participant')
ax4.set_xticks(range(1, len(participant_means_accuracy)+1))
ax4.set_ylim(0, 1)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create summary DataFrame
summary_df = pd.DataFrame({
    'Participant': [i+1 for i in range(5) for j in range(10) if delay_data[i][j] != 0],
    'Trial': [j+1 for i in range(5) for j in range(10) if delay_data[i][j] != 0],
    'Response_Time': filtered_delays,
    'Accuracy': filtered_accuracy
})

print("Sample of filtered data:")
print(summary_df.head(15))
print()
print("Summary Statistics:")
print(summary_df.groupby('Participant').agg({
    'Response_Time': ['count', 'mean', 'std'],
    'Accuracy': ['mean', 'sum']
}).round(3))
