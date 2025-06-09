"""
Enhanced Data Visualization Module for Haptic Experiment 1 Analysis
Imports and extends the ActuatorDataAnalyzer class for comprehensive visualization
Excludes Stage 3 from all analyses
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, anderson, kstest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Import the existing analyzer
try:
    from stat_analysis import ActuatorDataAnalyzer
except ImportError:
    print("Could not import ActuatorDataAnalyzer. Please ensure the original analysis file is available.")

class HapticDataVisualizer:
    def __init__(self, file_pattern="*_data.csv"):
        """
        Initialize the visualizer with data loading and basic analysis
        """
        self.analyzer = ActuatorDataAnalyzer()
        self.file_pattern = file_pattern
        self.aggregated_accuracy_data = {}
        self.sequence_groups = {}
        self.timing_data = {}
        self.stages_to_analyze = [1, 2, 4]  # Exclude Stage 3
        
        # Load and analyze data
        self._load_and_analyze_data()
        
    def _load_and_analyze_data(self):
        """
        Load data and run basic analysis
        """
        print("Loading data and running basic analysis...")
        self.analyzer.read_csv_files(self.file_pattern)
        self.analyzer.find_actuator_mismatches()
        self.analyzer.calculate_stage_accuracies()
        self.analyzer.analyze_direction_accuracy()
        
    def prepare_aggregated_accuracy_data(self):
        """
        Prepare aggregated accuracy data for normality testing
        """
        print("\nPreparing aggregated accuracy data...")
        
        for stage in self.stages_to_analyze:
            stage_accuracies = []
            
            for file_key, df in self.analyzer.dataframes.items():
                stage_data = df[df['Stage'] == stage]
                
                if len(stage_data) == 0:
                    continue
                
                # Calculate accuracy for this participant in this stage
                correct_count = 0
                total_count = len(stage_data)
                
                for _, row in stage_data.iterrows():
                    selected_list = self.analyzer.parse_actuators_to_list(row['Selected Actuators'])
                    actual_list = self.analyzer.parse_actuators_to_list(row['Actual Actuators'])
                    
                    if selected_list == actual_list:
                        correct_count += 1
                
                accuracy = correct_count / total_count if total_count > 0 else 0
                stage_accuracies.append({
                    'participant': file_key,
                    'stage': stage,
                    'accuracy': accuracy,
                    'correct': correct_count,
                    'total': total_count
                })
            
            self.aggregated_accuracy_data[stage] = stage_accuracies
        
        return self.aggregated_accuracy_data
    
    def test_normality(self):
        """
        Test normality of aggregated accuracy data using multiple tests
        """
        if not self.aggregated_accuracy_data:
            self.prepare_aggregated_accuracy_data()
        
        print("\n" + "="*60)
        print("NORMALITY TESTING FOR AGGREGATED ACCURACY DATA")
        print("="*60)
        
        normality_results = {}
        
        for stage in self.stages_to_analyze:
            stage_data = self.aggregated_accuracy_data[stage]
            accuracies = [item['accuracy'] for item in stage_data]
            
            if len(accuracies) < 3:
                print(f"\nStage {stage}: Insufficient data for normality testing (n={len(accuracies)})")
                continue
            
            print(f"\nStage {stage} Normality Tests (n={len(accuracies)}):")
            print("-" * 40)
            
            results = {}
            
            # Shapiro-Wilk test (best for small samples)
            if len(accuracies) <= 50:
                shapiro_stat, shapiro_p = shapiro(accuracies)
                results['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
                print(f"Shapiro-Wilk: W = {shapiro_stat:.4f}, p = {shapiro_p:.6f}")
                print(f"  Result: {'Normal' if shapiro_p > 0.05 else 'Not Normal'} (α = 0.05)")
            
            # D'Agostino's normality test
            if len(accuracies) >= 8:
                dagostino_stat, dagostino_p = normaltest(accuracies)
                results['dagostino'] = {'statistic': dagostino_stat, 'p_value': dagostino_p}
                print(f"D'Agostino: stat = {dagostino_stat:.4f}, p = {dagostino_p:.6f}")
                print(f"  Result: {'Normal' if dagostino_p > 0.05 else 'Not Normal'} (α = 0.05)")
            
            # Anderson-Darling test
            anderson_result = anderson(accuracies, dist='norm')
            results['anderson'] = {
                'statistic': anderson_result.statistic,
                'critical_values': anderson_result.critical_values,
                'significance_levels': anderson_result.significance_level
            }
            print(f"Anderson-Darling: stat = {anderson_result.statistic:.4f}")
            
            # Check significance levels
            for i, (cv, sl) in enumerate(zip(anderson_result.critical_values, anderson_result.significance_level)):
                result_text = "Normal" if anderson_result.statistic < cv else "Not Normal"
                print(f"  At {sl}% level: {result_text} (critical value: {cv:.4f})")
            
            # Kolmogorov-Smirnov test against normal distribution
            # First standardize the data
            standardized = stats.zscore(accuracies)
            ks_stat, ks_p = kstest(standardized, 'norm')
            results['ks'] = {'statistic': ks_stat, 'p_value': ks_p}
            print(f"Kolmogorov-Smirnov: D = {ks_stat:.4f}, p = {ks_p:.6f}")
            print(f"  Result: {'Normal' if ks_p > 0.05 else 'Not Normal'} (α = 0.05)")
            
            # Summary statistics
            print(f"\nDescriptive Statistics:")
            print(f"  Mean: {np.mean(accuracies):.4f}")
            print(f"  Std:  {np.std(accuracies, ddof=1):.4f}")
            print(f"  Skew: {stats.skew(accuracies):.4f}")
            print(f"  Kurt: {stats.kurtosis(accuracies):.4f}")
            
            normality_results[stage] = {
                'data': accuracies,
                'tests': results,
                'descriptive': {
                    'mean': np.mean(accuracies),
                    'std': np.std(accuracies, ddof=1),
                    'skewness': stats.skew(accuracies),
                    'kurtosis': stats.kurtosis(accuracies)
                }
            }
        
        # Overall assessment
        print(f"\n" + "="*60)
        print("OVERALL NORMALITY ASSESSMENT")
        print("="*60)
        
        for stage, results in normality_results.items():
            tests = results['tests']
            normal_count = 0
            total_tests = 0
            
            for test_name, test_result in tests.items():
                if test_name == 'anderson':
                    # For Anderson-Darling, check at 5% level
                    if test_result['statistic'] < test_result['critical_values'][2]:  # 5% level is usually index 2
                        normal_count += 1
                    total_tests += 1
                else:
                    if test_result['p_value'] > 0.05:
                        normal_count += 1
                    total_tests += 1
            
            assessment = "Likely Normal" if normal_count >= total_tests/2 else "Likely Not Normal"
            print(f"Stage {stage}: {normal_count}/{total_tests} tests suggest normality → {assessment}")
        
        self.normality_results = normality_results
        return normality_results
    
    def create_normality_plots(self):
        """
        Create visual plots to assess normality
        """
        if not hasattr(self, 'normality_results'):
            self.test_normality()
        
        n_stages = len(self.stages_to_analyze)
        # Change to single row: 1 row, n_stages columns
        fig, axes = plt.subplots(1, n_stages, figsize=(4*n_stages, 2))
        
        if n_stages == 1:
            axes = [axes]  # Make it a list for consistency
        
        for i, stage in enumerate(self.stages_to_analyze):
            if stage not in self.normality_results:
                continue
                
            accuracies = self.normality_results[stage]['data']
            
            # Histogram with normal overlay
            ax1 = axes[i]  # Now using single row
            
            # Better bin calculation for more coverage
            data_range = max(accuracies) - min(accuracies)
            if data_range == 0:  # Handle case where all values are the same
                n_bins = 1
                bin_width = 0.1  # Arbitrary small width
            else:
                # Use more aggressive binning for better coverage
                n_bins = max(4, min(6, len(accuracies) // 3))  # fewer bins = wider bins
            
            # Create bins that span the full range with some padding
            bin_edges = np.linspace(min(accuracies), 
                                   max(accuracies), 
                                   n_bins + 1)
            
            # This line controls bar width - rwidth parameter makes bars wider
            ax1.hist(accuracies, bins=bin_edges, density=True, alpha=0.7, 
                    color='skyblue', edgecolor='black', 
                    rwidth=0.95)  # rwidth=0.95 means bars take 95% of bin width (reduces gaps)
            
            # Overlay normal distribution
            x_range = np.linspace(min(accuracies) - data_range*0.1, 
                                 max(accuracies) + data_range*0.1, 100)
            normal_curve = stats.norm.pdf(x_range, np.mean(accuracies), np.std(accuracies, ddof=1))
            ax1.plot(x_range, normal_curve, 'r-', linewidth=2, label='Normal Distribution')
            
            ax1.set_xlabel('Accuracy', fontsize=14)
            ax1.set_ylabel('Density', fontsize=14)
            ax1.tick_params(axis='x', labelsize=14)  # X-axis tick labels (Stage 1, Stage 2, etc.)
            ax1.tick_params(axis='y', labelsize=14)  # Y-axis tick numbers (30, 40, 50, etc.)
            ax1.legend(fontsize=14)
            ax1.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
    def group_sequences_by_pattern(self):
        """
        Group sequences by their predefined experimental patterns
        """
        print("\nGrouping sequences by predefined experimental patterns...")
        
        # Define the expected sequences for each stage
        expected_sequences = {
            1: [
                [1, 5, 6], [6, 5, 1],                 # up/down
                [2, 1, 4], [4, 1, 2],                 # right/left  
                [2, 4], [4, 2],                       # right/left short
#                 [1, 2, 3],                            # back
                [1, 3],                               # back short
                [1, 2], [2, 3], [3, 4], [4, 1],       # diagonal
                [2, 5], [2, 6], [3, 5], [3, 6], [4, 5], [4, 6]  # diagonal
            ],
            2: [
                [1, 2, 4], [2, 6, 3], [3, 1, 5],
                [2, 5, 4], [3, 4, 6], [3, 5, 6],
                [1, 2, 3, 4], [2, 5, 4, 3],
                [5, 2, 3, 4], [4, 3, 2, 1],
                [2, 3, 4, 6], [1, 6, 4, 3],
                [6, 4, 3, 2], [3, 4, 6, 1],
                [1, 5, 3]
            ],
            4: [
                [1, 5, 6], [6, 5, 1],                 # up/down
                [2, 1, 4], [4, 1, 2],                 # right/left
                [2, 4], [4, 2],                       # right/left short
                [1, 2, 3],                            # back
                [3, 2, 1],                            # front
                [1, 3],                               # back short
                [1, 2], [2, 3],                       # diagonal subset
                [1, 2, 4], [2, 6, 3], [3, 1, 5], [2, 5, 4],  # stage 2 subset
                [3, 4, 6], [1, 2, 3, 4], [2, 5, 4, 3], [2, 3, 4, 6]  # stage 2 subset
            ]
        }
        
        for stage in self.stages_to_analyze:
            stage_groups = {}
            expected_seqs = expected_sequences.get(stage, [])
            
            # Initialize groups for all expected sequences
            for expected_seq in expected_seqs:
                pattern_key = tuple(expected_seq)
                stage_groups[pattern_key] = {
                    'actual_sequence': expected_seq,
                    'responses': [],
                    'participants': [],
                    'sequence_numbers': [],
                    'found_in_data': False
                }
            
            # Now populate with actual data
            for file_key, df in self.analyzer.dataframes.items():
                stage_data = df[df['Stage'] == stage]
                
                for _, row in stage_data.iterrows():
                    actual_list = self.analyzer.parse_actuators_to_list(row['Actual Actuators'])
                    selected_list = self.analyzer.parse_actuators_to_list(row['Selected Actuators'])
                    
                    pattern_key = tuple(actual_list)
                    
                    # Only include if it matches an expected sequence
                    if pattern_key in stage_groups:
                        stage_groups[pattern_key]['responses'].append(selected_list)
                        stage_groups[pattern_key]['participants'].append(file_key)
                        stage_groups[pattern_key]['sequence_numbers'].append(row['Sequence In Stage'])
                        stage_groups[pattern_key]['found_in_data'] = True
                    else:
                        print(f"Warning: Unexpected sequence {actual_list} found in Stage {stage} data")
            
#             # Report missing sequences
#             missing_sequences = [seq for pattern_key, data in stage_groups.items() 
#                                if not data['found_in_data']]
#             if missing_sequences:
#                 print(f"Warning: Expected sequences not found in Stage {stage} data:")
#                 for seq in missing_sequences:
#                     print(f"  {list(seq)}")
            
            # Only keep sequences that were found in data
            stage_groups = {k: v for k, v in stage_groups.items() if v['found_in_data']}
            
            self.sequence_groups[stage] = stage_groups
            print(f"Stage {stage}: Found {len(stage_groups)} out of {len(expected_seqs)} expected sequences")
        
        return self.sequence_groups
        
#     def create_sequence_scatter_plots(self):
#         """
#         Create scatter plots for ALL expected sequences (including empty ones)
#         """
#         if not self.sequence_groups:
#             self.group_sequences_by_pattern()
#         
#         for stage in self.stages_to_analyze:
#             stage_groups = self.sequence_groups[stage]
#             n_sequences = len(stage_groups)
#             
#             if n_sequences == 0:
#                 continue
#             
#             # Calculate optimal subplot arrangement
#             if n_sequences <= 6:
#                 n_cols = 3
#                 n_rows = 2
#             elif n_sequences <= 15:
#                 n_cols = 5
#                 n_rows = 3
#             elif n_sequences <= 20:
#                 n_cols = 5
#                 n_rows = 4
#             else:
#                 n_cols = 6
#                 n_rows = (n_sequences - 1) // n_cols + 1
#             
#             # Create figure
#             fig_width = max(12, n_cols * 2.5)
#             fig_height = max(8, n_rows * 2)
#             fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
#             fig.suptitle(f'Stage {stage}', 
#                         fontsize=14, fontweight='bold', y=0.95)
#             
#             # Handle subplot arrangement
#             if n_sequences == 1:
#                 axes = np.array([axes])
#             elif n_rows == 1:
#                 axes = axes.reshape(1, -1)
#             elif n_cols == 1:
#                 axes = axes.reshape(-1, 1)
#             
#             axes = axes.flatten()
#             
#             for idx, (pattern_key, group_data) in enumerate(stage_groups.items()):
#                 if idx >= len(axes):
#                     break
#                     
#                 ax = axes[idx]
#                 actual_sequence = group_data['actual_sequence']
#                 responses = group_data['responses']
#                 found_in_data = group_data['found_in_data']
#                 
#                 # Plot actual sequence
#                 if actual_sequence:
#                     x_actual = list(range(1, len(actual_sequence) + 1))
#                     color = 'ro-' if found_in_data else 'go-'  # Red if found, green if missing
#                     ax.plot(x_actual, actual_sequence, color, linewidth=2.5, markersize=8,
#                            label='Expected Sequence', alpha=0.9, zorder=3)
#                 
# #                 # Plot responses if data exists
# #                 if found_in_data and responses:
# #                     for pos_idx, pos in enumerate(range(1, len(actual_sequence) + 1)):
# #                         y_values = []
# #                         for response in responses:
# #                             if len(response) > pos_idx:
# #                                 y_values.append(response[pos_idx])
# #                         
# #                         if y_values:
# #                             x_values = [pos] * len(y_values)
# #                             ax.scatter(x_values, y_values, alpha=0.4, s=40, 
# #                                      edgecolors='darkblue', linewidth=0.5, zorder=2)
# 
#                 if found_in_data and responses:
#                     # Define 11 distinct colors for different shapes/markers
#                     colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 
#                               'gray', 'olive', 'cyan', 'magenta']
#                     markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', '<', '>']
#                     num_markers = len(markers)
#                     num_colors = len(colors)
# 
#                     for resp_idx, response in enumerate(responses):
#                         x_values = []
#                         y_values = []
# 
#                         for pos_idx, value in enumerate(response):
#                             if pos_idx < len(actual_sequence):  # Make sure within bounds
#                                 x_values.append(pos_idx + 1)  # Position starts from 1
#                                 y_values.append(value)
# 
#                         if y_values:
#                             marker = markers[resp_idx % num_markers]  # Cycle through markers
#                             color = colors[resp_idx % num_colors]     # Cycle through colors
#                             ax.scatter(x_values, y_values, alpha=0.7, s=60,
#                                        edgecolors=color, linewidth=1.4,
#                                        marker=marker, zorder=2, facecolors='none')  # No-fill shape with color
#                 # Set axes
#                 if actual_sequence:
#                     ax.set_xs(range(1, len(actual_sequence) + 1))
#                     ax.set_xlim(0.5, len(actual_sequence) + 0.5)
#                 
#                 ax.set_ylim(0.5, 6.5)
#                 ax.set_ylabel('Actuator Number', fontsize=12)
#                 
#                 # Title with status
#                 status = f"(n={len(responses)})" if found_in_data else "(MISSING)"
#                 title_color = 'black' if found_in_data else 'red'
#                 ax.set_title(f'{actual_sequence} {status}', 
#                             fontsize=12, pad=5, color=title_color)
#                 
#                 ax.grid(True, alpha=0.3)
#                 ax.set_ys(range(1, 7))
#             
#             # Hide empty subplots
#             for idx in range(n_sequences, len(axes)):
#                 axes[idx].set_visible(False)
#             
#             plt.tight_layout(rect=[0, 0, 1, 0.93])
#             plt.subplots_adjust(hspace=0.4, wspace=0.3)
#             plt.show()
    def create_sequence_scatter_plots(self):
        """
        Create scatter plots for ALL expected sequences (including empty ones)
        """
        if not self.sequence_groups:
            self.group_sequences_by_pattern()
        
        # Define error count markers and colors
        error_markers = {
            1: ('x', 'green'),      # Cross for 1 error - green
            2: ('s', 'orange'),     # Square for 2 errors - orange
            3: ('o', 'purple'),     # Circle for 3 errors - purple
            4: ('*', 'darkblue')    # Star for 4 errors - dark blue
        }
        
        for stage in self.stages_to_analyze:
            stage_groups = self.sequence_groups[stage]
            n_sequences = len(stage_groups)
            
            if n_sequences == 0:
                continue
            
            # Calculate optimal subplot arrangement
            if n_sequences <= 6:
                n_cols = 3
                n_rows = 2
            elif n_sequences <= 15:
                n_cols = 5
                n_rows = 3
            elif n_sequences <= 20:
                n_cols = 5
                n_rows = 4
            else:
                n_cols = 6
                n_rows = (n_sequences - 1) // n_cols + 1
            
            # Create figure
            fig_width = max(12, n_cols * 2.5)
            fig_height = max(8, n_rows * 2)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            
            # Handle subplot arrangement
            if n_sequences == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            axes = axes.flatten()
            
            # Track which error counts are actually used for legend
            used_error_counts = set()
            
            for idx, (pattern_key, group_data) in enumerate(stage_groups.items()):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                actual_sequence = group_data['actual_sequence']
                responses = group_data['responses']
                found_in_data = group_data['found_in_data']
                
                # Plot actual sequence with thicker red line
                if actual_sequence:
                    x_actual = list(range(1, len(actual_sequence) + 1))
                    color = 'ro-' if found_in_data else 'go-'  # Red if found, green if missing
                    ax.plot(x_actual, actual_sequence, color, linewidth=4, markersize=10,
                           label='Expected Sequence', alpha=0.9, zorder=3)
                
                # Plot error patterns
                if found_in_data and responses and actual_sequence:
                    # Count errors for each position and actuator
                    error_counts = {}  # {(position, actuator): count}
                    
                    for response in responses:
                        for pos_idx, selected_actuator in enumerate(response):
                            if pos_idx < len(actual_sequence):  # Within sequence bounds
                                correct_actuator = actual_sequence[pos_idx]
                                if selected_actuator != correct_actuator:
                                    # This is an error
                                    position = pos_idx + 1  # 1-based position
                                    key = (position, selected_actuator)
                                    error_counts[key] = error_counts.get(key, 0) + 1
                    
                    # Plot error markers
                    for (position, actuator), count in error_counts.items():
                        if count in error_markers:
                            marker, color = error_markers[count]
                            ax.scatter(position, actuator, marker=marker, s=120, 
                                     color=color, alpha=0.8, zorder=4)
                            used_error_counts.add(count)
                
                # Set axes
                if actual_sequence:
                    ax.set_xticks(range(1, len(actual_sequence) + 1))
                    ax.set_xlim(0.5, len(actual_sequence) + 0.5)
                
                ax.set_ylim(0.5, 6.5)
                ax.set_ylabel('Actuator Number', fontsize=12)
                
                # Title with status
                status = f"(n={len(responses)})" if found_in_data else "(MISSING)"
                title_color = 'black' if found_in_data else 'red'
                ax.set_title(f'{actual_sequence}', 
                            fontsize=12, pad=5, color=title_color)
                
                ax.grid(True, alpha=0.3)
                ax.set_yticks(range(1, 7))
                
                # Increase font sizes for axes
                ax.tick_params(axis='both', which='major', labelsize=12)
                            
            # Hide empty subplots
            for idx in range(n_sequences, len(axes)):
                axes[idx].set_visible(False)
            
            # Add legend for error counts (only for counts that were actually used)
            if used_error_counts:
                legend_elements = []
                for count in sorted(used_error_counts):
                    if count in error_markers:
                        marker, color = error_markers[count]
                        label = f'{count} error{"s" if count > 1 else ""}'
                        legend_elements.append(plt.Line2D([0], [0], marker=marker, color=color, 
                                                        linestyle='None', markersize=10, label=label))
                
                # Place legend outside the plot area (moved further right)
                fig.legend(handles=legend_elements, loc='center right', bbox_to_anchor=(0.98, 0.5))
            
            plt.tight_layout(rect=[0, 0, 0.90, 0.93])  # Leave more space for legend
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            plt.show()

    def plot_stage_accuracy_trend(self):
        """
        Plot stage accuracies as box and whisker plot using real data
        """
        if not self.aggregated_accuracy_data:
            self.prepare_aggregated_accuracy_data()
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Prepare real data for box plot
        stages = sorted([s for s in self.aggregated_accuracy_data.keys() if s in self.stages_to_analyze])
        data_list = []
        stage_labels = []
        
        for stage in stages:
            stage_data = self.aggregated_accuracy_data[stage]
            accuracies = [item['accuracy'] * 100 for item in stage_data]  # Convert to percentage
            data_list.append(accuracies)
            stage_labels.append(f'Stage {stage}')
        
        # Create box plot - REMOVE notch=True to get rectangular boxes
        box_plot = ax.boxplot(data_list, labels=stage_labels, patch_artist=True, 
                     notch=False, showmeans=True, widths=0.3)  # Add widths parameter to make boxes wider and closer
        
        # Customize box plot colors
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize other elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plot[element], color='black', linewidth=2)
        
        # Make means more visible
        plt.setp(box_plot['means'], marker='D', markerfacecolor='red', 
                 markeredgecolor='darkred', markersize=10)
        
        # Customize the plot
        ax.set_xlabel('Stage', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold')
        ax.tick_params(axis='x', labelsize=20)  # X-axis tick labels (Stage 1, Stage 2, etc.)
        ax.tick_params(axis='y', labelsize=20)  # Y-axis tick numbers (30, 40, 50, etc.)

        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(20, 100)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2, label='Median'),
            Line2D([0], [0], marker='D', color='red', linewidth=0, 
                   markerfacecolor='red', markersize=8, label='Mean'),
            Line2D([0], [0], color='black', linewidth=1, label='Q1/Q3 & Whiskers')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=20)
        
        plt.tight_layout()
        plt.show()
            
    def fit_learning_curves(self):
        """
        Fit various learning curves and distributions to the data - showing only averages
        """
        print("\n" + "="*60)
        print("FITTING LEARNING CURVES AND DISTRIBUTIONS")
        print("="*60)
        
        # Prepare sequence-level data for learning curve analysis
        learning_data = {}
        
        for stage in self.stages_to_analyze:
            stage_sequences = []
            
            for file_key, df in self.analyzer.dataframes.items():
                stage_data = df[df['Stage'] == stage].sort_values('Sequence In Stage')
                
                for idx, (_, row) in enumerate(stage_data.iterrows()):
                    selected_list = self.analyzer.parse_actuators_to_list(row['Selected Actuators'])
                    actual_list = self.analyzer.parse_actuators_to_list(row['Actual Actuators'])
                    is_correct = selected_list == actual_list
                    
                    stage_sequences.append({
                        'participant': file_key,
                        'trial_number': idx + 1,
                        'sequence_in_stage': row['Sequence In Stage'],
                        'accuracy': 1 if is_correct else 0
                    })
            
            learning_data[stage] = pd.DataFrame(stage_sequences)
        
        # Create single row of plots - only average learning curves
        fig, axes = plt.subplots(1, len(self.stages_to_analyze), figsize=(5*len(self.stages_to_analyze), 3))
        
        if len(self.stages_to_analyze) == 1:
            axes = [axes]
        
        for i, stage in enumerate(self.stages_to_analyze):
            df_stage = learning_data[stage]
            
            if df_stage.empty:
                continue
            
            # Aggregate by trial number (average across participants)
            trial_accuracy = df_stage.groupby('trial_number')['accuracy'].agg(['mean', 'std', 'count']).reset_index()
            
            x = trial_accuracy['trial_number'].values
            y = trial_accuracy['mean'].values
            yerr = trial_accuracy['std'].values / np.sqrt(trial_accuracy['count'].values)  # SEM
            
            # Plot learning curve with different fits
            ax = axes[i]
            ax.errorbar(x, y*100, yerr=yerr, fmt='o-', capsize=5, label='Observed', 
                       markersize=8, linewidth=2, color='black', alpha=0.8)
            
            # Linear fit
            if len(x) > 2:
                coeffs_linear = np.polyfit(x, y, 1)
                y_linear = np.polyval(coeffs_linear, x)
                r2_linear = r2_score(y, y_linear)
                ax.plot(x, y_linear*100, '--', label=f'Linear (R²={r2_linear:.3f})', 
                       linewidth=2, color='blue')
                
                # Exponential fit (learning curve): y = a * (1 - exp(-b*x)) + c
                try:
                    from scipy.optimize import curve_fit
                    
                    def exp_learning(x, a, b, c):
                        return a * (1 - np.exp(-b * x)) + c
                    
                    popt, _ = curve_fit(exp_learning, x, y, 
                                      bounds=([0, 0, 0], [1, np.inf, 1]),
                                      maxfev=1000)
                    y_exp = exp_learning(x, *popt)
                    r2_exp = r2_score(y, y_exp)
                    ax.plot(x, y_exp*100, '--', label=f'Exponential (R²={r2_exp:.3f})', 
                           linewidth=2, color='red')
                    
                except:
                    print(f"Could not fit exponential curve for Stage {stage}")
                
                # Power law fit: y = a * x^b + c
                try:
                    def power_law(x, a, b, c):
                        return a * np.power(x, b) + c
                    
                    popt_power, _ = curve_fit(power_law, x, y, 
                                            bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, 1]),
                                            maxfev=1000)
                    y_power = power_law(x, *popt_power)
                    r2_power = r2_score(y, y_power)
                    ax.plot(x, y_power*100, '--', label=f'Power Law (R²={r2_power:.3f})', 
                           linewidth=2, color='green')
                    
                except:
                    print(f"Could not fit power law for Stage {stage}")
            
            ax.set_xlabel('Trial Number', fontsize=13)
            ax.set_ylabel('Accuracy (%)', fontsize=13)
            ax.tick_params(axis='x', labelsize=13)  # X-axis tick labels (Stage 1, Stage 2, etc.)
            ax.tick_params(axis='y', labelsize=13)  # Y-axis tick numbers (30, 40, 50, etc.)
            ax.set_xticks(range(1, len(x) + 1, 4))  # Force x-axis to show only integer values
            ax.legend(fontsize=10, loc = 'lower left')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(20, 110)
        
        plt.tight_layout()
        plt.show()
        
        return learning_data
    
    def run_complete_analysis(self):
        """
        Run the complete visualization analysis
        """
        print("="*60)
        print("RUNNING COMPLETE HAPTIC DATA VISUALIZATION ANALYSIS")
        print("="*60)
        
        # 1. Test normality
        print("\n1. Testing normality of aggregated accuracy data...")
        self.test_normality()
        self.create_normality_plots()
        
#         # 2. Create sequence scatter plots
#         print("\n2. Creating sequence scatter plots...")
#         self.create_sequence_scatter_plots()
        
#         # 3. Plot stage accuracy trend
#         print("\n3. Plotting stage accuracy trends...")
#         self.plot_stage_accuracy_trend()
         
        # 4. Fit learning curves
        print("\n4. Fitting learning curves and distributions...")
        self.fit_learning_curves()
        
        print("\n" + "="*60)
        print("VISUALIZATION ANALYSIS COMPLETE")
        print("="*60)

if __name__ == "__main__":
    # Initialize visualizer
    visualizer = HapticDataVisualizer()
    visualizer.run_complete_analysis()   