'''
Enhanced data analysis for haptic experiment on wearable biomechatronic device
to provide haptic feedback on SRL end effector position
Revised: Excludes Stage 3 from analysis and adds vertical actuator confusion analysis
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import glob
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import defaultdict
from scipy.stats import chi2_contingency, f_oneway
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import mcnemar
import itertools

class ActuatorDataAnalyzer:
    def __init__(self):
        self.dataframes = {}
        self.mismatched_sequences = {
            'stage_1': {'selected': [], 'actual': []},
            'stage_2': {'selected': [], 'actual': []},
            'stage_4': {'selected': [], 'actual': []}
        }
        self.time_differences = {}
        self.direction_patterns = {
            'right': [[2,1,4], [2,4]],
            'left': [[4,1,2], [4,2]],
            'down': [[1,5,6]],
            'up': [[6,5,1]],
            'back': [[1,3], [1,2,3]],
            'forward': [[3,1], [3,2,1]],
            'diagonal': [[1, 2], [2, 3], [3, 4], [4, 1], [2, 5],
                         [2, 6], [3, 5], [3, 6], [4, 5], [4, 6]],
            'others': [[6, 5], [5, 3], [2, 5, 4], [4, 5, 2],
                       [4, 2, 3], [2, 1], [5, 4], [6, 4], [1, 6]]
        }
        
        self.direction_analysis = {}
        self.stage_accuracies = {}
        self.vertical_confusion_analysis = {}
        
    def read_csv_files(self, file_pattern="*data.csv"):
        """
        Read all CSV files matching the pattern and store them as dataframes
        """
        csv_files = glob.glob(file_pattern)
        
        if not csv_files:
            print(f"No CSV files found matching pattern: {file_pattern}")
            return
        
#         print(f"Found {len(csv_files)} CSV files:")
        
        for i, file_path in enumerate(csv_files):
            try:
                # Read CSV with proper handling of string columns
                df = pd.read_csv(file_path, dtype={'Selected Actuators': str, 'Actual Actuators': str})
                
                # Convert timestamp to datetime
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')
                
                # Store dataframe
                file_key = f"file_{i+1}_{os.path.basename(file_path)}"
                self.dataframes[file_key] = df
                
#                 print(f"  {i+1}. {file_path} - {len(df)} rows")
                
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
        
#         print(f"\nSuccessfully loaded {len(self.dataframes)} files")
        return self.dataframes
    
    def display_data_structure(self, file_key=None):
        """
        Display the structure of a specific file or the first file
        """
        if not self.dataframes:
            print("No data loaded. Please run read_csv_files() first.")
            return
        
        if file_key is None:
            file_key = list(self.dataframes.keys())[0]
        
        df = self.dataframes[file_key]
#         print(f"\nData structure for {file_key}:")
#         print(f"Shape: {df.shape}")
#         print(f"Columns: {list(df.columns)}")
#         print("\nFirst few rows:")
#         print(df.head())
#         print("\nData types:")
#         print(df.dtypes)
    
    def parse_actuators_to_list(self, actuator_str):
        """
        Convert actuator string to sorted list of integers
        """
        if pd.isna(actuator_str) or actuator_str == 'None':
            return []
        
        # Convert to string and clean up
        actuator_str = str(actuator_str).strip()
        
        # Extract numbers from the string
        actuator_numbers = []
        for part in actuator_str.split(','):
            part = part.strip()
            if part.isdigit():
                actuator_numbers.append(int(part))
        
        # Return list for consistent comparison
        return actuator_numbers
    
    def classify_direction(self, actuator_list):
        """
        Classify an actuator sequence into a direction based on predefined patterns
        """
        if not actuator_list:
            print(f"[Unknown] Empty actuator list: {actuator_list}")
            return 'unknown'
        
        for direction, patterns in self.direction_patterns.items():
            for pattern in patterns:
                if actuator_list == pattern:
                    return direction
        
        # If no match found
        print(f"[Unknown] Unmatched actuator sequence: {actuator_list}")
        return 'unknown'
    
    def analyze_vertical_actuator_confusion(self):
        """
        Analyze confusion patterns specifically for vertical actuators (1, 5, 6)
        """
        print("\n" + "="*60)
        print("VERTICAL ACTUATOR CONFUSION ANALYSIS")
        print("="*60)
        
        vertical_actuators = {1, 5, 6}
        confusion_data = {
            'stage_1': {'total': 0, 'confused': 0, 'details': []},
            'stage_2': {'total': 0, 'confused': 0, 'details': []},
            'stage_4': {'total': 0, 'confused': 0, 'details': []}
        }
        
        for stage in [1, 2, 4]:
            print(f"\nAnalyzing Stage {stage} vertical actuator confusion...")
            
            for file_key, df in self.dataframes.items():
                stage_data = df[df['Stage'] == stage]
                
                for _, row in stage_data.iterrows():
                    selected_list = self.parse_actuators_to_list(row['Selected Actuators'])
                    actual_list = self.parse_actuators_to_list(row['Actual Actuators'])
                    
                    # Check if any vertical actuators are involved in actual sequence
                    if any(act in vertical_actuators for act in actual_list):
                        confusion_data[f'stage_{stage}']['total'] += 1
                        
                        # Check for vertical actuator confusion
                        if selected_list != actual_list:
                            # Check if the confusion involves vertical actuators
                            selected_vertical = set(selected_list) & vertical_actuators
                            actual_vertical = set(actual_list) & vertical_actuators
                            
                            if selected_vertical != actual_vertical and len(selected_vertical) > 0:
                                confusion_data[f'stage_{stage}']['confused'] += 1
                                confusion_data[f'stage_{stage}']['details'].append({
                                    'actual': actual_list,
                                    'selected': selected_list,
                                    'actual_vertical': list(actual_vertical),
                                    'selected_vertical': list(selected_vertical)
                                })
            
            # Calculate confusion rate
            total = confusion_data[f'stage_{stage}']['total']
            confused = confusion_data[f'stage_{stage}']['confused']
            
            if total > 0:
                confusion_rate = confused / total * 100
                print(f"Stage {stage}: {confused}/{total} sequences with vertical actuator confusion ({confusion_rate:.1f}%)")
                
                # Show common confusion patterns
                if confused > 0:
                    print(f"  Common confusion patterns:")
                    for detail in confusion_data[f'stage_{stage}']['details'][:5]:  # Show first 5
                        print(f"    Actual: {detail['actual_vertical']} → Selected: {detail['selected_vertical']}")
            else:
                print(f"Stage {stage}: No sequences involving vertical actuators")
        
        self.vertical_confusion_analysis = confusion_data
        return confusion_data
    
#     def analyze_direction_accuracy(self, stages=[1]):
#         """
#         Analyze accuracy by direction for stage 1 (1D sequences only)
#         """
#         print("\n" + "="*60)
#         print("DIRECTION-BASED ACCURACY ANALYSIS (STAGE 1)")
#         print("="*60)
#         
#         direction_results = {}
#         
#         for stage in stages:
#             print(f"\nAnalyzing Stage {stage}...")
#             
#             # Collect all sequences for this stage
#             all_sequences = []
#             
#             for file_key, df in self.dataframes.items():
#                 stage_data = df[df['Stage'] == stage]
#                 
#                 for _, row in stage_data.iterrows():
#                     selected_list = self.parse_actuators_to_list(row['Selected Actuators'])
#                     actual_list = self.parse_actuators_to_list(row['Actual Actuators'])
#                     
#                     # Classify directions
#                     selected_direction = self.classify_direction(selected_list)
#                     actual_direction = self.classify_direction(actual_list)
#                     
#                     # Only include 1D sequences (those that match our direction patterns)
#                     if actual_direction != 'others':
#                         all_sequences.append({
#                             'actual_direction': actual_direction,
#                             'selected_direction': selected_direction,
#                             'actual_sequence': actual_list,
#                             'selected_sequence': selected_list,
#                             'correct': selected_list == actual_list
#                         })
#             
#             if not all_sequences:
#                 print(f"No 1D sequences found for Stage {stage}")
#                 continue
#             
#             # Calculate accuracy by direction
#             direction_stats = {}
#             
#             for direction in self.direction_patterns.keys():
#                 direction_sequences = [seq for seq in all_sequences if seq['actual_direction'] == direction]
#                 
#                 if direction_sequences:
#                     total = len(direction_sequences)
#                     correct = sum(1 for seq in direction_sequences if seq['correct'])
#                     accuracy = correct / total * 100
#                     
#                     direction_stats[direction] = {
#                         'total': total,
#                         'correct': correct,
#                         'incorrect': total - correct,
#                         'accuracy': accuracy
#                     }
#             
#             direction_results[f'stage_{stage}'] = {
#                 'sequences': all_sequences,
#                 'direction_stats': direction_stats
#             }
#             
#             # Display results with vertical actuator annotation
#             print(f"\nStage {stage} Direction Accuracy:")
#             print("-" * 40)
#             for direction, stats in direction_stats.items():
#                 annotation = " (vertical actuators - may mask confusion)" if direction in ['up', 'down'] else ""
#                 print(f"{direction.capitalize():>8}: {stats['correct']:>3}/{stats['total']:>3} = {stats['accuracy']:>5.1f}%{annotation}")
#         
#         self.direction_analysis = direction_results
#         return direction_results
#
    def analyze_direction_accuracy(self, stages=[1]):
        """
        Analyze accuracy by direction for stage 1 (1D sequences only)
        """
        print("\n" + "="*60)
        print("DIRECTION-BASED ACCURACY ANALYSIS (STAGE 1)")
        print("="*60)
        
        direction_results = {}
        
        for stage in stages:
            print(f"\nAnalyzing Stage {stage}...")
            
            # Collect all sequences for this stage
            all_sequences = []
            
            for file_key, df in self.dataframes.items():
                stage_data = df[df['Stage'] == stage]
                
                for _, row in stage_data.iterrows():
                    selected_list = self.parse_actuators_to_list(row['Selected Actuators'])
                    actual_list = self.parse_actuators_to_list(row['Actual Actuators'])
                    
                    # Classify directions
                    selected_direction = self.classify_direction(selected_list)
                    actual_direction = self.classify_direction(actual_list)
                    
                    # Only include 1D sequences (those that match our direction patterns)
                    if actual_direction != 'others':
                        all_sequences.append({
                            'actual_direction': actual_direction,
                            'selected_direction': selected_direction,
                            'actual_sequence': actual_list,
                            'selected_sequence': selected_list,
                            'correct': selected_list == actual_list,  # Exact sequence match
                            'direction_match': actual_direction == selected_direction  # Direction match
                        })
            
            if not all_sequences:
                print(f"No 1D sequences found for Stage {stage}")
                continue
            
            # Calculate accuracy by direction
            direction_stats = {}
            
            for direction in self.direction_patterns.keys():
                if direction == 'others':  # Skip 'others' since we filter them out
                    continue
                    
                direction_sequences = [seq for seq in all_sequences if seq['actual_direction'] == direction]
                
                if direction_sequences:
                    total = len(direction_sequences)
                    correct = sum(1 for seq in direction_sequences if seq['correct'])
                    direction_correct = sum(1 for seq in direction_sequences if seq['direction_match'])
                    accuracy = correct / total * 100
                    direction_accuracy = direction_correct / total * 100
                    
                    direction_stats[direction] = {
                        'total': total,
                        'correct': correct,  # Exact sequence matches
                        'direction_correct': direction_correct,  # Direction matches
                        'incorrect': total - correct,
                        'accuracy': accuracy,  # Exact sequence accuracy
                        'direction_accuracy': direction_accuracy  # Direction accuracy
                    }
            
            direction_results[f'stage_{stage}'] = {
                'sequences': all_sequences,
                'direction_stats': direction_stats
            }
            
            # Display results with both metrics
            print(f"\nStage {stage} Direction Accuracy:")
            print("-" * 70)
            print(f"{'Direction':<10} {'Exact Match':<15} {'Dir Match':<15} {'Exact %':<10} {'Dir %':<10}")
            print("-" * 70)
            for direction, stats in direction_stats.items():
                annotation = " (vertical)" if direction in ['up', 'down'] else ""
                print(f"{direction.capitalize():<10} {stats['correct']:>3}/{stats['total']:>3} ({stats['accuracy']:>5.1f}%) "
                      f"{stats['direction_correct']:>3}/{stats['total']:>3} ({stats['direction_accuracy']:>5.1f}%){annotation}")
        
        self.direction_analysis = direction_results
        return direction_results

    def chi_square_direction_test(self):
        """
        Perform chi-square tests to determine if accuracy differences between directions are significant
        """
        if not self.direction_analysis:
            print("No direction analysis found. Run analyze_direction_accuracy() first.")
            return
        
        print("\n" + "="*60)
        print("CHI-SQUARE TESTS FOR DIRECTION DIFFERENCES (STAGES 1, 2, 4)")
        print("="*60)
        
        chi_square_results = {}
        
        for stage_key, data in self.direction_analysis.items():
            stage_num = stage_key.split('_')[1]
            direction_stats = data['direction_stats']
            
            if len(direction_stats) < 2:
                print(f"\nStage {stage_num}: Not enough directions for comparison")
                continue
            
            print(f"\nStage {stage_num} Analysis:")
            print("-" * 30)
            
            # Create contingency table
            directions = list(direction_stats.keys())
            contingency_table = []
            
            for direction in directions:
                stats = direction_stats[direction]
                contingency_table.append([stats['correct'], stats['incorrect']])
            
            contingency_table = np.array(contingency_table)
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            print(f"Chi-square statistic: {chi2_stat:.4f}")
            print(f"p-value: {p_value:.6f}")
            print(f"Degrees of freedom: {dof}")
            print(f"Significance level: {'Significant' if p_value < 0.05 else 'Not significant'} (α = 0.05)")
            
            # Display contingency table
            print(f"\nContingency Table:")
            print(f"{'Direction':<10} {'Correct':<8} {'Incorrect':<10} {'Total':<8} {'Accuracy':<10}")
            print("-" * 56)
            
            for i, direction in enumerate(directions):
                correct = contingency_table[i, 0]
                incorrect = contingency_table[i, 1]
                total = correct + incorrect
                accuracy = correct / total * 100
                annotation = "*" if direction in ['up', 'down'] else ""
                print(f"{direction.capitalize():<10} {correct:<8} {incorrect:<10} {total:<8} {accuracy:<10.1f}%{annotation}")
            
            if any(direction in ['up', 'down'] for direction in directions):
                print("* Up/Down accuracy may be misleading due to vertical actuator confusion")
            
            chi_square_results[stage_key] = {
                'chi2_stat': chi2_stat,
                'p_value': p_value,
                'dof': dof,
                'expected': expected,
                'contingency_table': contingency_table,
                'directions': directions
            }
            
            # Post-hoc pairwise comparisons if significant
            if p_value < 0.05 and len(directions) > 2:
                print(f"\nPost-hoc Pairwise Comparisons (Bonferroni corrected):")
                self._pairwise_direction_comparisons(direction_stats, directions)
        
        self.chi_square_results = chi_square_results
        return chi_square_results
    
    def _pairwise_direction_comparisons(self, direction_stats, directions):
        """
        Perform pairwise comparisons between directions with Bonferroni correction
        """
        n_comparisons = len(directions) * (len(directions) - 1) // 2
        alpha_corrected = 0.05 / n_comparisons
        
        print(f"Number of comparisons: {n_comparisons}")
        print(f"Bonferroni corrected α: {alpha_corrected:.6f}")
        print()
        
        for i, dir1 in enumerate(directions):
            for j, dir2 in enumerate(directions[i+1:], i+1):
                stats1 = direction_stats[dir1]
                stats2 = direction_stats[dir2]
                
                # Create 2x2 contingency table for this pair
                contingency = np.array([
                    [stats1['correct'], stats1['incorrect']],
                    [stats2['correct'], stats2['incorrect']]
                ])
                
                chi2_stat, p_value, _, _ = chi2_contingency(contingency)
                
                significant = "***" if p_value < alpha_corrected else ""
                print(f"{dir1.capitalize()} vs {dir2.capitalize()}: "
                      f"χ² = {chi2_stat:.4f}, p = {p_value:.6f} {significant}")
    
    def calculate_stage_accuracies(self):
        """
        Calculate overall accuracy for each stage (EXCLUDING STAGE 3)
        """
        print("\n" + "="*60)
        print("STAGE ACCURACY CALCULATION (EXCLUDING STAGE 3)")
        print("="*60)
        
        stage_results = {}
        
        for stage in [1, 2, 4]:  # EXPLICITLY EXCLUDE STAGE 3
            all_sequences = []
            
            for file_key, df in self.dataframes.items():
                stage_data = df[df['Stage'] == stage]
                
                for _, row in stage_data.iterrows():
                    selected_list = self.parse_actuators_to_list(row['Selected Actuators'])
                    actual_list = self.parse_actuators_to_list(row['Actual Actuators'])
                    
                    all_sequences.append({
                        'correct': selected_list == actual_list,
                        'file': file_key
                    })
            
            if all_sequences:
                total = len(all_sequences)
                correct = sum(1 for seq in all_sequences if seq['correct'])
                accuracy = correct / total * 100
                
                stage_results[stage] = {
                    'total': total,
                    'correct': correct,
                    'incorrect': total - correct,
                    'accuracy': accuracy,
                    'sequences': all_sequences
                }
                
                print(f"Stage {stage}: {correct}/{total} = {accuracy:.1f}%")
        
        self.stage_accuracies = stage_results
        return stage_results
    
    def statistical_analysis_across_stages(self):
        """
        Perform statistical analysis comparing accuracy across stages using ANOVA and effect sizes
        (EXCLUDING STAGE 3)
        """
        if not self.stage_accuracies:
            self.calculate_stage_accuracies()
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS ACROSS STAGES (EXCLUDING STAGE 3)")
        print("="*60)
        
        # Prepare data for ANOVA (individual trial results)
        stage_data = {}
        all_accuracies = []
        stage_labels = []
        
        for stage, results in self.stage_accuracies.items():
            # Convert boolean correct/incorrect to 0/1 for each sequence
            accuracies = [1 if seq['correct'] else 0 for seq in results['sequences']]
            stage_data[stage] = accuracies
            all_accuracies.extend(accuracies)
            stage_labels.extend([stage] * len(accuracies))
        
        # One-way ANOVA
        stage_groups = [stage_data[stage] for stage in sorted(stage_data.keys())]
        f_stat, p_value = f_oneway(*stage_groups)
        
        print(f"One-way ANOVA Results (Stages 1, 2, 4 only):")
        print(f"F-statistic: {f_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significance: {'Significant' if p_value < 0.05 else 'Not significant'} (α = 0.05)")
        
        # Effect size calculation (Cohen's d for pairwise comparisons)
        if p_value < 0.05:
            print(f"\nEffect Sizes (Cohen's d) - Pairwise Comparisons:")
            print("-" * 50)
            
            stages = sorted(stage_data.keys())
            for i, stage1 in enumerate(stages):
                for stage2 in stages[i+1:]:
                    cohens_d = self._calculate_cohens_d(stage_data[stage1], stage_data[stage2])
                    effect_magnitude = self._interpret_cohens_d(cohens_d)
                    
                    acc1 = np.mean(stage_data[stage1]) * 100
                    acc2 = np.mean(stage_data[stage2]) * 100
                    
                    print(f"Stage {stage1} vs Stage {stage2}: d = {cohens_d:.3f} ({effect_magnitude})")
                    print(f"  Accuracies: {acc1:.1f}% vs {acc2:.1f}%")
        
        # Tukey's HSD for multiple comparisons
        if len(stage_data) > 2 and p_value < 0.05:
            print(f"\nTukey's HSD Multiple Comparisons:")
            print("-" * 40)
            
            # Prepare data for Tukey's test
            df_tukey = pd.DataFrame({
                'accuracy': all_accuracies,
                'stage': stage_labels
            })
            
            tukey_result = pairwise_tukeyhsd(df_tukey['accuracy'], df_tukey['stage'], alpha=0.05)
            print(tukey_result)
        
        # Create visualization
        self._plot_stage_accuracy_comparison()
        
        return {
            'anova_f': f_stat,
            'anova_p': p_value,
            'stage_data': stage_data,
            'stage_accuracies': {stage: np.mean(accs)*100 for stage, accs in stage_data.items()}
        }
    
    def _calculate_cohens_d(self, group1, group2):
        """
        Calculate Cohen's d effect size
        """
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        return cohens_d
    
    def _interpret_cohens_d(self, d):
        """
        Interpret Cohen's d effect size
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _plot_stage_accuracy_comparison(self):
        """
        Create visualization comparing accuracy across stages (EXCLUDING STAGE 3)
        """
        if not self.stage_accuracies:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # Bar plot
        stages = sorted(self.stage_accuracies.keys())
        accuracies = [self.stage_accuracies[stage]['accuracy'] for stage in stages]
        
        bars = ax1.bar(stages, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_xlabel('Stage')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy by Stage (Excluding Stage 3)')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom')
        
        # Box plot for distribution
        stage_data = []
        stage_labels = []
        
        for stage in stages:
            accuracies_binary = [1 if seq['correct'] else 0 for seq in self.stage_accuracies[stage]['sequences']]
            stage_data.extend([acc * 100 for acc in accuracies_binary])  # Convert to percentage
            stage_labels.extend([f'Stage {stage}'] * len(accuracies_binary))
        
        df_plot = pd.DataFrame({'Accuracy': stage_data, 'Stage': stage_labels})
        sns.boxplot(data=df_plot, x='Stage', y='Accuracy', ax=ax2)
        ax2.set_title('Accuracy Distribution by Stage (Excluding Stage 3)')
        ax2.set_ylabel('Accuracy (%)')
        
        plt.tight_layout()
        plt.show()
    
    def plot_direction_accuracy_comparison(self):
        """
        Create visualization comparing accuracy across directions
        """
        if not self.direction_analysis:
            print("No direction analysis found. Run analyze_direction_accuracy() first.")
            return
        
        fig, axes = plt.subplots(1, len(self.direction_analysis), figsize=(6*len(self.direction_analysis), 6))
        if len(self.direction_analysis) == 1:
            axes = [axes]
        
        for idx, (stage_key, data) in enumerate(self.direction_analysis.items()):
            ax = axes[idx]
            stage_num = stage_key.split('_')[1]
            direction_stats = data['direction_stats']
            
            directions = list(direction_stats.keys())
            accuracies = [direction_stats[dir]['accuracy'] for dir in directions]
            
            # Color code vertical directions differently
            colors = []
            for direction in directions:
                if direction in ['up', 'down']:
                    colors.append('orange')  # Highlight vertical directions
                else:
                    colors.append('lightblue')
            
            bars = ax.bar(directions, accuracies, color=colors)
            ax.set_xlabel('Direction')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Stage {stage_num} - Accuracy by Direction\n(Orange = Vertical, may mask confusion)')
            ax.set_ylim(0, 100)
            
            # Add value labels and sample sizes
            for bar, direction in zip(bars, directions):
                accuracy = direction_stats[direction]['accuracy']
                total = direction_stats[direction]['total']
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{accuracy:.1f}%\n(n={total})', ha='center', va='bottom')
            
            # Rotate x-axis labels if needed
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def find_actuator_mismatches(self):
        """
        Find sequences where selected actuators differ from actual actuators
        for stages 1, 2, and 4 (EXCLUDING STAGE 3)
        """
        all_mismatches = {
            'stage_1': {'selected': [], 'actual': []},
            'stage_2': {'selected': [], 'actual': []},
            'stage_4': {'selected': [], 'actual': []}
        }
        
        for file_key, df in self.dataframes.items():
            print(f"\nAnalyzing mismatches in {file_key}:")
            
            for stage in [1, 2, 4]:  # EXCLUDE STAGE 3
                stage_data = df[df['Stage'] == stage].copy()
                
                if len(stage_data) == 0:
                    print(f"  Stage {stage}: No data found")
                    continue
                
                # Find mismatches using list comparison
                mismatches = []
                mismatch_details = []
                
                for idx, row in stage_data.iterrows():
                    selected_list = self.parse_actuators_to_list(row['Selected Actuators'])
                    actual_list = self.parse_actuators_to_list(row['Actual Actuators'])
                    
                    if selected_list != actual_list:
                        mismatches.append(idx)
                        mismatch_details.append({
                            'sequence': row['Sequence In Stage'],
                            'selected_original': row['Selected Actuators'],
                            'actual_original': row['Actual Actuators'],
                            'selected_list': selected_list,
                            'actual_list': actual_list
                        })
                
                print(f"  Stage {stage}: {len(mismatches)} mismatches out of {len(stage_data)} sequences")
                
                if len(mismatches) > 0:
                    # Add to overall collection
                    all_mismatches[f'stage_{stage}']['selected'].extend(
                        [detail['selected_list'] for detail in mismatch_details]
                    )
                    all_mismatches[f'stage_{stage}']['actual'].extend(
                        [detail['actual_list'] for detail in mismatch_details]
                    )
        
        self.mismatched_sequences = all_mismatches
        return all_mismatches

#     def create_direction_confusion_matrices(self):
#         if not self.direction_analysis:
#             print("No direction analysis found. Run analyze_direction_accuracy() first.")
#             return
#                 
#         confusion_results = {}
#         
#         for stage_key, data in self.direction_analysis.items():
#             stage_num = stage_key.split('_')[1]
#             sequences = data['sequences']
#             
#             print(f"\n{'='*50}")
#             print(f"STAGE {stage_num} - DIRECTION CONFUSION MATRIX")
#             print(f"{'='*50}")
#             
#             # Get all unique directions that appear in actual sequences
#             all_directions = sorted(list(set([seq['actual_direction'] for seq in sequences])))
#             
#             # Create confusion matrix using the existing 'correct' field
#             confusion_matrix = defaultdict(lambda: defaultdict(int))
#             direction_totals = defaultdict(int)
#             direction_correct = defaultdict(int)
#             
#             # Populate confusion matrix
#             for seq in sequences:
#                 actual_dir = seq['actual_direction']
#                 selected_dir = seq['selected_direction']
#                 is_correct = seq['correct']  # Use the pre-calculated correct field
#                 
#                 confusion_matrix[actual_dir][selected_dir] += 1
#                 direction_totals[actual_dir] += 1
#                 
#                 # Count correct using the original 'correct' field
#                 if is_correct:
#                     direction_correct[actual_dir] += 1
#             
#             # Display confusion matrix
#             print(f"\nConfusion Matrix (Rows = Actual Direction, Columns = Selected Direction):")
#             print("-" * 80)
#             
#             # Header
#             print(f"{'Actual // Selected':<15}", end="")
#             for direction in all_directions:
#                 print(f"{direction.capitalize():<10}", end="")
#             print(f"{'Total':<8}{'True Acc%':<10}")
#             print("-" * 80)
#             
#             # Matrix rows
#             matrix_data = {}
#             for actual_dir in all_directions:
#                 matrix_data[actual_dir] = {}
#                 print(f"{actual_dir.capitalize():<15}", end="")
#                 
#                 row_total = direction_totals[actual_dir]
#                 correct_count = direction_correct[actual_dir]  # Use pre-calculated correct count
#                 
#                 for selected_dir in all_directions:
#                     count = confusion_matrix[actual_dir][selected_dir]
#                     matrix_data[actual_dir][selected_dir] = count
#                     print(f"{count:<10}", end="")
#                 
#                 true_accuracy = (correct_count / row_total * 100) if row_total > 0 else 0
#                 print(f"{row_total:<8}{true_accuracy:<10.1f}")
#             
#             confusion_results[stage_key] = {
#                 'confusion_matrix': dict(confusion_matrix),
#                 'matrix_data': matrix_data,
#                 'direction_totals': dict(direction_totals),
#                 'direction_correct': dict(direction_correct)  # Store the correct counts
#             }
#         
#         self.confusion_matrices = confusion_results
#         return confusion_results

    def create_direction_confusion_matrices(self):
        """
        Create confusion matrix based on direction classifications
        """
        if not self.direction_analysis:
            print("No direction analysis found. Run analyze_direction_accuracy() first.")
            return
                
        confusion_results = {}
        
        for stage_key, data in self.direction_analysis.items():
            stage_num = stage_key.split('_')[1]
            sequences = data['sequences']
            
            print(f"\n{'='*50}")
            print(f"STAGE {stage_num} - DIRECTION CONFUSION MATRIX")
            print(f"{'='*50}")
            
            # Get all unique directions that appear in actual sequences
            all_directions = sorted(list(set([seq['actual_direction'] for seq in sequences])))
            print(all_directions)
            # Create confusion matrix - this should count direction matches, not exact sequence matches
            confusion_matrix = defaultdict(lambda: defaultdict(int))
            direction_totals = defaultdict(int)
            direction_matches = defaultdict(int)  # Count where actual_dir == selected_dir
            exact_matches = defaultdict(int)      # Count where sequences are exactly the same
            
            # Populate confusion matrix
            for seq in sequences:
                actual_dir = seq['actual_direction']
                selected_dir = seq['selected_direction']
                
                # This is what goes in the confusion matrix
                confusion_matrix[actual_dir][selected_dir] += 1
                direction_totals[actual_dir] += 1
                
                # Count diagonal elements (where directions match)
                if actual_dir == selected_dir:
                    direction_matches[actual_dir] += 1
                
                # Count exact sequence matches (for reference)
                if seq['correct']:
                    exact_matches[actual_dir] += 1
            
            # Calculate total diagonal elements
            total_diagonal = sum(direction_matches.values())
            total_sequences = len(sequences)
            total_exact_matches = sum(exact_matches.values())
            
            print(f"\nTotal sequences: {total_sequences}")
            print(f"Total diagonal elements (direction matches): {total_diagonal}")
            print(f"Total exact sequence matches: {total_exact_matches}")
            print(f"Overall direction accuracy: {total_diagonal/total_sequences*100:.1f}%")
            print(f"Overall exact sequence accuracy: {total_exact_matches/total_sequences*100:.1f}%")
            
            # Display confusion matrix
            print(f"\nConfusion Matrix (Rows = Actual Direction, Columns = Selected Direction):")
            print("-" * 90)
            
            # Header
            print(f"{'Actual // Selected':<15}", end="")
            for direction in all_directions:
                print(f"{direction.capitalize():<10}", end="")
            print(f"{'Total':<8}{'Dir Acc%':<10}{'Exact Acc%':<12}")
            print("-" * 90)
            
            # Matrix rows
            matrix_data = {}
            for actual_dir in all_directions:
                matrix_data[actual_dir] = {}
                print(f"{actual_dir.capitalize():<15}", end="")
                
                row_total = direction_totals[actual_dir]
                dir_correct = direction_matches[actual_dir]
                exact_correct = exact_matches[actual_dir]
                
                for selected_dir in all_directions:
                    count = confusion_matrix[actual_dir][selected_dir]
                    matrix_data[actual_dir][selected_dir] = count
                    print(f"{count:<10}", end="")
                
                dir_accuracy = (dir_correct / row_total * 100) if row_total > 0 else 0
                exact_accuracy = (exact_correct / row_total * 100) if row_total > 0 else 0
                print(f"{row_total:<8}{dir_accuracy:<10.1f}{exact_accuracy:<12.1f}")
            
            confusion_results[stage_key] = {
                'confusion_matrix': dict(confusion_matrix),
                'matrix_data': matrix_data,
                'direction_totals': dict(direction_totals),
                'direction_matches': dict(direction_matches),  # Diagonal elements
                'exact_matches': dict(exact_matches),
                'total_diagonal': total_diagonal,
                'total_exact_matches': total_exact_matches,
                'total_sequences': total_sequences
            }
        
        self.confusion_matrices = confusion_results
        return confusion_results

    def plot_confusion_matrices_heatmap(self):
        """
        Create heatmap visualizations of the confusion matrices
        """
        if not hasattr(self, 'confusion_matrices'):
            print("No confusion matrices found. Run create_direction_confusion_matrices() first.")
            return
        
        n_stages = len(self.confusion_matrices)
        fig, axes = plt.subplots(1, n_stages, figsize=(8*n_stages, 6))
        
        if n_stages == 1:
            axes = [axes]
        
        for idx, (stage_key, data) in enumerate(self.confusion_matrices.items()):
            ax = axes[idx]
            stage_num = stage_key.split('_')[1]
            matrix_data = data['matrix_data']
            
            # Convert to DataFrame for heatmap
            directions = list(matrix_data.keys())
            all_selected = set()
            for actual_dir in directions:
                all_selected.update(matrix_data[actual_dir].keys())
            all_selected = sorted(list(all_selected))
            
            # Create matrix array
            matrix_array = np.zeros((len(directions), len(all_selected)))
            for i, actual_dir in enumerate(directions):
                for j, selected_dir in enumerate(all_selected):
                    matrix_array[i, j] = matrix_data[actual_dir].get(selected_dir, 0)
            
            # Create heatmap
            im = ax.imshow(matrix_array, cmap='Blues', aspect='auto')
            
            # Set labels
            ax.set_xticks(range(len(all_selected)))
            ax.set_yticks(range(len(directions)))
            ax.set_xticklabels([d.capitalize() for d in all_selected], rotation=45, ha='right')
            ax.set_yticklabels([d.capitalize() for d in directions])
            
            ax.set_xlabel('Selected Direction')
            ax.set_ylabel('Actual Direction')
            ax.set_title(f'Stage {stage_num} - Direction Confusion Matrix')
            
            # Add text annotations
            for i in range(len(directions)):
                for j in range(len(all_selected)):
                    count = int(matrix_array[i, j])
                    if count > 0:
                        # Color text based on whether it's on diagonal (correct) or off-diagonal (error)
                        text_color = 'white' if (i == j and count > matrix_array.max()/2) else 'black'
                        ax.text(j, i, str(count), ha='center', va='center', color=text_color, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
            
        plt.tight_layout()
        plt.show()
        
    def analyze_true_vs_apparent_accuracy(self):
        """
        Compare apparent accuracy (from original analysis) with true directional accuracy
        """
        if not hasattr(self, 'confusion_matrices'):
            print("No confusion matrices found. Run create_direction_confusion_matrices() first.")
            return
        
        print("\n" + "="*70)
        print("TRUE vs APPARENT ACCURACY COMPARISON")
        print("="*70)
        
        comparison_results = {}
        
        for stage_key, confusion_data in self.confusion_matrices.items():
            stage_num = stage_key.split('_')[1]
            matrix_data = confusion_data['matrix_data']
            direction_totals = confusion_data['direction_totals']
            
            print(f"\nSTAGE {stage_num}:")
            print("-" * 50)
            print(f"{'Direction':<12}{'Apparent Acc':<15}{'True Acc':<12}{'Difference':<12}{'Issue Detected'}")
            print("-" * 50)
            
            stage_results = {}
            
            for direction in matrix_data.keys():
                # Get apparent accuracy from original analysis
                apparent_acc = 0
                if hasattr(self, 'direction_analysis') and stage_key in self.direction_analysis:
                    if direction in self.direction_analysis[stage_key]['direction_stats']:
                        apparent_acc = self.direction_analysis[stage_key]['direction_stats'][direction]['accuracy']
                
                # Calculate true accuracy from confusion matrix
                total = direction_totals[direction]
                correct = matrix_data[direction].get(direction, 0)
                true_acc = (correct / total * 100) if total > 0 else 0
                
                difference = apparent_acc - true_acc
                
                # Detect issues
                issue = ""
                if abs(difference) > 5:  # More than 5% difference
                    issue = "⚠️ MISMATCH"
                
                # Special check for vertical directions
                if direction in ['up', 'down']:
                    # Check for opposite direction confusion
                    opposite = 'down' if direction == 'up' else 'up'
                    if opposite in matrix_data[direction]:
                        confusion_count = matrix_data[direction][opposite]
                        if confusion_count > 0:
                            issue += " 🔄 VERTICAL_CONFUSION"
                
                print(f"{direction.capitalize():<12}{apparent_acc:<15.1f}{true_acc:<12.1f}{difference:<12.1f}{issue}")
                
                stage_results[direction] = {
                    'apparent_accuracy': apparent_acc,
                    'true_accuracy': true_acc,
                    'difference': difference,
                    'total_trials': total,
                    'correct_trials': correct,
                    'issue_detected': issue
                }
            
            comparison_results[stage_key] = stage_results
        
        self.accuracy_comparison = comparison_results
        return comparison_results
    
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ActuatorDataAnalyzer()
    
    # Step 1: Read all CSV files
    print("Step 1: Reading CSV files...")
    analyzer.read_csv_files("*.csv")
    
    # Step 2: Display data structure
    print("\nStep 2: Displaying data structure...")
    analyzer.display_data_structure()
    
    # Step 3: Find actuator mismatches (EXCLUDING STAGE 3)
    print("\nStep 3: Finding actuator mismatches (Stages 1, 2, 4 only)...")
    analyzer.find_actuator_mismatches()
    
    # Step 4: Calculate stage accuracies (EXCLUDING STAGE 3)
    print("\nStep 4: Calculating stage accuracies (Stages 1, 2, 4 only)...")
    analyzer.calculate_stage_accuracies()
    
    # Step 5: Analyze vertical actuator confusion
    print("\nStep 5: Analyzing vertical actuator confusion...")
    analyzer.analyze_vertical_actuator_confusion()
     
    # Step 6: Direction-based analysis
    print("\nStep 6: Analyzing accuracy by direction (Stage 1 only)...")
    analyzer.analyze_direction_accuracy()
    
#     # Step 7: Chi-square tests for direction differences
#     print("\nStep 7: Chi-square tests for direction differences...")
#     analyzer.chi_square_direction_test()
#     
#     # Step 8: Statistical analysis across stages (EXCLUDING STAGE 3)
#     print("\nStep 8: Statistical analysis across stages (Stages 1, 2, 4 only)...")
#     analyzer.statistical_analysis_across_stages()
    
#     # Step 9: Create visualizations
#     print("\nStep 9: Creating visualizations...")
#     analyzer.plot_direction_accuracy_comparison()
    
#     # Step 10: Create detailed confusion matrices
#     print("\nStep 10: Creating detailed direction confusion matrices...")
#     analyzer.create_direction_confusion_matrices()
#     
#     # Step 11: Plot confusion matrix heatmaps
#     print("\nStep 11: Plotting confusion matrix heatmaps...")
#     analyzer.plot_confusion_matrices_heatmap()
#     
#     # Step 12: Analyze true vs apparent accuracy
#     print("\nStep 12: Comparing true vs apparent accuracy...")
#     analyzer.analyze_true_vs_apparent_accuracy()
