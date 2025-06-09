'''
Enhanced data analysis for haptic experiment on wearable biomechatronic device
to provide haptic feedback on SRL end effector position
Orthogonal directions only (5x5 confusion matrix)
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
from scipy import stats
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
        
        # CORRECTED: Only pure orthogonal directions
        self.direction_patterns = {
            'right': [[2,1,4], [2,4]],
            'left': [[4,1,2], [4,2]],
            'down': [[1,5,6]],
            'up': [[6,5,1]],
            'back': [[1,3], [1,2,3]],
            'forward': [[3,1], [3,2,1]]
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
        
        print(f"Found {len(csv_files)} CSV files:")
        
        for i, file_path in enumerate(csv_files):
            try:
                # Read CSV with proper handling of string columns
                df = pd.read_csv(file_path, dtype={'Selected Actuators': str, 'Actual Actuators': str})
                
                # Convert timestamp to datetime
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y %H:%M')
                
                # Store dataframe
                file_key = f"file_{i+1}_{os.path.basename(file_path)}"
                self.dataframes[file_key] = df
                
                print(f"  {i+1}. {file_path} - {len(df)} rows")
                
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
        
        print(f"\nSuccessfully loaded {len(self.dataframes)} files")
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
        print(f"\nData structure for {file_key}:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
    
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
        Only returns orthogonal directions or 'unknown'
        """
        if not actuator_list:
            return 'unknown'
        
        for direction, patterns in self.direction_patterns.items():
            for pattern in patterns:
                if actuator_list == pattern:
                    return direction
        
        # If no match found, return unknown (includes diagonal and other patterns)
        return 'unknown'
    
    def analyze_direction_accuracy(self, stages=[1]):
        """
        Analyze accuracy by direction for specified stages (orthogonal directions only)
        """
        print("\n" + "="*60)
        print("DIRECTION-BASED ACCURACY ANALYSIS (ORTHOGONAL DIRECTIONS ONLY)")
        print("="*60)
        
        direction_results = {}
        
        for stage in stages:
            print(f"\nAnalyzing Stage {stage}...")
            
            # Collect all sequences for this stage
            all_sequences = []
            unknown_sequences = []  # Track sequences that don't match orthogonal patterns
            
            for file_key, df in self.dataframes.items():
                stage_data = df[df['Stage'] == stage]
                
                for _, row in stage_data.iterrows():
                    selected_list = self.parse_actuators_to_list(row['Selected Actuators'])
                    actual_list = self.parse_actuators_to_list(row['Actual Actuators'])
                    
                    # Classify directions
                    selected_direction = self.classify_direction(selected_list)
                    actual_direction = self.classify_direction(actual_list)
                    
                    # Only include orthogonal sequences (exclude 'unknown')
                    if actual_direction != 'unknown':
                        all_sequences.append({
                            'actual_direction': actual_direction,
                            'selected_direction': selected_direction,
                            'actual_sequence': actual_list,
                            'selected_sequence': selected_list,
                            'correct': selected_list == actual_list,  # Exact sequence match
                            'direction_match': actual_direction == selected_direction  # Direction match
                        })
                    else:
                        unknown_sequences.append({
                            'actual_sequence': actual_list,
                            'selected_sequence': selected_list
                        })
            
            if not all_sequences:
                print(f"No orthogonal sequences found for Stage {stage}")
                continue
                
            print(f"Found {len(all_sequences)} orthogonal sequences")
            if unknown_sequences:
                print(f"Excluded {len(unknown_sequences)} non-orthogonal sequences:")
                for seq in unknown_sequences[:5]:  # Show first 5
                    print(f"  Actual: {seq['actual_sequence']}, Selected: {seq['selected_sequence']}")
                if len(unknown_sequences) > 5:
                    print(f"  ... and {len(unknown_sequences) - 5} more")
            
            # Calculate accuracy by direction
            direction_stats = {}
            
            for direction in self.direction_patterns.keys():
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
                'direction_stats': direction_stats,
                'unknown_sequences': unknown_sequences
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

    def create_direction_confusion_matrices(self):
        """
        Create 5x5 confusion matrix for pure orthogonal directions only
        """
        if not self.direction_analysis:
            print("No direction analysis found. Run analyze_direction_accuracy() first.")
            return
                
        confusion_results = {}
        
        # Define the order of orthogonal directions for consistent matrix display
        orthogonal_directions = ['up', 'down', 'left', 'right', 'back', 'forward']
        
        for stage_key, data in self.direction_analysis.items():
            stage_num = stage_key.split('_')[1]
            sequences = data['sequences']
            
            print(f"\n{'='*50}")
            print(f"STAGE {stage_num} - ORTHOGONAL DIRECTIONS CONFUSION MATRIX (5x5)")
            print(f"{'='*50}")
            
            # Get all unique directions that appear in actual sequences (should be orthogonal only)
            actual_directions_in_data = sorted(list(set([seq['actual_direction'] for seq in sequences])))
            print(f"Orthogonal directions found in data: {actual_directions_in_data}")
            
            # Create confusion matrix
            confusion_matrix = defaultdict(lambda: defaultdict(int))
            direction_totals = defaultdict(int)
            direction_matches = defaultdict(int)
            exact_matches = defaultdict(int)
            
            # Count sequences where selected direction is 'unknown' (wrong direction)
            wrong_direction_details = defaultdict(list)
            
            # Populate confusion matrix
            for seq in sequences:
                actual_dir = seq['actual_direction']
                selected_dir = seq['selected_direction']
                
                # Count all combinations (including 'others' selected directions)
                confusion_matrix[actual_dir][selected_dir] += 1
                direction_totals[actual_dir] += 1
                
                # Track wrong direction details
                if selected_dir == 'unknown':
                    wrong_direction_details[actual_dir].append({
                        'actual_sequence': seq['actual_sequence'],
                        'selected_sequence': seq['selected_sequence']
                    })
                
                # Count diagonal elements (where directions match)
                if actual_dir == selected_dir:
                    direction_matches[actual_dir] += 1
                
                # Count exact sequence matches
                if seq['correct']:
                    exact_matches[actual_dir] += 1
            
            # Calculate totals
            total_sequences = len(sequences)
            total_diagonal = sum(direction_matches.values())
            total_exact_matches = sum(exact_matches.values())
            
            print(f"\nTotal orthogonal sequences: {total_sequences}")
            print(f"Total diagonal elements (direction matches): {total_diagonal}")
            print(f"Total exact sequence matches: {total_exact_matches}")
            print(f"Overall direction accuracy: {total_diagonal/total_sequences*100:.1f}%")
            print(f"Overall exact sequence accuracy: {total_exact_matches/total_sequences*100:.1f}%")
            
            # Display confusion matrix (including 'unknown' column for wrong directions)
            print(f"\nConfusion Matrix (Rows = Actual Direction, Columns = Selected Direction):")
            print("-" * 100)
            
            # Use consistent ordering for both rows and columns to align diagonal
            display_directions = [d for d in orthogonal_directions if d in actual_directions_in_data]
            # For columns, use same order + 'others'
            column_directions = display_directions
            
            # Header
            print(f"{'Actual // Selected':<15}", end="")
            for direction in column_directions:
                print(f"{direction.capitalize():<10}", end="")
            print(f"{'Total':<8}{'Dir Acc%':<10}{'Exact Acc%':<12}")
            print("-" * 100)
            
            # Matrix rows
            matrix_data = {}
            for actual_dir in display_directions:
                matrix_data[actual_dir] = {}
                print(f"{actual_dir.capitalize():<15}", end="")
                
                row_total = direction_totals[actual_dir]
                dir_correct = direction_matches[actual_dir]
                exact_correct = exact_matches[actual_dir]
                
                for selected_dir in column_directions:
                    count = confusion_matrix[actual_dir][selected_dir]
                    matrix_data[actual_dir][selected_dir] = count
                    print(f"{count:<10}", end="")
                
                dir_accuracy = (dir_correct / row_total * 100) if row_total > 0 else 0
                exact_accuracy = (exact_correct / row_total * 100) if row_total > 0 else 0
                print(f"{row_total:<8}{dir_accuracy:<10.1f}{exact_accuracy:<12.1f}")
            
            # Show details of wrong directions
            print(f"\nDetails of Wrong Directions (Selected as 'Others'):")
            print("-" * 60)
            for actual_dir, wrong_list in wrong_direction_details.items():
                if wrong_list:
                    print(f"\n{actual_dir.capitalize()} direction misidentified as others:")
                    for i, item in enumerate(wrong_list[:3]):  # Show first 3 examples
                        print(f"  Example {i+1}: Actual {item['actual_sequence']} -> Selected {item['selected_sequence']}")
                    if len(wrong_list) > 3:
                        print(f"  ... and {len(wrong_list) - 3} more cases")
                  
            confusion_results[stage_key] = {
                'confusion_matrix': dict(confusion_matrix),
                'matrix_data': matrix_data,
                'direction_totals': dict(direction_totals),
                'direction_matches': dict(direction_matches),
                'exact_matches': dict(exact_matches),
                'wrong_direction_details': dict(wrong_direction_details),
                'total_diagonal': total_diagonal,
                'total_exact_matches': total_exact_matches,
                'total_sequences': total_sequences,
                'display_directions': display_directions,
                'column_directions': column_directions
            }
        
        self.confusion_matrices = confusion_results
        return confusion_results

    def plot_confusion_matrices_heatmap(self):
        """
        Create heatmap visualizations of the 5x5 confusion matrices
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
            display_directions = data['display_directions']
            column_directions = data['column_directions']
            
            # Create matrix array
            matrix_array = np.zeros((len(display_directions), len(column_directions)))
            for i, actual_dir in enumerate(display_directions):
                for j, selected_dir in enumerate(column_directions):
                    matrix_array[i, j] = matrix_data[actual_dir].get(selected_dir, 0)
            
            # Create heatmap
            im = ax.imshow(matrix_array, cmap='Blues', aspect='auto')
            
            # Set labels with increased fontsize
            ax.set_xticks(range(len(column_directions)))
            ax.set_yticks(range(len(display_directions)))
            ax.set_xticklabels([d.capitalize() for d in column_directions], rotation=45, ha='right', fontsize=14)  # Increased tick label fontsize
            ax.set_yticklabels([d.capitalize() for d in display_directions], fontsize=14)  # Increased tick label fontsize
            
            ax.set_xlabel('Selected Direction', fontsize=16)  # Increased axes label fontsize
            ax.set_ylabel('Actual Direction', fontsize=16)    # Increased axes label fontsize
            
            # Add text annotations
            for i in range(len(display_directions)):
                for j in range(len(column_directions)):
                    count = int(matrix_array[i, j])
                    if count > 0:
                        # Highlight diagonal elements (correct directions)
                        if i < len(column_directions) and display_directions[i] == column_directions[j]:
                            ax.text(j, i, str(count), ha='center', va='center', 
                                   color='white', fontweight='bold', fontsize=16)  # Diagonal numbers: white, bold, larger fontsize
                        else:
                            ax.text(j, i, str(count), ha='center', va='center', 
                                   color='black', fontsize=10)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
            
        plt.tight_layout()
        plt.show()
        
    def analyze_true_vs_apparent_accuracy(self):
        """
        Compare apparent accuracy with true directional accuracy for orthogonal directions
        """
        if not hasattr(self, 'confusion_matrices'):
            print("No confusion matrices found. Run create_direction_confusion_matrices() first.")
            return
        
        print("\n" + "="*70)
        print("TRUE vs APPARENT ACCURACY COMPARISON (ORTHOGONAL DIRECTIONS)")
        print("="*70)
        
        comparison_results = {}
        
        for stage_key, confusion_data in self.confusion_matrices.items():
            stage_num = stage_key.split('_')[1]
            matrix_data = confusion_data['matrix_data']
            direction_totals = confusion_data['direction_totals']
            
            print(f"\nSTAGE {stage_num}:")
            print("-" * 70)
            print(f"{'Direction':<12}{'Apparent Acc':<15}{'True Acc':<12}{'Difference':<12}{'Issue Detected'}")
            print("-" * 70)
            
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
                
                # Check for others direction selections
                others_count = matrix_data[direction].get('others', 0)
                if others_count > 0:
                    issue += f" ❓ {others_count}_OTHERS"
                
                print(f"{direction.capitalize():<12}{apparent_acc:<15.1f}{true_acc:<12.1f}{difference:<12.1f}{issue}")
                
                stage_results[direction] = {
                    'apparent_accuracy': apparent_acc,
                    'true_accuracy': true_acc,
                    'difference': difference,
                    'total_trials': total,
                    'correct_trials': correct,
                    'unknown_selections': others_count,
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
     
    # Step 6: Direction-based analysis
    print("\nStep 6: Analyzing accuracy by direction (Stage 1 only)...")
    analyzer.analyze_direction_accuracy()
    
    # Step 10: Create detailed confusion matrices
    print("\nStep 10: Creating detailed direction confusion matrices...")
    analyzer.create_direction_confusion_matrices()
    
    # Step 11: Plot confusion matrix heatmaps
    print("\nStep 11: Plotting confusion matrix heatmaps...")
    analyzer.plot_confusion_matrices_heatmap()
    
    # Step 12: Analyze true vs apparent accuracy
    print("\nStep 12: Comparing true vs apparent accuracy...")
    analyzer.analyze_true_vs_apparent_accuracy()