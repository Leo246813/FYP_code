'''
Experiment 2 data analysis and visualization:
Calculate accuracies of responses and perform statistical analysis
Create confusion matrices for directions and shape patterns
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import glob
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

class DataAnalyzer:
    def __init__(self, file_pattern="*_data.csv"):
        self.file_pattern = file_pattern
        self.data_files = []
        self.combined_data = None
        self.stage1_accuracies = []
        self.stage2_accuracies = []
        self.overall_accuracies = []
        
    def load_data(self):
        """Load all CSV files matching the pattern"""
        self.data_files = glob.glob(self.file_pattern)
        print(f"Found {len(self.data_files)} CSV files: {self.data_files}")
        
        if len(self.data_files) == 0:
            raise FileNotFoundError(f"No files found matching pattern: {self.file_pattern}")
        
        all_dataframes = []
        for i, file in enumerate(self.data_files):
            try:
                df = pd.read_csv(file)
                # Clean and standardize data
                df = df.fillna('/')  # Replace NaN with '/' for consistency
                
                # Ensure string columns are properly formatted
                string_columns = ['Selected Direction/Shape', 'Actual Direction', 'Actual Shape']
                for col in string_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip()
                        # Replace 'nan' string with '/'
                        df[col] = df[col].replace(['nan', 'NaN', 'None'], '/')
                
                df['File_ID'] = i + 1
                df['File_Name'] = file
                all_dataframes.append(df)
                print(f"Loaded {file}: {len(df)} rows")
                
                # Debug: Print first few rows to check data format
                print(f"  Sample data from {file}:")
                print(df[['Stage', 'Selected Direction/Shape', 'Actual Direction', 'Actual Shape']].head(3))
                
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        
        if not all_dataframes:
            raise Exception("No valid CSV files could be loaded")
            
        self.combined_data = pd.concat(all_dataframes, ignore_index=True)
        print(f"\nCombined dataset: {len(self.combined_data)} rows")
        
        # Print data types and unique values for debugging
        print(f"\nData types:")
        for col in ['Selected Direction/Shape', 'Actual Direction', 'Actual Shape']:
            if col in self.combined_data.columns:
                print(f"  {col}: {self.combined_data[col].dtype}")
                print(f"    Unique values: {sorted(self.combined_data[col].unique())}")
        
        return self.combined_data
    
    def calculate_accuracies(self):
        """Calculate accuracy metrics for each file"""
        print("\n" + "="*50)
        print("ACCURACY CALCULATIONS")
        print("="*50)
        
        for file_id in self.combined_data['File_ID'].unique():
            file_data = self.combined_data[self.combined_data['File_ID'] == file_id]
            file_name = file_data['File_Name'].iloc[0]
            
            # Stage 1 accuracy (directions) - exclude rows with '/' values
            stage1_data = file_data[file_data['Stage'] == 1]
            stage1_valid = stage1_data[
                (stage1_data['Selected Direction/Shape'] != '/') & 
                (stage1_data['Actual Direction'] != '/')
            ]
            
            if len(stage1_valid) > 0:
                stage1_matches = (stage1_valid['Selected Direction/Shape'] == stage1_valid['Actual Direction']).sum()
                stage1_accuracy = stage1_matches / len(stage1_valid)
            else:
                stage1_accuracy = 0
            
            # Stage 2 accuracy (shapes) - exclude rows with '/' values
            stage2_data = file_data[file_data['Stage'] == 2]
            stage2_valid = stage2_data[
                (stage2_data['Selected Direction/Shape'] != '/') & 
                (stage2_data['Actual Shape'] != '/')
            ]
            
            if len(stage2_valid) > 0:
                stage2_matches = (stage2_valid['Selected Direction/Shape'] == stage2_valid['Actual Shape']).sum()
                stage2_accuracy = stage2_matches / len(stage2_valid)
            else:
                stage2_accuracy = 0
            
            # Overall accuracy
            total_valid = len(stage1_valid) + len(stage2_valid)
            total_matches = (stage1_matches if len(stage1_valid) > 0 else 0) + (stage2_matches if len(stage2_valid) > 0 else 0)
            overall_accuracy = total_matches / total_valid if total_valid > 0 else 0
            
            self.stage1_accuracies.append(stage1_accuracy)
            self.stage2_accuracies.append(stage2_accuracy)
            self.overall_accuracies.append(overall_accuracy)
            
            print(f"\n{file_name}:")
            print(f"  Stage 1 (Directions): {stage1_accuracy:.3f} ({stage1_matches}/{len(stage1_valid)})")
            print(f"  Stage 2 (Shapes): {stage2_accuracy:.3f} ({stage2_matches}/{len(stage2_valid)})")
            print(f"  Overall: {overall_accuracy:.3f} ({total_matches}/{total_valid})")
    
    def statistical_analysis(self):
        """Perform statistical analysis on accuracies"""
        print("\n" + "="*50)
        print("STATISTICAL ANALYSIS")
        print("="*50)
        
        accuracies_dict = {
            'Stage 1 (Directions)': self.stage1_accuracies,
            'Stage 2 (Shapes)': self.stage2_accuracies,
            'Overall': self.overall_accuracies
        }
        
        for name, accuracies in accuracies_dict.items():
            print(f"\n{name} Accuracies:")
            print(f"  Mean: {np.mean(accuracies):.3f}")
            print(f"  Std: {np.std(accuracies, ddof=1):.3f}")
            print(f"  Min: {np.min(accuracies):.3f}")
            print(f"  Max: {np.max(accuracies):.3f}")
            print(f"  Median: {np.median(accuracies):.3f}")
            
            # Normality tests for small samples
            if len(accuracies) >= 3:
                # Shapiro-Wilk test (good for small samples)
                shapiro_stat, shapiro_p = stats.shapiro(accuracies)
                print(f"  Shapiro-Wilk test: statistic={shapiro_stat:.3f}, p-value={shapiro_p:.3f}")
                
                # Anderson-Darling test
                try:
                    ad_result = stats.anderson(accuracies, dist='norm')
                    print(f"  Anderson-Darling statistic: {ad_result.statistic:.3f}")
                except:
                    print("  Anderson-Darling test: Not applicable")
        
        # Compare Stage 1 vs Stage 2 accuracies
        if len(self.stage1_accuracies) >= 3 and len(self.stage2_accuracies) >= 3:
            print(f"\nComparison between Stage 1 and Stage 2:")
            try:
                # Wilcoxon signed-rank test (non-parametric, good for small samples)
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(self.stage1_accuracies, self.stage2_accuracies)
                print(f"  Wilcoxon signed-rank test: statistic={wilcoxon_stat:.3f}, p-value={wilcoxon_p:.3f}")
            except Exception as e:
                print(f"  Wilcoxon test failed: {e}")
            
            try:
                # Paired t-test
                ttest_stat, ttest_p = stats.ttest_rel(self.stage1_accuracies, self.stage2_accuracies)
                print(f"  Paired t-test: statistic={ttest_stat:.3f}, p-value={ttest_p:.3f}")
            except Exception as e:
                print(f"  T-test failed: {e}")
        else:
            print(f"\nNot enough data points for statistical comparison (need at least 3)")
            print(f"Stage 1 data points: {len(self.stage1_accuracies)}")
            print(f"Stage 2 data points: {len(self.stage2_accuracies)}")
    
    def create_confusion_matrices(self):
        """Create confusion matrices for directions and shapes"""
        print("\n" + "="*50)
        print("CONFUSION MATRICES")
        print("="*50)
        
        # Stage 1 - Directions
        stage1_data = self.combined_data[self.combined_data['Stage'] == 1]
        stage1_valid = stage1_data[
            (stage1_data['Selected Direction/Shape'] != 'Diagonal') & 
            (stage1_data['Actual Direction'] != 'Diagonal')&
            (stage1_data['Selected Direction/Shape'] != '/') & 
            (stage1_data['Actual Direction'] != '/')
        ]
        
        if len(stage1_valid) > 0:            
            y_true_directions = stage1_valid['Actual Direction'].astype(str)
            y_pred_directions = stage1_valid['Selected Direction/Shape'].astype(str)
            
            print("\nStage 1 - Directions Confusion Matrix:")
            cm_directions = confusion_matrix(y_true_directions, y_pred_directions)
            labels_directions = sorted(list(set(y_true_directions) | set(y_pred_directions)))
            
            # Create confusion matrix plot with larger fonts
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_directions, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels_directions, yticklabels=labels_directions,
                       annot_kws={'size': 20})
            plt.xlabel('Predicted Direction', fontsize=20)
            plt.ylabel('Actual Direction', fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.show()
            
            print("\nClassification Report - Directions:")
            try:
                print(classification_report(y_true_directions, y_pred_directions))
            except Exception as e:
                print(f"Could not generate classification report: {e}")
                print(f"Accuracy: {accuracy_score(y_true_directions, y_pred_directions):.3f}")
        
        # Stage 2 - Shapes
        stage2_data = self.combined_data[self.combined_data['Stage'] == 2]
        stage2_valid = stage2_data[
            (stage2_data['Selected Direction/Shape'] != '/') & 
            (stage2_data['Actual Shape'] != '/')
        ]
        
        if len(stage2_valid) > 0:
            y_true_shapes = stage2_valid['Actual Shape'].astype(str)
            y_pred_shapes = stage2_valid['Selected Direction/Shape'].astype(str)
            
            print("\nStage 2 - Shapes Confusion Matrix:")
            cm_shapes = confusion_matrix(y_true_shapes, y_pred_shapes)
            labels_shapes = sorted(list(set(y_true_shapes) | set(y_pred_shapes)))
            
            # Create confusion matrix plot with larger fonts
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_shapes, annot=True, fmt='d', cmap='Greens',
                       xticklabels=labels_shapes, yticklabels=labels_shapes,
                       annot_kws={'size': 20})
            plt.xlabel('Predicted Shape', fontsize=20)
            plt.ylabel('Actual Shape', fontsize=20)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.tight_layout()
            plt.show()
            
            print("\nClassification Report - Shapes:")
            try:
                print(classification_report(y_true_shapes, y_pred_shapes))
            except Exception as e:
                print(f"Could not generate classification report: {e}")
                print(f"Accuracy: {accuracy_score(y_true_shapes, y_pred_shapes):.3f}")
            
    def create_accuracy_plots(self):
        """Create plots for accuracy analysis"""
        print("\n" + "="*50)
        print("ACCURACY VISUALIZATIONS")
        print("="*50)
        
        # Debug prints to check data
        print(f"Stage 1 accuracies: {len(self.stage1_accuracies)} items: {self.stage1_accuracies}")
        print(f"Stage 2 accuracies: {len(self.stage2_accuracies)} items: {self.stage2_accuracies}")
        print(f"Overall accuracies: {len(self.overall_accuracies)} items: {self.overall_accuracies}")
        
        # Check if data exists
        if not all([self.stage1_accuracies, self.stage2_accuracies, self.overall_accuracies]):
            print("Error: One or more accuracy arrays are empty")
            return
            
        # Check if all arrays have the same length
        lengths = [len(self.stage1_accuracies), len(self.stage2_accuracies), len(self.overall_accuracies)]
        if len(set(lengths)) > 1:
            print(f"Error: Array lengths don't match: {lengths}")
            return
        
        # Accuracy comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Individual accuracy scores
        file_names = [f"Participant {i+1}" for i in range(len(self.stage1_accuracies))]
                
        # FIXED: Convert to numpy arrays and multiply by 100 properly
        accuracy_data = [
            np.array(self.stage1_accuracies) * 100,
            np.array(self.stage2_accuracies) * 100,
            np.array(self.overall_accuracies) * 100
        ]
        
        # Box plot of accuracies
        bp = axes[0].boxplot(accuracy_data, labels=['Stage 1', 'Stage 2', 'Overall'], 
                            showmeans=True, meanline=False)
        axes[0].set_ylabel('Accuracy (%)', fontsize=16)
        axes[0].set_ylim(40, 100)  # Changed from 0.4, 1 to 40, 100 since we're now using percentages
        axes[0].tick_params(axis='x', labelsize=16)
        axes[0].tick_params(axis='y', labelsize=16)
        
        # Customize box plot elements
        plt.setp(bp['whiskers'], color='black', linewidth=2)
        plt.setp(bp['fliers'], marker='o', markerfacecolor='gray', markersize=6)
        plt.setp(bp['medians'], color='blue', linewidth=3)
        plt.setp(bp['caps'], color='black', linewidth=2)
        plt.setp(bp['boxes'], color='lightblue', linewidth=2)
        
        # Make means more visible (red diamonds)
        plt.setp(bp['means'], marker='D', markerfacecolor='red', 
                 markeredgecolor='darkred', markersize=8)
        
        # Add statistical annotations
        labels = ['Stage 1', 'Stage 2', 'Overall']
        for i, data in enumerate(accuracy_data):
            mean_val = np.mean(data)
            median_val = np.median(data)
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            print(f"{labels[i]} - Mean: {mean_val:.1f}%, Median: {median_val:.1f}%, IQR: {iqr:.1f}%")
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=3, label='Median'),
            Line2D([0], [0], marker='D', color='red', linewidth=0, 
                   markerfacecolor='red', markersize=8, label='Mean'),
            Line2D([0], [0], color='black', linewidth=1, label='IQR & Whiskers')
        ]
        axes[0].legend(handles=legend_elements, loc='lower right', fontsize=14)
        
        # Line plot showing accuracy trends - FIXED: Convert to percentages here too
        axes[1].plot(file_names, np.array(self.stage1_accuracies) * 100, 'o-', 
                    label='Stage 1 (Directions)', linewidth=2, markersize=6)
        axes[1].plot(file_names, np.array(self.stage2_accuracies) * 100, 's-', 
                    label='Stage 2 (Shapes)', linewidth=2, markersize=6)
        axes[1].plot(file_names, np.array(self.overall_accuracies) * 100, '^-', 
                    label='Overall', linewidth=2, markersize=6)
        axes[1].set_xlabel('Participants', fontsize=14)
        axes[1].set_ylabel('Accuracy (%)', fontsize=16)
        axes[1].tick_params(axis='x', labelsize=16)
        axes[1].tick_params(axis='y', labelsize=16)
        axes[1].legend(fontsize=14)
        axes[1].set_ylim(40, 100)  # Changed to match percentage scale
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def additional_insights(self):
        """Provide additional insights and recommendations"""
        print("\n" + "="*50)
        print("ADDITIONAL INSIGHTS & RECOMMENDATIONS")
        print("="*50)
        
        # Performance comparison
        stage1_mean = np.mean(self.stage1_accuracies)
        stage2_mean = np.mean(self.stage2_accuracies)
        
        print(f"\nPerformance Summary:")
        print(f"  Stage 1 (Directions) average accuracy: {stage1_mean:.1%}")
        print(f"  Stage 2 (Shapes) average accuracy: {stage2_mean:.1%}")
        
        if stage1_mean > stage2_mean:
            print(f"  → Directions are recognized {((stage1_mean/stage2_mean - 1) * 100):.1f}% better than shapes")
        else:
            print(f"  → Shapes are recognized {((stage2_mean/stage1_mean - 1) * 100):.1f}% better than directions")
        
        # Consistency analysis
        stage1_std = np.std(self.stage1_accuracies, ddof=1)
        stage2_std = np.std(self.stage2_accuracies, ddof=1)
        
        print(f"\nConsistency Analysis:")
        print(f"  Stage 1 standard deviation: {stage1_std:.3f}")
        print(f"  Stage 2 standard deviation: {stage2_std:.3f}")
        
        if stage1_std < stage2_std:
            print(f"  → Direction recognition is more consistent")
        else:
            print(f"  → Shape recognition is more consistent")
        
        # Sample size recommendations
        print(f"\nStatistical Power Considerations:")
        print(f"  Current sample size: {len(self.overall_accuracies)} files")
        print(f"  For small samples (n<30), non-parametric tests are recommended")
        print(f"  Consider collecting more data for stronger statistical conclusions")
        
        # Best and worst performing files
        best_file = np.argmax(self.overall_accuracies) + 1
        worst_file = np.argmin(self.overall_accuracies) + 1
        
        print(f"\nPerformance Extremes:")
        print(f"  Best performing file: File {best_file} ({self.overall_accuracies[best_file-1]:.1%})")
        print(f"  Worst performing file: File {worst_file} ({self.overall_accuracies[worst_file-1]:.1%})")

def main():
    """Main analysis function"""
    print("Starting CSV Data Analysis...")
    
    # Initialize analyzer
    analyzer = DataAnalyzer("*_data.csv")
    
    try:
        # Load and analyze data
        analyzer.load_data()
        analyzer.calculate_accuracies()
#         analyzer.statistical_analysis()
        analyzer.create_confusion_matrices()
#         analyzer.create_accuracy_plots()
#         analyzer.additional_insights()
        
        print(f"\n{'='*50}")
        print("ANALYSIS COMPLETE!")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check that your CSV files are in the correct format and location.")

if __name__ == "__main__":
    main()