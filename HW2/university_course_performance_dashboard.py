import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# PART 1: Data Generation with NumPy
print("Generating Data...")

num_students = 100

# 1. Generate Midterm and Final scores (Normal distribution, clipped 0-100)
# Mean=70, Std=15 is a reasonable assumption for grades
midterm_scores = np.random.normal(70, 15, num_students)
final_scores = np.random.normal(75, 12, num_students)

# Clip values to ensure they are within 0-100 range
midterm_scores = np.clip(midterm_scores, 0, 100)
final_scores = np.clip(final_scores, 0, 100)

# 2. Generate Assignment scores (500 values, reshaped to 100x5)
assignment_scores = np.random.normal(80, 10, 500)
assignment_scores = np.clip(assignment_scores, 0, 100)
assignment_scores = assignment_scores.reshape(num_students, 5)

# 3. Generate Departments
departments = np.random.choice(['CS', 'EE', 'PHYS', 'CHEM', 'BIO'], num_students)

# 4. Introduce Missing Values (NaNs) - 5%
def inject_nans(array, percent=0.05):
    flat_arr = array.ravel() # Flatten to handle 1D and 2D arrays
    n = len(flat_arr)
    num_nans = int(n * percent)
    missing_indices = np.random.choice(n, size=num_nans, replace=False)
    
    # We need to copy to avoid modifying original array type issues if integer
    new_arr = flat_arr.astype(float) if array.dtype.kind != 'U' else flat_arr.astype(object)
    
    if array.dtype.kind != 'U': # If numeric
        new_arr[missing_indices] = np.nan
    else: # If string/object
        new_arr[missing_indices] = np.nan
        
    return new_arr.reshape(array.shape)

midterm_scores = inject_nans(midterm_scores)
final_scores = inject_nans(final_scores)
assignment_scores = inject_nans(assignment_scores)
# Note: Injecting NaNs into string array requires object type conversion in Pandas later
departments = inject_nans(departments) 

# PART 2: DataFrames with Pandas
print("Processing DataFrames...")

# Create Student IDs
student_ids = np.arange(1, num_students + 1)

# Create initial Dictionary
data = {
    'StudentID': student_ids,
    'Department': departments,
    'Midterm': midterm_scores,
    'Final': final_scores
}

# Create DataFrame df1
df1 = pd.DataFrame(data)

# Add Assignment columns
assignment_cols = [f'Assignment_{i+1}' for i in range(5)]
df_assignments = pd.DataFrame(assignment_scores, columns=assignment_cols)

# Combine
df = pd.concat([df1, df_assignments], axis=1)

print(f"Dataset shape with missing values: {df.shape}")

# --- Handle Missing Values ---

# 1. Fill Categorical (Department) with Mode
mode_dept = df['Department'].mode()[0]
df['Department'] = df['Department'].fillna(mode_dept)

# 2. Fill Numeric columns with the Mean of the specific Department
numeric_cols = ['Midterm', 'Final'] + assignment_cols

for col in numeric_cols:
    # Calculate mean per department and fill NaNs
    df[col] = df[col].fillna(df.groupby('Department')[col].transform('mean'))

# PART 3: Data Aggregation and Grouping
print("Calculating Grades and Aggregating...")

# Compute Average of Assignments
df['Assignments_Avg'] = df[assignment_cols].mean(axis=1)

# Compute Total Score
# Formula: Midterm (25%), Final (25%), Avg Assignments (50%)
df['Total_Score'] = (0.25 * df['Midterm']) + (0.25 * df['Final']) + (0.50 * df['Assignments_Avg'])

# Assign Letter Grade
# A: 90-100, B: 80-89, C: 70-79, D: 60-69, F: <60
bins = [-1, 59.9, 69.9, 79.9, 89.9, 100]
labels = ['F', 'D', 'C', 'B', 'A']
df['Grade'] = pd.cut(df['Total_Score'], bins=bins, labels=labels)

# Group by Department and compute stats
dept_stats = df.groupby('Department').agg({
    'Total_Score': ['mean', 'std'],
    'Midterm': 'mean',
    'Final': 'mean',
    'StudentID': 'count'
})

# Rename columns for clarity in output
dept_stats.columns = ['Total_Mean', 'Total_Std', 'Midterm_Mean', 'Final_Mean', 'Student_Count']
print("\nDepartment Statistics:")
print(dept_stats)

# PART 4: Visualization with Matplotlib
print("Generating Visualizations...")

# Define a color map for departments to be consistent across plots
dept_colors = {'CS': '#ff9999', 'EE': '#66b3ff', 'PHYS': '#99ff99', 'CHEM': '#ffcc99', 'BIO': '#c2c2f0'}
unique_depts = df['Department'].unique()

# --- Plot 1: Bar Chart (Average Total Score by Dept) ---
plt.figure(figsize=(8, 5))
avg_scores = df.groupby('Department')['Total_Score'].mean()
plt.bar(avg_scores.index, avg_scores.values, color=[dept_colors.get(x, '#333333') for x in avg_scores.index])
plt.title('Average Total Score by Department')
plt.xlabel('Department')
plt.ylabel('Average Score')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Plot 2: Histogram of Total Scores ---
plt.figure(figsize=(8, 5))
plt.hist(df['Total_Score'], bins=15, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Scores')
plt.xlabel('Total Score')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# --- Plot 3: Boxplot (Midterm, Final, Total) ---
plt.figure(figsize=(10, 6))
# Preparing data for boxplot
plot_data = [df['Midterm'], df['Final'], df['Total_Score']]
box = plt.boxplot(plot_data, patch_artist=True, labels=['Midterm', 'Final', 'Total'])

colors = ['#377eb8', '#e41a1c', '#4daf4a']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

plt.title('Score Distribution Across Exams')
plt.ylabel('Scores')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.show()

# --- Plot 4: Scatter Plot (Midterm vs Final) ---
plt.figure(figsize=(9, 6))
for dept in unique_depts:
    subset = df[df['Department'] == dept]
    plt.scatter(subset['Midterm'], subset['Final'], 
                label=dept, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)

plt.title('Midterm vs Final Scores by Department')
plt.xlabel('Midterm Score')
plt.ylabel('Final Score')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# --- Plot 5: Pie Chart (Student Distribution) ---
plt.figure(figsize=(7, 7))
dept_counts = df['Department'].value_counts()
plt.pie(dept_counts, labels=dept_counts.index, autopct='%1.1f%%', startangle=90, 
        colors=[dept_colors.get(x) for x in dept_counts.index])
plt.title('Student Distribution by Department')
plt.show()

# --- Plot 6: Heatmap of Correlation Matrix ---
plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df[['StudentID', 'Midterm', 'Final'] + assignment_cols + ['Total_Score']]
corr_matrix = numeric_df.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numeric Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout() # Adjust layout to make room for labels
plt.show()

# --- Plot 7: Stacked Bar Chart (Grade Distribution) ---
grade_dist = pd.crosstab(df['Department'], df['Grade'])

# Reorder columns to F, D, C, B, A (or A, B, C, D, F depending on preference)
grade_dist = grade_dist[['B', 'C', 'D', 'F', 'A']] # Based on image supplied, A might be rare
# Let's ensure all grades are present in columns even if count is 0
for g in labels:
    if g not in grade_dist.columns:
        grade_dist[g] = 0
grade_dist = grade_dist[labels[::-1]] # A, B, C, D, F

ax = grade_dist.plot(kind='bar', stacked=True, figsize=(10, 6), width=0.5)
plt.title('Grade Distribution by Department')
plt.xlabel('Department')
plt.ylabel('Number of Students')
plt.xticks(rotation=45)
plt.legend(title='Grade')
plt.tight_layout()
plt.show()

print("Script execution complete.")
