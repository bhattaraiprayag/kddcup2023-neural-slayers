# plot_tuning.py

import pandas as pd
import io
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


file_path = '/home/prayag/side_projects/kdd_cup/outputs/tuning/tuning_results.csv'
df = pd.read_csv(file_path)
dpi = 600
selected_cols = ['lr', 'weight_decay', 'hidden_channels_factor', 'scheduler',
                 'trial_id', 'locale', 'best_val_auc', 'test_auc', 'training_time_s']
df_selected = df[selected_cols]


# Visualization 1: Scatter Plot: best_val_auc vs. training_time_s
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_selected, x='training_time_s', y='best_val_auc')
plt.title('Best Validation AUC vs. Training Time')
plt.xlabel('Training Time (s)')
plt.ylabel('Best Validation AUC')
plt.grid(True)
plt.savefig(os.path.join("outputs", "tuning", "plots", f"1. best_val_auc_vs_training_time.png"), dpi=dpi)
plt.close()


# Visualization 2: Scatter Plot: test_auc vs. best_val_auc
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_selected, x='best_val_auc', y='test_auc')
plt.title('Test AUC vs. Best Validation AUC')
plt.xlabel('Best Validation AUC')
plt.ylabel('Test AUC')
plt.grid(True)
plt.savefig(os.path.join("outputs", "tuning", "plots", f"2. test_auc_vs_best_val_auc.png"), dpi=dpi)
plt.close()


# Visualization 3: Box Plot: best_val_auc by scheduler
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_selected, x='scheduler', y='best_val_auc')
plt.title('Distribution of Best Validation AUC by Scheduler')
plt.xlabel('Scheduler')
plt.ylabel('Best Validation AUC')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"3. best_val_auc_by_scheduler.png"), dpi=dpi)
plt.close()


# Visualization 4: Box Plot: test_auc by scheduler
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_selected, x='scheduler', y='test_auc')
plt.title('Distribution of Test AUC by Scheduler')
plt.xlabel('Scheduler')
plt.ylabel('Test AUC')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"4. test_auc_by_scheduler.png"), dpi=dpi)
plt.close()


# Visualization 5: Pair Plot of numerical variables
numerical_cols = ['lr', 'weight_decay', 'hidden_channels_factor', 'best_val_auc', 'test_auc', 'training_time_s']
sns.pairplot(df_selected[numerical_cols])
plt.suptitle('Pair Plot of Numerical Variables', y=1.02)
plt.savefig(os.path.join("outputs", "tuning", "plots", f"5. pair_plot_numerical_vars.png"), dpi=dpi)
plt.close()


# Visualization 6: Bar Plot: Average best_val_auc by locale
avg_best_val_auc_by_locale = df_selected.groupby('locale')['best_val_auc'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_best_val_auc_by_locale, x='locale', y='best_val_auc', hue='locale', palette='viridis', legend=False) # Changed
plt.title('Average Best Validation AUC by Locale')
plt.xlabel('Locale')
plt.ylabel('Average Best Validation AUC')
plt.grid(axis='y')
plt.savefig(os.path.join("outputs", "tuning", "plots", f"6. avg_best_val_auc_by_locale.png"), dpi=dpi)
plt.close()


# Visualization 7: Bar Plot: Average test_auc by locale
avg_test_auc_by_locale = df_selected.groupby('locale')['test_auc'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=avg_test_auc_by_locale, x='locale', y='test_auc', hue='locale', palette='viridis', legend=False) # Changed
plt.title('Average Test AUC by Locale')
plt.xlabel('Locale')
plt.ylabel('Average Test AUC')
plt.grid(axis='y')
plt.savefig(os.path.join("outputs", "tuning", "plots", f"7. avg_test_auc_by_locale.png"), dpi=dpi)
plt.close()


# Visualization 8: Heatmap: Correlation Matrix of numerical variables
correlation_matrix = df_selected[numerical_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"8. correlation_heatmap.png"), dpi=dpi)
plt.close()


# Visualization 9: Scatter Plot: lr vs. best_val_auc (colored by scheduler)
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df_selected, x='lr', y='best_val_auc', hue='scheduler', style='scheduler', s=100, palette='deep')
plt.title('Best Validation AUC vs. Learning Rate (by Scheduler)')
plt.xlabel('Learning Rate (lr)')
plt.ylabel('Best Validation AUC')
plt.grid(True)
plt.legend(title='Scheduler', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"9. best_val_auc_vs_lr_by_scheduler.png"), dpi=dpi)
plt.close()


# Visualization 10: Scatter Plot: weight_decay vs. best_val_auc (colored by scheduler)
plt.figure(figsize=(12, 7))
sns.scatterplot(data=df_selected, x='weight_decay', y='best_val_auc', hue='scheduler', style='scheduler', s=100, palette='deep')
plt.title('Best Validation AUC vs. Weight Decay (by Scheduler)')
plt.xlabel('Weight Decay')
plt.ylabel('Best Validation AUC')
plt.grid(True)
plt.legend(title='Scheduler', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"10. best_val_auc_vs_weight_decay_by_scheduler.png"), dpi=dpi)
plt.close()


# Visualization 11: Faceted Scatter Plots
g = sns.relplot(
    data=df_selected,
    x='lr',
    y='best_val_auc',
    col='scheduler',      # Create columns based on the scheduler
    col_wrap=2,           # Wrap the columns after 2 plots
    kind='scatter',       # relplot correctly uses kind='scatter'
    palette='viridis',
    hue='lr',             # Color points by learning rate
    s=80,                 # Point size
    height=4,             # Set the height of each facet
    aspect=1.2            # Set the aspect ratio of each facet
)
g.fig.suptitle('Best Validation AUC vs. Learning Rate (Faceted by Scheduler)', y=1.03)
g.set_axis_labels('Learning Rate (lr)', 'Best Validation AUC')
g.set_titles("Scheduler: {col_name}")
g.set(xscale="log")
g.fig.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"11. revamped_1_auc_vs_lr_by_scheduler.png"), dpi=dpi)
plt.close()


# Visualization 12: Hyperparameter Interaction Heatmap
pivot_df = df_selected.pivot_table(
    index='lr',
    columns='weight_decay',
    values='best_val_auc',
    aggfunc='mean' # Use the mean AUC for each combination
)
plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_df,
    annot=True,          # Show the AUC values on the map
    cmap='viridis',      # Use a sequential colormap where lighter is better
    fmt=".3f",           # Format annotations to 3 decimal places
    linewidths=.5
)
plt.title('Heatmap of Best Validation AUC by LR and Weight Decay')
plt.xlabel('Weight Decay')
plt.ylabel('Learning Rate')
plt.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"12. revamped_2_heatmap_lr_vs_wd.png"), dpi=dpi)
plt.close()


# Visualization 13: Multi-variable Scatter Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(
    x=df_selected['weight_decay'],
    y=df_selected['hidden_channels_factor'],
    c=df_selected['best_val_auc'], # Color by performance (AUC)
    s=df_selected['training_time_s'] / 5, # Size by training time
    cmap='viridis',
    alpha=0.7,
    edgecolors='black'
)
cbar = plt.colorbar(scatter)
cbar.set_label('Best Validation AUC', rotation=270, labelpad=15)
for time in [200, 600, 1000]:
    plt.scatter([], [], c='gray', alpha=0.5, s=time/5, label=f'{time}s')
plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Training Time')
plt.title('Performance vs. Weight Decay and Hidden Channels')
plt.xlabel('Weight Decay')
plt.ylabel('Hidden Channels Factor')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"13. revamped_3_wd_vs_hidden_channels.png"), dpi=dpi)
plt.close()


# Visualization 14: Violin Plot with Hue
plt.figure(figsize=(14, 8))
sns.violinplot(
    data=df_selected,
    x='scheduler',
    y='best_val_auc',
    hue='hidden_channels_factor', # Split violins by this variable
    split=True,                   # Show the two hues on opposite sides of the same violin
    inner='quartile',             # Show quartiles inside the violin
    palette='pastel'
)
plt.title('AUC Distribution by Scheduler and Hidden Channels Factor')
plt.xlabel('Scheduler')
plt.ylabel('Best Validation AUC')
plt.xticks(rotation=15)
plt.legend(title='Hidden Channels Factor', loc='upper right')
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join("outputs", "tuning", "plots", f"14. revamped_4_violin_scheduler_vs_hidden.png"), dpi=dpi)
plt.close()


# Visualization 15: Parallel Coordinates Plot
parallel_cols = [
    'lr', 'weight_decay', 'hidden_channels_factor',
    'best_val_auc', 'training_time_s'
]
df_parallel = df_selected.copy()
df_parallel['scheduler_id'] = pd.Categorical(df_parallel['scheduler']).codes
parallel_cols_with_scheduler = ['scheduler_id'] + parallel_cols
fig = px.parallel_coordinates(
    df_parallel,
    dimensions=parallel_cols_with_scheduler,
    color="best_val_auc", # Color lines by performance
    color_continuous_scale=px.colors.sequential.Viridis,
    labels={
        "scheduler_id": "Scheduler",
        "lr": "Learning Rate",
        "weight_decay": "Weight Decay",
        "hidden_channels_factor": "Hidden Channels",
        "best_val_auc": "Best Val AUC",
        "training_time_s": "Training Time (s)"
    },
    title="Parallel Coordinates Plot of Hyperparameter Tuning"
)
fig.update_layout(
    xaxis=dict(
        tickvals=list(range(len(df_parallel['scheduler'].unique()))),
        ticktext=df_parallel['scheduler'].unique()
    )
)
fig.write_html("15. revamped_5_parallel_coordinates.html")  # To display in a notebook, you would just call: fig.show()
