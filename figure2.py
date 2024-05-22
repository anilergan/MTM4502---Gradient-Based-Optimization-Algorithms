import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Veriler
algos = ['Newton-Raphson', 'Conjugate Gradient\nHestenes-Stiefel', 'Conjugate Gradient\nPolak-Ribiere', 'Conjugate Gradient\nFletcher-Reeves']
initial_points = ['1st', '2nd', '3rd']
execution_times = [
    [2.63353, 0.585, 1.318,  0.903],
    [2.655, 0.616,  0.989, 3.432],
    [1.394, 0.386, 0.845, 0.305]
]

# Ortalama iterasyon sayıları
avg_execution_times = [np.mean([execution_times[j][i] for j in range(len(initial_points))]) for i in range(len(algos))]

# Veri çerçevesini oluşturma
data = {'Algorithm': algos * len(initial_points), 
        'Initial Point': initial_points * len(algos),
        'Execution Times': [item for sublist in execution_times for item in sublist]}

# Seaborn barplot
plt.figure(figsize=(12, 8))
ax = sns.barplot(data=data, x='Algorithm', y='Execution Times', hue='Initial Point', palette='flare')
plt.title('Execution Times Comparison by Algorithm and Initial Point', fontsize=14)
plt.xlabel('')
plt.ylabel('Execution Times', fontsize=12)
plt.xticks(rotation=0)

# Ortalama iterasyon sayılarını yıldız işareti ile göster
for i, algo in enumerate(algos):
    ax.scatter(i, avg_execution_times[i], color='gold', marker='*', s=500, zorder=5)

# Add legend for the average star marker (only once outside the loop)
handles, labels = ax.get_legend_handles_labels()

star_marker = plt.Line2D([0], [0], marker='*', color='gold', linestyle='None', markersize=10)
handles.append(star_marker)
labels.append('Average Point(All Initial Points)')
ax.legend(handles=handles, labels=labels, title='Initial Point', fontsize=10, title_fontsize=12)

plt.grid(axis='y', alpha=0.4)  # Add a subtle horizontal grid
plt.tight_layout()

# Göster
plt.show()
