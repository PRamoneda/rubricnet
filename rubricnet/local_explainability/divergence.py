import matplotlib.pyplot as plt

# Values extracted from the image
# values = [0.99, 1.0, 0.36, 0.99, -0.9, -0.93, -1.0, -1.0, 0.99, 0.95, 1.0, -0.43] # chopin
# values = [+0.78, +0.98, -0.94, +0.17, -0.92, -0.89, -0.97, -0.99, +0.84, +0.91, +0.67, -0.99]
# liszt
# values = [1.0, 1.0, 0.68, 0.99, 0.05, 0.03, 0.0, 0.0, 1.0, 0.98, 1.0, 0.29]
# grade_divergence = [0.04, 0.0, 0.41, 0.1, 0.01, -0.01, -0.03, -0.06, 0.04, 0.01, 0.06, 0.25]
# bach
# values = [0.89, 0.99, 0.03, 0.58, 0.04, 0.06, 0.01, 0.01, 0.92, 0.95, 0.84, 0.0]
# grade_divergence = [0.12, 0.12, 0.0, 0.12, 0.0, -0.0, -0.02, -0.04, 0.0, -0.0, 0.05, 0.0]
# liszt
values = [0.1, -0.52, 0.05, -0.39, 0.16, -0.01, 0.04, 0.0, -0.08, 0.09,  0.05, -0.34]
# Column names
new_columns = [
    'P. Entropy (R)', 'P. Entropy (L)',
    'P. Range (R)', 'P. Range (L)',
    'Avg P. (R)', 'Avg P. (L)',
    'Avg IOI (R)', 'Avg IOI (L)',
    'Disp. Rate (R)', 'Disp. Rate (L)',
    'P. Set LZ (R)', 'P. Set LZ (L)'
]

# Create a minimal plot
plt.figure(figsize=(10, 3))
bars = plt.bar(new_columns, values, color='#D8E2DC')  # Use a neutral color for the values

# Optionally, you can connect the value and the mean with a line
# for i, (bar, mean) in enumerate(zip(bars, means)):
#     plt.plot([bar.get_x() + bar.get_width() / 2, bar.get_x() + bar.get_width() / 2], [bar.get_height(), mean], 'k-')

plt.xticks(rotation=45, fontsize=15)  # bold
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.yticks(fontsize=15)  # Adjust font size of y-ticks
plt.subplots_adjust(left=0.07)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
# Add a horizontal line at y = 0
plt.axhline(y=0, color='gray', linestyle='--')

# Save the plot
plot_path = 'chopin2.pdf'
plt.savefig(plot_path)
plt.show()
plt.close() # Close the plot to prevent display in output