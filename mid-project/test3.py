import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Create sample data
t = np.linspace(0, 2*np.pi, 100)
x = np.sin(t)
y = np.cos(t)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

# Calculate angles for coloring
angles = np.arctan2(Y, X)  # Calculate angles in radians

# Create custom colormap with specified colors
colors = [(1, 0, 0), (1,0.5,0), (1, 1, 0),(0,1,0), (0, 0, 1), (0,0.5,1), (0.5, 0, 0.5), (1,0,0)]  
cmap_name = 'custom_color_map'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Normalize angles to [0, 1] range for colormap mapping
angle_normalized = (angles + np.pi) / (2 * np.pi)

# Create a contour plot with colored lines
contour = plt.contourf(X, Y, Z, levels=100, cmap='viridis')  # Grayscale background
plt.colorbar(contour, label='Radius')

# Draw colored lines connecting the center to the edge
for i in range(len(x)):
    for j in range(len(y)):
        color = cm(angle_normalized[j, i])  # Get color from colormap
        plt.plot([0, X[j, i]], [0, Y[j, i]], color=color)

plt.gca().set_aspect('equal', adjustable='box')  # Make the plot square
plt.title('Colored Lines from Center to Edge')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()