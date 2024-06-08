# Point cloud segmentation

# Exercise 1.
import open3d as o3d
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots

# Run the next cell in Colab to download the bunny mesh
#!wget https://graphics.stanford.edu/~mdfisher/Data/Meshes/bunny.obj

# Get the bunny mesh
bunny_mesh = o3d.io.read_triangle_mesh("bunny.obj")
num_points = 30000

def mesh2cloud(mesh_obj: o3d.geometry.TriangleMesh, num_points: int) -> np.array:
    point_cloud = mesh_obj.sample_points_uniformly(number_of_points=num_points)
    return np.asarray(point_cloud.points)

bunny_points = mesh2cloud(bunny_mesh, num_points)

# Let's colour our bunny by assigning a unique colour to each point
# The colour will change gradually along all 3 axis simultaneously
colors = np.linspace(0, 1, num_points)
bunny_points = np.concatenate([bunny_points, colors.reshape(-1, 1)], axis=-1)

# Rotation matricies around X, Y or Z-axis, theta given in radians
def R_axis(theta, axis):
    assert axis in ['x', 'y', 'z'], "Axis must be either x, y or z"
    axis = axis.lower()
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
      ] )
    if axis == 'y':
        return np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    if axis == 'z':
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])


# Rotate the bunny, so it's looking good from the start :)
bunny_points[:, :-1] = bunny_points[:, :-1] @ \
    R_axis(-3 * np.pi / 4, axis='y') @ \
    R_axis(-np.pi / 2, axis='x')

  # Visualization function
def visualize_point_clouds(
      points1: np.ndarray,
      points2: np.ndarray,
      title1: str,
      title2: str,
      show_both: bool=True
  ):
      fig = make_subplots(
          rows=1, cols=2,
          specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
          subplot_titles=[title1, title2]
      )

      fig.add_trace(
          go.Scatter3d(
              x=points1[:, 0], y=points1[:, 1], z=points1[:, 2], 
              mode='markers',
              marker=dict(size=1.5, color=points1[:, -1], colorscale='Viridis'),
              name=title1
          ),
          row=1, col=1
      )

      if show_both:
          fig.add_trace(
              go.Scatter3d(
                  x=points2[:, 0], y=points2[:, 1], z=points2[:, 2], 
                  mode='markers',
                  marker=dict(size=1.5, color=points2[:, -1], colorscale='Viridis'),
                  name=title2
              ),
              row=1, col=2
          )

      fig.update_layout(
          scene1=dict(
              xaxis=dict(visible=False),
              yaxis=dict(visible=False),
              zaxis=dict(visible=False),
              aspectmode='data'
          ),
          scene2=dict(
              xaxis=dict(visible=False),
              yaxis=dict(visible=False),
              zaxis=dict(visible=False),
              aspectmode='data'
          )
      )

      fig.show()

# Let's permute our bunny but leave the coloring of each point unchanged
permuted_indices = np.random.permutation(num_points)
permuted_bunny_points = bunny_points[:, :-1][permuted_indices]
permuted_bunny_points = np.concatenate([permuted_bunny_points, colors.reshape(-1, 1)], axis=-1)

# Visualize both
visualize_point_clouds(bunny_points, permuted_bunny_points, "Original Bunny", "Permuted Bunny", show_both=True)


