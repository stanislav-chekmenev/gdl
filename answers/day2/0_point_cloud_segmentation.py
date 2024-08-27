# ++++++++++++++ START OF EXERCISE 1  ++++++++++++++ 
# Run the next cell in Colab to download the bunny mesh, uncomment the #-symbol
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

# ++++++++++++++ END OF EXERCISE 1  ++++++++++++++++ 


# ++++++++++++++ START OF EXERCISE 2  ++++++++++++++ 

# Create a small dataset
fixed_points_transform = FixedPoints(num=10000, replace=False)
tiny_kitti = TinyVKittiDataset(root="data/train", size=2, transform=fixed_points_transform, log=False)
data = tiny_kitti[0]

# Run FPS to sample 10 centroids
fps_idx = fps(data.pos, ratio=0.001)

# Get src and dest node indices
dest_idx_knn, src_idx_knn = knn(x=data.pos, y=data.pos[fps_idx], k=32)
dest_idx_radius, src_idx_radius = radius(x=data.pos, y=data.pos[fps_idx], r=0.1, max_num_neighbors=32)

# Create edge_index tensors
edge_index_knn = torch.stack([src_idx_knn, fps_idx[dest_idx_knn]], dim=0)
edge_index_bq = torch.stack([src_idx_radius, fps_idx[dest_idx_radius]], dim=0)

# Visualize the graphs
visualize(
    point_cloud_graph1=data,
    point_cloud_graph2=data,
    edge_indices=[edge_index_knn, edge_index_bq],
    show_both=True,
    name1="kNN graph",
    name2="BQ graph",
)

# ++++++++++++++ END OF EXERCISE 2  ++++++++++++++++ 


# ++++++++++++++ START OF EXERCISE 3  ++++++++++++++ 

# Compose a series of rotations around all 3 axes
sample_random_rotate = Compose([
    FixedPoints(num=10_000, replace=False),
    RandomRotate(degrees=30, axis=0),
    RandomRotate(degrees=30, axis=1),
    RandomRotate(degrees=30, axis=2),
])

# Create the dataset
rotated_test_dataset = TinyVKittiDataset(root="data/test/", size=1, transform=sample_random_rotate)

# Visualize point clouds
data_rotated = rotated_test_dataset[0]
data_original = test_dataset[0]
visualize(
    point_cloud_graph1=data_original,
    point_cloud_graph2=data_rotated,
    name1="Original Point Cloud",
    name2="Rotated Point Cloud",
    show_both=True
)

# Run inference and compute the accuracy
rotated_loader = DataLoader(rotated_test_dataset, batch_size=1, shuffle=False)

# Run inference
print(f"Test accurcy for the rotated point cloud: {test(rotated_loader)}")

# ++++++++++++++ END OF EXERCISE 3  ++++++++++++++++


# ++++++++++++++ START OF EXERCISE 4  ++++++++++++++ 

class PPFPointNetPP(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        seed(12345)
        # Dim 4 is for the PPF features (r^2, angle_1, angle_2, angle_3)
        self.sa1_module = PPFSAModule(ratio=0.2, r=0.2, nn=MLP([4, 32, 32]))
        self.sa2_module = PPFSAModule(ratio=0.25, r=0.4, nn=MLP([32 + 4, 32, 64]))
        # Extra 3 for the normals
        self.sa3_module = PPFGlobalSAModule(MLP([64 + 3 + 3, 64, 128]))

        self.fp3_module = PPFFPModule(2, MLP([128 + 64, 64]))
        self.fp2_module = PPFFPModule(3, MLP([64 + 32, 32]))
        self.fp1_module = PPFFPModule(3, MLP([32, 128]))

        self.mlp = MLP([128, 64, num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        # Adding normals
        sa0_out = (data.x, data.pos, data.norm, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x)

# ++++++++++++++ END OF EXERCISE 4  ++++++++++++++++
