
# +++++++++++++++++++ Dataset +++++++++++++++++++

class TinyVKittiDataset(Dataset):

    def __init__(self, root: str, size, pre_transofrm: callable=NormalizeScale(), transform: callable=None, **kwargs):
       """TinyVKitti Dataset class.
       Args:
           root (str): Root directory where the dataset should be saved.
           size (int): The number of total VKITTI frame files in the root directory.
           transform (callable, optional): A function/transform that takes in a
               :class:`~torch_geometric.data.Data` or
               :class:`~torch_geometric.data.HeteroData` object and returns a
               transformed version.
               The data object will be transformed before every access.
               (default: :obj:`None`)
           pre_transform (callable, optional): A function/transform that takes in
               a :class:`~torch_geometric.data.Data` or
               :class:`~torch_geometric.data.HeteroData` object and returns a
               transformed version.
               The data object will be transformed before being saved to disk.
               (default: :obj:`None`)
       """
       self.size = size
       super().__init__(root, pre_transform=pre_transofrm, transform=transform, **kwargs)
      
    @property
    def raw_file_names(self) -> list[str]:
       points = [f'frame_{i}.npy' for i in range(self.size)]
       return points


    @property
    def processed_file_names(self) -> list[str]:
       graphs = [f'graph_{i}.pt' for i in range(self.size)]
       return graphs
  
    def process(self) -> None:
       # Read each point cloud file and process it
       for num, raw_file_name in enumerate(self.raw_file_names):
           print(f'Processing {num}th file')


           # Read the raw data
           raw_file_path = os.path.join(self.raw_dir, raw_file_name)
           if not raw_file_path.endswith('.npy'):
               raise ValueError("File must be a .npy file!")


           points = np.load(raw_file_path)
           points_labels = points[:, [0, 1, 2, -1]] # Throw away the RGB channel values




           # Create one pos-variable for the graph holding all three coordinates
           pos = points_labels[:, :3]


           # Create a y-variable for for the graph holding all labels
           y = points_labels[:, 3]


           # Create a graph
           pos = torch.tensor(pos, dtype=torch.float)
           y = torch.tensor(y, dtype=torch.int)
           data = Data(pos=pos, y=y)


           # Scale and normalize the data
           if self.pre_transform is not None:
               data = self.pre_transform(data)


           # Save the processed data
           torch.save(data, os.path.join(self.processed_dir, f'graph_{num}.pt'))


    def download(self) -> None:
       raise NotImplementedError("Specify custom logic to download your dataset")
   
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'graph_{idx}.pt'))
        return data


# +++++++++++++++++++ PointNet Layer +++++++++++++++++++

class PointNetLayer(MessagePassing):
   def __init__(self, nn: torch.nn.Module):
       # Message passing with "max" aggregation.
       super().__init__(aggr='max')


       # Initialization of the MLP:
       # Here, the number of input features correspond to the hidden node
       # dimensionality plus point dimensionality (=3).
       self.nn = nn


   def forward(
           self, x: Tuple[torch.Tensor, ...], pos: Tuple[torch.Tensor, ...], edge_index: torch.Tensor
       ) -> torch.Tensor:
       # Start propagating messages, the bipartite nature of the graphs is taking into account inside the method
       return self.propagate(edge_index, x=x, pos=pos)


   def message(self, x_j: torch.Tensor, pos_j: torch.Tensor, pos_i: torch.Tensor) -> torch.Tensor:
       # Translate all positions to the coordinate frame of the centroids
       input = pos_j - pos_i


       if x_j is not None:
           # In the first layer, we may not have any hidden node features,
           # so we only combine them in case they are present.
           input = torch.cat([x_j, input], dim=-1)


       return self.nn(input)  # Apply our final neural network.
   
   
# +++++++++++++++++++ Local SA Module +++++++++++++++++++

class SAModule(torch.nn.Module):
   def __init__(self, ratio, r, nn):
       super().__init__()
       self.ratio = ratio
       self.r = r
       self.conv = PointNetLayer(nn)


   def forward(self, x: Optional[torch.Tensor], pos: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
       # Apply FPS to get the centroids for each graph in the batch
       idx = fps(pos, batch, ratio=self.ratio)


       # Group the neighbours of the centroids for each graph in the batch
       dest_idx, src_idx = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=32)
       edge_index = torch.stack([src_idx, dest_idx], dim=0)


       # See if we have any hidden node features that we received from the previous layer
       x_dest = None if x is None else x[idx]


       # Apply PointNetLayer using the features of the neighbouring nodes and the centroids
       # The output is the updated node features for the centroids
       # We feed tuples to differentiate between the features of the neighbouring nodes and the centroids
       x = self.conv((x, x_dest), (pos, pos[idx]), edge_index)


       # Get the centroid positions for each batch
       pos, batch = pos[idx], batch[idx]
       return x, pos, batch


# +++++++++++++++++++ Global SA Module +++++++++++++++++++

class GlobalSAModule(torch.nn.Module):
   def __init__(self, nn):
       super().__init__()
       self.nn = nn
   def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
       # Concatenate all the node features and positions and run though a neural network
       x = self.nn(torch.cat([x, pos], dim=1))
       # Pool across all the node features in the batch to produce graph embeddings of dim [num_graphs, F_x]
       x = global_max_pool(x, batch)
       # Create an empty tensor of positions for each graph embedding we got at the previous step
       pos = pos.new_zeros((x.size(0), 3))
       # Create a new batch tensor for each graph embedding
       batch = torch.arange(x.size(0), device=batch.device)
       return x, pos, batch
   

# +++++++++++++++++++ FP Module +++++++++++++++++++

class FPModule(torch.nn.Module):
   def __init__(self, k, nn):
       super().__init__()
       self.k = k
       self.nn = nn


   def forward(
           self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, x_skip: torch.Tensor,
           pos_skip: torch.Tensor, batch_skip: torch.Tensor) -> Tuple[torch.Tensor, ...]:
       # Perform the interpolation
       x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)


       # Check if there was any previous SA layer output to concatenate with
       if x_skip is not None:
           x = torch.cat([x, x_skip], dim=1)


       # Encode the features with a neural network
       x = self.nn(x)
       return x, pos_skip, batch_skip
   

# +++++++++++++++++++ PointNet++  +++++++++++++++++++

class PointNetPP(torch.nn.Module):
   def __init__(self, num_classes):
       super().__init__()
       seed(12345)
       # Input channels account for both `pos` and node features.
       # Perform the downsampling and feature aggregation
       # Sample 20% of input points, group them within the radius of 0.2 and encode the features into a 16-dim vector
       self.sa1_module = SAModule(ratio=0.2, r=0.2, nn=MLP([3, 32, 32]))
       # Sample 25 % of the downsampled points, group within the radius of 0.4 (since the points are more sparse now)
       # and encode them into a 32-dim vector
       self.sa2_module = SAModule(ratio=0.25, r=0.4, nn=MLP([32 + 3, 32, 64]))
       # Take each point positions and features, encode them into a 64-dim vector and then max-pool across all graphs
       self.sa3_module = GlobalSAModule(MLP([64 + 3, 64, 128]))


       # Perform upsampling and feature propagation
       # Interpolate output features from sa3_module and concatenate with the sa2_module output features
       # Input features are 64-dim from sa3_module and 32-dim from sa2_module
       self.fp3_module = FPModule(2, MLP([128 + 64, 64]))
       # Interpolate upsampled features from fp3_module and concatenate with sa1_module output features
       # Input features are 32-dim from fp3_module and 16-dim from sa1_module
       self.fp2_module = FPModule(3, MLP([64 + 32, 32]))
       # Interpolate upsampled output features from fp2_module and encode them into a 128-dim vector
       self.fp1_module = FPModule(3, MLP([32, 128]))


       # Apply the final MLP network to perform label segmentation for each point, using their propagated features
       self.mlp = MLP([128, 64, num_classes], dropout=0.5, norm=None)


   def forward(self, data):
       sa0_out = (data.x, data.pos, data.batch)
       sa1_out = self.sa1_module(*sa0_out) # x, pos, batch after the 1st local SA layer
       sa2_out = self.sa2_module(*sa1_out) # x, pos, batch after the 2nd local SA layer
       sa3_out = self.sa3_module(*sa2_out) # x, pos, batch after the 3rd global SA layer (pos are all zeros here!)


       fp3_out = self.fp3_module(*sa3_out, *sa2_out) # x, pos, batch after upsampling in the 3rd FP layer
       fp2_out = self.fp2_module(*fp3_out, *sa1_out) # x, pos, batch after upsampling in the 2nd FP layer
       x, _, _ = self.fp1_module(*fp2_out, *sa0_out) # x - node embeddings for each point in the original point clouds


       # Generate final label predictions for each data point in each batch
       return self.mlp(x)
   

# +++++++++++++++++++ Train & Test  +++++++++++++++++++

def train(model, loader):
   model.train()
   loss_all = 0


   for data in loader:
       data.to(device) # send tensors to GPU/CPU
       optimizer.zero_grad() # remove all grads from the previous step
       logits = model(data) # run inference
       loss = criterion(logits, data.y) # compute loss
       loss.backward() # compute gradients with respect to each model parameter
       loss_all += loss.item() * data.num_graphs # multiply loss by N graphs in a batch and add to the total loss
       optimizer.step() # apply grads and update the weights
   return loss_all / len(train_loader.dataset) # return avg loss per the whole dataset

@torch.no_grad()
def test(model, loader):
   model.eval()
   total_correct = 0
   total_nodes = 0
   for data in loader:
       data.to(device)
       logits = model(data)
       total_correct += logits.argmax(dim=1).eq(data.y).sum().item()
       total_nodes += data.num_nodes


   return total_correct / total_nodes


# +++++++++++++++++++ PPF Modules  +++++++++++++++++++

class PPFSAModule(torch.nn.Module):
   def __init__(self, ratio, r, nn):
       super().__init__()
       self.ratio = ratio
       self.r = r
       self.conv = PPFConv(nn, add_self_loops=False)


   def forward(
           self, x: Optional[torch.Tensor], pos: torch.Tensor, normal: torch.Tensor, batch: torch.Tensor
       ) -> Tuple[torch.Tensor, ...]:
       idx = fps(pos, batch, ratio=self.ratio)
       dest_idx, src_idx = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=32)
       edge_index = torch.stack([src_idx, dest_idx], dim=0)
       x_dest = None if x is None else x[idx]
       # Here we add normals!
       x = self.conv((x, x_dest), (pos, pos[idx]), (normal, normal[idx]), edge_index)
       pos, normal, batch = pos[idx], normal[idx], batch[idx]
       return x, pos, normal, batch


class PPFGlobalSAModule(torch.nn.Module):
   def __init__(self, nn):
       super().__init__()
       self.nn = nn


   def forward(
           self, x: torch.Tensor, pos: torch.Tensor, normal: torch.Tensor, batch: torch.Tensor
       ) -> Tuple[torch.Tensor, ...]:
       # Concatenate normals
       x = self.nn(torch.cat([x, pos, normal], dim=1))
       x = global_max_pool(x, batch)
       pos = pos.new_zeros((x.size(0), 3))
       normal = normal.new_zeros((x.size(0), 3))
       batch = torch.arange(x.size(0), device=batch.device)
       return x, pos, normal, batch




class PPFFPModule(torch.nn.Module):
   def __init__(self, k, nn):
       super().__init__()
       self.k = k
       self.nn = nn
   
   def forward(
           self, x: torch.Tensor, pos: torch.Tensor, normal:torch.Tensor, batch: torch.Tensor,
           x_skip: torch.Tensor, pos_skip: torch.Tensor, normal_skip, batch_skip: torch.Tensor
       ) -> Tuple[torch.Tensor, ...]:


       # Perform the interpolation
       x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)


       # Check if there was any previous SA layer output to concatenate with
       if x_skip is not None:
           x = torch.cat([x, x_skip], dim=1)


       # Encode the features with a neural network
       x = self.nn(x)
       return x, pos_skip, normal_skip, batch_skip