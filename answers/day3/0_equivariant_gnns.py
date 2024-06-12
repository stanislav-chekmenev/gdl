# ++++++++++++++ START OF EXERCISE 1  ++++++++++++++ 

class CoordMPNNModel(MPNNModel):
   def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
       super().__init__()
       pos_dim = 3
       self.lin_in = torch.nn.Linear(in_dim + pos_dim, emb_dim)

       # Stack of MPNN layers
       self.convs = torch.nn.ModuleList()
       for layer in range(num_layers):
           self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

       # Global pooling/readout function `R` (mean pooling)
       # PyG handles the underlying logic via `global_mean_pool()`
       self.pool = global_mean_pool

       # Linear prediction head
       # dim: d -> out_dim
       self.lin_pred =torch.nn.Linear(emb_dim, out_dim)

   def forward(self, data):
       # Concatenate positions and coordinates
       h = torch.concat([data.x, data.pos], dim=-1) 
       h = self.lin_in(h)
       for conv in self.convs:
           h = h + conv(h, data.edge_index, data.edge_attr) 
       h_graph = self.pool(h, data.batch)
       out = self.lin_pred(h_graph)
       return out.view(-1)
   
coord_mpnn = CoordMPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1)

# ++++++++++++++ END OF EXERCISE 1  +++++++++++++++ 

# ++++++++++++++ START OF EXERCISE 2 ++++++++++++++ 

def rot_trans_invariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN model/layer) is
    rotation and translation invariant.
    """
    it = iter(dataloader)
    data = next(it)

    # Forward pass on original example
    # Note: A conditional forward pass allows to run the same unit test for both the GNN model as well as the layer.
    if isinstance(module, MPNNModel):
        out_1 = module(data)
    else: # if ininstance(module, MessagePassing):
        out_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    # Get a random rotation matrix
    Q = random_rotation_matrix(dim=3)
    # Get a random translation
    t = torch.rand(3)

    # Rotate and translate the positions
    data.pos = data.pos @ Q + t

    # Forward pass on rotated + translated example
    if isinstance(module, MPNNModel):
        out_2 = module(data)
    else: # if ininstance(module, MessagePassing):
        out_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    # Check whether output varies after applying transformations
    return torch.allclose(out_1, out_2, atol=1e-04)

# ++++++++++++++ END OF EXERCISE 2  +++++++++++++++ 

# ++++++++++++++ START OF EXERCISE 3 ++++++++++++++ 
class InvariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        """Message Passing Neural Network Layer

        This layer is invariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.rel_dist_dim = 1

        self.mlp_msg = MLP([2 * emb_dim + self.rel_dist_dim + edge_dim, emb_dim, emb_dim])
        
        self.mlp_upd = MLP([2 * emb_dim, emb_dim, emb_dim])

    def forward(self, h, pos, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        """
        The `message()` function constructs messages from source nodes j
        to destination nodes i for each edge (i, j) in `edge_index`.

        The arguments can be a bit tricky to understand: `message()` can take
        any arguments that were initially passed to `propagate`. Additionally,
        we can differentiate destination nodes and source nodes by appending
        `_i` or `_j` to the variable name, e.g. for the node features `h`, we
        can use `h_i` and `h_j`.

        This part is critical to understand as the `message()` function
        constructs messages for each edge in the graph. The indexing of the
        original node features `h` (or other node variables) is handled under
        the hood by PyG.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features

        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        node_edge_attr = torch.cat([h_i, h_j, edge_attr], dim=-1)
        dist_attr = torch.norm(pos_j - pos_i, dim=-1, keepdim=True)
        feature = torch.cat([node_edge_attr, dist_attr], dim=-1)
        return self.mlp_msg(feature)

    def aggregate(self, inputs, index):
        """The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        """The `update()` function computes the final node features by combining the
        aggregated messages with the initial node features.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
        """
        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class InvariantMPNNModel(MPNNModel):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations.

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = torch.nn.Linear(in_dim, emb_dim)

        # Stack of invariant MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(InvariantMPNNLayer(emb_dim, edge_dim, aggr='add'))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = torch.nn.Linear(emb_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x) # (n, d_n) -> (n, d)

        for conv in self.convs:
            # Note: here the conv layer takes 3D positions, which we previously were concatenating to the features
            h = h + conv(h, data.pos, data.edge_index, data.edge_attr) # (n, d) -> (n, d)

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)
    

inv_layer = InvariantMPNNLayer(emb_dim=11, edge_dim=4, aggr="add")
inv_mpnn = InvariantMPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1)

# ++++++++++++++ END OF EXERCISE 3  +++++++++++++++ 

# ++++++++++++++ START OF EXERCISE 4 ++++++++++++++ 

def rot_trans_equivariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN layer) is 
    rotation and translation equivariant.
    """
    it = iter(dataloader)
    data = next(it)
    out_1, pos_1 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    Q = random_rotation_matrix(dim=3)
    t = torch.rand(3)

    # Rotate and translate the postitions 
    data.pos = data.pos @ Q + t

    # Forward pass on rotated + translated example
    out_2, pos_2 = module(data.x, data.pos, data.edge_index, data.edge_attr)

    # Rotate and translate the original output position embeddings
    pos_1_rot_t = pos_1 @ Q + t
    
    return torch.allclose(out_1, out_2, atol=1e-04), torch.allclose(pos_1_rot_t, pos_2, atol=1e-04)

# ++++++++++++++ END OF EXERCISE 4  +++++++++++++++ 

# ++++++++++++++ START OF EXERCISE 5 ++++++++++++++

class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr='add'):
        """Message Passing Neural Network Layer

        This layer is equivariant to 3D rotations and translations.

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.rel_dist_dim = 1
        self.pos_msg_weight_dim = 1

        # MLP to create messages for node features
        self.mlp_msg_h = MLP([2 * emb_dim + edge_dim + self.rel_dist_dim, emb_dim, emb_dim])
        
        # MLP to create weights for relative positions
        self.mlp_msg_pos = MLP([emb_dim, emb_dim, self.pos_msg_weight_dim])

        # MLP to update node features
        self.mlp_upd_h = MLP([2 * emb_dim, emb_dim, emb_dim])

    def forward(self, h, pos, edge_index, edge_attr):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_attr: (e, d_e) - edge features

        Returns:
            out: [(n, d),(n,3)] - updated node features
        """

        out = self.propagate(edge_index=edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        """
        The `message()` function constructs messages from source nodes j
        to destination nodes i for each edge (i, j) in `edge_index`.

        The arguments can be a bit tricky to understand: `message()` can take
        any arguments that were initially passed to `propagate`. Additionally,
        we can differentiate destination nodes and source nodes by appending
        `_i` or `_j` to the variable name, e.g. for the node features `h`, we
        can use `h_i` and `h_j`.

        This part is critical to understand as the `message()` function
        constructs messages for each edge in the graph. The indexing of the
        original node features `h` (or other node variables) is handled under
        the hood by PyG.

        Args:
            h_i: (e, d) - destination node features
            h_j: (e, d) - source node features
            edge_attr: (e, d_e) - edge features

        Returns:
            msg: (e, d) - messages `m_ij` passed through MLP `\psi`
        """
        node_edge_features = torch.cat([h_i, h_j, edge_attr], dim=-1)
        # Get distance between the nodes
        pos_dist = torch.norm(pos_j - pos_i, dim=-1, keepdim=True)
        # Create node features
        node_feature = torch.cat([node_edge_features, pos_dist], dim=-1)

        # Calculate edge_embeddings for each edge
        edge_emb = self.mlp_msg_h(node_feature)

        # Calculate scalar weights for each relative position
        pos_msg_weights = self.mlp_msg_pos(edge_emb)
        # Compute weighted relative positions
        pos_msg = (pos_j - pos_i) * pos_msg_weights
        return edge_emb, pos_msg

    def aggregate(self, inputs, index):
        """The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        # Get all the messages for the node features
        aggr_out_h = inputs[0]
        # Get all the messages for the node coordinates
        aggr_out_pos = inputs[1]
        # Aggregate all node features
        aggr_out_h = scatter(src=aggr_out_h, index=index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate all node coordinates (Note! We use mean hear, as in the paper)
        aggr_out_pos = scatter(src=aggr_out_pos, index=index, dim=self.node_dim, reduce="mean")
        return aggr_out_h, aggr_out_pos

    def update(self, aggr_out, h, pos):
        # Get aggegated node features and coordnates
        aggr_out_h, aggr_out_pos = aggr_out[0], aggr_out[1]
        # Concatenate node features with the aggregated ones
        upd_out_h = torch.cat([h, aggr_out_h], dim=-1)
        # Update positions simply adding the aggregated positions (this ensures equivariance)
        upd_out_pos = pos + aggr_out_pos
        return self.mlp_upd_h(upd_out_h), upd_out_pos

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class EquivariantMPNNModel(MPNNModel):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations (the constituent MPNN layers
        are equivariant to 3D rotations and translations).

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in_x = torch.nn.Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(EquivariantMPNNLayer(emb_dim, edge_dim, aggr='add'))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = torch.nn.Linear(emb_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns:
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in_x(data.x) # (n, d_n) -> (n, d)
        pos = data.pos

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, data.edge_index, data.edge_attr)

            # Update node features
            h = h + h_update 

            # Update node coordinates
            pos = pos + pos_update 

        h_graph = self.pool(h, data.batch) 

        out = self.lin_pred(h_graph)

        return out.view(-1)

# ++++++++++++++ END OF EXERCISE 5  +++++++++++++++ 





