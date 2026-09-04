# +++++++++++++++++++ MPNN Layer +++++++++++++++++++


class MPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr="add"):
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        # MLP `\psi` for computing messages `m_ij`
        # dims: (2d + d_e) -> d
        self.mlp_msg = MLP([2 * emb_dim + edge_dim, emb_dim, emb_dim])

        # MLP `\phi` for computing updated node features `h_i^{l+1}`
        # Implemented as a stack of Linear->BN->ReLU->Linear->BN->ReLU
        # dims: 2d -> d
        self.mlp_upd = MLP([2 * emb_dim, emb_dim, emb_dim])

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):

        upd_out = torch.cat([h, aggr_out], dim=-1)
        return self.mlp_upd(upd_out)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"


# +++++++++++++++++++ MPNN Model +++++++++++++++++++


class MPNNModel(torch.nn.Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        super().__init__()
        # Linear encoder
        # dim: in_dim -> d = emb_dim
        self.lin_in = torch.nn.Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr="add"))

        # Global pooling/readout function `R` (mean pooling)
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = torch.nn.Linear(emb_dim, out_dim)

    def forward(self, data):
        h = self.lin_in(data.x)

        for conv in self.convs:
            # Note that we add a residual connection after each MPNN layer
            # (n, d) -> (n, d)
            h = h + conv(h, data.edge_index, data.edge_attr)

        # (n, d) -> (batch_size, d)
        h_graph = self.pool(h, data.batch)

        # (batch_size, d) -> (batch_size, 1)
        out = self.lin_pred(h_graph)

        return out.view(-1)


# +++++++++++++++++++ Unit tests +++++++++++++++++++


seed(12345)


def permute_graph(data, perm):
    # Permute the node attribute ordering
    data.x = data.x[perm]
    data.pos = data.pos[perm]
    data.z = data.z[perm]
    data.batch = data.batch[perm]

    # Permute the edge index
    adj = to_dense_adj(data.edge_index)
    adj = adj[:, perm, :]
    adj = adj[:, :, perm]
    data.edge_index = dense_to_sparse(adj)[0]
    return data


def permutation_invariance_unit_test(module, dataloader):
    it = iter(dataloader)
    data = next(it)

    # Set edge_attr to dummy values (for simplicity)
    data.edge_attr = torch.zeros(data.edge_attr.shape)

    # Forward pass on original example
    out_1 = module(data)

    # Create random permutation
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)

    # Forward pass on permuted example
    out_2 = module(data)

    # Check whether output varies after applying transformations
    return torch.allclose(out_1, out_2, atol=1e-04)


def permutation_equivariance_unit_test(module, dataloader):
    it = iter(dataloader)
    data = next(it)
    data.edge_attr = torch.zeros(data.edge_attr.shape)
    out_1 = module(data.x, data.edge_index, data.edge_attr)
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)

    out_2 = module(data.x, data.edge_index, data.edge_attr)
    return torch.allclose(out_1[perm], out_2, atol=1e-04)


# +++++++++++++++++++ E(n) Equivariant GNN Layer +++++++++++++++++++


class EquivariantMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=4, aggr="add"):
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.edge_dim = edge_dim
        self.rel_dist_dim = 1  # +1 dim for the rel distance
        self.pos_weight_dim = 1  # +1 dim for the weights of rel positions

        # MLP to create messages for node features
        self.mlp_msg = MLP([2 * emb_dim + edge_dim + self.rel_dist_dim, emb_dim, emb_dim])
        # MLP to create weights for relative positions
        self.mlp_pos_weight = MLP([emb_dim, emb_dim, self.pos_weight_dim])
        # MLP to update node features
        self.mlp_upd_h = MLP([2 * emb_dim, emb_dim, emb_dim])

    def forward(self, h, pos, edge_index, edge_attr):
        # Propagate positions too
        out = self.propagate(edge_index=edge_index, h=h, pos=pos, edge_attr=edge_attr)
        return out

    def message(self, h_i, h_j, pos_i, pos_j, edge_attr):
        node_edge_features = torch.cat([h_i, h_j, edge_attr], dim=-1)
        # Get distance between the nodes
        dist = torch.norm(pos_j - pos_i, dim=-1, keepdim=True)
        # Create node features
        node_features = torch.cat([node_edge_features, dist], dim=-1)

        # Calculate messages for each edge
        msg = self.mlp_msg(node_features)
        # Calculate scalar weights for each relative position
        pos_msg_weights = self.mlp_pos_weight(msg)
        # Compute weighted relative positions
        pos_msg = (pos_j - pos_i) * pos_msg_weights
        return msg, pos_msg

    def aggregate(self, inputs, index):
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
        # Get aggregated node features and coordinates
        aggr_out_h, aggr_out_pos = aggr_out[0], aggr_out[1]
        # Concatenate node features with the aggregated ones
        upd_out_h = torch.cat([h, aggr_out_h], dim=-1)
        # Update positions simply adding the aggregated positions (this ensures equivariance)
        upd_out_pos = pos + aggr_out_pos
        return self.mlp_upd_h(upd_out_h), upd_out_pos

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"
