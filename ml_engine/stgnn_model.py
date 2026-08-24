import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network (STGNN) for EEG and Time-Series Signal Processing.
    Combines Temporal Convolutions (1D CNN) with Spatial Graph Convolution Networks (GCN).
    """
    def __init__(self, in_channels: int = 1, spatial_hidden: int = 32, temporal_hidden: int = 64, num_classes: int = 2):
        super(SpatioTemporalGNN, self).__init__()
        
        self.temporal_conv = nn.Conv1d(in_channels, temporal_hidden, kernel_size=3, padding=1)
        self.spatial_weight = nn.Parameter(torch.FloatTensor(temporal_hidden, spatial_hidden))
        nn.init.xavier_uniform_(self.spatial_weight)
        self.classifier = nn.Linear(spatial_hidden, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, seq_len = x.shape
        x_reshaped = x.view(-1, 1, seq_len)
        temp_out = F.relu(self.temporal_conv(x_reshaped))
        temp_out = temp_out.view(batch_size, num_nodes, -1)
        
        support = torch.matmul(temp_out, self.spatial_weight)
        spatial_out = torch.matmul(adj, support)
        spatial_out = F.relu(spatial_out)
        
        global_repr = torch.mean(spatial_out, dim=1)
        logits = self.classifier(global_repr)
        return logits

def generate_adjacency_matrix(num_nodes: int = 19, threshold: float = 0.4) -> torch.Tensor:
    synthetic_signals = torch.randn(num_nodes, 500)
    corr_matrix = torch.corrcoef(synthetic_signals)
    adj = torch.where(torch.abs(corr_matrix) > threshold, 1.0, 0.0)
    adj.fill_diagonal_(1.0)
    
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    deg_mat = torch.diag(deg_inv_sqrt)
    return torch.mm(torch.mm(deg_mat, adj), deg_mat)

if __name__ == "__main__":
    adj = generate_adjacency_matrix()
    model = SpatioTemporalGNN()
    sample_data = torch.randn(8, 19, 256)
    output = model(sample_data, adj)
    print(f"[ML Engine] Pipeline Verification Successful. Output Logits Shape: {output.shape}")
