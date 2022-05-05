from typing import Union

import torch
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.typing import Adj, OptTensor, OptPairTensor
from torch import Tensor


class GCNConvBatch(GCNConv):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, bias: bool = True, 
                 **kwargs):
        super(GCNConvBatch, self).__init__(in_channels, out_channels,
                                           improved=improved, cached=cached, add_self_loops=add_self_loops,
                                           bias=bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out = torch.matmul(out, self.weight)

        return out
