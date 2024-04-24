import torch.nn as nn
from models.networks.utils import UnormGPS
from torch.nn.functional import tanh, sigmoid, softmax


class AuxHead(nn.Module):
    def __init__(self, aux_data=[], use_tanh=False):
        super().__init__()
        self.aux_data = aux_data
        self.unorm = UnormGPS()
        self.use_tanh = use_tanh

    def forward(self, x):
        """Forward pass of the network.
        x : Union[torch.Tensor, dict] with the output of the backbone.
        """
        if self.use_tanh:
            gps = tanh(x["gps"])
        gps = self.unorm(gps)
        output = {"gps": gps}
        if "land_cover" in self.aux_data:
            output["land_cover"] = softmax(x["land_cover"])
        if "road_index" in self.aux_data:
            output["road_index"] = x["road_index"]
        if "drive_side" in self.aux_data:
            output["drive_side"] = sigmoid(x["drive_side"])
        if "climate" in self.aux_data:
            output["climate"] = softmax(x["climate"])
        if "soil" in self.aux_data:
            output["soil"] = softmax(x["soil"])
        if "dist_sea" in self.aux_data:
            output["dist_sea"] = x["dist_sea"]
        return output
