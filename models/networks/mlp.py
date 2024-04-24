import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        aux_data=[],
    ):
        """
        Initializes an MLP Classification Head
        Args:
            hidden_dim (list): list of hidden dimensions for the MLP
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.aux_data = aux_data
        self.aux = len(self.aux_data) > 0
        if self.aux:
            hidden_dim_aux = hidden_dim
            hidden_dim_aux[-1] = 128
            final_dim_aux_dict = {
                "land_cover": 12,
                "climate": 30,
                "soil": 14,
                "road_index": 1,
                "drive_side": 1,
                "dist_sea": 1,
            }
            self.idx = {}
            final_dim_aux = 0
            for col in self.aux_data:
                self.idx[col] = [
                    final_dim_aux + i for i in range(final_dim_aux_dict[col])
                ]
                final_dim_aux += final_dim_aux_dict[col]
            dim = [initial_dim] + hidden_dim_aux + [final_dim_aux]
            args = self.init_layers(dim, norm, activation)
            self.mlp_aux = nn.Sequential(*args)
        dim = [initial_dim] + hidden_dim + [final_dim]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                # args.append(norm(dim[i + 1]))
                args.append(norm(4, dim[i + 1]))
                args.append(activation())
        return args

    def forward(self, x):
        """Predicts GPS coordinates from an image.
        Args:
            x: torch.Tensor with features
        """
        if self.aux:
            out = {"gps": self.mlp(x[:, 0, :])}
            x = self.mlp_aux(x[:, 0, :])
            for col in list(self.idx.keys()):
                out[col] = x[:, self.idx[col]]
            return out
        return self.mlp(x[:, 0, :])

class MLPResNet(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        aux_data=[],
    ):
        """
        Initializes an MLP Classification Head
        Args:
            hidden_dim (list): list of hidden dimensions for the MLP
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.aux_data = aux_data
        self.aux = len(self.aux_data) > 0
        if self.aux:
            hidden_dim_aux = hidden_dim
            hidden_dim_aux[-1] = 128
            final_dim_aux_dict = {
                "land_cover": 12,
                "climate": 30,
                "soil": 14,
                "road_index": 1,
                "drive_side": 1,
                "dist_sea": 1,
            }
            self.idx = {}
            final_dim_aux = 0
            for col in self.aux_data:
                self.idx[col] = [
                    final_dim_aux + i for i in range(final_dim_aux_dict[col])
                ]
                final_dim_aux += final_dim_aux_dict[col]
            dim = [initial_dim] + hidden_dim_aux + [final_dim_aux]
            args = self.init_layers(dim, norm, activation)
            self.mlp_aux = nn.Sequential(*args)
        dim = [initial_dim] + hidden_dim + [final_dim]
        args = self.init_layers(dim, norm, activation)
        self.mlp = nn.Sequential(*args)

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                # args.append(norm(dim[i + 1]))
                args.append(norm(4, dim[i + 1]))
                args.append(activation())
        return args

    def forward(self, x):
        """Predicts GPS coordinates from an image.
        Args:
            x: torch.Tensor with features
        """
        if self.aux:
            out = {"gps": self.mlp(x[:, 0, :])}
            x = self.mlp_aux(x[:, 0, :])
            for col in list(self.idx.keys()):
                out[col] = x[:, self.idx[col]]
            return out
        return self.mlp(x)


class MLPCentroid(nn.Module):
    def __init__(
        self,
        initial_dim=512,
        hidden_dim=[128, 32, 2],
        final_dim=2,
        norm=nn.InstanceNorm1d,
        activation=nn.ReLU,
        aux_data=[],
    ):
        """
        Initializes an MLP Classification Head
        Args:
            hidden_dim (list): list of hidden dimensions for the MLP
            norm (nn.Module): normalization layer
            activation (nn.Module): activation layer
        """
        super().__init__()
        self.aux_data = aux_data
        self.aux = len(self.aux_data) > 0
        dim = [initial_dim] + hidden_dim + [final_dim // 3]
        args = self.init_layers(dim, norm, activation)
        self.classif = nn.Sequential(*args)
        dim = [initial_dim] + hidden_dim + [2 * final_dim // 3]
        args = self.init_layers(dim, norm, activation)
        self.reg = nn.Sequential(*args)
        # torch.nn.init.normal_(self.reg.weight, mean=0.0, std=0.01)
        if self.aux:
            self.dim = [initial_dim] + hidden_dim
            self.predictors = {"gps": self.mlp}
            self.init_aux(dim, norm, activation)

    def init_layers(self, dim, norm, activation):
        """Initializes the MLP layers."""
        args = [nn.LayerNorm(dim[0])]
        for i in range(len(dim) - 1):
            args.append(nn.Linear(dim[i], dim[i + 1]))
            if i < len(dim) - 2:
                # args.append(norm(dim[i + 1]))
                args.append(norm(4, dim[i + 1]))
                args.append(activation())
        return args

    def init_aux(self, dim, norm, activation):
        final_dim_aux = {
            "land_cover": 12,
            "climate": 30,
            "soil": 14,
            "road_index": 1,
            "drive_side": 1,
            "dist_sea": 1,
        }
        if "land_cover" in self.aux_data:
            args = self.init_layers(
                self.dim + [final_dim_aux["land_cover"]], norm, activation
            )
            self.land_cover = nn.Sequential(*args)
            self.predictors["land_cover"] = self.land_cover
        if "road_index" in self.aux_data:
            args = self.init_layers(
                self.dim + [final_dim_aux["road_index"]], norm, activation
            )
            self.road_index = nn.Sequential(*args)
            self.predictors["road_index"] = self.road_index
        if "drive_side" in self.aux_data:
            args = self.init_layers(
                self.dim + [final_dim_aux["drive_side"]], norm, activation
            )
            self.drive_side = nn.Sequential(*args)
            self.predictors["drive_side"] = self.drive_side
        if "climate" in self.aux_data:
            args = self.init_layers(
                self.dim + [final_dim_aux["climate"]], norm, activation
            )
            self.climate = nn.Sequential(*args)
            self.predictors["climate"] = self.climate
        if "soil" in self.aux_data:
            args = self.init_layers(
                self.dim + [final_dim_aux["soil"]], norm, activation
            )
            self.soil = nn.Sequential(*args)
            self.predictors["soil"] = self.soil
        if "dist_sea" in self.aux_data:
            args = self.init_layers(
                self.dim + [final_dim_aux["dist_sea"]], norm, activation
            )
            self.dist_sea = nn.Sequential(*args)
            self.predictors["dist_sea"] = self.dist_sea

    def forward(self, x):
        """Predicts GPS coordinates from an image.
        Args:
            x: torch.Tensor with features
        """
        if self.aux:
            return {
                col: self.predictors[col](x[:, 0, :]) for col in self.predictors.keys()
            }
        return torch.cat([self.classif(x[:, 0, :]), self.reg(x[:, 0, :])], dim=1)


class Identity(nn.Module):
    def __init__(
        self
    ):
        """
        Initializes an Identity module
        """
        super().__init__()

    def forward(self, x):
        """
        Return same as input
        """
        return x