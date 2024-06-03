import torch
from torch import nn

from model.resnet_3d import generate_model


class DeepCariesClassifer(nn.Module):
    def __init__(self, output_type, num_inp_channels, num_fmap_channels, att_dim, num_classes, patch_size, patch_stride,
                 k_min):
        super().__init__()
        self.num_classes = num_classes
        self.k_min = k_min

        self.backbone = generate_model(18, n_classes=num_classes)  # avail: 10, 18, 34, 50, 101, 152, 200
        self.patch_extractor = nn.AvgPool3d(patch_size, patch_stride)

        fea_ = nn.Linear(num_fmap_channels, num_classes)
        fea_att = nn.Softmax(dim=-1) if output_type == 'multiclass' else nn.Sigmoid()
        self.output = nn.Sequential(
            fea_,
            fea_att
        )

        self.att_tanh = nn.Sequential(
            nn.Linear(num_fmap_channels, att_dim),
            nn.Tanh()
        )
        self.att_sigm = nn.Sequential(
            nn.Linear(num_fmap_channels, att_dim),
            nn.Sigmoid()
        )
        self.att_outer = nn.Sequential(
            nn.Linear(att_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, img, output_heatmaps=False):
        x = self.backbone(img)
        x = self.patch_extractor(x)

        b, c, _, _, _ = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1)
        b, k, c = x.shape
        x = x.reshape(-1, c)

        x_local = self.output(x).view(b, k, self.num_classes)

        ## attention
        x_weight = self.att_outer(self.att_tanh(x) * self.att_sigm(x)).view(b, k, 1)
        pred = torch.sum(x_local * x_weight, dim=1) / torch.clamp(torch.sum(x_weight, dim=1), min=self.k_min)

        return pred
