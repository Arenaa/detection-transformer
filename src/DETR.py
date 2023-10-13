from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet50
import torchvision.transforms as T
torch.set_grad_enabled(False)

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                  num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        self.backbone = resnet50

        del self.backbone.fc

        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        self.transforemr = nn.Transformer(hidden_dim, nheads, num_encoder_layers,
                                          num_decoder_layers)

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone.conv1(input)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        h = self.conv(x)

        H, W = h.shape[-2:]
        pos = torch.cat([self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                         self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                         ], dim=1).flatten(0, 1).unsqueeze(1)


