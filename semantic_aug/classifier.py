import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights

class ClassificationModel(nn.Module):
    
    def __init__(self, num_classes: int, backbone: str = "resnet50"):
        
        super(ClassificationModel, self).__init__()

        self.backbone = backbone
        if backbone == "resnet50":
            self.base_model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.out = nn.Linear(2048, num_classes)
        elif backbone == "vit":
            self.base_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            self.base_model.heads = nn.Linear(768, num_classes)
        elif backbone == "vit_l":
            self.base_model = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
            self.base_model.heads = nn.Linear(1024, num_classes)  # ViT-L has 1024-dim features instead of 768

    def forward(self, image):
        
        x = image

        if self.backbone == "resnet50":
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)

            return self.out(x)
        
        elif self.backbone == "vit" or self.backbone == "vit_l":
            return self.base_model(x)