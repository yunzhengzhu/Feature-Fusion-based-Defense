from torchvision import models
import torch
import torch.nn as nn

class Medical_Vgg_Model(nn.Module):
    def __init__(self, pretrain=False, num_feat_layers=None, feat_embedding_layer=None, in_feats=None, num_classes=2, freeze=False, image_free=False):
        super().__init__()
        if num_feat_layers is not None:
            self.num_feat_layers = num_feat_layers
            self.in_feats = in_feats
            self.feat_group_layers = nn.ModuleList()
            for i in range(num_feat_layers):
                self.feat_group_layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_feats[i], 1, 3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                    )
                )
        else:
            self.feat_group_layers = None
        
        self.image_free = image_free
        if feat_embedding_layer is not None:
            if num_feat_layers is None:
                self.feat_embedding_layer = nn.Sequential(nn.Conv2d(in_feats, 3, 3, stride=1, padding=1), nn.ReLU(inplace=True))
            else:
                in_feats = num_feat_layers if image_free else num_feat_layers + 1
                self.feat_embedding_layer = nn.Sequential(nn.Conv2d(in_feats, 3, 3, stride=1, padding=1), nn.ReLU(inplace=True))
        else:
            self.feat_embedding_layer = None

        self.vgg16 = models.vgg16(pretrained=pretrain)

        if freeze:
            for param in self.vgg16.parameters():
                param.reqires_grad = False

        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = torch.nn.Linear(num_features, 1024)
        self.vgg16.classifier.add_module("7", torch.nn.Linear(1024, num_classes))

    def forward(self, x):
        if self.feat_group_layers is not None:
            feats = x[:, 1:]
            split_feats = torch.split(feats, self.in_feats, dim=1)

        x = x[:, 0:1]
       
        if self.feat_group_layers is not None:
            for i, l in enumerate(self.feat_group_layers):
                if i == 0:
                    feat_embed = l(split_feats[0])
                else:
                    feat_embed = torch.cat((l(split_feats[i]), feat_embed), dim=1)
           
            if self.image_free:
                x = feat_embed
            else:
                x = torch.cat((x, feat_embed), dim=1)

        if self.feat_embedding_layer:
            x = self.feat_embedding_layer(x)

        #if x.shape[1] < self.vgg16.features[0].in_channels:
            #print("Warning: Using the hard way to replicate the channels to match pretrained imagenet model with 3!")
        #    x = x.repeat(1, self.vgg16.features[0].in_channels, 1, 1)
        #elif x.shape[1] > self.vgg16.features[0].in_channels:
        #    raise NotImplementedError
        #    #ids = torch.randperm(x.shape[1])
        #    #x = x[:, ids[:3]]
        x = self.vgg16(x)

        return x
