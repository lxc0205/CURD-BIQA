import torch
import random
import torchvision  
import numpy as np
from tqdm import tqdm 
from timm.models.vision_transformer import vit_base_patch16_224_in21k as VisionTransformer

class feature_framework():
    def __init__(self, iqa_net, backbone=None):
        super(feature_framework, self).__init__()
        # load iqa_network
        self.function = iqa_net
        # load backbone model
        self.backbone_name = backbone
        if self.backbone_name == 'vit':
            self.net = VisionTransformer(num_classes=10).cuda().eval()
        elif self.backbone_name == 'vgg16':
            self.net = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).cuda().features.eval()
        elif self.backbone_name is None:
            self.net = None
            print("No multiscale backbone model is used, you can use the original iqa network directly.")
        else:
            raise ValueError("Unsupported backbone. Choose 'vgg16' or 'vit'.")

        # transform for deep feature maps
        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((224, 224)),
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor()])
        self.to_pil = torchvision.transforms.ToPILImage()

    def extract_feature_VIT(self, image):
        convlayer_id = [1, 3, 5, 7, 9]

        image = torch.as_tensor(image).cuda()
        feature_maps = [image]
        for i, layer in enumerate(self.net.children()):
            if layer.__class__.__name__ == 'Sequential':
                for j, sub_layer in enumerate(layer.children()):
                    image = sub_layer(image)
                    if j in convlayer_id:
                        feature = image
                        feature = feature.reshape(1, 196, 24, 32)
                        random_channels = [random.randint(0, feature.size(1) - 1) for _ in range(3)] # 生成随机的三个通道索引
                        temp = torch.cat([torch.as_tensor(self.transform(self.to_pil(feature[:, c,:,:]))).unsqueeze(1) for c in random_channels], dim=1).cuda()
                        feature_maps.append(temp)
            else:
                image = layer(image)
        return feature_maps

    def extract_feature_VGG16(self, image):
        convlayer_id = [1, 5, 9, 18, 27]
        sr = np.array([64, 128, 256, 512, 512])

        image = torch.as_tensor(image).cuda()
        feature_maps = [image]
        cnt = 0
        for i, layer in enumerate(self.net.children()):
            image = layer(image)
            if i in convlayer_id:
                feature = image
                for j in range(feature.shape[1]):
                    if j % sr[cnt] == 0:
                        random_channels = [random.randint(0, feature.size(1) - 1) for _ in range(3)] # 生成随机的三个通道索引
                        temp = torch.cat([torch.as_tensor(self.transform(self.to_pil(feature[:, c,:,:]))).unsqueeze(1) for c in random_channels], dim=1).cuda()
                        feature_maps.append(temp)
                cnt = cnt + 1
        return feature_maps

    def origin(self, image):
        return self.function(torch.as_tensor(image).cuda())

    def multiscale(self, image):
         # Extract feature maps
        if self.backbone_name == 'vit':
            feature_maps = self.extract_feature_VIT(image)
        elif self.backbone_name == 'vgg16':
            feature_maps = self.extract_feature_VGG16(image)
        elif self.backbone_name is None:
            raise ValueError("No multiscale backbone model is used, you can only use the original iqa network directly.")
        else:
            raise ValueError("Unsupported backbone. Choose 'vgg16' or 'vit'.")
        
        layer_scores = []

        for feature in feature_maps:
            pred = self.function(feature)
            score = float(pred.item())
            layer_scores.append(score)

        layer_scores = torch.tensor(layer_scores)
        return layer_scores
    
    def origin_loader(self, data):
        scores, labels = [], []
        for image, label in tqdm(data):
            score = self.origin(image)
            scores.append(float(score.item()))
            labels = labels + label.tolist()
        scores, labels = torch.tensor(scores), torch.tensor(labels)
        return scores, labels

    def multiscale_loader(self, data):
        matrix = []
        for img, label in tqdm(data):
            layer_scores = self.multiscale(img)
            matrix.append(torch.hstack((layer_scores, torch.tensor(label))))
        matrix = torch.stack(matrix)
        scores, labels = matrix[:,:-1], matrix[:,-1].unsqueeze(1)
        return scores, labels