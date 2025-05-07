import torch
import random
import torchvision  
import numpy as np

class feature_framework():
    def __init__(self, iqa_net):
        super(feature_framework, self).__init__()
        # load iqa_network
        self.function = iqa_net
        # load VGG16 model
        self.net = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).cuda().features.eval()
        # Feature Layers ID
        self.convlayer_id = [1, 5, 9, 18, 27]
        # sample rate 
        self.sr = np.array([64, 128, 256, 512, 512])
        # transform for deep feature maps
        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor()])
        self.to_pil = torchvision.transforms.ToPILImage()
        
    def extract_feature(self, image):
        image = torch.as_tensor(image).cuda()
        feature_maps = [image]
        cnt = 0
        for i, layer in enumerate(self.net.children()):
            image = layer(image)
            if i in self.convlayer_id:
                feature = image
                for j in range(feature.shape[1]):
                    if j % self.sr[cnt] == 0:
                        random_channels = [random.randint(0, feature.size(1) - 1) for _ in range(3)] # 生成随机的三个通道索引
                        temp = torch.cat([torch.as_tensor(self.transform(self.to_pil(feature[:, c,:,:]))).unsqueeze(1) for c in random_channels], dim=1).cuda()
                        feature_maps.append(temp)
                cnt = cnt + 1
        return feature_maps

    def origin(self, image):
        return self.function(torch.as_tensor(image).cuda())

    def multiscale(self, image):
        feature_maps = self.extract_feature(image) # Extract feature map
        layer_scores = []

        for feature in feature_maps:
            pred = self.function(feature)
            score = float(pred.item())
            layer_scores.append(score)
        return layer_scores