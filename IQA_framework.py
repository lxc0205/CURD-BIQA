import random
import torch
import torchvision  
import numpy as np

class IQA_framework():
    def __init__(self, iqa_net):
        super(IQA_framework, self).__init__()
        # load iqa_network
        self.func = iqa_net
        # Load VGG16 model
        self.net = torchvision.models.vgg16(weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1).cuda().features.eval()
        # Feature Layers ID
        self.convlayer_id = [0, 2, 5, 7, 10]
        # Sample Rate
        self.sr = np.array([64, 128, 256, 512, 512])
        # Transform for feature maps
        self.transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=224),
                    torchvision.transforms.ToTensor()])
        self.to_pil = torchvision.transforms.ToPILImage()
        
    def extractFeature(self, img):
        img = torch.as_tensor(img).cuda()
        feat_map = [img]
        cnt = 0
        for i, layer in enumerate(self.net.children()):
            img = layer(img)
            if i in self.convlayer_id:
                img0 = img
                for j in range(img0.shape[1]):
                    if j % self.sr[cnt] == 0:
                        random_channels = [random.randint(0, img0.size(1) - 1) for _ in range(3)] # 生成随机的三个通道索引
                        temp = torch.cat([torch.as_tensor(self.transform(self.to_pil(img0[:, c,:,:]))).unsqueeze(1) for c in random_channels], dim=1).cuda()
                        feat_map.append(temp)
                cnt = cnt + 1
        return feat_map

    def origin_framework(self, img, repeat_mean=False):
        img = torch.as_tensor(img).cuda()
        if repeat_mean:
            pred_scores = []
            for _ in range(10):
                pred = self.func(img)
                pred_scores.append(float(pred.item()))
            return np.mean(pred_scores)
        else:
            return self.func(img)

    def multiscale_framework(self, img, repeat_mean=False):
        feat_map = self.extractFeature(img) # Extract feature map
        layer_scores = []
        for feat in feat_map:
            if repeat_mean:
                pred_scores = []
                for _ in range(10):
                    pred = self.func(feat)
                    pred_scores.append(float(pred.item()))
                score = np.mean(pred_scores)
            else:
                pred = self.func(feat)
                score = float(pred.item())
            layer_scores.append(score)
        return layer_scores
