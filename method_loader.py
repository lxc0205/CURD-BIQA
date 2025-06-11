import sys
import torch
import pyiqa
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MethodLoader(object):
    """Method class for CURD-BIQA"""
    def __init__(self, method, pyiqa=True, ckpt=None):
        self.method = method
        self.pyiqa = pyiqa
        self.ckpt = ckpt

    def __call__(self):
        return self.load_methods()

    def __str__(self):
        return f'MethodLoader(method={self.method}, pyiqa={self.pyiqa}, ckpt={self.ckpt})'


    def load_methods(self):
        if self.pyiqa:
            iqa_method = self.load_methods_pyiqa(self.method)
            transform_mode = 'pyiqa'
            return iqa_method, transform_mode
        else:
            if self.method == 'maniqa':
                iqa_method = self.load_maniqa(self.ckpt)
                transform_mode = 'maniqa'
                return iqa_method, transform_mode
            elif self.method == 'arniqa':
                iqa_method = self.load_arniqa()
                transform_mode = 'arniqa'
                return iqa_method, transform_mode
            else:
                print('The method is not supported.')
                self.print_pyiqa_list()
                return

    def print_pyiqa_list(self):
        print('The available metrics in pyiqa are:')
        print(pyiqa.list_models())

    def load_methods_pyiqa(self, method):
        iqa_net = pyiqa.create_metric(method)
        flag = 'lower' if iqa_net.lower_better else 'higher'
        print(f'The {flag} value of the metric {method} is better.')
        def scaled_iqa_net(image, scale = 100):
            with torch.no_grad():
                score = iqa_net(image)
                return score * scale
        return scaled_iqa_net

    def load_arniqa(self):
        model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA", regressor_dataset="kadid10k").eval().cuda()
        def iqa_net(img, scale=1):
            with torch.no_grad(), torch.cuda.amp.autocast():
                img_ds = F.interpolate(img, size=(img.size()[2] // 2, img.size()[3] // 2), mode='bilinear', align_corners=False)
                score = model(img, img_ds, return_embedding=False, scale_score=True)
                return score * scale
        return iqa_net

    def load_maniqa(self, ckpt_path):
        if './models/' not in sys.path: sys.path.insert(0, './models/')
        if './models/maniqa' not in sys.path: sys.path.insert(0, './models/maniqa')
        from maniqa.models.maniqa import MANIQA
        iqa_net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224, window_size=4, depths=[2,2], num_heads=[4,4], num_tab=2, scale=0.8)
        iqa_net.load_state_dict(torch.load(ckpt_path))
        iqa_net = iqa_net.cuda()
        iqa_net.eval()
        return iqa_net
