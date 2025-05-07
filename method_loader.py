import sys
import torch
import pyiqa

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_methods(method, ckpt = None):
    if ckpt is None:
        iqa_net = load_methods_pyiqa(method)
        transform_mode = 'pyiqa'
        return iqa_net, transform_mode
    elif method == 'maniqa':
        iqa_net = load_maniqa(ckpt)
        transform_mode = 'maniqa'
        return iqa_net, transform_mode
    else:
        print('The method is not supported.')
        return

def print_pyiqa_list():
    print('The available metrics in pyiqa are:')
    print(pyiqa.list_models())

def load_methods_pyiqa(method):
    print_pyiqa_list()
    iqa_net = pyiqa.create_metric(method, device)
    flag = 'lower' if iqa_net.lower_better else 'higher'
    print(f'The {flag} value of the metric {method} is better.')
    def scaled_iqa_net(image, scale = 100):
        with torch.no_grad():
            score = iqa_net(image)
            return score * scale
    return scaled_iqa_net

def load_maniqa(ckpt_path):
    if './models/' not in sys.path: sys.path.insert(0, './models/')
    if './models/maniqa' not in sys.path: sys.path.insert(0, './models/maniqa')
    from maniqa.models.maniqa import MANIQA
    iqa_net = MANIQA(embed_dim=768, num_outputs=1, dim_mlp=768, patch_size=8, img_size=224, window_size=4, depths=[2,2], num_heads=[4,4], num_tab=2, scale=0.8)
    iqa_net.load_state_dict(torch.load(ckpt_path))
    iqa_net = iqa_net.cuda()
    iqa_net.eval()
    return iqa_net
