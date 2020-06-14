import torch
from  collections import OrderedDict

saved_state_dict = torch.load('../checkpoints/init/resnet101_stem.pth')

new_state=OrderedDict()
for k, v in saved_state_dict.items():
    if k=='conv1.0.weight':
        new_state.update({'conv1.weight':v})
    elif k=='conv1.1.weight':
        new_state.update({'bn1.weight': v})
    elif k=='conv1.1.bias':
        new_state.update({'bn1.bias': v})
    elif k=='conv1.1.running_mean':
        new_state.update({'bn1.running_mean': v})
    elif k=='conv1.1.running_var':
        new_state.update({'bn1.running_var': v})
    elif k=='conv1.3.weight':
        new_state.update({'conv2.weight': v})
    elif k=='conv1.4.weight':
        new_state.update({'bn2.weight':v})
    elif k=='conv1.4.bias':
        new_state.update({'bn2.bias': v})
    elif k=='conv1.4.running_mean':
        new_state.update({'bn2.running_mean': v})
    elif k=='conv1.4.running_var':
        new_state.update({'bn2.running_var': v})
    elif k=='conv1.6.weight':
        new_state.update({'conv3.weight':v})
    elif k=='bn1.weight':
        new_state.update({'bn3.weight': v})
    elif k=='bn1.bias':
        new_state.update({'bn3.bias': v})
    elif k=='bn1.running_mean':
        new_state.update({'bn3.running_mean': v})
    elif k=='bn1.running_var':
        new_state.update({'bn3.running_var': v})
    else:
        new_state.update({k: v})



torch.save(new_state, '../checkpoints/init/new_resnet101_stem.pth')







