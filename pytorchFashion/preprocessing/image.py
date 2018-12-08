#encoding:utf-8
import torch
import torch.nn.functional as F

class normalize(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        if not F._is_tensor_image(tensor):
            raise TypeError('tensor is not a torch image.')
        # TODO: make efficient
        for t in tensor:
            t.sub_(0.5).multi_(2.0)
        return tensor
