
import torch
from torch import nn
from torch.nn.modules.utils import _single, _pair, _triple


class MedianPool1d(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.pooling_op_1d = MedianPool1DFunction()
    
    def forward(self, input):
        ic = input.size()[1]

        new_ic = torch.zeros(1, dtype=torch.int16, requires_grad=False)
        new_ic.fill_(ic)
        ph, _ = torch.max(torch.tensor(2), 0)

        padding = ph
        # if ~torch.eq(padding, 0):
        padding = input[:,:padding]
        x = torch.cat((input, padding), dim=1)
        # else: x = input
        windowed_input = x.unfold(dimension=1, size=3, step=1)
        # unfolded_input = unfolded_input.contiguous().view(unfolded_input.size()[:2] + (-1,))
        unfolded_input = windowed_input.unfold(dimension=-1, size=2, step=1)
        # unfolded_input = unfolded_input.contiguous().view(unfolded_input.size()[:3] + (-1,))
        first, _ = torch.max(unfolded_input, dim=-1)
        second, _ = torch.min(unfolded_input, dim=-1)
        maxs = torch.clone(first[:,:,0])
        mins = torch.clone(second[:,:,1])
        new_tensor = torch.stack((maxs, mins), dim=-1)
        output, _ = torch.max(new_tensor, dim=-1)
        return output

                



class MyConv2d(torch.nn.modules.conv._ConvNd):
    """
    Implements a standard convolution layer that can be used as a regular module
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1,
                 bias=True, padding_mode='zeros', groups=0):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        

    def conv2d_forward(self, input):
        return self.myconv2d(input)

    def forward(self, input):
        return self.conv2d_forward(input)


    def myconv2d(self, input):
        """
        Function to process an input with a standard convolution
        """
        batch_size, in_channels, in_h, in_w = input.shape
        out_channels, _, kh, kw = self.weight.shape
        new_in_h = torch.zeros(1, dtype=torch.int16, requires_grad=False)
        new_in_h.fill_(in_h)

        new_in_w = torch.zeros(1, dtype=torch.int16, requires_grad=False)
        new_in_w.fill_(in_w)
        # kh = torch.tensor(kh)

        # kw = torch.tensor(kw)

        out_h = int(torch.add(torch.div(torch.add(torch.sub(new_in_h, kh), torch.mul(self.padding[0], 2)), self.stride[0]), 1))

        out_w = int(torch.add(torch.div(torch.add(torch.sub(new_in_w, kw), torch.mul(self.padding[1], 2)), self.stride[1]), 1))
        # out_w = int((in_w - kw + 2 * self.padding[1]) / self.stride[1] + 1)
        unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=self.dilation, padding=self.padding, stride=self.stride)
        inps_unf=unfold(input)
        # print(input.shape)
        # print(inps_unf.shape)
        inps_unf = self.median_pool2d(inps_unf)
        # print(inps_unf.shape)
        w_ = self.weight.view(self.weight.size(0), -1).t()
        if self.bias is None:
            out_unf = inps_unf.transpose(1,2).matmul(w_).transpose(1, 2)
        else:
            out_unf = torch.add(inps_unf.transpose(1,2).matmul(w_), self.bias).transpose(1, 2)
        
        out = out_unf.view(batch_size, out_channels, out_h, out_w)

        return out.float()

    def median_pool2d(self, input):
        ic = input.size()[1]
        new_ic = torch.zeros(1, dtype=torch.int16, requires_grad=False)
        new_ic.fill_(ic)

        # if torch.eq(torch.fmod(new_ic,1),0):
            # comp = 2
            # new_comp = torch.zeros(1, dtype=torch.int16, requires_grad=False)
            # new_comp.fill_(comp)
            
        ph, _ = torch.max(torch.tensor(2), 0)
        # else:
        #     ph, _ = max(3 - (new_ic % 1), 0)

        # padding depth
        # padding = 2
        padding = ph
        # if ~torch.eq(padding, 0):
        padding = input[:,:padding,:]
        x = torch.cat((input, padding), dim=1)
        # else: x = input
        windowed_input = x.unfold(dimension=1, size=3, step=1)
        # unfolded_input = unfolded_input.contiguous().view(unfolded_input.size()[:2] + (-1,))
        unfolded_input = windowed_input.unfold(dimension=-1, size=2, step=1)
        # unfolded_input = unfolded_input.contiguous().view(unfolded_input.size()[:3] + (-1,))
        first, _ = torch.max(unfolded_input, dim=-1)
        second, _ = torch.min(unfolded_input, dim=-1)
        maxs = torch.clone(first[:,:,:,0])
        mins = torch.clone(second[:,:,:,1])
        new_tensor = torch.stack((maxs, mins), dim=-1)

        output, _ = torch.max(new_tensor, dim=-1)
        return output

def rec_dnn_exploration_mf(model, new_layer=None):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # self, in_channels, out_channels, kernel_size, stride=1,
            #      padding=0, dilation=1,
            #      bias=True, padding_mode='zeros', groups=0
            new_layer = MyConv2d(in_channels=module.in_channels, 
                                       out_channels=module.out_channels, 
                                       kernel_size=module.kernel_size, 
                                       stride=module.stride, 
                                       padding=module.padding, 
                                       dilation=module.dilation,
                                       groups=module.groups,
                                       bias=True,
                                       padding_mode=module.padding_mode)
            new_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_layer.bias = torch.nn.parameter.Parameter(module.bias)
            setattr(model, name, new_layer)
        elif isinstance(module, nn.Linear):
            median_pool = MedianPool1d()
            new_layer = nn.Sequential(*[median_pool, module])
            setattr(model, name, new_layer)
        elif name!='layer1':
            rec_dnn_exploration_mf(module, new_layer)
    return model


def rec_dnn_exploration(model, new_layer=None):
    for idx in range(len(model._modules.values())):
        module = model._modules[list(model._modules.keys())[idx]]
        name = list(model._modules.keys())[idx]
        if isinstance(module, torch.nn.BatchNorm2d):
            # Check if the next layer is ReLU
            if idx != len(model._modules.values())-1:
                next_module = model._modules[list(model._modules.keys())[idx+1]]
                next_name = list(model._modules.keys())[idx+1]
                if isinstance(next_module, torch.nn.ReLU):
                    # Exchange positions of BatchNorm and ReLU
                    setattr(model, name, torch.nn.ReLU(inplace=True))
                    setattr(model, next_name, module)

        else:
            # Recursively explore the next layer
            rec_dnn_exploration(module)
 
    return model