import os
import torch
import tqdm
import argparse
import subprocess
import numpy as np
from torch2trt import torch2trt
from torch2trt.torch2trt import *
from torch2trt import tensorrt_converter
from modules.model import *

@tensorrt_converter('XFeatModel._unfold2d')
def convert_unfold2d(ctx):
    # Retrieve input arguments
    x = ctx.method_args[1]
    ws = ctx.method_args[2] if len(ctx.method_args) > 2 else 2
    
    # Retrieve output tensor
    output_tensor = ctx.method_return
    
    # Extract input tensor shape
    B, C, H, W = x.shape

    # Calculate output dimensions
    out_H = H // ws
    out_W = W // ws
    out_C = C * ws**2

    # Reshape input tensor
    x_reshaped = trt.reshape(x._trt, (-1, C, H, W))

    # Unfold along height and width dimensions
    x_unfolded_H = ctx.network.add_slice(x_reshaped, (0, 0, 0, 0), (1, C, -1, -1), (1, C, ws, W))
    x_unfolded_W = ctx.network.add_slice(x_unfolded_H.get_output(0), (0, 0, 0, 0), (1, C, -1, -1), (1, C, -1, ws))

    # Reshape to desired output shape
    x_reshaped = trt.reshape(x_unfolded_W.get_output(0), (-1, out_C, out_H, out_W))

    # Set output tensor
    output_tensor._trt = x_reshaped

@tensorrt_converter('torch.nn.BatchNorm2d.forward')
def convert_BatchNorm2d(ctx):
    module = ctx.method_args[0]
    input = ctx.method_args[1]
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    output = ctx.method_return
    if module.affine:
        scale = module.weight.detach().cpu().numpy() / np.sqrt(module.running_var.detach().cpu().numpy() + module.eps)
        bias = module.bias.detach().cpu().numpy() - module.running_mean.detach().cpu().numpy() * scale
    else:
        scale = 1.0 / np.sqrt(module.running_var.detach().cpu().numpy() + module.eps)
        bias = - module.running_mean.detach().cpu().numpy() * scale
    power = np.ones_like(scale)
    
    # reshape to 2D
    # layer = ctx.network.add_shuffle(input_trt)
    
    # if len(input.shape) == 2:
    #     layer.reshape_dims = (0, 0, 1, 1)
    # else:
    #     layer.reshape_dims = (0, 0, 0, 1)
    
    layer = ctx.network.add_scale(input_trt, trt.ScaleMode.CHANNEL, bias, scale, power)

    # reshape back to 1D
    # layer = ctx.network.add_shuffle(layer.get_output(0))
    # if len(input.shape) == 2:
    #     layer.reshape_dims = (0, 0)
    # else:
    #     layer.reshape_dims = (0, 0, 0)
    
    output._trt = layer.get_output(0)


def build_trt_engine(
        weights: str,
        imgsz: tuple = (480,640),
        fp16_mode: bool = False,
        int8_mode: bool = False) -> None: 

    if weights.endswith(".pt"):
        # Replace ".pt" with ".onnx"
        trt_weight = weights[:-3] + ".engine"
    else:
        raise Exception("File path does not end with '.pt'.")
    net = XFeatModel().float().eval().cuda()
    net.load_state_dict(torch.load(weights, map_location='cuda'))
    x = torch.ones(1,3,*imgsz, dtype=torch.float32).cuda()
    trt_net = None
    if fp16_mode:
        trt_net = torch2trt(net, [x], fp16_mode=fp16_mode, max_batch_size=1)
    elif int8_mode:
        trt_net = torch2trt(net, [x], int8_mode=int8_mode, max_batch_size=1)
    else:
        trt_net = torch2trt(net, [x], max_batch_size=1)
    
    torch.save(trt_net.state_dict(), trt_weight)

def main():
    parser = argparse.ArgumentParser(description='Create ONNX and TensorRT export for XFeat.')
    parser.add_argument('--weights', type=str, default=f'{os.path.abspath(os.path.dirname(__file__))}/weights/xfeat.pt', help='Path to the weights pt file to process')
    parser.add_argument('--imgsz', type=tuple, default=(480,640), help='Input image size')
    parser.add_argument("--fp16_mode", type=bool, default=False)
    parser.add_argument("--int8_mode", type=bool, default=False)
    args = parser.parse_args()
    weights = args.weights
    imgsz = args.imgsz
    fp16_mode = args.fp16_mode
    int8_mode = args.int8_mode
    build_trt_engine(weights, imgsz, fp16_mode, int8_mode)

if __name__ == '__main__':
    main()