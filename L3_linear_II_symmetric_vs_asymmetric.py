import torch

from helper import linear_q_with_scale_and_zero_point
from helper import linear_dequantization, plot_quantization_errors
from helper import quantization_error


## ------------------------------------------------------##
def get_q_scale_symmetric(tensor, dtype = torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max

    return r_max / q_max

## ------------------------------------------------------##
test_tensor = torch.randn((4, 4))

## ------------------------------------------------------##
test_tensor # print()

## ------------------------------------------------------##
get_q_scale_symmetric(test_tensor)

## ------------------------------------------------------##
def linear_q_symmetric(tensor, dtype = torch.int8):
    scale = get_q_scale_symmetric(tensor)

    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale = scale,
                                                        zero_point = 0, dtype = dtype)

    return quantized_tensor, scale

## ------------------------------------------------------##
quantized_tensor, scale = linear_q_symmetric(test_tensor)

## ------------------------------------------------------##
dequantized_tensor = linear_dequantization(quantized_tensor, scale, 0)

## ------------------------------------------------------##
plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)

## ------------------------------------------------------##
print(f"""Quantization Error : {quantization_error(test_tensor, dequantized_tensor)}""")
