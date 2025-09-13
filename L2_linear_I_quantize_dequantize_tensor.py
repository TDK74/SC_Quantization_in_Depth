import torch

from helper import plot_quantization_errors


## ------------------------------------------------------##
def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype = torch.int8):
    scaled_and_shifted_tensor = tensor / scale + zero_point

    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min, q_max).to(dtype)

    return q_tensor

## ------------------------------------------------------##
### a dummy tensor to test the implementation
test_tensor = torch.tensor(
                        [[191.6, -13.5, 728.6],
                        [92.14, 295.5,  -184],
                        [0,     684.6, 245.5]]
                        )

## ------------------------------------------------------##
scale = 3.5
zero_point = -70

## ------------------------------------------------------##
quantized_tensor = linear_q_with_scale_and_zero_point(test_tensor, scale, zero_point)

## ------------------------------------------------------##
quantized_tensor

## ------------------------------------------------------##
dequantized_tensor = scale * (quantized_tensor.float() - zero_point)

## ------------------------------------------------------##
dequantized_tensor

## ------------------------------------------------------##
scale * (quantized_tensor - zero_point)

## ------------------------------------------------------##
def linear_dequantization(quantized_tensor, scale, zero_point):
    return scale * (quantized_tensor.float() - zero_point)

## ------------------------------------------------------##
dequantized_tensor = linear_dequantization(quantized_tensor, scale, zero_point)

## ------------------------------------------------------##
dequantized_tensor

## ------------------------------------------------------##
plot_quantization_errors(test_tensor, quantized_tensor, dequantized_tensor)

## ------------------------------------------------------##
dequantized_tensor - test_tensor

## ------------------------------------------------------##
(dequantized_tensor - test_tensor).square()

## ------------------------------------------------------##
(dequantized_tensor - test_tensor).square().mean()
