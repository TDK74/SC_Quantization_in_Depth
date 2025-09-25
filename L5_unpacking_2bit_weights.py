import torch


## ------------------------------------------------------##
def unpack_weights(uint8tensor, bits):
    num_values = uint8tensor.shape[0] * 8 // bits

    num_steps = 8 // bits

    unpacked_tensor = torch.zeros((num_values), dtype = torch.uint8)

    unpacked_idx = 0

    mask = 2 ** bits - 1

    for i in range(uint8tensor.shape[0]):
        for j in range(num_steps):
            unpacked_tensor[unpacked_idx] |= uint8tensor[i] >> (bits * j)
            unpacked_idx += 1

    unpacked_tensor &= mask

    return unpacked_tensor

## ------------------------------------------------------##
unpacked_tensor = torch.tensor([177, 255], dtype = torch.uint8)

## ------------------------------------------------------##
unpack_weights(unpacked_tensor, 2)
