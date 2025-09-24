import torch


## ------------------------------------------------------##
def pack_weights(uint8tensor, bits):
    if uint8tensor.shape[0] * bits % 8 != 0:
        raise ValueError(f"The input shape needs to be a mutiple "
                        f"of {8 / bits} - got {uint8tensor.shape[0]}")

    num_values = uint8tensor.shape[0] * bits // 8

    num_steps = 8 // bits

    unpacked_idx = 0

    packed_tensor = torch.zeros((num_values), dtype = torch.uint8)

    for i in range(num_values):
        for j in range(num_steps):
            packed_tensor[i] |= uint8tensor[unpacked_idx] << (bits * j)
            unpacked_idx += 1

    return packed_tensor

## ------------------------------------------------------##
unpacked_tensor = torch.tensor([1, 0, 3, 2], dtype = torch.uint8)

## ------------------------------------------------------##
pack_weights(unpacked_tensor, 2)

## ------------------------------------------------------##
unpacked_tensor = torch.tensor([1, 0, 3, 2, 3, 3, 3, 3], dtype = torch.uint8)

## ------------------------------------------------------##
pack_weights(unpacked_tensor, 2)
