# mlp files were becoming a mess, so I decided to separate the matrix multiplication
import torch

weight_mat = torch.arange(8, dtype=torch.float32).view(2, 4)

print("weight_mat")
print(weight_mat)

input_vec = torch.ones(4, dtype=torch.float32).view(4, 1)

print("input_vec")
print(input_vec)

out = weight_mat @ input_vec

print("out")
print(out)

with open('./mats.bin', 'wb') as f:
    f.write(weight_mat.numpy().tobytes())
    f.write(input_vec.numpy().tobytes())

