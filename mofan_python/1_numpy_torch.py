import torch
import numpy as np

# 1
# np_data = np.arange(6).reshape((2, 3))
# torch_data = torch.from_numpy(np_data)
# tensor2array = torch_data.numpy()
# print(
#     '\nnumpy:', np_data,
#     '\ntorch:', torch_data,
#     '\ntensor2array', tensor2array,
# )


# 2  abs
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)  # 32bit floating point
# http://pytorch.org/docs/torch.html#math-operations
# print abs()
# print(
#     '\nabs',
#     '\nnumpy:', np.abs(data),   # [1 2 1 2]
#     '\ntorch:', tensor.abs(tensor)       # [1 2 1 2]
# )
# print sin()
# print(
#     '\nsin',
#     '\nnumpy:', np.sin(data),   # [1 2 1 2]
#     '\ntorch:', torch.sin(tensor)       # [1 2 1 2]
# )
# print(
#     '\nmean',
#     '\nnumpy:', np.mean(data),   # [1 2 1 2]
#     '\ntorch:', torch.mean(tensor)       # [1 2 1 2]
# )

# 3 mutiply
# data = [[1, 2], [3, 4]]
# tensor = torch.FloatTensor(data)  # 32-bit floating point
# data = np.array(data)
# print(
#     # '\ndata', np.matmul(data, data)
#     '\nnumpy:', data.dot(data),

#     '\ntorch:', sensor.mm(tensor)
#     '\ntorch:', torch.mm(tensor, tensor)

#     '\ntorch:', torch.dot(tensor)   # different
# )

##################################################
# variable applement
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)  # requires_grad use to define if infer backward(gradient)

t_out = torch.mean(tensor*tensor)  # x^2
v_out = torch.mean(variable*variable)

# print(tensor)
# print(variable)
print(t_out)
print(v_out)
# calculate gradient
v_out.backward()
# v_out = 1/4*sum(var*var)
# d(v_out)/d(var) = 1/4*2*variable=variable/2
print(variable.grad)
print(variable.data)
print(variable.data.numpy())
print(variable)


