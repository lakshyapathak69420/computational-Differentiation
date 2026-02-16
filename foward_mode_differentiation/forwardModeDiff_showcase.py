from tensor import Tensor
from tensorFunctions import Tfs


def foo(x):
    return Tfs.sine(x) + x**2 

z = Tensor(2)
y = foo(z)
grad_of_y = y.get_gradient()
print(grad_of_y)