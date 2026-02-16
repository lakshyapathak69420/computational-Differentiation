import numpy as np
class Tensor:
    """
    BASIC DESIGN CHOICES:
        every thing has exact two dimensions np array.
        every vector is a row vector 
        with every value there is also stored the transpose of it 

    """
    def __init__(self, value, operation = None, *parents):
        self.data_type = np.float32
        if np.array(value).ndim == 0:
            self.value = np.array([[value]], dtype=self.data_type)  
        elif np.array(value).ndim == 1:
            self.value =  np.array([value], dtype= self.data_type)
        elif np.array(value).ndim == 2:
            self.value = np.array(value, dtype= self.data_type)
        else:
            raise ValueError("Cant handle more that 2 dimension")
        
        self.transpose_value = self.value.T
        self.grad = np.zeros_like(self.value, dtype=np.float32)  # d self / d final_node
        self.operation = operation
        self.parents = parents

    def __add__(self, tensor):
        operation = "+"
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)        
        value = self.value + tensor.value
        return Tensor(value, operation, self, tensor)
    
    def __sub__(self, tensor):
        operation = "-"
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)        
        value = self.value - tensor.value
        return Tensor(value, operation, self, tensor)
    
    def __mul__(self, tensor):
        operation = "*"
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor) 
        value = self.value + tensor.value
        return Tensor(value, operation, self, tensor)
    
    def __truediv__(self, tensor):
        operation = "/"
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)
        if (tensor.value == 0).all():
            raise ZeroDivisionError("Denominatior is zero")
        value = self.value / tensor.value
        return Tensor(value, operation, self, tensor)
    

    def _process_gradient(self):
        """
            so it takes the current node for it updates the disjoint/gradient of there parents
            if the there are two parents p_1 and p_1 the it updates them as follows
            disjoint(p_i) = dijoint(self) * df{p_1, p_2} / dp_i for i = {1 , 2} and here self = f{p_1, p_2}

            if there is only one parent p_i then 
            disjoint(p_i) = dijoint(self) * df{p_i} / dp_i 

            if no parent then do nothing and return  
        """
        if self.parents is None:
                return

        if len(self.parents) == 2:
            p1, p2 = self.parents
            if self.operation == "+":
                p1.grad += self.grad
                p2.grad += self.grad
            
            elif self.operation == "-":
                p1.grad += self.grad
                p2.grad -= self.grad

            elif self.operation == "*":
                
                p1.grad += self.grad * p2.value
                p2.grad += self.grad * p1.value

            elif self.operation == "/":
                pass

            elif self.operation == "matmul":
                p1.grad += np.matmul(self.grad, p2.transpose_value)
                p2.grad += np.matmul(p1.transpose_value, self.grad)

            elif self.operation == "CE":
                logits, labels = self.parents
                shifted_logits = logits.value - np.max(logits.value)
                exps = np.exp(shifted_logits)
                softmax_probs = exps / np.sum(exps)
                
                
                local_gradient = softmax_probs - labels.value
                logits.grad += self.grad * local_gradient

        elif len(self.parents) == 1:
        
                p = self.parents[0]
                
                if self.operation == "relu":
            
                    # Create a mask: 1.0 where value > 0, else 0.0
                    # Works for scalars, vectors (dim 1, 2, etc.)
                    mask = (p.value > 0).astype(self.data_type)
                    
                    p.grad += mask * self.grad
                    
                elif self.operation == "tanh":
                    # Derivative is (1 - tanh^2)
                    p.grad += (1 - np.tanh(p.value)**2) * self.grad


    def backward(self):
        # 1. Iterative Topological Sort (DFS Post-Order)
        ordering = []
        visited = set()
        processed = set() # To track when we've explored all parents of a node
        stack = [self]

        while stack:
            node = stack[-1] # Look at the top
            
            if node not in visited:
                # First time seeing this node, mark visited and add parents to stack
                visited.add(node)
                for p in node.parents:
                    if p not in visited:
                        stack.append(p)
            else:
                
                stack.pop()
                if node not in processed:
                    ordering.append(node)
                    processed.add(node) 

        self.grad = np.ones_like(self.value)
        for node in reversed(ordering):
            node._process_gradient()

