from pyhessian import hessian

class HessianWrapper:
    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True):
        self.hessian_calculator = hessian(model, criterion, data, dataloader, cuda)

    def compute_top_eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        return self.hessian_calculator.eigenvalues(maxIter, tol, top_n)

    def compute_trace(self, maxIter=100, tol=1e-3):
        return self.hessian_calculator.trace(maxIter, tol)

    def compute_eigenvalue_density(self, iter=100, n_v=1):
        return self.hessian_calculator.density(iter, n_v)

# Example Usage:
# Assuming you have a model, criterion, and data/dataloader defined
# model = ...
# criterion = ...
# data = ...

# Create an instance of the wrapper
wrapper = HessianWrapper(model, criterion, data=data, cuda=True)

# Use the wrapper functions
top_eigenvalues, _ = wrapper.compute_top_eigenvalues(top_n=1)
trace = wrapper.compute_trace()
eigenvalue_density = wrapper.compute_eigenvalue_density()

