import torch
import torch.nn.functional as F
from preds.likelihoods import GaussianLh
from preds.laplace import Laplace

# We only calculate the diagonal elements of the hessian
def logistic_hessian(f):
    f = f[:, :]
    pi = torch.sigmoid(f)
    return pi*(1-pi)


def softmax_hessian(f):
    s = F.softmax(f, dim=-1)
    return s - s*s

# Select memorable points ordered by their lambda values (descending=True picks most important points)
def select_memorable_points(data, model, num_points=10,
                            use_cuda=False, label_set=None, descending=True):

    f, _ = model.get_action(data[0], data[1], data[2], data[3], data[4])
    
    lh = GaussianLh()  # likelihood: GaussianLh for regression, CategoricalLh for classification 
    prior_precision = 1.  # prior
    posterior = Laplace(model, prior_precision, lh)
    posterior.infer([(data, f)], cov_type='kron', dampen_kron=False)   
    _, lamb = posterior.predictive_samples_glm(data, n_samples=1000)     
    _, indices = lamb.sort(descending=descending)
    
    good = data[indices[:num_points]]
    good_target = f[indices[:num_points]]
    bad = data[indices[-num_points:]]
    bad_target = f[indices[-num_points:]]

    return [good, good_target, bad, bad_target]