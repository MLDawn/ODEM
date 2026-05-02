import torch
from torch.autograd.functional import jacobian
import inspect
def compute(gen_mu,f, q_theta_mu, device):
    '''
    :param gen_mu: The tensor of generalised posterior expectations (mu_x, mu_x_dot, mu_x_dotdot, etc.)
    :param f: The function describing flow on state dynamics
    :param kx: The number of generalised coordinates in x
    :param dx: The dimension of hidden state x
    :param q_theta_mu: The list of tensor parameters
    :return: The calculated error e_x
    '''
    #Crucial to set create_graph=True: This argument ensures that the operations involved
    # in computing the Jacobian are tracked in the computational graph. As a result, you
    # can backpropagate through jacob_f_eval to calculate the gradients with respect to
    # the inputs (like q_theta_mu) during parameter learning
    kx, dx = gen_mu.shape[0], gen_mu.shape[1]
    jacob_f_eval = torch.autograd.functional.jacobian(lambda x: f(x, q_theta_mu), gen_mu[0], create_graph=True)

    pred = []
    for i in range(kx):
        # i=0 then we are generating predictions for x', which is f(mu_x)
        if i == 0:
            pred = [f(gen_mu[0], q_theta_mu)]
        # if i > 0, then it means we are generating predictions for x'',x''', etc.
        # This requires the Jacobian of f() evaluated at mu_x
        else:
            pred.append(torch.matmul(jacob_f_eval, gen_mu[i]))
    pred = torch.stack(pred)
    #Build the ground-truth vector (it should have a zero vector at the end, which is the regulariser)
    gt = torch.cat((gen_mu[1:], torch.zeros(1,dx).to(device)),dim=0)
    e_x = gt - pred
    return e_x
