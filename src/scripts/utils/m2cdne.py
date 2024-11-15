import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# import globalvar as gl


class FlipGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def flip_gradient(x, lambda_=1.0):
    return FlipGradient.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    def __init__(self, n_emb, Ada_lambda):
        super(DomainDiscriminator, self).__init__()
        self.Ada_lambda = Ada_lambda
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(n_emb, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Define the final domain classification layer
        self.W_domain = nn.Parameter(torch.Tensor(128, 2))
        self.b_domain = nn.Parameter(torch.Tensor(2))
        
        # Initialize weights
        nn.init.trunc_normal_(self.W_domain, std=1.0 / math.sqrt(128.0 / 2.0))
        nn.init.constant_(self.b_domain, 0.1)

    def forward(self, x, d_label):
        h_grl = flip_gradient(x, self.Ada_lambda)

        # Forward pass through fully connected layers
        h_dann_1 = F.relu(self.fc1(h_grl))
        h_dann_2 = F.relu(self.fc2(h_dann_1))
        
        # Domain classification layer
        d_logit = torch.matmul(h_dann_2, self.W_domain) + self.b_domain
        
        # Softmax
        d_softmax = F.softmax(d_logit, dim=1)
        
        # Cross-entropy loss
        domain_loss = F.cross_entropy(d_logit, d_label)
        
        return d_softmax, domain_loss


class LocalDiscriminator(nn.Module):
    def __init__(self, n_emb, Ada_lambda, num_class):
        super(LocalDiscriminator, self).__init__()
        self.Ada_lambda = Ada_lambda
        self.num_class = num_class

        # Define the fully connected layers
        self.fc1 = nn.ModuleList([nn.Linear(n_emb, 128) for _ in range(num_class)])
        self.fc2 = nn.ModuleList([nn.Linear(128, 128) for _ in range(num_class)])
        
        # Define the final domain classification layers for each class
        self.W_domain = nn.ParameterList([nn.Parameter(torch.Tensor(128, 2)) for _ in range(num_class)])
        self.b_domain = nn.ParameterList([nn.Parameter(torch.Tensor(2)) for _ in range(num_class)])
        
        # Initialize weights
        for W, b in zip(self.W_domain, self.b_domain):
            nn.init.trunc_normal_(W, std=1.0 / math.sqrt(128.0 / 2.0))
            nn.init.constant_(b, 0.1)

    def forward(self, x, pred_prob, d_label):
        self.local_loss = 0
        tmpd_c = 0

        for i in range(self.num_class):
            h_grl = flip_gradient(x, self.Ada_lambda)
            p_source = pred_prob[:, i]
            ps = p_source.view(-1, 1).detach()
            fs = ps * h_grl # h_grl[i]
            
            # Forward pass through fully connected layers for each class
            h_daan_1 = F.relu(self.fc1[i](fs))
            h_daan_2 = F.relu(self.fc2[i](h_daan_1))
            
            # Domain classification layer for each class
            d_logit1 = torch.matmul(h_daan_2, self.W_domain[i]) + self.b_domain[i]
            
            # Softmax
            self.d_softmax1 = F.softmax(d_logit1, dim=1)
            
            # Cross-entropy loss calculation
            local_loss_i = F.cross_entropy(d_logit1, d_label)
            self.local_loss += local_loss_i
            tmpd_c += 2 * (1 - 2 * local_loss_i.item())  # Adjust the calculation similar to the original logic

        tmpd_c /= self.num_class

        return self.local_loss, tmpd_c


# def init_D_MU():
#     global D_M, D_C, MU
#     gl._init()
#     gl.set_value('D_M', 0)
#     gl.set_value('D_C', 0)
#     gl.set_value('MU', 0)


# def init_D_set_MU():
#     D_M = gl.get_value('D_M')
#     D_C = gl.get_value('D_C')
#     MU = gl.get_value('MU')

#     if D_M == 0 and D_C == 0 and MU == 0:
#         MU = 0.5
#     else:
#         D_M = D_M / 5
#         D_C = D_C / 5
#         MU = 1 - D_M / (D_M + D_C)
#         print(MU)
#     D_M=0
#     D_C=0
#     gl.set_value('D_M', D_M)
#     gl.set_value('D_C', D_C)
#     gl.set_value('MU', MU)


# def set_D(domain_loss, tmpd_c):
#     D_M = gl.get_value('D_M')
#     D_C = gl.get_value('D_C')

#     D_C = D_C + tmpd_c
#     D_M = D_M+2*(1-2*domain_loss)
#     gl.set_value('D_M', D_M)
#     gl.set_value('D_C', D_C)