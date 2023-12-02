import numpy as np
import torch
import torch.nn.functional as F


class LocalKernelLayer(torch.nn.Module):
    """
    Class of Local Kernel used in pre-training stage of Global-Local Kernel model.
    Then the pre-trained Local Kernel is loaded in the fine-tuning stage to

    Partially taken from pytorch adaptation of official GLocal-Kernel code.
    Source of the adaptation can be accessed by any of the links in
    data/external/'GLocal-K implementation torch-adapted source.txt'
    Source of the original tensorflow-based implementation can be accessed by the link in
    data/external/'GLocal-K official tensorflow-based implementation.txt'
    """
    def __init__(self, n_in, n_hid, n_dim, activation, lambda_s, lambda_2):
        super(LocalKernelLayer, self).__init__()
        self.activation = activation

        self.W = torch.nn.parameter.Parameter(
            torch.rand(size=(n_in, n_hid)) * 2 * np.sqrt(6 / (n_in + n_hid))
            - np.sqrt(6 / (n_in + n_hid)),
            requires_grad=True,
        )
        self.u = torch.nn.parameter.Parameter(
            torch.normal(0, 1e-3, size=(n_in, 1, n_dim)), requires_grad=True
        )
        self.v = torch.nn.parameter.Parameter(
            torch.normal(0, 1e-3, size=(1, n_hid, n_dim)), requires_grad=True
        )
        self.b = torch.nn.parameter.Parameter(
            torch.rand(size=(n_hid,)), requires_grad=True
        )

        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2

    def forward(self, z):
        x, reg_loss = z
        w_hat = torch.maximum(torch.Tensor([0.0]),
                              torch.Tensor([1.0]) - torch.linalg.norm(self.u - self.v, ord=2, dim=2)**2)
        y = torch.matmul(x, self.W * w_hat) + self.b
        y = self.activation(y)

        return y, reg_loss + self.lambda_s * (torch.sum(w_hat**2) / 2) + self.lambda_2 * (torch.sum(self.W**2) / 2)


class LocalKernel(torch.nn.Module):
    """
    class of Local Kernel (kernel of kernel-based AE for pre-training)

    Partially taken from pytorch adaptation of official GLocal-Kernel code.
    Source of the adaptation can be accessed by any of the links in
    data/external/'GLocal-K implementation torch-adapted source.txt'
    Source of the original tensorflow-based implementation can be accessed by the link in
    data/external/'GLocal-K official tensorflow-based implementation.txt'
    """
    def __init__(self, n_layers, n_u, n_hid, n_dim, activation, lambda_s, lambda_2):
        super(LocalKernel, self).__init__()

        self.hidden_layers = torch.nn.Sequential(
            *[LocalKernelLayer(n_u, n_hid, n_dim, activation, lambda_s, lambda_2)]
            + [
                LocalKernelLayer(n_hid, n_hid, n_dim, activation, lambda_s, lambda_2)
                for _ in range(n_layers - 1)
            ]
        )
        self.out_layer = LocalKernelLayer(
            n_hid, n_u, n_dim, lambda x: x, lambda_s, lambda_2
        )

    def forward(self, x):
        reg_loss = 0
        y, reg_loss = self.hidden_layers((x, reg_loss))
        y, reg_loss = self.out_layer((y, reg_loss))
        return y, reg_loss


class GlobalKernel(torch.nn.Module):
    """
    class of GlobalKernel, used for Fine-Tuning after the Pre-training of AE with local kernel.
    After Fine-tuning, model gives best results, so more focus on fine-tuning (more epochs than for pre-training)

    Partially taken from pytorch adaptation of official GLocal-Kernel code.
    Source of the adaptation can be accessed by any of the links in
    data/external/'GLocal-K implementation torch-adapted source.txt'
    Source of the original tensorflow-based implementation can be accessed by the link in
    data/external/'GLocal-K official tensorflow-based implementation.txt'
    """
    def __init__(self, n_kernel, gk_size, dot_scale):
        super(GlobalKernel, self).__init__()
        self.gk_size = gk_size
        self.dot_scale = dot_scale

        conv_kernel_ = torch.empty(size=(n_kernel, self.gk_size**2))
        torch.nn.init.trunc_normal_(conv_kernel_, 0, 0.1)

        self.conv_kernel = torch.nn.parameter.Parameter(
            conv_kernel_, requires_grad=True
        )

    def forward(self, x):
        avg_pooling = torch.reshape(torch.mean(x, dim=1), (1, -1))

        gk = (torch.matmul(avg_pooling, self.conv_kernel) * self.dot_scale)
        gk = torch.reshape(gk, (1, 1, self.gk_size, self.gk_size))

        return gk


def global_conv(x, W):
    conv2d = F.relu(F.conv2d(torch.reshape(x, [1, 1, x.shape[0], x.shape[1]]), W, stride=1, padding="same"))
    return torch.reshape(conv2d, (conv2d.shape[2], conv2d.shape[3]))
