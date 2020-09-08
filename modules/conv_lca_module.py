import torch
import torch.nn as nn
import torch.nn.functional as F

from DeepSparseCoding.modules.lca_module import LcaModule
import DeepSparseCoding.utils.data_processing as dp


class ConvLcaModule(LcaModule):
    """
    Parameters
    -----------------------------
    data_shape [list of int] by default it is set to [h, w, c], however pytorch conv wants [c, h, w] so it is permuted in this module
        Assumes h = w (i.e. square inputs)
    in_channels [int] - Number of channels in the input image
        Automatically set to params.num_pixels
    out_channels [int] - Number of channels produced by the convolution
        Automatically set to params.num_latent
    kernel_size [int] - Edge size of the square convolving kernel
    stride [int] - Vertical and horizontal stride of the convolution.
    padding [int] - Zero-padding added to both sides of the input.
    """
    def setup_module(self, params):
        self.params = params
        self.params.data_shape = [self.params.data_shape[2], self.params.data_shape[0], self.params.data_shape[1]]
        self.input_shape = [self.params.batch_size] + self.params.data_shape
        self.w_shape = [
            self.params.out_channels,
            self.params.in_channels,
            self.params.kernel_size,
            self.params.kernel_size
        ]
        dilation = 1
        conv_hout = int(1 + (self.input_shape[2] + 2 * self.params.padding - dilation * (self.params.kernel_size - 1) - 1) / self.params.stride)
        conv_wout = conv_hout # Assumes square images
        self.output_shape = [self.params.batch_size, self.params.out_channels, conv_hout, conv_wout]
        w_init = torch.randn(self.w_shape)
        w_init_normed = dp.l2_normalize_weights(w_init, eps=self.params.eps)
        self.w = nn.Parameter(w_init_normed, requires_grad=True)

    def preprocess_data(self, input_tensor):
        return input_tensor.permute(0, 3, 1, 2)

    def get_recon_from_latents(self, a_in):
        recon = F.conv_transpose2d(
            input=a_in,
            weight=self.w,
            bias=None,
            stride=self.params.stride,
            padding=self.params.padding
        )
        return recon

    def step_inference(self, input_tensor, u_in, a_in, step):
        recon = self.get_recon_from_latents(a_in)
        recon_error = input_tensor - recon
        error_injection = F.conv2d(
            input=recon_error,
            weight=self.w,
            bias=None,
            stride=self.params.stride,
            padding=self.params.padding
        )
        du = error_injection + a_in - u_in
        u_out = u_in + self.params.step_size * du
        return u_out

    def infer_coefficients(self, input_tensor):
        u_list = [torch.zeros(self.output_shape, device=self.params.device)]
        a_list = [self.threshold_units(u_list[0])]
        for step in range(self.params.num_steps-1):
            u = self.step_inference(input_tensor, u_list[step], a_list[step], step)
            u_list.append(u)
            a_list.append(self.threshold_units(u))
        return (u_list, a_list)

    def get_encodings(self, input_tensor):
        u_list, a_list = self.infer_coefficients(input_tensor)
        return a_list[-1]

    def forward(self, input_tensor):
        latents = self.get_encodings(input_tensor)
        return latents
