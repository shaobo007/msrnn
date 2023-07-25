""" Full assembly of the parts to form the complete network """
import torch
from select import select
from .unet_parts import *
import einops
#from .attention import *
from .utils import init_modules

class DCRNN(nn.Module):
    def __init__(self, n_channels, n_classes, num_frame=4, base_num_channels=32, num_encoders=4, bilinear=False, with_ca=False, recurrent_block_type='convlstm'):
        super(DCRNN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_frame = num_frame
        self.bilinear = bilinear
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.with_ca = with_ca
        self.recurrent_block_type = recurrent_block_type
        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [
            self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]
        self.inc = DoubleConv(n_channels, self.base_num_channels)
        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(DCRNN_Down(input_size, output_size, recurrent_block_type=self.recurrent_block_type, with_ca= self.with_ca))

        self.decoders = nn.ModuleList()
        for input_size, output_size in zip(reversed(self.encoder_output_sizes), reversed(self.encoder_input_sizes)):
            self.decoders.append(Up(input_size, output_size, bilinear))
        self.outc = OutConv(self.base_num_channels, self.n_classes)
        init_modules(self)

    def sub_forward(self, x, prev_states=None):  #x.shape (b 1) c h w
        x = self.inc(x)
        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            blocks.append(x)
            x, state = encoder(x, prev_states[i])
            states.append(state)
        for i, decoder in enumerate(self.decoders):
            x = decoder(
                x, blocks[self.num_encoders - i - 1])
        logits = torch.sigmoid(self.outc(x))
        return logits, states
    
    def forward(self, batch_images, prev_states=None):  #batch_images.shape (b n) c h w
        batch_images = einops.rearrange(batch_images, '(b n) c h w -> b n c h w', n = self.num_frame, c=self.n_channels) #(B, n, c, h, w)
        out_list = list()
        for i in range(self.num_frame):
            x = batch_images[:, i]  #b c h w
            x, prev_states = self.sub_forward(x, prev_states)  #b 1 h w
            out_list.append(x) 
        return torch.stack(out_list, dim=1)  #b n 1 h w

if __name__ == '__main__':
    input = torch.randn(12, 3, 256, 336)
    net = DCRNN(3, 1)
    out = net(input)
    print(out.shape)
    