""" Full assembly of the parts to form the complete network """
import torch
from select import select
from .unet_parts import *
import einops
from .attention import *
from .utils import init_modules

class AFNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_frame=4, base_num_channels=32, num_encoders=4, bilinear=False):
        super(AFNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_frame = num_frame
        self.bilinear = bilinear
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.encoder_input_sizes = []
        self.conv_c_i = DoubleConv(1, 3, self.base_num_channels)  #corr way of image
        self.conv_c_e = DoubleConv(self.n_channels, 3, self.base_num_channels)  #corr way of events
        self.conv_i = DoubleConv(1, 3, self.base_num_channels)  
        self.conv_e = DoubleConv(self.n_channels, 3, self.base_num_channels)
        self.in_conv = DoubleConv(3, self.base_num_channels)

        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [
            self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]
        #self.inc = DoubleConv(n_channels, self.base_num_channels)
        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(recurrentDown(input_size, output_size))

        self.decoders = nn.ModuleList()
        for input_size, output_size in zip(reversed(self.encoder_output_sizes), reversed(self.encoder_input_sizes)):
            self.decoders.append(Up(input_size, output_size, bilinear))
        self.outc = OutConv(self.base_num_channels, self.n_classes)
        init_modules(self)

    def sub_forward(self, event, image, prev_states=None):  #x.shape (b 1) c h w
        #event = self.inc(event)
        corr_i = self.conv_c_i(image)
        corr_e = self.conv_c_e(event)
        corr_score = torch.sigmoid(torch.mul(corr_i, corr_e))  #map to [0,1] b 3 h w
        image = self.conv_i(image)
        event = self.conv_e(event)
        corr_image = torch.mul(corr_score, image)
        event = torch.add(corr_image, event)  #b, 3, h, w
        event = self.in_conv(event)

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            blocks.append(event)
            event, state = encoder(event, prev_states[i])
            states.append(state)
        for i, decoder in enumerate(self.decoders):
            event = decoder(
                event, blocks[self.num_encoders - i - 1])
        logits = torch.sigmoid(self.outc(event))
        return logits, states
    
    def forward(self, batch_events, batch_images, prev_states=None):  #batch_images.shape (b n) c h w
        batch_events = einops.rearrange(batch_events, '(b n) c h w -> b n c h w', n = self.num_frame, c=self.n_channels) #(B, n, c, h, w)
        batch_images = einops.rearrange(batch_images, '(b n) c h w -> b n c h w', n = self.num_frame, c=1) #(B, n, 1, h, w)
        out_list = list()
        for i in range(self.num_frame):
            event = batch_events[:, i]  #b c h w
            image = batch_images[:, i]  #b c h w
            out, prev_states= self.sub_forward(event, image, prev_states)  #b 1 h w
            out_list.append(out) 
        return torch.stack(out_list, dim=1) #b n 1 h w


class FNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_frame=4, base_num_channels=32, num_encoders=4, bilinear=False):
        super(FNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_frame = num_frame
        self.bilinear = bilinear
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.encoder_input_sizes = []
        self.conv_i = DoubleConv(1, self.base_num_channels, self.base_num_channels)  
        self.conv_e = DoubleConv(self.n_channels, self.base_num_channels, self.base_num_channels)  

        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [
            self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]
        #self.inc = DoubleConv(n_channels, self.base_num_channels)
        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(recurrentDown(input_size, output_size))

        self.decoders = nn.ModuleList()
        for input_size, output_size in zip(reversed(self.encoder_output_sizes), reversed(self.encoder_input_sizes)):
            self.decoders.append(Up(input_size, output_size, bilinear))
        self.outc = OutConv(self.base_num_channels, self.n_classes)
        init_modules(self)

    def sub_forward(self, event, image, prev_states=None):  #x.shape (b 1) c h w
        image = self.conv_i(image)
        event = self.conv_e(event)
        event = torch.add(image, event)  #b, 32, h, w

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            blocks.append(event)
            event, state = encoder(event, prev_states[i])
            states.append(state)
        for i, decoder in enumerate(self.decoders):
            event = decoder(
                event, blocks[self.num_encoders - i - 1])
        logits = torch.sigmoid(self.outc(event))
        return logits, states
    
    def forward(self, batch_events, batch_images, prev_states=None):  #batch_images.shape (b n) c h w
        batch_events = einops.rearrange(batch_events, '(b n) c h w -> b n c h w', n = self.num_frame, c=self.n_channels) #(B, n, c, h, w)
        batch_images = einops.rearrange(batch_images, '(b n) c h w -> b n c h w', n = self.num_frame, c=1) #(B, n, 1, h, w)
        out_list = list()
        for i in range(self.num_frame):
            event = batch_events[:, i]  #b c h w
            image = batch_images[:, i]  #b c h w
            out, prev_states = self.sub_forward(event, image, prev_states)  #b 1 h w
            out_list.append(out) 
        return torch.stack(out_list, dim=1)  #b n 1 h w

class AFNet_test(nn.Module):
    def __init__(self, n_channels, n_classes, num_frame=4, base_num_channels=32, num_encoders=4, bilinear=False):
        super(AFNet_test, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.num_frame = num_frame
        self.bilinear = bilinear
        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.encoder_input_sizes = []
        self.conv_c_i = DoubleConv(1, self.base_num_channels, self.base_num_channels)  #corr way of image
        self.conv_c_e = DoubleConv(self.n_channels, self.base_num_channels, self.base_num_channels)  #corr way of events
        self.conv_i = DoubleConv(1, self.base_num_channels, self.base_num_channels)  
        self.conv_e = DoubleConv(self.n_channels, self.base_num_channels, self.base_num_channels)  

        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [
            self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]
        #self.inc = DoubleConv(n_channels, self.base_num_channels)
        self.encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.encoders.append(recurrentDown(input_size, output_size))

        self.decoders = nn.ModuleList()
        for input_size, output_size in zip(reversed(self.encoder_output_sizes), reversed(self.encoder_input_sizes)):
            self.decoders.append(Up(input_size, output_size, bilinear))
        self.outc = OutConv(self.base_num_channels, self.n_classes)
        init_modules(self)

    def sub_forward(self, event, image, prev_states=None):  #x.shape (b 1) c h w
        #event = self.inc(event)
        corr_i = self.conv_c_i(image)
        corr_e = self.conv_c_e(event)
        corr_score = torch.sigmoid(torch.mul(corr_i, corr_e))  #map to [0,1]
        image = self.conv_i(image)
        event = self.conv_e(event)
        corr_image = torch.mul(corr_score, image)
        event = torch.add(corr_image, event)  #b, 32, h, w


        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # encoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.encoders):
            blocks.append(event)
            event, state = encoder(event, prev_states[i])
            states.append(state)
        for i, decoder in enumerate(self.decoders):
            event = decoder(
                event, blocks[self.num_encoders - i - 1])
        logits = torch.sigmoid(self.outc(event))
        return logits, states, corr_score
    
    def forward(self, batch_events, batch_images, prev_states=None):  #batch_images.shape (b n) c h w
        batch_events = einops.rearrange(batch_events, '(b n) c h w -> b n c h w', n = self.num_frame, c=self.n_channels) #(B, n, c, h, w)
        batch_images = einops.rearrange(batch_images, '(b n) c h w -> b n c h w', n = self.num_frame, c=1) #(B, n, 1, h, w)
        out_list = list()
        for i in range(self.num_frame):
            event = batch_events[:, i]  #b c h w
            image = batch_images[:, i]  #b c h w
            out, prev_states, corr = self.sub_forward(event, image, prev_states)  #b 1 h w
            out_list.append(out) 
        return torch.stack(out_list, dim=1)  #b n 1 h w

if __name__ == '__main__':
    event = torch.randn(12, 3, 256, 336)
    image = torch.randn(12, 1, 256, 336)
    net = AFNet(3, 1)
    out = net(event, image)
    print(out.shape)
    