import torch
import torch.nn as nn
import math

def make_mlp(params, input_dim, output_dim, output_layer=True):
    layer_sizes = [input_dim]+ params.ARCHITECTURE +[output_dim]
    #outs = params.ARCHITECTURE +[output_dim]
    acts = params.ACTIVATION
    bnorm = params.BATCHNORM
    dout = params.DROPOUT
    init = params.INIT
    output_size = layer_sizes[-2] 
    layers = [mlp_layer(in_, 
                    out_, 
                    activation_=acts,
                    batchnorm=bnorm,
                    dropout=dout) for in_, out_ in zip(layer_sizes[:-2], layer_sizes[1:-1])] #.apply(inits[init])
    if output_layer:
        output_size = layer_sizes[-1]
        layers.append(mlp_layer(layer_sizes[-2], layer_sizes[-1]))#.apply(inits[init]))
    return unwrap_layers(nn.Sequential(*layers)), output_size

def make_cnn(params, channels, kernels, strides, paddings, pool=False):
    if params:
        acts = params.ACTIVATION
        bnorm = params.BATCHNORM
        dout = params.DROPOUT
        init = params.INIT
    else:
        acts = 'ReLU' # nn.Le
        bnorm = True
        dout = False
        init = 'xavier'

    layers = []
    for in_, out_, ker_, stride_, pad_ in zip(channels[:-1], channels[1:], kernels, strides, paddings): 
        layers.append(conv_layer(in_,
                                out_,
                                ker_,
                                stride_,
                                pad_,
                                activation=acts,
                                batchnorm=bnorm,
                                dropout=dout, pool=pool).apply(inits[init]))
    cnn = unwrap_layers(nn.Sequential(*layers))
    return cnn

def make_decoder(params, channels, kernels, strides, paddings, pool=False):
    if params:
        acts = params.ACTIVATION
        bnorm = params.BATCHNORM
        dout = params.DROPOUT
        init = params.INIT
    else:
        acts = 'ReLU' # nn.Le
        bnorm = True
        dout = False
        init = 'xavier'

    layers = []
    for in_, out_, ker_, stride_, pad_ in zip(channels[:-1], channels[1:], kernels, strides, paddings): 
        layers.append(deconv_layer(in_,
                                out_,
                                ker_,
                                stride_,
                                pad_,
                                activation=acts,
                                batchnorm=bnorm,
                                dropout=dout, pool=pool).apply(inits[init]))
    cnn = unwrap_layers(nn.Sequential(*layers))
    return cnn

def mlp_layer(in_, out_, activation_=None, dropout=None, batchnorm=False):
    l = nn.ModuleList([nn.Linear(in_, out_)])
    if batchnorm:
        l.append(nn.BatchNorm1d(out_))
    if activation_ is not None:
        activation = getattr(nn.modules.activation, activation_)()
        l.append(activation)
    if dropout:
        l.append(nn.Dropout()) 
    return l

def conv_layer(in_, out_, ker_, stride_, pad_, bias=True, activation=nn.ReLU(), batchnorm=False, dropout=None, pool = False):
    l = nn.ModuleList([nn.Conv2d(in_,out_,kernel_size=ker_,stride=stride_,padding=pad_, bias=bias)])
    if batchnorm:
        l.append(nn.BatchNorm2d(out_))
    if activation is not None:
        if activation == 'LeakyReLU':
            activation =nn.LeakyReLU(0.02)
        else:
            activation = getattr(nn.modules.activation, activation)()
        l.append(activation)
    if dropout:
        l.append(nn.Dropout())
    if pool:
        l.append(nn.MaxPool2d(kernel_size=3, stride=2))
    return l

def deconv_layer(in_, out_, ker_, stride_, pad_, bias=True, activation=nn.ReLU(), batchnorm=False, dropout=None, pool = False):
    l = nn.ModuleList([nn.ConvTranspose2d(in_,out_,kernel_size=ker_,stride=stride_,output_padding=pad_, bias=bias)])
    if batchnorm:
        l.append(nn.BatchNorm2d(out_))
    if activation is not None:
        if activation == 'LeakyReLU':
            activation =nn.LeakyReLU(0.02)
        else:
            activation = getattr(nn.modules.activation, activation)()
        l.append(activation)
    if dropout:
        l.append(nn.Dropout())
    if pool:
        l.append(nn.MaxPool2d(kernel_size=3, stride=2))
    return l

def unwrap_layers(model):
    l = []
    def recursive_wrap(model):
        for m in model.children():
            if isinstance(m, nn.Sequential): recursive_wrap(m)
            elif isinstance(m, nn.ModuleList): recursive_wrap(m)
            else: l.append(m)
    recursive_wrap(model)
    return nn.Sequential(*l)

def naive(m):
    if isinstance(m, nn.Linear):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        nn.init.uniform_(m.weight, a=-math.sqrt(1.0 / float(fan_in)), b=math.sqrt(1.0 / float(fan_in)))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        nn.init.uniform_(m.weight, a=-math.sqrt(1.0 / float(fan_in)), b=math.sqrt(1.0 / float(fan_in)))

def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

def kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

def orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        try:
            nn.init.zeros_(m.bias)
        except:
            pass

def delta_orthogonal(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


inits = {
    'naive' : naive,
    'xavier': xavier,
    'kaiming': kaiming,
    'orthogonal': orthogonal
}