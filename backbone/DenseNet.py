import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

model_urls = {
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth"
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_channels, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(input_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)

class Transition(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Transition, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DenseNet(nn.Module):
    def __init__(self, growth_rate, block_config, num_classes=1000):
        super(DenseNet, self).__init__()
        num_init_features = 2 * growth_rate

        # Initial convolution
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transitions
        self.dense_blocks = []
        input_channels = num_init_features
        for i, num_layers in enumerate(block_config):
            self.dense_blocks.append(DenseBlock(num_layers, input_channels, growth_rate))
            input_channels += num_layers * growth_rate
            if i != len(block_config) - 1:  # Last block
                transition = Transition(input_channels, input_channels // 2)
                self.dense_blocks.append(transition)
                input_channels //= 2

        self.dense_blocks = nn.Sequential(*self.dense_blocks)
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dense_blocks(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def densenet121(pretrained=False, num_classes=1000):
    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16), num_classes=num_classes)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls["densenet121"])
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    return model

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
def Backbone_DenseNet121(pretrained=True):
    if pretrained:
        print("The backbone model loads the pretrained parameters...")
    net = densenet121(pretrained=pretrained)
    div_2 = nn.Sequential(*list(net.children())[:3])
    div_4 = net.dense_blocks[0]  # First dense block
    div_8 = net.dense_blocks[1]  # First transition
    div_16 = net.dense_blocks[2]  # Second dense block
    div_32 = net.dense_blocks[3]  # Second transition

    return div_2, div_4, div_8, div_16, div_32

if __name__ == "__main__":
    div_2, div_4, div_8, div_16, div_32 = Backbone_DenseNet121()
    indata = torch.rand(4, 3, 320, 320)
    print(div_2)
