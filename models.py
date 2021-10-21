from torch.nn.modules.batchnorm import BatchNorm2d
from pwlu import *
from utils import *


def conv3(in_channels, out_channels=None, activation='relu', norm=BatchNorm2d, bn_zeros=False, pool=False,
          conv_function=nn.Conv2d, depthwise=False):
    '''
    :param in_channels: number of input channels
    :param out_channels: number of output channels (defaults to in_channels)
    :param activation: activation function (defaults to relu)
    :param norm: normalization function (defaults to batchnorm)
    :param pool: whether to pool (defaults to False)
    :param conv_function: convolution function (defaults to nn.Conv2d)
    '''
    if out_channels is None:
        out_channels = in_channels
    activation = get_activation(activation)
    layers = [conv_function(in_channels, out_channels, kernel_size=3, padding=1, groups=in_channels if depthwise else 1,
                            bias=not norm)]
    if norm is not None:
        if not isinstance(activation, PWLU):
            layers.append(norm(out_channels))
            if bn_zeros:
                assert isinstance(layers[-1], BatchNorm2d)
                nn.init.constant_(layers[-1].weight, 0)
    if activation is not None:
        layers.append(activation)
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv1(in_channels, out_channels=None, activation='relu', norm=BatchNorm2d, bn_zeros=False, pool=False,
          conv_function=nn.Conv2d):
    '''
    :param in_channels: number of input channels
    :param out_channels: number of output channels (defaults to in_channels)
    :param activation: activation function (defaults to relu)
    :param norm: normalization function (defaults to batchnorm)
    :param pool: whether to pool (defaults to False)
    :param conv_function: convolution function (defaults to nn.Conv2d)
    '''
    if out_channels is None:
        out_channels = in_channels
    activation = get_activation(activation)
    layers = [conv_function(in_channels, out_channels, kernel_size=1, bias=not norm)]
    if norm is not None:
        layers.append(
            norm(out_channels, affine=(activation != SquareActivation and not isinstance(activation, PWLU))))
        if bn_zeros:
            assert isinstance(layers[-1], BatchNorm2d)
            nn.init.constant_(layers[-1].weight, 0)
    if activation is not None:
        layers.append(activation)
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = nn.functional.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


class PWLUResBlock(nn.Module):
    def __init__(self, channels: int):
        '''
        :param channels: number of channels in the input and output
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.activ1 = PWLU(channels, normed=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.activ2 = PWLU(channels, normed=True)
        self.bn = nn.BatchNorm2d(channels)
        nn.init.constant_(self.bn.weight, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.activ1(out)
        out = self.conv2(out)
        out = self.activ2(out)
        out = self.bn(out)
        out = out + x
        return out


class ResBlock(nn.Module):

    def __init__(self, channels: int, use_square: bool = False):
        '''
        :param channels: number of channels in the input and output
        :param use_square: if True, use square activation function
        '''
        super().__init__()
        self.conv1 = conv3(channels, activation='square' if use_square else 'relu')
        self.conv2 = conv3(channels, activation=None, bn_zeros=True)

        self.bn = nn.BatchNorm2d(channels)
        self.activation = get_activation('relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.bn(out)
        out = self.activation(out)
        return out


class SquareResBlock(nn.Module):

    def __init__(self, channels: int):
        '''
        :param channels: number of channels in the input and output
        :param use_square: if True, use square activation function
        '''
        super().__init__()
        self.conv1 = conv3(channels, channels, activation='square')
        self.conv2 = conv3(channels, channels, activation=None, bn_zeros=True)
        self.bn = nn.BatchNorm2d(channels)
        self.activation = get_activation('relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        out = self.bn(out)
        out = self.activation(out)
        return out


class TestBlock(nn.Module):
    def __init__(self, channels: int):
        '''
        :param channels: number of channels in the input and output
        :param use_square: if True, use square activation function
        '''
        super().__init__()
        self.conv1 = conv1(channels, channels, activation=PWLU(channels, normed=True))

        self.bn = nn.BatchNorm2d(channels)
        self.activation = get_activation('relu')

    def forward(self, x):
        out = self.conv1(x)
        out = out + x
        out = self.bn(out)
        out = self.activation(out)
        return out


class ResNet(ImageClassificationBase):
    def __init__(self, size=3, in_channels=3, n_classes=100, layer_lengths=[3, 3, 3], use_squares=[False, True, True]):
        '''
        :param size: number of blocks per layer
        :param in_channels: number of input channels
        :param n_classes: number of output classes
        '''

        super().__init__()

        self.layers = []

        self.layers.append(conv3(in_channels, 16))

        #self.layers.extend(ResBlock(16, use_squares[0]) for _ in range(layer_lengths[0]))

        self.layers.append(conv3(16, 32, pool=True))

        #self.layers.extend(SquareResBlock(32) for _ in range(layer_lengths[1]))

        self.layers.append(conv3(32, 64, pool=True))
        self.layers.extend(PWLUResBlock(64) for _ in range(3))

        self.layers.append(BatchNorm2d(64))

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(64 * 8 * 8, n_classes))

        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.network(x)
