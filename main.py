import torch.jit
from torch.utils.data import Dataset
from torchvision import datasets
import os
from torchsummary import summary
import torchvision.transforms as tt
import tensorflow.summary as tfsummary
import matplotlib.pyplot as plt
from models import *

batch_size = 64

DATA_FOLDER = '~/datasets/cifar100'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Make preprocessing transforms
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
trn_tfm = torch.nn.Sequential(tt.RandomCrop(32, padding=[4], padding_mode='reflect'),
                              tt.RandomHorizontalFlip(),
                              tt.RandomRotation(10),
                              tt.Normalize(*stats, inplace=True)
                              )
tst_tfm = tt.Normalize(*stats)

trn_tfm = torch.jit.script(trn_tfm)
tst_tfm = torch.jit.script(tst_tfm)


class DataLoader:
    def __init__(self, ds, n_images: int = 64, shuffle: bool = True):
        '''
        :param ds: Dataset
        :param n_images: int, number of images (default: 64)
        :param shuffle: bool, whether to shuffle the dataset (default: True)

        '''
        self.ds = ds
        self.n_images = n_images
        self.ds_size = len(self.ds)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            shuffler = np.random.permutation(self.ds_size)
        position = 0
        while position < self.ds_size:
            # n images is size of batch, might be less than self.n_images
            n_images = min(self.n_images, self.ds_size - position)
            start, end = position, position + n_images
            indices = np.arange(start, end)
            if self.shuffle:
                indices = shuffler[indices]

            ret = self.ds[indices]
            if end == len(self.ds):
                return ret
            yield ret
            position += n_images

    def __len__(self):
        return self.ds_size // self.n_images


class CIFARDataset(Dataset):
    def __init__(self, x, y, n_copies=1, transform=None):
        '''
        :param x: numpy array, images
        :param y: numpy array, labels
        :param n_copies: int, number of copies of each image (default: 1)
        :param transform: torchvision.transforms.Compose, transform to apply to images
        '''
        self.x = torch.tensor(np.moveaxis(x, -1, 1).astype(np.float32) / 255)
        self.y = torch.tensor(y)
        self.transform = transform
        self.n_copies = n_copies

    def __getitem__(self, idx):
        x = self.x[idx].to(device)
        y = self.y[idx].repeat(self.n_copies).to(device)
        xs = [self.transform(x) for _ in range(self.n_copies)]
        xs = torch.stack(xs, dim=0).reshape(-1, *xs[0].shape[1:])

        return xs, y

    def __len__(self):
        return len(self.x)


def make_dls():
    trn_cifar = datasets.CIFAR100(DATA_FOLDER, download=True, train=True)

    trn_images = trn_cifar.data
    trn_classes = trn_cifar.targets
    tst_cifar = datasets.CIFAR100(DATA_FOLDER, download=True, train=False)
    tst_images = tst_cifar.data
    tst_classes = tst_cifar.targets

    n_copies = 1
    trn_ds = CIFARDataset(trn_images, trn_classes, n_copies=n_copies, transform=trn_tfm)
    trn_dl = DataLoader(trn_ds, n_images=batch_size // n_copies, shuffle=True)

    tst_ds = CIFARDataset(tst_images, tst_classes, transform=tst_tfm)
    tst_dl = DataLoader(tst_ds, n_images=1024)
    return trn_dl, tst_dl


def get_logdir():
    root_logdir = os.path.join(os.curdir, 'runs')
    logdir = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    logdir = os.path.join(root_logdir, logdir)
    return logdir


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(epochs, max_lr, model, train_loader, val_loader,
          weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, **kwargs):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay, **kwargs)
    # optimizer = SAM(model.parameters(), opt_func, max_lr, weight_decay=weight_decay, **kwargs)

    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))

    pwlus = [layer for layer in model.modules() if isinstance(layer, PWLU)]

    # sched = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    for epoch in range(epochs):
        start = time.perf_counter()
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        batch_num = 0

        for batch in train_loader:
            if batch_num % 3 == 0:
                for pwlu in pwlus:
                    pwlu.normalize_points()
            

            def closure():
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            closure()
            for pwlu in pwlus:
                pwlu.spread_gradient()

            # Step
            optimizer.step()
            optimizer.zero_grad()
            for pwlu in pwlus:
                pwlu.approximate_with_quadratic(16)

            batch_num += 1
            sched.step()

        lrs.append(get_lr(optimizer))
        train_time = time.perf_counter() - start
        start = time.perf_counter()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
        valid_time = time.perf_counter() - start

        with writer.as_default():
            tfsummary.scalar('valid_acc', result['val_acc'], step=epoch)
            tfsummary.scalar('valid_loss', result['val_loss'], step=epoch)
        print(f'{train_time=:.2f}s, {valid_time=:.2f}s')
        if epoch % 5 == 4:
            torch.save(model.state_dict(), 'newpwlutest' + str(epoch) + '.pth')
            pass

    '''
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=5, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiles/00'),
    ) as prof: 
    '''

    return history


if __name__ == '__main__':
    model = ResNet().to(device)
    trn_dl, tst_dl = make_dls()
    evaluate(model, tst_dl)
    writer = tfsummary.create_file_writer(get_logdir())
    summary(model, (3, 32, 32), batch_size=batch_size)
    epochs = 20
    max_lr = 0.001
    grad_clip = 0.1
    weight_decay = 1e-4
    optimizer = torch.optim.Adam

    # model.load_state_dict(torch.load('newpwlutest4.pth'))
    history = [evaluate(model, tst_dl)]

    print(history)

    history += train(epochs, max_lr, model, trn_dl, tst_dl, grad_clip=grad_clip, weight_decay=weight_decay,
                     opt_func=optimizer, betas=(0.9, 0.999))

    print('Finished')

    pwlus = [layer for layer in model.modules() if isinstance(layer, PWLU)]


    def plot(i, *j):
        for index in i:
            pwlu = pwlus[index]
            data = pwlu.get_plottable()
            if j:
                for jindex in j:
                    plt.plot(data[0][jindex], data[1][jindex], label=f'{index}.{jindex} {pwlu.init=}')
            else:
                plt.plot(data[0], data[1], label=f'{index} {pwlu.init=}')
        plt.legend()
        plt.show()


    def plot_avg(*i):
        for index in i:
            pwlu = pwlus[index]
            assert pwlu.channelwise
            data = pwlu.get_plottable()
            plt.plot(data[0][0], np.average(data[1], axis=0), label=f'{index}.avg {pwlu.init=}')
        plt.legend()
        plt.show()
