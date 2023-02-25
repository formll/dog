"""
    Minimal example to show how to use the DoG optimizer.
    Based on https://github.com/pytorch/examples/blob/main/mnist/main.py
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from dog_optimizer import DoG, LDoG, PolynomialDecayAverager


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def format_pgroup_dog_state(dog_param_group_state):
    """
    A helper function to format the state of a DoG parameter group into a loggable string,
    describing the distance from initial point, the sum of gradient squared norms, and the step size.

    Note: for LDoG, those value are the mean across layers
    @param dog_param_group_state: A state_dict of a single param group of a DoG optimizer
    @return: A printable string
    """
    # among other things, the state dict contains the following keys:
    #    'rbar' is a tensor holding the distance (maximum distance observed so far) from the initial point
    #            For DoG this is a single value, while for LDoG this is a vector of size equal to the number of layers
    #    'G' is a tensor holding the sum gradient squared norms
    #            For DoG this is a single value, while for LDoG this is a vector of size equal to the number of layers
    #    'eta' is a list of scalar tensors holding the step size for each layer (i.e., pytorch Parameter)

    rbar = torch.mean(dog_param_group_state['rbar'].detach()).item()
    G = torch.mean(dog_param_group_state['G'].detach()).item()
    # in DoG, eta has the same value for all layers
    eta = torch.mean(torch.stack(dog_param_group_state['eta'])).detach().item()
    return f'rbar={rbar:E}, G={G:E}, eta={eta:E}'


def train(args, model, averager, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        averager.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.log_state and isinstance(optimizer, DoG):
                opt_state = optimizer.state_dict()
                for i, p in enumerate(opt_state['param_groups']):
                    prefix = f"DoG's state for param group {i}" if not args.ldog else \
                        f"LDoG's state for param group {i} (mean values across layers)"
                    print(f'\t - {prefix}: {format_pgroup_dog_state(p)}')


def test(model, device, test_loader, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set ({}): Loss = {:.4f}, Accuracy = {:.2f}% ({}/{})\n'.format(
        model_name, test_loss, 100. * correct / len(test_loader.dataset),
        correct, len(test_loader.dataset),
        ))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--data-root', type=str, default='../data', metavar='N',
                        help='data root (default: "../data")')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--ldog', action='store_true', default=False,
                        help='If set to true, will use LDoG rather than DoG')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='base learning rate (default: 1.0) - should not be changed!')
    parser.add_argument('--reps_rel', type=float, default=1e-6, metavar='M',
                        help='normalized version of the r_epsilon parameter (default: 1e-6)')
    parser.add_argument('--init_eta', type=float, default=0, metavar='M',
                        help='if above 0, will use this value as the initial eta instead of the result of '
                             'reps_rel (default: 0)')
    parser.add_argument('--avg_gamma', type=float, default=8, metavar='M',
                        help='Polynomial decay averager gamma (default: 8)')
    parser.add_argument('--weight_decay', type=float, default=0, metavar='M',
                        help='weight decay coefficient (default: 0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='Save the current Model')
    parser.add_argument('--no-log-state', action='store_false', default=True, dest='log_state',
                        help='Suppress logging the state of the optimizer at each log_interval')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST(args.data_root, train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST(args.data_root, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    opt_class = LDoG if args.ldog else DoG
    # Creating the optimizer
    optimizer = opt_class(model.parameters(), reps_rel=args.reps_rel, lr=args.lr,
                          init_eta=(args.init_eta if args.init_eta > 0 else None), weight_decay=args.weight_decay)
    averager = PolynomialDecayAverager(model, gamma=args.avg_gamma)  # Creating the averager
    # Note - there is no lr scheduler

    for epoch in range(1, args.epochs + 1):
        train(args, model, averager, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, 'base model')  # get test results for the base model
        test(averager.averaged_model, device, test_loader, 'averaged model')  # get test results for the averaged model

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
        torch.save(averager.averaged_model.state_dict(), "mnist_cnn_averaged.pt")


if __name__ == '__main__':
    main()
