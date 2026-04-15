import torch
from torch import nn, optim
import torchvision.transforms as transforms
import numpy as np

# Import from the original GitHub code you copied over
from .nets.models import DigitModel
from .utils import data_utils

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def prepare_data(clientnum=5, setnum=10, classnum=10, batch_size=32, seed=0, noniid=False):
    setup_seed(seed)
    
    # Data Augmentation
    rotate_degree = 20
    train_transform = transforms.Compose([
        transforms.RandomRotation([-rotate_degree, rotate_degree]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
    ])

    # Get splited data
    if classnum == 2:
        client_train_data, client_test_data, prior_test, client_priors_corr, client_Pi = \
            data_utils.MNIST_SET(data_path="./data", clientnum=clientnum, setnum_perclient=setnum)
        client_validation_data = client_test_data 
    else:
        client_train_data, client_validation_data, client_test_data, prior_test, client_priors_corr, client_Pi = \
            data_utils.MNIST_SET_Multiclass(
                data_path="./data", clientnum=clientnum, setnum_perclient=setnum, noniid=noniid)

    train_loaders, validation_loaders, test_loaders = [], [], []

    for i in range(len(client_train_data)):
        train_set = data_utils.BaiscDataset(client_train_data[i], transform=train_transform)
        val_set = data_utils.BaiscDataset(client_validation_data[i], transform=test_transform)
        test_set = data_utils.BaiscDataset(client_test_data[i], transform=test_transform)

        train_loaders.append(torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True))
        validation_loaders.append(torch.utils.data.DataLoader(val_set, batch_size=batch_size * 5, shuffle=False))
        test_loaders.append(torch.utils.data.DataLoader(test_set, batch_size=batch_size * 5, shuffle=False))

    return train_loaders, validation_loaders, test_loaders, prior_test, client_priors_corr, client_Pi

def L1_Regularization(model):
    L1_reg = 0
    for param in model.parameters():
        L1_reg += torch.sum(torch.abs(param))
    return L1_reg

def train(model, train_loader, optimizer, loss_fun, device, Pi, priors_corr, prior_test, wdecay=0.0):
    model.train()
    loss_all = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        x, y = x.to(device).float(), y.to(device).long()
        priors_corr, Pi = priors_corr.to(device).float(), Pi.to(device).float()

        output = model(x, Pi, priors_corr, prior_test)
        loss = loss_fun(output, y) + L1_Regularization(model) * wdecay
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

    return loss_all / len(train_loader)

def test(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).long()
            output = model.predict(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.view(-1)).sum().item()
            total += target.size(0)

    return (total - correct) / total