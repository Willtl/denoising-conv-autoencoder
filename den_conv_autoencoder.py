import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import swats

from model3 import ConvAutoEncoder as ConvAE3
from model4 import ConvAutoEncoder as ConvAE4
from model5 import ConvAutoEncoder as ConvAE5
from model6 import ConvAutoEncoder as ConvAE6
from noisy import NoisyDataset, create_noisy_dataset


def train(args, model, device, train_loader, optimizer, epoch, criterion1, criterion2):
    model.train()
    mean_loss1, mean_loss2, mean_loss3 = [], [], []
    start = time.time()
    for batch_idx, (b_x, b_y, b_z) in enumerate(train_loader):
        b_x, b_y, b_z = b_x.to(device), b_y.to(device), b_z.to(device)
        optimizer.zero_grad()   # clear gradients for this training step
        encoded, decoded, cls = model(b_x)  # Feed data
        loss1 = criterion1(decoded, b_y)    # reconstruction loss (mean square error)
        loss2 = criterion2(cls, b_z)    # classifier loss (negative log likelihood)
        loss3 = (0.1 * loss1) + (1.0 * loss2)   # combined loss (reconstruction + classifier)
        loss3.backward()    # backpropagation, compute gradients
        optimizer.step()    # apply gradients

        if args.log_epoch:
            mean_loss1.append(loss1.to(torch.device("cpu")).item())  # used to calculate the autoencoder epoch mean loss
            mean_loss2.append(loss2.to(torch.device("cpu")).item())  # used to calculate the classifier epoch mean loss
            mean_loss3.append(loss3.to(torch.device("cpu")).item())  # used to calculate the weighted mean loss

        if batch_idx % args.log_batch_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(b_x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss3.item()))

    if args.log_epoch:
        elapsed_time = time.time() - start
        print(f'\nMean loss 1: {np.mean(np.array(mean_loss1))}, epoch time: {elapsed_time}')
        print(f'Mean loss 2: {np.mean(np.array(mean_loss2))}, epoch time: {elapsed_time}')
        print(f'Mean loss 3: {np.mean(np.array(mean_loss3))}, epoch time: {elapsed_time}\n')


def validate(args, model, device, valid_loader, criterion1, criterion2):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for b_x, b_y, b_z in valid_loader:
            b_x, b_y, b_z = b_x.to(device), b_y.to(device), b_z.to(device)
            encoded, decoded, cls = model(b_x)
            loss2 = criterion2(cls, b_z)  # classifier loss (negative log likelihood)
            val_loss += loss2.item()  # sum up classifier batch loss
            # get the index of the max log-probability
            pred = cls.argmax(dim=1, keepdim=True)
            correct += pred.eq(b_z.view_as(pred)).sum().item()

    val_loss /= len(valid_loader.dataset)
    accuracy = correct / len(valid_loader.dataset)

    print('\nValidation, average loss: {:.10f}, accuracy: {}/{} ({:.5f}%)\n'.format(
        val_loss, correct, len(valid_loader.dataset),
        100. * accuracy))

    return val_loss, accuracy


def test(model, device, test_loader):
    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b_x, b_y, b_z in test_loader:
            b_x, b_y, b_z = b_x.to(device), b_y.to(device), b_z.to(device)
            encoded, decoded, cls = model(b_x)
            # Loop through outputs and check if it is correct or not
            for idx, i in enumerate(cls):
                if torch.argmax(i) == b_z[idx]:
                    correct += 1
                total += 1

        print("Accuracy: ", round(correct / total, 6))


def denoise(args, model, data_loader):
    """ Visualize the output of the decoder, the denoised images """
    # Define GPU and CPU devices
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    # Pick a random batch for visualization
    iterator = iter(data_loader)
    x_batch, y_batch, z_batch = iterator.next()
    view_data = Variable(x_batch).to(cpu)
    view_norm = Variable(y_batch).to(cpu)

    # Plotting decoded image
    with torch.no_grad():
        # Set the model to evaluation mode
        model = model.eval()

        # Encode and decode view_data to visualize the outcome
        encoded_data, decoded_data, classf_data = model(view_data.to(gpu))
        decoded_data = decoded_data.to(cpu)

        # initialize figure
        f, a = plt.subplots(3, args.denoise_images, figsize=(5, 2))

        for i in range(args.denoise_images):
            a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())

        for i in range(args.denoise_images):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(view_norm.data.numpy()[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())

        for i in range(args.denoise_images):
            a[2][i].clear()
            a[2][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
            a[2][i].set_xticks(())
            a[2][i].set_yticks(())
        plt.show()


def load_data(set_type):
    """ Load datasets from file"""
    # Load noise, normal, and labels
    noise_data = torch.load('noisy-mnist/0.75/' + set_type + '/noisy.pt')
    # index = random.randint(0, 60000)
    # plot_one(noise_data[index + 3])
    normal_data = torch.load('noisy-mnist/0.75/' + set_type + '/normal.pt')
    # plot_one(normal_data[index + 3])
    label_data = torch.load('noisy-mnist/0.75/' + set_type + '/label.pt')

    # Create training and validation sets
    return NoisyDataset(noise_data, normal_data, label_data)


def main(args):
    # Define GPU and CPU devices
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load train and split into train and valid loaders
    data = load_data('train')
    train_set, valid_set = torch.utils.data.random_split(data, [54000, 6000])
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    # Load test data
    test_set = load_data('test')
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=True, pin_memory=True)

    # Load model and move it to the GPU
    # model, bkp_model = ConvAE3(), ConvAE3()     # 69.55 %
    # model, bkp_model = ConvAE4(), ConvAE4()     # 69.64 %
    # model, bkp_model = ConvAE5(), ConvAE5()   # 69.71 %
    model, bkp_model = ConvAE6(), ConvAE6()   # 70.55 %
    model.to(device)
    # Define optimizer (must be done after moving the model to the GPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)   # 0.9903
    # Criterion for the autoencoder
    criterion1 = nn.MSELoss().to(device)
    # Criterion for the classifier
    criterion2 = nn.NLLLoss().to(device)
    # criterion2 = nn.CrossEntropyLoss().to(device)
    # Learning rate scheduling
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # , min_lr=1e-5)
    # Plotting loss and accuracy before training
    validate(args, model, device, valid_loader, criterion1, criterion2)
    # Training loop
    best_loss, loss_counter, best_accuracy = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        print(f'Current learning rate ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
        train(args, model, device, train_loader, optimizer, epoch, criterion1, criterion2)
        val_loss, accuracy = validate(args, model, device, valid_loader, criterion1, criterion2)
        # Store the parameters that lead to the best accuracy
        if accuracy > best_accuracy:
            bkp_model.load_state_dict(model.state_dict())
            best_accuracy = accuracy
        # Adjust learning rate
        scheduler.step(val_loss)
        # Stop the training if loss_counter is higher than args.max_epochs
        if best_loss > val_loss:
            best_loss = val_loss
            loss_counter = 0
        else:
            loss_counter += 1
            if loss_counter == args.max_epochs:
                break
            print(f'{loss_counter} epochs without improving best loss {best_loss}')

    # Restore the best parameters
    model.load_state_dict(bkp_model.state_dict())

    validate(args, model, device, valid_loader, criterion1, criterion2)
    # Plot denoised images
    if args.denoise_images > 0:
        denoise(args, model, valid_loader)
    # Test classification accuracy over the test set
    test(model, device, test_loader)


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='stop training after max-epochs without improvement')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=300,
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log-epoch', action='store_true', default=True,
                        help='log mean loss after each epoch')
    parser.add_argument('--log-batch-interval', type=int, default=10,
                        help='log training status every log-batch-interval (default: 10)')
    parser.add_argument('--denoise-images', type=int, default=20,
                        help='display log-batch-interval denoised images at the end (default: 0)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--create-noisy', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--noise-prob', type=float, default=0.50,
                        help='probability of noise to each pixel (default: 0.5)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Fix random seeds for reproducibility
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # Create noisy dataset
    if args.create_noisy:
        create_noisy_dataset(args.noise_prob)

    main(args)
