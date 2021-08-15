import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UnetAutoencoder, Autoencoder
from noisy import NoisyDataset, create_noisy_dataset
from utils import init_weights


def train(args, model, device, train_loader, optimizer, epoch, criterion1, criterion2):
    model.train()
    mean_loss1, mean_loss2, mean_loss3 = [], [], []
    start = time.time()
    with tqdm(train_loader, unit="batch") as train_epoch:
        # Add description to the tqdm bar
        train_epoch.set_description(f"Training Phase - Epoch {epoch}")

        for batch_idx, (b_x, b_y, b_z) in enumerate(train_epoch):
            b_x, b_y, b_z = b_x.to(device), b_y.to(device), b_z.to(device)
            # Feed forward
            encoded, decoded, cls = model(b_x)

            # Reconstruction loss (mean square error) and classifier loss (negative log likelihood)
            loss1 = criterion1(decoded, b_y)
            loss2 = criterion2(cls, b_z)

            # Calculate L1 loss penalty
            if args.l1:
                l1_reg = torch.tensor(0., requires_grad=True).to(device)
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        l1_reg = l1_reg + torch.norm(param, 1)
                # Combined loss (reconstruction + classifier) + l1
                loss3 = loss1 + loss2 + (1e-5 * l1_reg)
            else:
                # Combined loss (reconstruction + classifier)
                loss3 = loss1 + loss2

            # Clear gradients, backpropagate, and apply gradients
            optimizer.zero_grad()
            loss3.backward()
            optimizer.step()

            # Keep loss of each batch to calculate the mean over the epoch
            mean_loss1.append(loss1.to(torch.device("cpu")).item())  # used to calculate the autoencoder epoch mean loss
            mean_loss2.append(loss2.to(torch.device("cpu")).item())  # used to calculate the classifier epoch mean loss
            mean_loss3.append(loss3.to(torch.device("cpu")).item())  # used to calculate the weighted mean loss

            # Calculate spend time so far
            elapsed_time = time.time() - start

            # Update the bar with the time and mean loss info
            train_epoch.set_postfix(time=elapsed_time, mean_rec_loss=np.mean(np.array(mean_loss1)),
                                    mean_cls_loss=np.mean(np.array(mean_loss2)),
                                    mean_loss=np.mean(np.array(mean_loss3)))
        time.sleep(0.1)  # to avoid tqdm bar problems


def validate(args, model, device, valid_loader, criterion1, criterion2):
    model.eval()
    with torch.no_grad():
        val_loss = []
        correct = 0
        with tqdm(valid_loader, unit="batch") as valid_epoch:
            # Add description to the tqdm bar
            valid_epoch.set_description(f"Validation Phase")

            for b_x, b_y, b_z in valid_epoch:
                b_x, b_y, b_z = b_x.to(device), b_y.to(device), b_z.to(device)
                encoded, decoded, cls = model(b_x)

                loss1 = criterion1(decoded, b_y)
                loss2 = criterion2(cls, b_z)
                loss3 = loss1 + loss2

                val_loss.append(loss3.item())
                mean_loss = np.mean(np.array(val_loss))

                # Get the index of the max log-probability
                pred = cls.argmax(dim=1, keepdim=True)
                correct += pred.eq(b_z.view_as(pred)).sum().item()
                accuracy = correct / len(valid_loader.dataset)

                # Update the bar with the time and mean loss info
                valid_epoch.set_postfix(val_loss=mean_loss, accuracy=accuracy)
        time.sleep(0.1)  # to avoid tqdm bar problems
        val_loss = np.mean(np.array(val_loss))
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
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        print('denoising1')
        # Encode and decode view_data to visualize the outcome
        encoded_data, decoded_data, classf_data = model(view_data.to(gpu))
        decoded_data = decoded_data.to(cpu)
        print('denoising2')
        # initialize figure
        f, a = plt.subplots(3, args.denoise_images, figsize=(5, 2))
        print('denoising3')
        for i in range(args.denoise_images):
            a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (32, 32)), cmap='gray')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())

        for i in range(args.denoise_images):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(view_norm.data.numpy()[i], (32, 32)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())

        for i in range(args.denoise_images):
            a[2][i].clear()
            a[2][i].imshow(np.reshape(decoded_data.data.numpy()[i], (32, 32)), cmap='gray')
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

    # Instantiate models
    if args.model_name == 'unet':
        model, bkp_model = UnetAutoencoder(), UnetAutoencoder()
    else:
        model, bkp_model = Autoencoder(), Autoencoder()

    init_weights(model, init_type='normal')
    model.to(device)

    # Define optimizer (must be done after moving the model to the GPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # , weight_decay=1e-5)

    # Reconstruction and classification loss
    criterion1 = nn.MSELoss().to(device)
    criterion2 = nn.NLLLoss().to(device)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # , min_lr=1e-5)

    # Plotting loss and accuracy before training
    validate(args, model, device, valid_loader, criterion1, criterion2)

    # Main loop
    best_loss, loss_counter, best_accuracy = float("inf"), 0, 0
    for epoch in range(1, args.epochs + 1):
        print(f'Current learning rate ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
        time.sleep(0.1)  # to avoid tqdm bar problems

        # Train the model for this epoch
        train(args, model, device, train_loader, optimizer, epoch, criterion1, criterion2)

        # Validate the model for this epoch
        val_loss, accuracy = validate(args, model, device, valid_loader, criterion1, criterion2)

        # Store the parameters that lead to the best accuracy
        if accuracy > best_accuracy:
            bkp_model.load_state_dict(model.state_dict())
            best_accuracy = accuracy

        # Adjust learning rate
        scheduler.step(val_loss)
        # If val_loss is improved keep a bkp of the model
        if val_loss < best_loss:
            print(f"New best val. loss: {val_loss} < {best_loss}")
            best_loss = val_loss
            bkp_model.load_state_dict(model.state_dict())
            loss_counter = 0
        # Check number of epochs without improvement and stop if equals to args.max_epochs
        else:
            loss_counter += 1
            if loss_counter == args.max_epochs:
                break
            print(f'{loss_counter} epochs without improving best loss {best_loss}')

    # Restore the best parameters
    print("Restoring model which lead to the best val. accuracy.")
    model.load_state_dict(bkp_model.state_dict())

    print("Comparing val. accuracy with previous model.")
    validate(args, model, device, valid_loader, criterion1, criterion2)

    # Store model
    print('Save model')
    if not os.path.exists('trained_models/'):
        os.makedirs(f'trained_models/')
        torch.save(model.state_dict(), 'trained_models/autoencoder.pt')

    print('Plot images')
    # Plot denoised images
    if args.denoise_images > 0:
        denoise(args, model, valid_loader)

    # Test classification accuracy over the test set
    # test(model, device, test_loader)


def parse_args():
    """ Parse arguments """
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # Model name
    parser.add_argument('--model-name', type=str, default='unet', help='[normal | unet] (default: unet)')

    # Random params
    parser.add_argument('--seed', type=int, default=-1, help='random seed (default: 1)')

    # Training params
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--max-epochs', type=int, default=10, help='stop training after max-epochs without improvement')
    parser.add_argument('--test-batch-size', type=int, default=300, help='input batch size for testing (default: 1000)')
    parser.add_argument('--l1', default=False, help='use L1 regularization')
    parser.add_argument('--save-model', default=False, help='For Saving the current Model')

    # Logging params
    parser.add_argument('--log-epoch', default=True, help='log mean loss after each epoch')
    parser.add_argument('--log-batch-interval', type=int, default=10,
                        help='log training status every log-batch-interval (default: 10)')
    parser.add_argument('--denoise-images', type=int, default=20,
                        help='display log-batch-interval denoised images at the end (default: 0)')

    # Noisy dataset params
    parser.add_argument('--img-size', type=int, default=32, help='size of the image (default: 32x32)')
    parser.add_argument('--create-noisy', default=True, help='For Saving the current Model')
    parser.add_argument('--noise-prob', type=float, default=0.75, help='noise probability to each pixel (default: 0.5)')
    return parser.parse_args()


def set_reproducible_mode(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_performance_mode():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.autograd.detect_anomaly = False
    torch.autograd.profiler.emit_nvtx = False
    torch.autograd.profiler.profile = False
    torch.autograd.gradcheck = False


if __name__ == '__main__':
    args = parse_args()

    # Fix random seeds for reproducibility
    if args.seed != -1:
        set_reproducible_mode(args)
    else:
        set_performance_mode()

    # Create a noisy version of the MNIST handwritten dataset given the argument 'noise_prob'
    # Alternatively, Gaussian noise could be added to the input during forward pass, where 'noise_prob' controls the std
    # or, using dropout at the input layer
    if args.create_noisy:
        create_noisy_dataset(args.img_size, args.noise_prob)

    main(args)
