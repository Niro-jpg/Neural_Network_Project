import torch
from torch.utils.data import DataLoader
from Utils import *
import sys
from RNNs import *
from VARIABLES import *
from tqdm import tqdm


def Test():
    batch_size = 32
    if "-b" in sys.argv:
        index = sys.argv.index("-b")
        batch_size = int(sys.argv[index + 1])
    save = False
    if "-s" in sys.argv:
        save = True
    plot = False
    # choosing the path of the dataset
    path = ARCHIVE_PATH
    if "-d" in sys.argv:
        index = sys.argv.index("-d")
        path = sys.argv[index + 1]
    if "-p" in sys.argv:
        plot = True
    _, features, X, t = InitDataset(path)
    # choosing the hidden size
    hidden_size = 256
    if "-h" in sys.argv:
        index = sys.argv.index("-h")
        hidden_size = int(sys.argv[index + 1])
    # choosing the number of epochs
    epochs = 15000
    if "-e" in sys.argv:
        index = sys.argv.index("-e")
        epochs = int(sys.argv[index + 1])
    models = []
    models_name = []
    # choosing the model
    if "-m" in sys.argv:
        index = sys.argv.index("-m")
        arg = sys.argv[index + 1]
        if "r" in arg:
            models.append(RNN(features, hidden_size, batch_size=batch_size))
            models_name.append("RNN")
            print("rnn")
        if "s" in arg:
            models.append(SRNN(features, hidden_size, batch_size=batch_size))
            models_name.append("SRNN")
            print("srnn")
        if "g" in arg:
            models.append(GRU(features, hidden_size, batch_size=batch_size))
            models_name.append("GRU")
            print("gru")
        if "l" in arg:
            models.append(LSTM(features, hidden_size, batch_size=batch_size))
            models_name.append("LSTM")
            print("lstm")
        if "t" in arg:
            models.append(Net2(features, hidden_size, batch_size=batch_size))
            models_name.append("TEST2")
            print("test2")
            models.append(Net3(features, hidden_size, batch_size=batch_size))
            models_name.append("TEST3")
            print("test3")
    if len(models) == 0:
        models.append(RNN(features, hidden_size, batch_size=batch_size))
        models_name.append("RNN")
    # choosing the learning rate
    learning_rate = 0.00001
    if "-l" in sys.argv:
        index = sys.argv.index("-l")
        learning_rate = float(sys.argv[index + 1])
    train_dataset = Data(path)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    j = -1
    total_losses = []
    for model in models:
        losses = []
        j += 0
        # loss function
        criterion = nn.BCELoss()
        # optimier algorithm
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # training process
        for i in tqdm(range(epochs)):
            for local_batch, local_labels in train_dataloader:
                outputs, hiddens = model.forward(local_batch.float())
                loss = criterion(outputs.float(), torch.unsqueeze(
                    local_labels.float(), 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                A = X[:15]
                miar = t[:15]
                output, _ = model.forward(A.float())
                print(accuracy(miar, output))
                losses.append(loss.detach().numpy())
        # adding the loss to the list
        total_losses.append(losses)
    # plot the loss
    j = 0
    if plot == True:
        for x in total_losses:
            plt.plot(x, label=models_name[j])
            j += 1
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()
    # save the models
    j = 0
    if save == True:
        for model in models:
            Save(model, MODELS_PATH+models_name[j])
            j += 1
    A = X[:15]
    miar = t[:15]
    for models in models:
        output, _ = model.forward(A.float())
        print(accuracy(miar, output))
