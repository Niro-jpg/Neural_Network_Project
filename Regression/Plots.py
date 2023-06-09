from RNNs import *
from Utils import *
from tqdm import tqdm
from VARIABLES import *


def test_with_parameters(plot_name,
                         plot_value,
                         hidden_size=128,
                         batch_size=32,
                         epochs=100,
                         learning_rate=0.0001,
                         sequence_length=100,
                         path=ARCHIVE_PATH,
                         ):

    # we need the size to init the neural network
    _, size, _, _ = InitDataset(path)
    # initializing the dataset and the dataloader
    train_dataset = Data(path, sequence_length=sequence_length)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # adding the models we want to plot to a list
    models = []
    models_name = []
    models.append(SRNN(size, size, hidden_size, batch_size=batch_size))
    models_name.append("SRNN")
    models.append(LSTM(size, size, hidden_size, batch_size=batch_size))
    models_name.append("LSTM")
    #models.append(Net2(size,size,hidden_size, batch_size = batch_size))
    models.append(GRU(size, size, hidden_size, batch_size=batch_size))
    models_name.append("GRU")
    total_losses = []
    j = 0
    for model in models:
        losses = []
        j += 0
        # loss function
        criterion = nn.MSELoss()
        # optimier algorithm
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # training step
        for i in tqdm(range(epochs)):
            for local_batch, local_labels in train_dataloader:
                outputs, _ = model.forward(local_batch.float())
                loss = criterion(outputs.float(), local_labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())
        # adding the loss to the list
        total_losses.append(losses)
    # ploting the results
    j = 0
    for x in total_losses:
        plt.plot(x, label=models_name[j])
        j += 1
    plt.title(plot_name + str(plot_value), fontsize=15)
    # Adding axis titles
    plt.ylabel('MSE', fontsize=12)
    plt.xlabel('Training steps', fontsize=12)
    plt.grid()
    plt.legend()
    plt.savefig(PLOTS_PATH + plot_name + str(plot_value) +
                '.png', bbox_inches='tight')
    plt.clf()


def shift_variation_test(hidden_size=128,
                         batch_size=32,
                         epochs=100,
                         learning_rate=0.0001,
                         sequence_length=100,
                         path=ARCHIVE_PATH,
                         ):
    # we need the sizes of the neural network
    _, size, _, _ = InitDataset(path)
    # initializing the dataset and dataloader
    train_dataset = Data(path, sequence_length=sequence_length)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # adding the models to plot
    models = []
    models_name = []
    models.append(SRNN(size, size, hidden_size,
                  batch_size=batch_size, shift=0))
    models_name.append("SRNN with shift = 0")
    models.append(SRNN(size, size, hidden_size,
                  batch_size=batch_size, shift=1))
    models_name.append("SRNN with shift = 1")
    models.append(SRNN(size, size, hidden_size,
                  batch_size=batch_size, shift=2))
    models_name.append("SRNN with shift = 2")
    models.append(SRNN(size, size, hidden_size,
                  batch_size=batch_size, shift=3))
    models_name.append("SRNN with shift = 3")
    total_losses = []
    j = 0
    for model in models:
        losses = []
        j += 0
        # loss function
        criterion = nn.MSELoss()
        # optimier algorithm
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # training process
        for i in tqdm(range(epochs)):
            for local_batch, local_labels in train_dataloader:
                outputs, hiddens = model.forward(local_batch.float())
                loss = criterion(outputs.float(), local_labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())
        # adding the loss to the list
        total_losses.append(losses)
    # ploting the results
    j = 0
    for x in total_losses:
        plt.plot(x, label=models_name[j])
        j += 1
    plt.title('shift variation', fontsize=15)
    # Adding axis title
    plt.ylabel('MSE', fontsize=12)
    plt.xlabel('Training steps', fontsize=12)
    plt.grid()
    plt.legend()
    plt.savefig(PLOTS_PATH + 'Shift Variation.png', bbox_inches='tight')
    plt.clf()
# in this function we do 3 test by changing 1 variable each time.
# the first test is changing the sequence lenght, the second one
# is changing the hidden size and the third one is changing the
# shift variation of srrn.


def Plot():
    sequences_length = [50, 100, 200, 400]
    for sequence_length in sequences_length:
        test_with_parameters("Sequence Length - ",
                             sequence_length, sequence_length=sequence_length)
    hidden_sizes = [64, 128, 256, 512]
    for hidden_size in hidden_sizes:
        test_with_parameters("Hidden size - ", hidden_size,
                             hidden_size=hidden_size)
    shift_variation_test()
