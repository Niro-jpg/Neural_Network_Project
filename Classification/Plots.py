from RNNs import *
from Utils import *
from tqdm import tqdm
from VARIABLES import *
from torch.utils.data import DataLoader


def test_with_parameters(plot_name,
                         plot_value,
                         hidden_size=128,
                         batch_size=32,
                         epochs=1000,
                         learning_rate=0.0001,
                         sequence_length=100,
                         path=ARCHIVE_PATH,
                         ):

    _, features, X, t = InitDataset(path)

    train_dataset = Data(path)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)

    models = []
    models_name = []
    models.append(SRNN(features, hidden_size, batch_size=batch_size))
    models_name.append("SRNN")
    models.append(LSTM(features, hidden_size, batch_size=batch_size))
    models_name.append("LSTM")
    models.append(Net2(features, hidden_size, batch_size=batch_size))
    #models.append(GRU(features, hidden_size, batch_size = batch_size))
    models_name.append("GRU")

    total_losses = []

    j = 0

    for model in models:
        losses = []
        j += 0
        # loss function
        criterion = nn.BCELoss()
        # optimier algorithm
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # initializing hidden layer
        hidden = model.init_hidden()

        for i in tqdm(range(epochs)):
            for local_batch, local_labels in train_dataloader:
                outputs, hiddens = model.forward(local_batch.float())
                loss = criterion(outputs.float(), torch.unsqueeze(
                    local_labels.float(), 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())

        total_losses.append(losses)

    j = 0
    for x in total_losses:
        plt.plot(x, label=models_name[j])
        j += 1

    plt.title(plot_name + str(plot_value), fontsize=15)
    # Adding axis title
    plt.ylabel('BCE', fontsize=12)
    plt.xlabel('Training steps', fontsize=12)
    plt.grid()
    plt.legend()
    plt.savefig(PLOTS_PATH + plot_name + str(plot_value) +
                '.png', bbox_inches='tight')
    plt.clf()


def shift_variation_test(
    hidden_size=128,
    batch_size=32,
    epochs=1000,
    learning_rate=0.001,
    path=ARCHIVE_PATH
):

    _, features, X, t = InitDataset(path)

    train_dataset = Data(path)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)

    models = []
    models_name = []
    models.append(SRNN(features, hidden_size, batch_size=batch_size, shift=0))
    models_name.append("SRNN with shift = 0")
    models.append(SRNN(features, hidden_size, batch_size=batch_size, shift=1))
    models_name.append("SRNN with shift = 1")
    models.append(SRNN(features, hidden_size, batch_size=batch_size, shift=2))
    models_name.append("SRNN with shift = 2")
    models.append(SRNN(features, hidden_size, batch_size=batch_size, shift=3))
    models_name.append("SRNN with shift = 3")

    total_losses = []

    j = 0

    for model in models:
        losses = []
        mses = []
        j += 0
        # loss function
        criterion = nn.BCELoss()
        # optimier algorithm
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for i in tqdm(range(epochs)):
            for local_batch, local_labels in train_dataloader:
                outputs, hiddens = model.forward(local_batch.float())
                loss = criterion(outputs.float(), torch.unsqueeze(
                    local_labels.float(), 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())

        total_losses.append(losses)

    j = 0
    for x in total_losses:
        plt.plot(x, label=models_name[j])
        j += 1

    plt.title('shift variation', fontsize=15)
    # Adding axis title
    plt.ylabel('BCE', fontsize=12)
    plt.xlabel('Training steps', fontsize=12)
    plt.grid()
    plt.legend()
    plt.savefig(PLOTS_PATH + 'Shift Variation.png', bbox_inches='tight')
    plt.clf()


def Plot():

    #hidden_sizes = [64, 128, 256, 512, 1024]
    # for hidden_size in hidden_sizes:
    #    test_with_parameters("Hidden size - ", hidden_size,
    #                         hidden_size=hidden_size, epochs=100)

    shift_variation_test(epochs=10, hidden_size=512)
