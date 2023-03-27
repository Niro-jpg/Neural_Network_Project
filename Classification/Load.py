from Utils import *
import sys
from RNNs import *
from VARIABLES import *


def Load():
    models = []
    models_name = []

    path = ARCHIVE_PATH
    if "-d" in sys.argv:
        index = sys.argv.index("-d")
        path = sys.argv[index + 1]

    _, _, X, t = InitDataset(path)

    # choosing the model
    if "-m" in sys.argv:
        index = sys.argv.index("-m")
        arg = sys.argv[index + 1]

        if "r" in arg:
            model = Load_s(RNN, MODELS_PATH+"RNN")
            models.append(model)
            models_name.append("RNN")
            print("rnn")

        if "s" in arg:
            model = Load_s(SRNN, MODELS_PATH+"SRNN")
            models.append(model)
            models_name.append("SRNN")
            print("srnn")

        if "g" in arg:
            model = Load_s(GRU, MODELS_PATH+"GRU")
            models.append(model)
            models_name.append("GRU")
            print("gru")

        if "l" in arg:
            model = Load_s(LSTM, MODELS_PATH+"LSTM")
            models.append(model)
            models_name.append("LSTM")
            print("lstm")

    A = X[:15]
    miar = t[:15]
    for models in models:
        output, _ = model.forward(A.float())
        print(accuracy(miar, output))
