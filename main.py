from RNNs import *
from Utils import *
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm

def main():

   model = RNN(4,4,256)
   arg = None

   if "-m" in sys.argv:
      index = sys.argv.index("-m")
      arg = sys.argv[index + 1]

      if arg == "srnn" or arg == "s" or arg == "SRNN":
         model = SRNN(4,4,256)
         print("srnn")

      if arg == "gru" or arg == "g" or arg == "GRU":
         model = GRU(4,4,256)   
         print("gru")

      if arg == "lstm" or arg == "l" or arg == "LSTM":
         model = LSTM(4,4,256)   
         print("lstm")   

   path = "../Archive/DailyDelhiClimateTrain.csv"  
   if "-p" in sys.argv:
      index = sys.argv.index("-p")
      path = sys.argv[index + 1]

   dataset, miao = InitDataset(path)
   losses = []

   criterion = nn.MSELoss()
   learning_rate = 0.00001
   optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

   hidden = model.init_hidden()

   loss = 0

   for i in tqdm(range(2000)):

    j = 4 + random.randint(0,5)

    x, y = dataset.GetItems(j)

    output, hidden = model.forward(x.float())

    loss += criterion(output, y.float())

    if (i % 35 == 0 and i != 0):
      losses.append(loss.detach().numpy())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss = 0
  

   plt.plot(losses)
   plt.grid()
   plt.show()  

   print(miao[3])
   print(model.forward(miao[0:3].float()))




if __name__ == "__main__":
   main()


