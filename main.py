from RNNs import *
from Utils import *
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm

def main():

   #choosing the path of the dataset
   path = "../Archive/DailyDelhiClimateTrain.csv"  
   if "-p" in sys.argv:
      index = sys.argv.index("-p")
      path = sys.argv[index + 1]

   size, dataset, miao = InitDataset(path)

   #choosing the hidden size
   hidden_size = 256
   if "-h" in sys.argv:
      index = sys.argv.index("-h")
      hidden_size = int(sys.argv[index + 1])

   #choosing the number of epochs
   epochs = 15000
   if "-e" in sys.argv:
      index = sys.argv.index("-e")
      epochs = int(sys.argv[index + 1])

   model = RNN(size,size,hidden_size)
   arg = None

   #choosing the model
   if "-m" in sys.argv:
      index = sys.argv.index("-m")
      arg = sys.argv[index + 1]

      if arg == "srnn" or arg == "s" or arg == "SRNN":
         model = SRNN(size,size,hidden_size)
         print("srnn")

      if arg == "gru" or arg == "g" or arg == "GRU":
         model = GRU(size,size,hidden_size)   
         print("gru")

      if arg == "lstm" or arg == "l" or arg == "LSTM":
         model = LSTM(size,size,hidden_size)   
         print("lstm") 

      if arg == "test" or arg == "t" or arg == "TEST":
         model = Net(size,size,hidden_size)   
         print("test")    

   #choosing the learning rate
   learning_rate = 0.00001
   if "-l" in sys.argv:
      index = sys.argv.index("-l")
      learning_rate = float(sys.argv[index + 1])        


   losses = []

   #loss function
   criterion = nn.MSELoss()
   #optimier algorithm
   optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

   hidden = model.init_hidden()

   loss = 0

   #training
   for i in tqdm(range(epochs)):

    j = size + random.randint(0,5)

    x, y = dataset.GetItems(j)

    output, hidden = model.forward(x.float())

    loss += criterion(output, y.float())

    if (i % 35 == 0 and i != 0):
      losses.append(loss.detach().numpy())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      loss = 0
  
   #plot the loss
   plt.plot(losses)
   plt.grid()
   plt.show()  

   print(miao[8])
   print(model.forward(miao[0:8].float()))




if __name__ == "__main__":
   main()


