from RNNs import *
from Utils import *
import matplotlib.pyplot as plt
import sys
import random
from tqdm import tqdm

def main():

   batch_size = 32
   if "-b" in sys.argv:
      index = sys.argv.index("-b")
      batch_size = int(sys.argv[index + 1])

   plot = False

   #choosing the path of the dataset
   path = "../Archive/DailyDelhiClimateTrain.csv"  
   if "-d" in sys.argv:
      index = sys.argv.index("-d")
      path = sys.argv[index + 1]

   if "-p" in sys.argv:   
      plot = True

   _,size, dataset, miao = InitDataset(path)

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

   models = []
   models_name = []

   #choosing the model
   if "-m" in sys.argv:
      index = sys.argv.index("-m")
      arg = sys.argv[index + 1]

      if "r" in arg:
         models.append(RNN(size,size,hidden_size, batch_size = batch_size))
         models_name.append("RNN")
         print("rnn")


      if "s" in arg:
         models.append(SRNN(size,size,hidden_size, batch_size = batch_size))
         models_name.append("SRNN")
         print("srnn")

      if "g" in arg:
         models.append(GRU(size,size,hidden_size, batch_size = batch_size))
         models_name.append("GRU")
         print("gru")

      if "l" in arg:
         models.append(LSTM(size,size,hidden_size, batch_size = batch_size))
         models_name.append("LSTM")
         print("lstm") 

      if "t" in arg:
         models.append(Net2(size,size,hidden_size, batch_size = batch_size))   
         models_name.append("TEST2")
         print("test2")    
         models.append(Net3(size,size,hidden_size, batch_size = batch_size))   
         models_name.append("TEST3")
         print("test3")    

   if len(models) == 0: models.append(RNN(size,size,hidden_size, batch_size = batch_size))         

   #choosing the learning rate
   learning_rate = 0.00001
   if "-l" in sys.argv:
      index = sys.argv.index("-l")
      learning_rate = float(sys.argv[index + 1])        


   train_dataset = Data("../Archive/DailyDelhiClimateTrain.csv")
   train_dataloader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)

   j = -1

   total_losses = []

   for model in models:
      losses = []
      j+=0
      #loss function
      criterion = nn.MSELoss()
      #optimier algorithm
      optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
      #initializing hidden layer
      hidden = model.init_hidden()

      for i in tqdm(range(epochs)):
         for local_batch, local_labels in train_dataloader:
            outputs, hiddens = model.forward(local_batch.float())
            loss = criterion(outputs.float(),local_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach().numpy())

      total_losses.append(losses)      

  
   #plot the loss
   j = 0
   if plot == True:
      for x in total_losses:
         plt.plot(x, label = models_name[j])
         j += 1

      plt.legend(loc="upper left")
      plt.grid()
      plt.show()


   j = 0
   for model in models:
      print(models_name[j])
      j += 1
      print("actual value: ",miao[15])
      output, _ = model.forward(miao[:15].unsqueeze(0).float())
      print("predicted one: ", output)




if __name__ == "__main__":
   main()

