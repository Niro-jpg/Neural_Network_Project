from RNNs import *
from Utils import *
import matplotlib.pyplot as plt

dataset, miao = InitDataset("./Archive/DailyDelhiClimateTrain.csv")

model = SRNN(4,4,256)

losses = []

criterion = nn.MSELoss()
learning_rate = 0.00001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

hidden = model.init_hidden()

loss = 0

for i in range(10000):

  print(i)

  j = 4 #+ random.randint(0,5)

  x, y = dataset.GetItems(j)

  output, hidden = model.forward(x.float())

  hidden = model.init_hidden()
  
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