import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import CustomDataset
from src.model import Static_reconstruction, CustomLoss

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
input_file = 'data/sdf_dataset'
target_file = 'data/observed_points_dataset'
save_dir = 'out/model/'
save_file = 'model.pth'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, save_file)


dataset = CustomDataset(input_file, target_file)
# input_sample, target_sample = dataset[0]
# print("Input sample:", input_sample)
# print("Target sample:", target_sample)


batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

input_size = 100
hidden_size1 = 512
hidden_size2 = 512
hidden_size3 = 512
output_size = 6
num_epochs = 2000
# Define your neural network src
model = Static_reconstruction(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
criterion = CustomLoss()  # Choose an appropriate loss function
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)  # Choose an optimizer
train_loss_history = []  # loss
model.train()

for epoch in range(num_epochs):

    running_loss = 0.0

    # Iterating through the minibatches of the data

    for i, data in enumerate(dataloader, 0):

        # data is a tuple of (inputs, labels)
        X, y = data

        # Reset the parameter gradients  for the current  minibatch iteration
        optimizer.zero_grad()

        y_pred = model(X)  # Perform a forward pass on the network with inputs
        loss = criterion(y_pred, y)  # calculate the loss with the network predictions and ground Truth
        loss.backward()  # Perform a backward pass to calculate the gradients
        optimizer.step()  # Optimize the network parameters with calculated gradients

        # Accumulate the loss and calculate the accuracy of predictions
        running_loss += loss.item()

        # Print statistics to console
        if i % 100 == 99:  # print every 5 mini-batches
            running_loss /= 100
            print("[Epoch %d, Iteration %5d] loss: %.3f " % (epoch + 1, i + 1, running_loss))
            train_loss_history.append(running_loss)
            running_loss = 0.0
torch.save(model.state_dict(), save_path)
print('FINISH.')
# plt.plot(train_loss_history)
# plt.title("sdf")
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.legend(['loss'])
# plt.show()

plt.plot(train_loss_history)
plt.title("sdf")
plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(['loss'])
plt.savefig('out/loss_plot.png')
plt.close()
