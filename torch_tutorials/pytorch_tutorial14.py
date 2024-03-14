import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import torch.nn.functional as F
import sys
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist2")


# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper params
input_size = 28 * 28
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

# MNIST
train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                           transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST(root="./data", train=False,
                                           transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, 
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, 
                                           shuffle = False)

examples = iter(train_loader)
example_data, example_targets = next(examples)


for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(example_data[i][0]
               #, cmap="red"
               )
# plt.show()
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image("mnist_images", img_grid)
# writer.close()

# sys.exit()


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # we dont need Softmax function. Because we use Cross Entropy loss which already applies Softmax for us
        return out
    

model = NeuralNet(input_size, hidden_size, num_classes)
# loss & optimizer
criterion = nn.CrossEntropyLoss() # this one applies Softmax for us
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

writer.add_graph(model, example_data.reshape(-1, 28*28))
# writer.close()
# sys.exit()
# training loop

n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images.shape = 100, 1, 28, 28
        # so we need to reshape
        # input size is 784 = 28 * 28
        # so we need to bring it into: 100, 784 format
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predictions = torch.max(outputs, 1)  # 1 is to denote the dimension rowwise or columnwise
        running_correct += (predictions==labels).sum().item()

        if (i+1) % 100 == 0:
            print(f"epoch: {epoch+1} / {num_epochs}, step: {i+1} / {n_total_steps}, loss: {loss.item():.4f}")
            writer.add_scalar("training loss", running_loss/100, epoch * n_total_steps + i)
            writer.add_scalar("accuracy", running_correct/100, epoch * n_total_steps + i)

            running_loss = 0.0
            running_correct = 0


labels = []
preds = []


# testing / evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels1 in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels1 = labels1.to(device)  # this is the actual labels
        outputs = model(images)

        # it actually returns values, index
        _, predictions = torch.max(outputs, 1)  # 1 is to denote the dimension rowwise or columnwise
        n_samples += labels1.shape[0]  # this gives us the number of samples in the current batch
        n_correct += (predictions == labels1).sum().item()


        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_predictions)
        labels.append(predictions)

    
    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels)


    acc = 100.0 * n_correct / n_samples
    print(f"accuracy: {acc}")


    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
