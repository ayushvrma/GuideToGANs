import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch import optim as optim
from matplotlib import pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


training_parameters = {'batch_size': 100, 'n_epochs':100}

os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

num_batches = len(data_loader)
print("Number of batches: ", num_batches)

for x._ in dataloader:
    plt.imshow(x.numpy()[0][0], cmap='gray')
    break

class Generator (nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        input_dim = 100+10 #10 extra for embedding
        output_dim = 784
        self.label_embedding = nn.Embedding(10,10)

        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim,256), nn.LeakyReLU(0.2))
        self.hidden_layer2 = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU(0.2))
        self.hidden_layer3 = nn.Sequential(nn.Linear(512,1024), nn.LeakyReLU(0.2))
        self.hidden_layer4 = nn.Sequential(nn.Linear(1024,output_dim), nn.tanH())
    
    def forward(self, x, label):
        c = self.label_embedding(label)
        x = torch.cat([x,c],1)
        output = self.hidden_layer1(x)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        return output.to(device)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        input_dim = 784+10
        output_dim = 1
        self.label_embedding = nn.Embedding(10,10)
        
        self.hidden_layer1 = nn.Sequential(nn.Linear(input_dim,1024), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.hidden_layer2 = nn.Sequential(nn.Linear(1024,512), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.hidden_layer3 = nn.Sequential(nn.Linear(512,256), nn.LeakyReLU(0.2), nn.Dropout(0.3))
        self.hidden_layer4 = nn.Sequential(nn.Linear(256,output_dim), nn.Sigmoid())

        def forward(self, x, labels):
            c = self.label_embedding(labels)
            x = torch.cat([x,c], 1)
            output = self.hidden_layer1(x)
            output = self.hidden_layer2(output)
            output = self.hidden_layer3(output)
            output = self.hidden_layer4(output)

            return output.to(device)

