import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda

def get_dataloader(batch_size: int):
    transform = Compose([ToTensor, Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root = './data/mnist', transform = transform)

    return DataLoader(dataset, batch_size = batch_size, shuffle = True)



