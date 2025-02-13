import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda

def get_dataloader(batch_size: int):
    # 加载数据时要先对图像数据归一化到[-1, 1], 与高斯噪声分布匹配
    transform = Compose([ToTensor, Lambda(lambda x: (x - 0.5) * 2)])
    dataset = torchvision.datasets.MNIST(root = './data/mnist', transform = transform)

    return DataLoader(dataset, batch_size = batch_size, shuffle = True)



