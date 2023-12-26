import torch
import torchvision 
image_path='./'
'''
celeba_dataset = torchvision.datasets.CelebA(image_path, split='train', target_type='attr', download=True)
print(celeba_dataset)
'''

mnist_dataset = torchvision.datasets.MNIST(image_path, 'train', download=True)
assert isinstance(mnist_dataset, torch.utils.data.Dataset)
example = next(iter(mnist_dataset))
print(example)

