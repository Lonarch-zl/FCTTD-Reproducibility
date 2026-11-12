from torchvision import transforms

from data_source import get_cifar10

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def image_transform(training=False):
    operations = [transforms.Resize(256)]
    if training:
        operations.extend([transforms.RandomCrop(224), transforms.RandomHorizontalFlip()])
    else:
        operations.append(transforms.CenterCrop(224))
    operations.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(operations)


def dataset(root, training, augment=False, download=True):
    return get_cifar10(root, train=training, transform=image_transform(augment), download=download)
