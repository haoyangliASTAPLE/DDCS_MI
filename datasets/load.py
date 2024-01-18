import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms, datasets

class ImageFolderWithName(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[index]
        filename = path.split('/')[-1]  # get file name
        return image, target, filename

def get_dataset(
        subset='target',
        name='umdfaces',
        crop=None,
        resolution=None,
        with_name=False,
        flip=True,
        normalize=True,
):
    transform_train, transform_test = [], []
    if resolution is not None:
        transform_train.append(transforms.Resize((resolution)))
        transform_test.append(transforms.Resize((resolution)))
    if crop is not None:
        transform_train.append(transforms.CenterCrop(crop))
        transform_test.append(transforms.CenterCrop(crop))
    if flip:
        transform_train.append(transforms.RandomHorizontalFlip(p=0.5))
    transform_train.append(transforms.ToTensor())
    transform_test.append(transforms.ToTensor())

    # normalize to [-1,1]
    if normalize:
        transform_train.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        transform_test.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    # get subset dataset
    if subset == 'target':
        if with_name:
            train = ImageFolderWithName(root=f'./.data/{name}/train', transform=transform_train)
            test = ImageFolderWithName(root=f'./.data/{name}/test', transform=transform_test)
        else:
            train = datasets.ImageFolder(root=f'./.data/{name}/train', transform=transform_train)
            test = datasets.ImageFolder(root=f'./.data/{name}/test', transform=transform_test)
        return train, test
    elif subset == 'aux':
        if with_name:
            train = ImageFolderWithName(root=f'./.data/{name}/{subset}', transform=transform_test)
        else:
            train = datasets.ImageFolder(root=f'./.data/{name}/{subset}', transform=transform_test)
        return train

def get_recovery(
        folder_path,
        normalize=True,
):
    transform = [transforms.ToTensor()]
    if normalize:
        transform.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform)

    dataset = ImageFolderWithName(root=folder_path, transform=transform)
    return dataset

class NpyImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.file_list = os.listdir(root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.root, self.file_list[idx])
        image = np.load(file_name)[0] # 3, 1024, 1024
        image = np.transpose(image, (1,2,0)) # 1024, 1024, 3

        if self.transform:
            image = self.transform(image)

        label = 0  # Set the label for all images to 0

        return image, label

    def resize_save(self, resolution, output_dir):
        # Create a directory for the resized data
        os.makedirs(output_dir, exist_ok=True)

        for idx in tqdm(range(len(self))):
            file_name = os.path.join(self.root, self.file_list[idx])
            image = np.load(file_name)[0] # 3, 1024, 1024
            image = np.transpose(image, (1,2,0)) # 1024, 1024, 3

            # Define the resize transformation
            resize_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((resolution))
            ])

            # Apply the resize transformation
            resized_image = resize_transform(image) # 3, 256, 256

            # Save the resized image
            output_file = os.path.join(output_dir, self.file_list[idx])
            np.save(output_file, np.expand_dims(resized_image.numpy(), axis=0))

def get_celebahq(
        crop=None,
        resolution=256,
        normalize=True,
):
    transform = [transforms.ToTensor()] # to tensor first, then crop and resize
    if resolution is not None:
        transform.append(transforms.Resize((resolution)))
    if crop is not None:
        transform.append(transforms.CenterCrop(crop))
    # transform.append(transforms.ToTensor())
    if normalize:
        transform.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform)

    train = NpyImageFolder(root='./.data/celebA-HQ', transform=transform)
    return train

def get_tsinghuadogs(
        crop=256,
        resolution=256,
        normalize=True,
):
    transform = []
    if resolution is not None:
        transform.append(transforms.Resize((resolution)))
    if crop is not None:
        transform.append(transforms.CenterCrop(crop))
    transform.append(transforms.ToTensor())

    # normalize to [-1,1]
    if normalize:
        transform.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    transform = transforms.Compose(transform)
    train = datasets.ImageFolder(root=f'./.data/tsinghuadogs', transform=transform)

    return train