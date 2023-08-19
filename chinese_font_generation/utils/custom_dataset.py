import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, filename, image_dir, transform=None, repeat=1):
        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.transform = transform
        self.repeat = repeat

    def __getitem__(self, i):
        index = i % self.len
        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = self.load_data(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        if self.repeat is None:
            return 100000000
        else:
            return self.len * self.repeat

    def read_file(self, filename):
        image_label_list = []
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                content = line.rstrip().split(" ")
                name = content[0]
                labels = []

                for val in content[1:]:
                    labels.append(int(val))
                image_label_list.append((name, labels))
        return image_label_list

    def load_data(self, path):
        image = Image.open(path)
        return image


# def main():
#     transform = transforms.Compose(
#         [
#             transforms.Resize((64, 64)),
#             transforms.ToTensor(),
#             # transforms.Grayscale(),
#         ]
#     )
#     dataset = CustomDataset('font_files/3500_style.txt', 'data/', transform=transform)
#     dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, num_workers=4)
#     a = dataset[0]
#     print(a[1][1])


# if __name__ == "__main__":
#     main()
