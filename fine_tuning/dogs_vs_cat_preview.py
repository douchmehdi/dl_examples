import torch.utils.data as data
from PIL import Image
import os
from torchvision.models import vgg16
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchsummary import summary
import requests


class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)


LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

# Let's get our class labels.
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}

base_dir = "/home/mehdi/Documents/DL/Datasets/Dogs_vs_Cats"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

train_imagefolder = ImageFolder(root=train_dir)
vgg = vgg16(pretrained=True)
vgg.train()
summary(vgg, (3, 224, 224))

min_img_size = 224
transform_pipeline = transforms.Compose([
    transforms.Resize(min_img_size),
    transforms.CenterCrop((min_img_size, min_img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

idx = 113
img, filename = train_im
agefolder[idx]
img.show()
im = transform_pipeline(img)
im_ = im.unsqueeze(0)

im_ = Variable(im_)

prediction = vgg(im_)  # Returns a Tensor of shape (batch, num class labels)
# prediction = im_.data.numpy().argmax()  # Our prediction will be the index of the class label with the largest value.
prediction = prediction.argmax()  # Our prediction will be the index of the class label with the largest value.
print(labels[int(prediction)])  # Converts the index to a string using our labels dict
