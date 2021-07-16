import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils
from CNN import CNN
import cv2



# data

test_data = dataset.MNIST(root="mnist", train=False, transform=transforms.ToTensor(), download=False)
# batchsize

test_loader = data_utils.DataLoader(dataset=test_data,
                                    batch_size=64,
                                    shuffle=True)

cnn = torch.load("model/mnist_model.pkl")
cnn = cnn.cuda()  # 放到GPU上

loss_test = 0
accuracy = 0
for i, (images, labels) in enumerate(test_loader):
    images = images.cuda()
    labels = labels.cuda()
    outputs = cnn(images)
    _, pred = outputs.max(1)
    accuracy += (pred == labels).sum().item()
    images = images.cpu().numpy()
    labels = labels.cup().numpy()
    pred = pred.cpu().numpy()
    #batchsize * 1 * 28 * 28
    for idx in range(images.shape[0]):
        im_data = images[idx]
        im_label = labels[idx]
        im_pred = pred[idx]
        im_data = im_data.transpose(1, 2, 0)
        print("label", im_label)
        print("pred", im_pred)
        cv2.imshow("imdata", im_data)
        cv2.waitKey(0)
accuracy = accuracy / len(test_data)
print(accuracy)
