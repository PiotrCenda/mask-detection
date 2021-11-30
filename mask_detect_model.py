from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tt

transforms = tt.Compose([tt.Resize((256, 256)),
                        tt.ToTensor(),
                        tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def predict(tensor, model):
    prediction = model(tensor.unsqueeze(0))
    prediction = prediction.clone().detach()
    return prediction


def frame_classification(frame_tensor, model):
    classes = ["mask", "no_mask"]

    model.eval()

    with torch.no_grad():
        pred = predict(frame_tensor, model)
        predicted = classes[pred[0].argmax(0)]
        print(f"Predicted: \"{predicted}\"")


def frame_to_tensor(frame):
    frame = Image.fromarray(frame, mode='RGB')
    img_tensor = transforms(frame).float()
    return img_tensor.clone().detach()


def model_load():
    model = CNN_net(in_channels=3, num_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device) 
    return model


class CNN_net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN_net, self).__init__()

        # input: 3 X 256 x 256
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(4)) # out: 16 x 64 x 64

        self.conv2 = nn.Sequential(nn.Conv2d(16, 64, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(8)) # out: 128 x 8 x 8

        self.res1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True)) # out: 128 x 8 x 8

        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1),
                                  nn.BatchNorm2d(128),
                                  nn.ReLU(inplace=True),
                                  nn.MaxPool2d(8)) # out: 128 x 1 x 1

        self.classifier = nn.Sequential(nn.Flatten(),  # out: 128
                                        nn.Linear(128, num_classes))  # out: 2

    def forward(self, inpt):
        out = self.conv1(inpt)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.classifier(out)
        return out
