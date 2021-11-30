from PIL import Image
import torch
import torchvision.transforms as tt

from cnn_model import mask_net

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
    model = mask_net(in_channels=3, num_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device) 
    return model
