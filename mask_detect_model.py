from PIL import Image
import torch
import torchvision
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
    
    return pred[0].argmax(0)


def frame_to_tensor(frame):
    frame = Image.fromarray(frame, mode='RGB')
    img_tensor = transforms(frame).float()
    return img_tensor.clone().detach()


def model_load():
    model = torchvision.models.resnet18()
    num_outs = model.fc.in_features
    model.fc = torch.nn.Linear(num_outs, 2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device) 
    return model
