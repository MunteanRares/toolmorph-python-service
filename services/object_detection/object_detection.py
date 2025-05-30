from services.object_detection.cnn_model import SimpleCNN
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

class ObjectDetection:
    def __init__(self, image):
        self.image = Image.open(image).convert("RGB")
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.512, 0.5, 0.250), (0.229, 0.200, 0.254))
        ])
        self.trainset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=self.transform)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def guess_image(self):
        model = SimpleCNN()
        model.load_state_dict(torch.load("services/object_detection/simplecnn_cifar100_72.pth"))
        model.to(self.device)
        model.eval()

        img_tensor = self.transform(self.image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = model(img_tensor)

        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(outputs)
        top_probabilities, top_classes = torch.topk(probabilities, k=5)

        results = []
        for i in range(top_probabilities.size(1)):
            class_name = self.trainset.classes[top_classes[0][i]]
            probability = top_probabilities[0][i].item() * 100
            results.append({
                "class": class_name,
                "probability": round(probability, 2)
            })
        output = {
            "Predictions": results
        }

        print(output)
        return output
