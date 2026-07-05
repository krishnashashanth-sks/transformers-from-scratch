import torch
from PIL import Image

def predict_image(model, image_path, transform, device, class_names):
    model.eval() # Set the model to evaluation mode
    image = Image.open(image_path).convert('RGB')
    image_transformed = transform(image)
    image_transformed = image_transformed.unsqueeze(0) # Add batch dimension
    image_transformed = image_transformed.to(device)

    with torch.no_grad():
        outputs = model(image_transformed)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        predicted_prob = probabilities[0, predicted_idx.item()].item()

    return predicted_class, predicted_prob

def predict_tensor_image(model, image_tensor, class_names, device):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0) # Add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        predicted_prob = probabilities[0, predicted_idx.item()].item()

    return predicted_class, predicted_prob