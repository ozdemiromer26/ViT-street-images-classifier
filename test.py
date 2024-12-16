import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torchvision.transforms.functional import InterpolationMode

def load_model(model_path: str, device: torch.device, num_classes: int = 3):
    """
    Load the trained model from the given path.
    Adjust the architecture and final layer to match your training setup.
    """
    
    weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
    # Using ViT as in your original code. Adjust if needed. 
    model = models.vit_l_16(weights=weights)
    # Replace final layer (originally model.heads) to match your training setup
    # For ViT in torchvision, the final layer is model.heads.head
    in_features = model.heads.head.in_features
    model.heads.head = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    # Configuration
    test_csv = "data/test_data.csv"
    test_dir = "data/test"
    model_path = "best_model_09523_loss_02776.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label mappings - adjust if your classes differ
    index_to_label = {0: 'Istanbul', 1: 'Ankara', 2: 'Izmir'}

    # Load the model
    model = load_model(model_path, device, num_classes=len(index_to_label))

    # # Define the transformations (must match what was used during training)
    # transform = transforms.Compose([
    #     transforms.Resize((512, 512),interpolation=InterpolationMode.BICUBIC), # Resize to fixed size
    #     transforms.CenterCrop((512,512)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1
    transform = weights.transforms()

    # Read test.csv
    df = pd.read_csv(test_csv)

    # Prepare a list to store predictions
    predictions = []

    # Iterate through each row in test.csv
    for _, row in df.iterrows():
        filename = row['filename']
        image_path = os.path.join(test_dir, f"{filename}")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output = model(image)
            _, pred_idx = torch.max(output, 1)
            pred_city = index_to_label[pred_idx.item()]

        predictions.append(pred_city)

    # Add the predictions as a new column in the DataFrame
    df['city'] = predictions

    # Save the updated CSV
    df.to_csv(test_csv, index=False)
    print("Predictions saved to test.csv!")

if __name__ == "__main__":
    main()