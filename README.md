# ViT-street-images-classifier
For ITU DatathonAI 2024 DatathonAI Qualification Round

#Architecture
ViT_l_16

#File Locations
test_csv = "data/test_data.csv"
test_dir = "data/test"
train_csv = "data/train_data.csv"
train_dir = "data/train"

#Parameters
batch_size = 32
num_epochs_feature_extraction = 7
num_epochs_fine_tuning = 13
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Best working model name
model_path = "best_model_09523_loss_02776.pth"