import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import ConvAutoencoder

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ajout de transformations pour rendre le modèle robuste aux variations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor()
    ])

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, '..', 'data', 'train')

    dataset = datasets.ImageFolder(root=train_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ConvAutoencoder().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 60  # Passage à 30 époques
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, _ in dataloader:
            inputs = data.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Époque {epoch+1}/{epochs} — Perte (MSE) : {running_loss/len(dataloader):.5f}")

    torch.save(model.state_dict(), 'autoencoder_anomaly.pth')
    print("Modèle sauvegardé avec succès.")

if __name__ == '__main__':
    train()