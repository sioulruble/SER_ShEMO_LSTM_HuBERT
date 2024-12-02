import torch
from tqdm import tqdm
import torch.optim as optim

def train_model(model, train_dataloader, val_dataloader, num_epochs, device, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for padded_audio, attention_mask, labels in tqdm(train_dataloader):
            padded_audio = padded_audio.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            labels = torch.argmax(labels, dim=1)

            optimizer.zero_grad()
            outputs = model(padded_audio, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/total}, Accuracy: {100 * correct/total}%")

    return model
