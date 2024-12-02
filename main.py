import torch
from torch.utils.data import DataLoader, random_split
from preprocessing import NADataset, custom_collate_fn
from models import EmotionClassifier
from training import train_model
from transformers import AutoConfig

# Configurations
male_data_directory = '/tmp/male'
female_data_directory = '/tmp/female'
target_sample_rate = 22000
batch_size = 6
num_epochs = 10

def main():
    # shemo daset
    shemo_NA_dataset = NADataset(
        male_directory=male_data_directory,
        female_directory=female_data_directory,
        target_sample_rate=target_sample_rate
    )


    total_size = len(shemo_NA_dataset)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(shemo_NA_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # Initialization
    labels = ['A', 'N']
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    config = AutoConfig.from_pretrained("facebook/hubert-base-ls960", num_labels=2, label2id=label2id, id2label=id2label)

    model = EmotionClassifier(config, classifier_type='LSTM', num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # freeze HuBERT weights
    for param in model.hubert.parameters():
        param.requires_grad = False

    # training
    trained_model = train_model(model, train_dataloader, val_dataloader, num_epochs, device)

if __name__ == "__main__":
    main()
