import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from dataset import MedicalDataset
from model_utils import MultiModalModel
from torchvision import models
import time

def train():
    # Hyperparameters
    num_epochs = 5
    batch_size = 4
    learning_rate = 5e-5
    tokenizer_name = 'path_to_tokenizer'  # Replace with actual path or model name
    language_model_name = 'path_to_falcon_model'  # Replace with actual path or model name

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = MedicalDataset(
        annotations_file='path_to_annotations.csv',
        img_dir='path_to_images',
        tokenizer_name=tokenizer_name,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    vision_model = models.resnet50(pretrained=True)
    model = MultiModalModel(
        vision_model=vision_model,
        language_model_name=language_model_name
    )
    model.to(device)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                image=batch['image'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss/len(dataloader):.4f}")

        # Save the model after each epoch
        model_save_path = f'model_epoch_{epoch+1}.pt'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train()