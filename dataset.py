import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer
import torch
from torchvision import transforms

class MedicalDataset(Dataset):
    def __init__(self, annotations_file, img_dir, tokenizer_name, max_length=512, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Load text data
        symptoms = self.img_labels.iloc[idx]['symptoms']
        diagnosis = self.img_labels.iloc[idx]['diagnosis']

        # Tokenize input text
        inputs = self.tokenizer(
            f"Symptoms: {symptoms} Diagnosis:",
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )

        # Tokenize labels
        labels = self.tokenizer(
            diagnosis,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
        )

        # Squeeze tensors to remove extra dimensions
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = labels['input_ids'].squeeze()

        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }