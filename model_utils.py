import torch
from torch import nn
from transformers import LlamaForCausalLM

class MultiModalModel(nn.Module):
    def __init__(self, vision_model, language_model_name):
        super(MultiModalModel, self).__init__()
        # Vision Model (e.g., ResNet)
        self.vision_model = vision_model
        self.vision_model.fc = nn.Identity()  # Remove classification layer
        self.image_embedding_dim = self.vision_model.fc.in_features

        # Language Model (e.g., Falcon)
        self.language_model = LlamaForCausalLM.from_pretrained(language_model_name)

        # Project image embeddings to language model's hidden size
        self.image_projection = nn.Linear(
            self.image_embedding_dim,
            self.language_model.config.hidden_size
        )

    def forward(self, image, input_ids, attention_mask, labels=None):
        # Extract image features
        image_features = self.vision_model(image)
        image_embeddings = self.image_projection(image_features).unsqueeze(1)  # Add sequence dimension

        # Get embeddings from language model
        text_embeddings = self.language_model.model.embed_tokens(input_ids)

        # Concatenate image and text embeddings
        inputs_embeds = torch.cat((image_embeddings, text_embeddings), dim=1)
        attention_mask = torch.cat((torch.ones((attention_mask.size(0), 1), device=attention_mask.device), attention_mask), dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs