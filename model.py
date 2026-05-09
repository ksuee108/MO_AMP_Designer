import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class MICDataset(Dataset):

    def __init__(self, X, bacteria_ids, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.bacteria_ids = torch.tensor(
            bacteria_ids,
            dtype=torch.long
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return (
            self.X[idx],
            self.bacteria_ids[idx],
            self.y[idx]
        )

# =========================================================
# Feature Augmentation
# =========================================================

def augment_features(x,
                     noise_std=0.05,
                     drop_prob=0.1):

    noise = torch.randn_like(x) * noise_std

    x_aug = x + noise

    mask = (
        torch.rand_like(x_aug) > drop_prob
    ).float()

    x_aug = x_aug * mask

    return x_aug

# =========================================================
# NT-Xent Loss
# =========================================================

class NTXentLoss(nn.Module):

    def __init__(self,
                 temperature=0.2):

        super().__init__()

        self.temperature = temperature

    def forward(self, z1, z2):

        batch_size = z1.shape[0]

        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)

        similarity_matrix = torch.matmul(
            representations,
            representations.T
        )

        mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=z1.device
        )

        similarity_matrix = similarity_matrix[
            ~mask
        ].view(2 * batch_size, -1)

        positives = torch.sum(z1 * z2, dim=1)

        positives = torch.cat(
            [positives, positives],
            dim=0
        )

        logits = similarity_matrix / self.temperature

        positives = positives / self.temperature

        labels = torch.zeros(
            2 * batch_size,
            dtype=torch.long,
            device=z1.device
        )

        logits = torch.cat(
            [positives.unsqueeze(1), logits],
            dim=1
        )

        loss = F.cross_entropy(logits, labels)

        return loss

# =========================================================
# Base MultiTask Encoder
# =========================================================

class MultiTaskEncoder(nn.Module):

    def __init__(self,
                 input_dim,
                 num_bacteria=4,
                 embed_dim=16):

        super().__init__()

        self.bacteria_embedding = nn.Embedding(
            num_bacteria,
            embed_dim
        )

        self.encoder = nn.Sequential(

            nn.Linear(input_dim + embed_dim, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )

    def encode(self, x, bacteria_id):

        b_embed = self.bacteria_embedding(
            bacteria_id
        )

        x = torch.cat([x, b_embed], dim=1)

        latent = self.encoder(x)

        return latent

# =========================================================
# CNN
# =========================================================

class CNNModel(MultiTaskEncoder):

    def __init__(self,
                 input_dim,
                 num_bacteria=4):

        super().__init__(
            input_dim,
            num_bacteria
        )

        self.conv = nn.Sequential(

            nn.Conv1d(1, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.head = nn.Sequential(

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)
        )

    def forward(self, x, bacteria_id):

        latent = self.encode(x, bacteria_id)

        latent = latent.unsqueeze(1)

        latent = self.conv(latent)

        latent = latent.squeeze(-1)

        out = self.head(latent)

        return out.squeeze(1)

# =========================================================
# AE
# =========================================================

class AutoEncoderRegressor(MultiTaskEncoder):

    def __init__(self,
                 input_dim,
                 num_bacteria=4,
                 latent_dim=64):

        super().__init__(
            input_dim,
            num_bacteria
        )

        self.bottleneck = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 128),
            nn.GELU(),

            nn.Linear(128, input_dim)
        )

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            for _ in range(num_bacteria)
        ])

    def forward(self, x, bacteria_id):

        latent = self.encode(x, bacteria_id)

        z = self.bottleneck(latent)

        recon = self.decoder(z)

        outputs = []

        for i in range(len(x)):

            task_id = bacteria_id[i].item()

            out = self.task_heads[task_id](z[i])

            outputs.append(out)

        pred = torch.stack(outputs)

        return pred.squeeze(1), recon

# =========================================================
# DAE
# =========================================================

class DenoisingAutoEncoderRegressor(
    AutoEncoderRegressor
):

    def __init__(self,
                 input_dim,
                 num_bacteria=4,
                 latent_dim=64,
                 noise_std=0.1):

        super().__init__(
            input_dim,
            num_bacteria,
            latent_dim
        )

        self.noise_std = noise_std

    def forward(self, x, bacteria_id):

        if self.training:

            noise = (
                torch.randn_like(x)
                * self.noise_std
            )

            x = x + noise

        return super().forward(x, bacteria_id)

# =========================================================
# VAE
# =========================================================

class VariationalAutoEncoderRegressor(MultiTaskEncoder):

    def __init__(self,
                 input_dim,
                 num_bacteria=4,
                 latent_dim=32):

        super().__init__(
            input_dim,
            num_bacteria
        )

        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(

            nn.Linear(latent_dim, 128),
            nn.GELU(),

            nn.Linear(128, input_dim)
        )


        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            for _ in range(num_bacteria)
        ])

    def reparameterize(self,
                       mu,
                       logvar):

        std = torch.exp(0.5 * logvar)

        eps = torch.randn_like(std)

        return mu + eps * std

    def forward(self, x, bacteria_id):

        latent = self.encode(x, bacteria_id)

        mu = self.mu_layer(latent)

        logvar = self.logvar_layer(latent)

        z = self.reparameterize(mu, logvar)

        recon = self.decoder(z)

        outputs = []

        for i in range(len(x)):

            task_id = bacteria_id[i].item()

            out = self.task_heads[task_id](z[i])

            outputs.append(out)

        pred = torch.stack(outputs)

        return pred.squeeze(1), recon, mu, logvar

# =========================================================
# Contrastive Model
# =========================================================

class ContrastiveEncoder(MultiTaskEncoder):

    def __init__(self,
                 input_dim,
                 num_bacteria=4,
                 latent_dim=128):

        super().__init__(
            input_dim,
            num_bacteria
        )

        self.projector = nn.Sequential(

            nn.Linear(128, latent_dim),
            nn.GELU(),

            nn.Linear(latent_dim, latent_dim)
        )

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1)
            )
            for _ in range(num_bacteria)
        ])

    def forward(self, x, bacteria_id):

        latent = self.encode(x, bacteria_id)

        z = F.normalize(latent, dim=1)

        projection = self.projector(z)

        outputs = []

        for i in range(len(x)):

            task_id = bacteria_id[i].item()

            out = self.task_heads[task_id](z[i])

            outputs.append(out)

        pred = torch.stack(outputs)

        return pred.squeeze(1), z, projection


# =========================================================
# Linear Probe for Bacteria Classification
# =========================================================

class LinearProbe(nn.Module):

    def __init__(self, encoder, hidden_dim, num_bacteria=4):

        super().__init__()

        self.encoder = encoder

        # freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(hidden_dim, num_bacteria)

    def forward(self, x, bacteria_id):

        with torch.no_grad():
            latent = self.encoder.encode(x, bacteria_id)

        logits = self.classifier(latent)

        return logits
    