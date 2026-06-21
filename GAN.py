"""GAN-based data augmentation for co-crystal regression data."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

import config


class Generator(nn.Module):
    """Simple MLP generator."""

    def __init__(self, input_dim: int, output_dim: int):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Discriminator(nn.Module):
    """Simple MLP discriminator."""

    def __init__(self, input_dim: int):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train_gan(
    data_tensor: torch.Tensor,
    input_dim: int,
    output_dim: int,
    num_epochs: int = 5000,
    batch_size: int = 32,
    lr: float = 0.0002,
) -> Generator:
    """Train a GAN on the provided data tensor and return the generator."""
    generator = Generator(input_dim, output_dim)
    discriminator = Discriminator(output_dim)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    num_samples = len(data_tensor)

    for epoch in range(num_epochs):
        z = torch.randn(batch_size, input_dim)
        generated_data = generator(z)
        labels_real = torch.ones(batch_size, 1)
        labels_fake = torch.zeros(batch_size, 1)

        # Generator step
        optimizer_G.zero_grad()
        fake_output = discriminator(generated_data)
        loss_G = criterion(fake_output, labels_real)
        loss_G.backward()
        optimizer_G.step()

        # Discriminator step
        optimizer_D.zero_grad()
        real_indices = np.random.choice(num_samples, batch_size, replace=False)
        real_data = data_tensor[real_indices]
        real_output = discriminator(real_data)
        loss_D_real = criterion(real_output, labels_real)

        fake_output = discriminator(generated_data.detach())
        loss_D_fake = criterion(fake_output, labels_fake)

        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch}, Generator Loss: {loss_G.item():.4f}, "
                f"Discriminator Loss: {loss_D.item():.4f}"
            )

    return generator


def generate_samples(
    generator: Generator,
    scaler: StandardScaler,
    feature_columns: list,
    num_samples: int = 2000,
    input_dim: int = 10,
) -> pd.DataFrame:
    """Generate synthetic samples using the trained generator."""
    z = torch.randn(num_samples, input_dim)
    generated_data = generator(z)

    generated_X = generated_data[:, :-1].detach().numpy()
    generated_X = scaler.inverse_transform(generated_X)
    X_df = pd.DataFrame(generated_X, columns=feature_columns)

    generated_y = generated_data[:, -1].detach().numpy()
    y_df = pd.DataFrame(generated_y, columns=["Regression"])

    generated_df = pd.concat([X_df, y_df], axis=1)
    return generated_df


def run_gan_pipeline(
    data_path: Path,
    output_path: Path,
    noise_dim: int = 10,
    num_epochs: int = 5000,
    batch_size: int = 32,
    num_new_samples: int = 2000,
) -> None:
    """Load data, train a GAN, generate synthetic samples, and save the combined dataset.

    Parameters
    ----------
    data_path : Path
        Path to the preprocessed regression data.
    output_path : Path
        Path where the augmented dataset will be saved.
    noise_dim : int
        Dimension of the generator input noise vector.
    num_epochs : int
        Number of GAN training epochs.
    batch_size : int
        Mini-batch size for training.
    num_new_samples : int
        Number of synthetic samples to generate.
    """
    df = pd.read_excel(data_path)

    # Features are from the 8th column onward; target is 'Regression'
    X = df.iloc[:, 7:].values
    y = df["Regression"].values
    feature_columns = df.columns[7:].tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    data_tensor = torch.cat([X_tensor, y_tensor], dim=1)

    output_dim = data_tensor.size(1)
    generator = train_gan(
        data_tensor,
        input_dim=noise_dim,
        output_dim=output_dim,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    generated_df = generate_samples(
        generator,
        scaler,
        feature_columns,
        num_samples=num_new_samples,
        input_dim=noise_dim,
    )

    df["source"] = "original"
    generated_df["source"] = "generated"
    combined_df = pd.concat([df, generated_df], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_excel(output_path, index=False)
    print(f"Augmented dataset saved to: {output_path}")


if __name__ == "__main__":
    run_gan_pipeline(
        config.PROCESSED_REGRESSION_DATA,
        config.GENERATED_DATA,
    )
