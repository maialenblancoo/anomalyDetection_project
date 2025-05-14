import torch
import torch.nn as nn
import numpy as np
import logging
import json
import os

logger = logging.getLogger(__name__)

# LSTM Autoencoder Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=n_features,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.output_layer = nn.Linear(n_features, n_features)

    def forward(self, x):
        x, (hidden_n, cell_n) = self.encoder(x)
        decoder_input = x.clone()
        x, (hidden_n, cell_n) = self.decoder(decoder_input)
        x = self.output_layer(x)
        return x

# GRU Autoencoder Model
class GRUAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64, dropout=0.2):
        super(GRUAutoencoder, self).__init__()

        self.encoder = nn.GRU(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.decoder = nn.GRU(
            input_size=embedding_dim,
            hidden_size=n_features,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.output_layer = nn.Linear(n_features, n_features)

    def forward(self, x):
        x, hidden = self.encoder(x)
        decoder_input = x.clone()
        x, hidden = self.decoder(decoder_input)
        x = self.output_layer(x)
        return x

# Transformer Autoencoder Model
class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, n_features, d_model=64, nhead=4, dim_feedforward=128, dropout=0.2, num_layers=2):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model

        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, n_features)

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)
        x = self.output_projection(x)
        return x

# LSTM Predictor Model
class LSTMPredictor(nn.Module):
    def __init__(self, seq_len, n_features, output_size, hidden_dim=64, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.linear = nn.Linear(hidden_dim, output_size)
      
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        # Usar el último estado oculto para la predicción
        out = self.linear(hidden[-1])
        return out

# Función para guardar modelos entrenados
def save_trained_models(models_dict, results_dict=None, config=None, base_path="models/autoencoder/"):
    """Guarda los modelos entrenados en el directorio especificado."""
    os.makedirs(base_path, exist_ok=True)
    
    for name, model in models_dict.items():
        model_path = os.path.join(base_path, f"{name}_model.pt")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Modelo {name} guardado en {model_path}")
    
    # Guardar también los metadatos (umbrales, configuración, etc.)
    if results_dict and config:
        metadata = {
            "thresholds": {
                name: float(results_dict[name]['threshold']) for name in results_dict
            },
            "sequence_length": config.get("sequence_length", 150),
            "embedding_dim": config.get("embedding_dim", 64)
        }
        
        with open(os.path.join(base_path, "model_metadata.json"), "w") as f:
            json.dump(metadata, f)
        logger.info(f"Metadatos guardados en {os.path.join(base_path, 'model_metadata.json')}")

# Función para cargar modelos entrenados
def load_trained_models(seq_len=150, n_features=1, base_path="models/autoencoder/"):
    """Carga modelos entrenados desde el directorio especificado."""
    # Cargar metadatos
    with open(os.path.join(base_path, "model_metadata.json"), "r") as f:
        metadata = json.load(f)
    
    embedding_dim = metadata.get("embedding_dim", 64)
    
    # Inicializar modelos
    lstm_model = LSTMAutoencoder(seq_len=seq_len, n_features=n_features, embedding_dim=embedding_dim)
    gru_model = GRUAutoencoder(seq_len=seq_len, n_features=n_features, embedding_dim=embedding_dim)
    transformer_model = TransformerEncoder(seq_len=seq_len, n_features=n_features, d_model=embedding_dim)
    
    # Cargar pesos
    lstm_model.load_state_dict(torch.load(os.path.join(base_path, "lstm_model.pt")))
    gru_model.load_state_dict(torch.load(os.path.join(base_path, "gru_model.pt")))
    transformer_model.load_state_dict(torch.load(os.path.join(base_path, "transformer_model.pt")))
    
    logger.info(f"Modelos cargados desde {base_path}")
    
    return {
        "lstm": lstm_model,
        "gru": gru_model,
        "transformer": transformer_model
    }, metadata

# Entrenamiento con Early Stopping
def train_model(model, train_loader, val_loader, n_epochs=50, learning_rate=1e-3, device='cpu',
                patience=7, min_delta=0.0001, weight_decay=1e-5):
    logger.info(f"Starting model training {model.__class__.__name__}")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction='mean')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, verbose=True)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    best_val_loss = float('inf')
    no_improve_count = 0
    best_model_state = None

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            seq_true = batch[0].to(device)
            optimizer.zero_grad()
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                seq_true = batch[0].to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        logger.info(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}')

        scheduler.step(val_loss)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break

    if epoch == n_epochs - 1 and best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f'Training completed. Restoring the best model with validation loss: {best_val_loss:.5f}')

    return history, model

# Clase para ensemble de modelos
class EnsembleModel:
    def __init__(self, models, names, weights=None):
        self.models = models
        self.names = names

        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            self.weights = np.array(weights) / np.sum(weights)

        logger.info(f"Ensemble created with {len(models)} models and weights {self.weights}")

    def get_weighted_errors(self, data_loaders, device='cpu'):
        individual_errors = []

        for i, (model, loader) in enumerate(zip(self.models, data_loaders)):
            errors, _, _ = get_reconstruction_errors(model, loader, device)

            mean_error = np.mean(errors)
            std_error = np.std(errors)
            normalized_errors = (errors - mean_error) / std_error

            individual_errors.append(normalized_errors)
            logger.info(f"Model {self.names[i]}: Mean Error {mean_error:.5f}, Std {std_error:.5f}")

        weighted_errors = np.zeros_like(individual_errors[0])
        for i, errors in enumerate(individual_errors):
            weighted_errors += self.weights[i] * errors

        return weighted_errors, individual_errors

    def optimize_weights(self, val_errors_list, val_labels):
        from sklearn.metrics import roc_auc_score
        
        normalized_errors = []
        for errors in val_errors_list:
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            normalized_errors.append((errors - mean_error) / std_error)

        aucs = []
        for errors in normalized_errors:
            auc_score = roc_auc_score(val_labels, errors)
            aucs.append(auc_score)

        auc_weights = np.array(aucs) / np.sum(aucs)
        self.weights = auc_weights

        logger.info(f"Weights optimized based on AUC: {self.weights}")
        return self.weights

def get_reconstruction_errors(model, data_loader, device='cpu'):
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction='none')

    reconstruction_errors = []
    original_data = []
    reconstructed_data = []

    with torch.no_grad():
        for batch in data_loader:
            seq_true = batch[0].to(device)
            seq_pred = model(seq_true)

            error = criterion(seq_pred, seq_true).mean(dim=(1,2))
            reconstruction_errors.extend(error.cpu().numpy())

            original_data.append(seq_true.cpu().numpy())
            reconstructed_data.append(seq_pred.cpu().numpy())

    original_data = np.vstack([x for x in original_data])
    reconstructed_data = np.vstack([x for x in reconstructed_data])

    return np.array(reconstruction_errors), original_data, reconstructed_data