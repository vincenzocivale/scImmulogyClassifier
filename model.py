import torch
import torch.nn as nn
from typing import Optional, List
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class MLPConfig(PretrainedConfig):
    """
    Configurazione per l'AdvancedMLPClassifier.
    
    Args:
        input_dim (int): Dimensione dell'input.
        hidden_dims (List[int]): Lista delle dimensioni dei layer nascosti.
        output_dim (int): Dimensione dell'output.
        dropout_rate (float): Tasso di dropout.
        use_residual_in_hidden (bool): Se usare o meno connessioni residuali nei layer nascosti.
    """
    model_type = "mlp_classifier"

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 2,
        dropout_rate: float = 0.2,
        use_residual_in_hidden: bool = True,
        id2label: Optional[dict] = None, 
        label2id: Optional[dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256]
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.use_residual_in_hidden = use_residual_in_hidden
        self.id2label = id2label
        self.label2id = label2id


class MLPBlock(nn.Module):
    """
    Singolo blocco MLP con supporto opzionale a residual connection.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.2, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual and (input_dim == output_dim)

        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.linear(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + identity
        return x


class MLPClassifier(PreTrainedModel):
    """
    MLP classifier compatibile con Hugging Face.
    """
    config_class = MLPConfig
    base_model_prefix = "mlp_classifier"

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        # BatchNorm iniziale sugli input
        self.input_bn = nn.BatchNorm1d(config.input_dim)

        # Costruzione dei blocchi hidden
        all_dims = [config.input_dim] + config.hidden_dims
        layers = []
        for i in range(len(all_dims) - 1):
            layers.append(
                MLPBlock(
                    input_dim=all_dims[i],
                    output_dim=all_dims[i + 1],
                    dropout_rate=config.dropout_rate,
                    use_residual=config.use_residual_in_hidden and (all_dims[i] == all_dims[i + 1])
                )
            )
        self.hidden_network = nn.Sequential(*layers)

        # Proiezione finale
        self.output_layer = nn.Linear(all_dims[-1], config.output_dim)

        # Inizializzazione pesi
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> SequenceClassifierOutput:
        """
        Forward compatibile con il Trainer di HF.
        """
        # Aggiunta di una gestione del batch per la BatchNorm, che richiede un input 2D.
        # Riferimento: https://github.com/huggingface/transformers/issues/24795
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.ndim > 2:
            input_ids = input_ids.view(input_ids.size(0), -1)

        x = self.input_bn(input_ids)
        x = self.hidden_network(x)
        logits = self.output_layer(x)

        loss = None
        if labels is not None:
            # Per CrossEntropyLoss, i logits devono essere di forma (batch_size, num_classes)
            # e le etichette di forma (batch_size,)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.output_dim), labels.view(-1))

        if not return_dict:
            output = (logits, )
            if loss is not None:
                output = (loss,) + output
            return output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )