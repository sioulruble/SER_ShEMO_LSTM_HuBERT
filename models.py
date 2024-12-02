import torch
import torch.nn as nn
from transformers import HubertModel

class EmotionClassifier(nn.Module):
    def __init__(self, hubert_config, classifier_type='LSTM', num_labels=2, hidden_size=768):
        super(EmotionClassifier, self).__init__()
        self.hubert = HubertModel.from_pretrained(hubert_config)
        self.classifier_type = classifier_type
        self.hidden_size = hidden_size

        if classifier_type == 'LSTM':
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=256,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(256 * 2, num_labels)
        self.init_weights()

    def init_weights(self):
        if self.classifier_type == 'LSTM':
            nn.init.xavier_uniform_(self.fc.weight)
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, input_values, attention_mask=None, lengths=None):
        outputs = self.hubert(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        if self.classifier_type == 'LSTM':
            if lengths is not None:
                packed_hidden = nn.utils.rnn.pack_padded_sequence(
                    hidden_states, lengths, batch_first=True, enforce_sorted=False
                )
                lstm_output, _ = self.lstm(packed_hidden)
                lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
            else:
                lstm_output, _ = self.lstm(hidden_states)
            x = torch.mean(lstm_output, dim=1)
            x = self.dropout(x)
            logits = self.fc(x)
            return logits
