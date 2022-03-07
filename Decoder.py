import Attention as Attn
import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size,
                 output_size, n_layers=1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        # Define layers
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          n_layers,
                          bidirectional=True)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn.Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        rnn_output = rnn_output[:, :,
                                :self.hidden_size] + rnn_output[:, :,
                                                                self.hidden_size:]
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        '''
        Multiply attention weights to encoder outputs 
        to get new "weighted sum" context vector
        '''
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        '''
        Concatenate weighted context vector and GRU output using Luong eq. 5
        '''
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
