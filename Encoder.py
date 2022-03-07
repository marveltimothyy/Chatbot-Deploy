import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        '''
        Initialize GRU; the input_size and hidden_size params 
        are both set to 'hidden_size' because our input 
        size is a word embedding with number of features == hidden_size
        '''
        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          n_layers,
                          bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):

        embedded = self.embedding(input_seq)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        outputs, hidden = self.gru(packed, hidden)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        outputs = outputs[:, :,
                          :self.hidden_size] + outputs[:, :,
                                                       self.hidden_size:]

        return outputs, hidden
