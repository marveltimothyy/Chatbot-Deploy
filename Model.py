# from Evaluate import GreedySearchDecoder
import torch
from torch import optim
import torch.nn as nn
import os
import Encoder
import Decoder
import Voc
import re
import unicodedata


class Model():

    def __init__(self, checkpoint, loadFilename, hidden_size, batch_size,
                 learning_rate, device, max_length, voc):
        self.device = device

        self.model_name = 'cb_model'
        self.attn_model = 'dot'

        self.voc = voc

        self.max_length = max_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loadFilename = loadFilename

        self.embedding = nn.Embedding(voc.num_words, hidden_size)
        self.encoder = Encoder.EncoderRNN(self.hidden_size, self.embedding)
        self.decoder = Decoder.LuongAttnDecoderRNN(self.attn_model,
                                                   self.embedding,
                                                   hidden_size,
                                                   voc.num_words)

        self.encoder_optimizer = optim.Adam(
            self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.Adam(
            self.decoder.parameters(), lr=learning_rate)

        # load checkpoint
        # checkpoint = torch.load(loadFilename)
        checkpoint = torch.load(
            self.loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

        # Initialize word embeddings
        self.embedding.load_state_dict(embedding_sd)

        # Initialize encoder & decoder models
        self.encoder.load_state_dict(encoder_sd)
        self.decoder.load_state_dict(decoder_sd)

        # Use appropriate device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        self.decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        # self.searcher =  GreedySearchDecoder(self.encoder,self.decoder)

    def searcher(self, input_seq, input_length, max_length):
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # print('encoder state dict', self.decoder.state_dict())
        decoder_hidden = encoder_hidden
        # decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device,
                                   dtype=torch.long) * 1  # 1 is value of SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs)
            # print('decoder_out', decoder_output)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(s):
        s = Model.unicodeToAscii(s.lower().strip())
        # s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r'[^a-zA-Z]', ' ', s)
        # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def indexesFromSentence(self, sentence):
        # 2 is value of EOS_TOKEN
        return [self.voc.word2index[word] for word in sentence.split(' ')] + [2]

    def evaluate(self, sentence):
        indexes_batch = [self.indexesFromSentence(sentence)]

        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])

        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(self.device)
        lengths = lengths.to("cpu")
        # Decode sentence with searcher
        tokens, scores = self.searcher(input_batch, lengths, self.max_length)
        # indexes -> words
        decoded_words = [self.voc.index2word[token.item()] for token in tokens]
        return decoded_words

    # def evaluateInput(self):
    #     input_sentence = ''
    #     while(1):
    #         try:
    #             # Get input sentence
    #             input_sentence = input('You: ',)
    #             if input_sentence == 'q' or input_sentence == 'quit':
    #                 break
    #             input_sentence = Model.normalizeString(input_sentence)

    #             output_words = self.evaluate(input_sentence)
    #             # print(output_words)
    #             outword = []
    #             for i in output_words:
    #                 if i == 'EOS':
    #                     break
    #                 elif i != 'PAD':
    #                     outword.append(i)

    #             string = ' '.join(outword)
    #             string = re.sub(' ll ', "'ll ", string)
    #             string = re.sub(' t ', "'t ", string)
    #             string = re.sub(' d ', "'d ", string)
    #             string = re.sub(' re ', "'re ", string)
    #             string = re.sub(' s ', "'s ", string)
    #             string = re.sub(' m ', "'m ", string)
    #             string = re.sub(' ve ', "'ve ", string)
    #             # print(self.voc.__dict__)
    #             print('BOT: {}'.format(string))
    #         except KeyError:
    #             print(
    #                 "I am sorry, as a bot i have limited vocabulary to understand. Please use another word or fix your typo word")

    def response(self, input_sentence):
        try:
            input_sentence = Model.normalizeString(input_sentence)

            output_words = self.evaluate(input_sentence)
            # print(output_words)
            outword = []
            for i in output_words:
                if i == 'EOS':
                    break
                elif i != 'PAD':
                    outword.append(i)

            string = ' '.join(outword)
            string = re.sub(' ll ', "'ll ", string)
            string = re.sub(' t ', "'t ", string)
            string = re.sub(' d ', "'d ", string)
            string = re.sub(' re ', "'re ", string)
            string = re.sub(' s ', "'s ", string)
            string = re.sub(' m ', "'m ", string)
            string = re.sub(' ve ', "'ve ", string)
            return string

        except KeyError:
            return(
                "I am sorry, as a bot i have limited vocabulary to understand. Please use another word or fix your typo word")
