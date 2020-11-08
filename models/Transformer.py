import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#first level
class First_TransformerModel(nn.Module):

    def __init__(self, ninp=300, nhead=4, nhid=128, nlayers=6, dropout=0.5):
        super(First_TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        # self.decoder = nn.Linear(ninp, ntoken)

    def _generate_square_subsequent_mask(self, src, lenths):
        '''
        padding_mask
        src:max_lenth,num,300
        lenths:[lenth1,lenth2...]
        '''

        # mask num_of_sens x max_lenth
        mask = torch.ones(src.size(1), src.size(0)) == 1
        for i in range(len(lenths)):
            lenth = lenths[i]
            for j in range(lenth):
                mask[i][j] = False

        # mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        #mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, mask):
        '''
        src:num_of_all_sens,max_lenth,300
        '''
        self.src_mask = mask

        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=self.src_mask)
        output = output[0,:,:]
        #output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# second level

class Second_TransformerModel(nn.Module):

    def __init__(self, ninp=300, nhead=4, nhid=256, nlayers=6, dropout=0.1):
        super(Second_TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

    def _generate_square_subsequent_mask(self, src, lenths):
        '''
        padding_mask
        src:num_of_sentence x batch(文章数) x 300
        lenths:[lenth1,lenth2...]
        '''

        # mask num_of_sens x max_lenth
        mask = torch.ones(src.size(1), src.size(0)) == 1
        for i in range(len(lenths)):
            lenth = lenths[i]
            for j in range(lenth):
                mask[i][j] = False

        return mask

    def forward(self, src, mask):
        '''

        src:max_sentence_num x batch(文章数) x 300

        '''
        self.src_mask = mask

        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=self.src_mask)
        # output = self.decoder(output)
        return output


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class segmentmodel(nn.Module):
    def __init__(self, ninp=300, nhead=4, nhid=256, nlayers=6, dropout=0.1):
        super(segmentmodel, self).__init__()
        self.first_layer = First_TransformerModel(ninp, nhead, nhid, nlayers, dropout)
        self.second_layer = Second_TransformerModel(ninp, nhead, nhid, nlayers, dropout)
        self.linear = nn.Linear(ninp, 2)
        self.criterion = torch.nn.CrossEntropyLoss()

    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = torch.tensor(s.unsqueeze(0).unsqueeze(0))
        padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0, 0, max_document_length - d_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def forward(self, batch):
        batch_size = len(batch)

        sentences_per_doc = []
        all_batch_sentences = []
        for document in batch:
            all_batch_sentences.extend(document)
            sentences_per_doc.append(len(document))

        lengths = [s.size()[0] for s in all_batch_sentences]

        max_length = max(lengths)
        # logger.debug('Num sentences: %s, max sentence length: %s',
        # sum(sentences_per_doc), max_length)

        padded_sentences = [self.pad(s, max_length) for s in all_batch_sentences]
        big_tensor = torch.cat(padded_sentences, 1).cuda()  # (max_length, batch size, 300)

        mask = self.first_layer._generate_square_subsequent_mask(big_tensor,
                                                                 lengths).cuda()

        firstlayer_out = self.first_layer(src=big_tensor, mask=mask)
        # 句子数 x 300

        # padded_output  batch x 300
        # 将各个文章中的句子分别取出来
        encoded_documents = []
        index = 0
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(firstlayer_out[index: end_index, :])
            index = end_index

        # docuemnt_padding
        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        padded_docs = [self.pad_document(d, max_doc_size) for d in encoded_documents]
        docs_tensor = torch.cat(padded_docs, 1)
        # docs_tensor max_doc_size x batch x 300

        mask = self.second_layer._generate_square_subsequent_mask(docs_tensor, doc_sizes).cuda()
        second_layer_out = self.second_layer(src=docs_tensor, mask=mask)
        # 去除最后一个句子
        doc_outputs = []

        for i, doc_len in enumerate(doc_sizes):
            doc_outputs.append(second_layer_out[0:doc_len - 1, i, :])  # -1 to remove last predic
        sentence_outputs = torch.cat(doc_outputs, 0)
        # 句子数 x 300

        out = self.linear(sentence_outputs)
        return out


def create():

    return segmentmodel()
