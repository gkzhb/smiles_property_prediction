r"""undocumented
这个页面的代码很大程度上参考(复制粘贴)了https://github.com/huggingface/pytorch-pretrained-BERT的代码， 如果你发现该代码对你
    有用，也请引用一下他们。
"""

__all__ = [
    "BertModel",
]

import copy
import json
import math

import torch
from torch import nn
import numpy as np

from fastNLP.io.file_utils import _get_file_name_base_on_postfix
from fastNLP.io.file_utils import _get_bert_dir
from fastNLP.core import logger
from fastNLP.core.utils import _build_args


import fastNLP as fnlp
from fastNLP.io.file_utils import _get_embedding_url, cached_path, \
    PRETRAINED_BERT_MODEL_DIR, _get_file_name_base_on_postfix
from fastNLP.core import logger
from fastNLP.modules.decoder import ConditionalRandomField, allowed_transitions
import torch.nn.functional as F
import pdb

CONFIG_FILE = 'bert_config.json'

BERT_KEY_RENAME_MAP_1 = {
    'gamma': 'weight',
    'beta': 'bias',
    'distilbert.embeddings': 'bert.embeddings',
    'distilbert.transformer': 'bert.encoder',
}

BERT_KEY_RENAME_MAP_2 = {
    'q_lin': 'self.query',
    'k_lin': 'self.key',
    'v_lin': 'self.value',
    'out_lin': 'output.dense',
    'sa_layer_norm': 'attention.output.LayerNorm',
    'ffn.lin1': 'intermediate.dense',
    'ffn.lin2': 'output.dense',
    'output_layer_norm': 'output.LayerNorm',
}

class BertCRFForTokenClassification(fnlp.models.base_model.BaseModel):
    """
    BERT model for token classification.

    """
    def __init__(self, embed , num_labels, tag_vocab, dropout=0.1):
        """
        
        :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
        :param int num_labels: 序列标注标签数目，无默认值.
        :param float dropout: dropout的大小，默认值为0.1.
        """
        super(BertCRFForTokenClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        trans = allowed_transitions(tag_vocab, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)

        if self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = False
            warn_msg = "Bert for token classification excepts BertEmbedding `include_cls_sep` False, " \
                       "but got True. FastNLP has changed it to False."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward_(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, seq_len, num_labels]
        """
        sequence_output = self.bert(words)  # [batch_size, seq_len, embed_dim]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return {fnlp.Const.OUTPUT: logits}

    def predict__(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size, seq_len]
        """
        #pdb.set_trace()
        if hasattr(self.bert, 'predict'):
            sequence_output = self.bert.predict(words) #[fnlp.Const.OUTPUT]
            logits = self.classifier(sequence_output)
        else:
            logits = self.forward(words)[fnlp.Const.OUTPUT]
        return {fnlp.Const.OUTPUT: torch.argmax(logits, dim=-1)}

    def _forward(self, words, target):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, seq_len, num_labels]
        """

        mask = words.ne(0)
        distill_loss = 0
        sequence_output = self.bert(words)  # [batch_size, seq_len, embed_dim]
        if isinstance(sequence_output, tuple):
            sequence_output, distill_loss = sequence_output
        #pdb.set_trace()
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        logits = F.log_softmax(logits, dim=-1)

        if target is None:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, mask)
            return {'loss': loss+distill_loss}

    def forward(self, words, target):
        return self._forward(words, target)

    def predict(self, words):
        return self._forward(words, target=None)

        #return {fnlp.Const.OUTPUT: logits}


class BertForTokenClassification(fnlp.models.base_model.BaseModel):
    """
    BERT model for token classification.

    """
    def __init__(self, embed , num_labels, dropout=0.1):
        """
        
        :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
        :param int num_labels: 序列标注标签数目，无默认值.
        :param float dropout: dropout的大小，默认值为0.1.
        """
        super(BertForTokenClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = False
            warn_msg = "Bert for token classification excepts BertEmbedding `include_cls_sep` False, " \
                       "but got True. FastNLP has changed it to False."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, seq_len, num_labels]
        """
        sequence_output = self.bert(words)  # [batch_size, seq_len, embed_dim]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return {fnlp.Const.OUTPUT: logits}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size, seq_len]
        """
        #pdb.set_trace()
        if hasattr(self.bert, 'predict'):
            sequence_output = self.bert.predict(words) #[fnlp.Const.OUTPUT]
            logits = self.classifier(sequence_output)
        else:
            logits = self.forward(words)[fnlp.Const.OUTPUT]
        return {fnlp.Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForSequenceClassification(fnlp.models.base_model.BaseModel):
    r"""
    BERT model for classification.

    """

    def __init__(self, embed, num_labels: int = 2, dropout=0.1):

        super(BertForSequenceClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for sequence classification excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warning(warn_msg)

    def forward(self, words):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, num_labels]
        """
        hidden = self.dropout(self.bert(words))
        cls_hidden = hidden[:, 0]
        logits = self.classifier(cls_hidden)
        #pdb.set_trace()

        return {fnlp.Const.OUTPUT: logits}

    def predict(self, words):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        #pdb.set_trace()
        if hasattr(self.bert, 'predict'):
            hidden = self.bert.predict(words)
            cls_hidden = hidden[:, 0]
            logits = self.classifier(cls_hidden)
        else:
            logits = self.forward(words)[fnlp.Const.OUTPUT]
        if logits.size(1) == 1:
            return {fnlp.Const.OUTPUT: logits}
        else:
            return {fnlp.Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertConfig(object):
    r"""Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        r"""Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        r"""Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        r"""Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        r"""Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        r"""Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        r""" Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


# class BertLayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         r"""Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(BertLayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps
#
#     def forward(self, x):
#         u = x.mean(-1, keepdim=True)
#         s = (x - u).pow(2).mean(-1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)
#         return self.weight * x + self.bias

BertLayerNorm = torch.nn.LayerNorm


class DistilBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(DistilBertEmbeddings, self).__init__()

        def create_sinusoidal_embeddings(n_pos, dim, out):
            position_enc = np.array([
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ])
            out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
            out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
            out.detach_()
            out.requires_grad = False

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(n_pos=config.max_position_embeddings,
                                         dim=config.hidden_size,
                                         out=self.position_embeddings.weight)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        r"""
        Parameters
        ----------
        input_ids: torch.tensor(bs, max_seq_length)
            The token ids to embed.
        token_type_ids: no used.
        Outputs
        -------
        embeddings: torch.tensor(bs, max_seq_length, dim)
            The embedded tokens (plus position embeddings, no token_type embeddings)
        """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device) # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                      # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(input_ids)                   # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(position_ids)        # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)             # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)               # (bs, max_seq_length, dim)
        return embeddings


class BertEmbeddings(nn.Module):
    r"""Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, words_embeddings=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if words_embeddings is None:
            words_embeddings = self.word_embeddings(input_ids)
        else:
            assert input_ids.size() == words_embeddings.size()[: -1]
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(6)])#config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertEncoderTheseus(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.successor = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers // 2)])
        self.successor_rate = 0.
        self.bernoulli = False
        self.uniform = False
        self.predict_with_successor = True
        self.compress_ratio = 2

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        #pdb.set_trace()
        all_encoder_layers = []
        group = [[self.layer[idx * 2], self.layer[idx * 2 + 1]] for idx in range(len(self.layer) // 2)]

        #pdb.set_trace()
        if not self.bernoulli:

            for idx, group_module in enumerate(group):
                encoder_hidden = hidden_states
                for i in range(self.compress_ratio):
                    encoder_hidden = group_module[i](encoder_hidden, attention_mask)

                #encoder_hidden = group_module[1](encoder_hidden, attention_mask)
                '''if output_all_encoded_layers:
                    all_encoder_layers.append(encoder_hidden)'''

                if self.successor_rate > 0:
                    successor_hidden = self.successor[idx](hidden_states, attention_mask)
                else:
                    successor_hidden = 0
                hidden_states = (1 - self.successor_rate) * encoder_hidden + self.successor_rate * successor_hidden
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            if not output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        else:
            #print('bernoulli!\n')
            #pdb.set_trace()
            
            original = True if random.random() > self.successor_rate else False

            for idx, (encoder, successor) in enumerate(zip(group, self.successor)):
                if not self.uniform:
                    original = True if random.random() > self.successor_rate else False
                if original:
                    encoder_hidden = hidden_states
                    for i in range(self.compress_ratio):
                        encoder_hidden = encoder[i](encoder_hidden, attention_mask)
                    hidden_states = encoder_hidden
                else:
                    hidden_states = successor(hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
            if not output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

        return all_encoder_layers

    def predict(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        # warnings.warn('Start predict successor!')
        #pdb.set_trace()
        all_encoder_layers = []
        for idx, layer_module in enumerate(self.successor if (self.predict_with_successor and self.successor_rate > 0) else self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertOrderedDistillationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.candidate = copy.deepcopy(layer)
        self.loss_func = torch.nn.MSELoss()
        self.distill = False

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        candidate_hidden = None
        for idx, layer_module in enumerate(self.layer):
            if self.distill and self.training and idx == len(self.layer) - 2:
                candidate_hidden = self.candidate(hidden_states.detach(), attention_mask)
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if self.distill and self.training and candidate_hidden is not None:
            loss = self.loss_func(candidate_hidden, hidden_states.detach())
        else:
            loss = None

        if loss is not None:
            return (all_encoder_layers,) + (loss,)

        return all_encoder_layers

    def predict(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

    def update_layer(self):
        if len(self.layer) < 2:
            return
        self.layer = nn.ModuleList([self.layer[idx] for idx in range(len(self.layer) - 2)])
        self.layer.append(copy.deepcopy(self.candidate))
        self.candidate.load_state_dict(self.layer[-2].state_dict())


class BertKeepOrderedDistillationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.candidate = copy.deepcopy(layer)
        self.loss_func = torch.nn.MSELoss()
        self.distill = False
        self.idx_layer = len(self.layer) - 2

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        candidate_hidden = None
        #pdb.set_trace()
        for idx, layer_module in enumerate(self.layer[:self.idx_layer + 1]):
            if self.distill and self.training and idx == self.idx_layer:
                candidate_hidden = self.candidate(hidden_states.detach(), attention_mask)
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if self.distill and self.training and candidate_hidden is not None:
            loss = self.loss_func(candidate_hidden, hidden_states.detach())
        else:
            loss = None
        #pdb.set_trace()
        if loss is not None:
            return (all_encoder_layers,) + (loss,)

        return all_encoder_layers

    def predict(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for idx, layer_module in enumerate(self.layer):
            if idx == self.idx_layer:
                hidden_states = self.candidate(hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                break
            else:
                hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

    def update_layer(self):
        #pdb.set_trace()
        if self.idx_layer > 1:
            self.layer[self.idx_layer].load_state_dict(self.candidate.state_dict())
        self.idx_layer = max(self.idx_layer - 1, 1)
        self.candidate.load_state_dict(self.layer[self.idx_layer].state_dict())


class BertTransitionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        self.confidence = nn.ModuleList([nn.Linear(config.hidden_size, config.num_hidden_layers - remain_layer)
                                         for remain_layer in range(config.num_hidden_layers - 1)])
        self.only_cls = True
        self.is_classification_task = True

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, target=None,
                classifier_func=None):
        assert self.is_classification_task
        batch_size = hidden_states.size(0)
        all_encoder_layers = []
        transition_outputs = []
        for idx, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if idx != len(self.layer) - 1:
                transition_output = self.confidence[idx](hidden_states)
                transition_outputs.append(transition_output)
            all_encoder_layers.append(hidden_states)

        if target is not None:
            seq_len_mask = attention_mask[:, :, 0].eq(0).float()  # [batch, seq_len]
            for idx in range(len(self.layer) - 1):
                scores = torch.zeros_like(transition_outputs[idx])
                # if

                for jdx in range(idx + 1, len(self.layer)):
                    pass

        # torch.nn.CosineSimilarity

        if not output_all_encoded_layers:
            all_encoder_layers = [all_encoder_layers[-1]]
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    r"""
    BERT(Bidirectional Embedding Representations from Transformers).

    用预训练权重矩阵来建立BERT模型::

        model = BertModel.from_pretrained(model_dir_or_name)

    用随机初始化权重矩阵来建立BERT模型::

        model = BertModel()

    :param int vocab_size: 词表大小，默认值为30522，为BERT English uncase版本的词表大小
    :param int hidden_size: 隐层大小，默认值为768，为BERT base的版本
    :param int num_hidden_layers: 隐藏层数，默认值为12，为BERT base的版本
    :param int num_attention_heads: 多头注意力头数，默认值为12，为BERT base的版本
    :param int intermediate_size: FFN隐藏层大小，默认值是3072，为BERT base的版本
    :param str hidden_act: FFN隐藏层激活函数，默认值为``gelu``
    :param float hidden_dropout_prob: FFN隐藏层dropout，默认值为0.1
    :param float attention_probs_dropout_prob: Attention层的dropout，默认值为0.1
    :param int max_position_embeddings: 最大的序列长度，默认值为512，
    :param int type_vocab_size: 最大segment数量，默认值为2
    :param int initializer_range: 初始化权重范围，默认值为0.02
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        super(BertModel, self).__init__()
        self.config = config
        self.hidden_size = self.config.hidden_size
        self.model_type = 'bert'
        if hasattr(config, 'sinusoidal_pos_embds'):
            self.model_type = 'distilbert'
        elif 'model_type' in kwargs:
            self.model_type = kwargs['model_type'].lower()

        if self.model_type == 'distilbert':
            self.embeddings = DistilBertEmbeddings(config)
        else:
            self.embeddings = BertEmbeddings(config)

        self.encoder = BertEncoder(config)
        if self.model_type != 'distilbert':
            self.pooler = BertPooler(config)
        else:
            logger.info('DistilBert has NOT pooler, will use hidden states of [CLS] token as pooled output.')
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        r""" Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,
                target=None, classifier_func=None):
        """

        :param torch.LongTensor input_ids: bsz x max_len的输入id
        :param torch.LongTensor token_type_ids: bsz x max_len，如果不输入认为全为0，一般第一个sep(含)及以前为0, 一个sep之后为1
        :param attention_mask: 需要attend的为1，不需要为0
        :param bool output_all_encoded_layers: 是否输出所有层，默认输出token embedding(包含bpe, position以及type embedding)
            及每一层的hidden states。如果为False，只输出最后一层的结果
        :param torch.LongTensor target: label
        :param torch.nn.Module classifier_func: classifier
        :return: encode_layers: 如果output_all_encoded_layers为True，返回list(共num_layers+1个元素)，每个元素为
            bsz x max_len x hidden_size否则返回bsz x max_len x hidden_size的tensor;
            pooled_output: bsz x hidden_size为cls的表示，可以用于句子的分类
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        # encoded_layers = self.encoder(embedding_output,
        #                               extended_attention_mask,
        #                               output_all_encoded_layers=output_all_encoded_layers)

        input_args = {'hidden_states': embedding_output,
                      'attention_mask': extended_attention_mask,
                      'output_all_encoded_layers': output_all_encoded_layers,
                      'target': target,
                      'classifier_func': classifier_func
                      }

        if not self.training and hasattr(self.encoder, 'predict'):
            input_args = _build_args(self.encoder.predict, **input_args)
            encoded_results = self.encoder.predict(**input_args)
        else:
            input_args = _build_args(self.encoder.forward, **input_args)
            encoded_results = self.encoder(**input_args)

        if isinstance(encoded_results, tuple):
            encoded_layers, loss = encoded_results
        else:
            encoded_layers = encoded_results
            loss = None

        sequence_output = encoded_layers[-1]
        if self.model_type != 'distilbert':
            pooled_output = self.pooler(sequence_output)
        else:
            pooled_output = sequence_output[:, 0]
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        else:
            encoded_layers.insert(0, embedding_output)

        results = (encoded_layers, pooled_output)
        if loss is not None:
            results = results + (loss,)

        return results

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        kwargs.pop('cache_dir', None)
        kwargs.pop('from_tf', None)

        # get model dir from name or dir
        pretrained_model_dir = _get_bert_dir(model_dir_or_name)

        # Load config
        config_file = _get_file_name_base_on_postfix(pretrained_model_dir, '.json')
        config = BertConfig.from_json_file(config_file)

        if state_dict is None:
            weights_path = _get_file_name_base_on_postfix(pretrained_model_dir, '.bin')
            state_dict = torch.load(weights_path, map_location='cpu')
        else:
            logger.error(f'Cannot load parameters through `state_dict` variable.')
            raise RuntimeError(f'Cannot load parameters through `state_dict` variable.')

        model_type = 'BERT'
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'bert' not in key:
                new_key = 'bert.' + key
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            for key_name in BERT_KEY_RENAME_MAP_1:
                if key_name in key:
                    new_key = key.replace(key_name, BERT_KEY_RENAME_MAP_1[key_name])
                    if 'distilbert' in key:
                        model_type = 'DistilBert'
                    break
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            for key_name in BERT_KEY_RENAME_MAP_2:
                if key_name in key:
                    new_key = key.replace(key_name, BERT_KEY_RENAME_MAP_2[key_name])
                    break
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Instantiate model.
        model = cls(config, model_type=model_type, *inputs, **kwargs)

        encoder_type = kwargs.pop('encoder_type', 'original')
        encoder_type = encoder_type.lower()
        assert encoder_type in ['original', 'order-distillation', 'keep-order-distillation',
                                'transition']
        if 'order-distillation' in encoder_type:
            if encoder_type == 'order-distillation':
                model.encoder = BertOrderedDistillationEncoder(config)
            elif encoder_type == 'keep-order-distillation':
                model.encoder = BertKeepOrderedDistillationEncoder(config)
            else:
                raise RuntimeError
            old_keys = []
            new_keys = []
            num_layers = len(model.encoder.layer)
            for key in state_dict.keys():
                if f'layer.{num_layers - 2}' in key:
                    new_key = key.replace(f'layer.{num_layers - 2}', 'candidate')
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = copy.deepcopy(state_dict[old_key])
        elif encoder_type == 'transition':
            model.encoder = BertTransitionEncoder(config)
            update_state_dict = {n: p for n, p in model.named_parameters() if 'confidence' in n}
            for n, p in update_state_dict.items():
                state_dict[n] = p

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.warning("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.debug("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))

        logger.info(f"Load pre-trained {model_type} parameters from file {weights_path}.")
        return model
