# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import Optional, Sequence, Union

import torch


# 添加新分词器
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os
import sys

import numpy as np
import regex as re

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE
    # tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func

class Vocab:
    def __init__(self, path):
        import json
        with open(path, "r") as f:
            self.vocab = json.load(f)
        self.tokens = [""] * len(self.vocab)
        for key, value in self.vocab.items():
            self.tokens[value] = key
    
    def get_idx(self, tokens):
        if type(tokens) == str:
            if len(tokens) > 2 and tokens[0] == "[" and tokens[-1] == "]":
                tokens = tokens[1:-1]
            tokens = tokens.split(", ")
            tokens = map(lambda t: t.strip(), tokens)
        return list(map(lambda t: self.vocab[t], tokens))

    def get_tokens(self, idx):
        if isinstance(idx, (int, np.int64)):
            return self.tokens[idx]
        return list(map(lambda i: self.tokens[i], idx))


@lru_cache()
def bytes_to_unicode():
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class GPTTokenizer(object):

    def __init__(self, vocab_dir, errors='replace', max_len=None, print_info=False):
        
        tokenizer_info = torch.load(os.path.join(vocab_dir, "aixTokenizer"), map_location="cpu")
        self.encoder = tokenizer_info.get("vocab")
        bpe_data = tokenizer_info.get("merges_files")
        special_tokens_dict = tokenizer_info.get("special_tokens_dict")
        self.special_tokens_dict = special_tokens_dict
        # Yocot : 此处新加了一个stop words的字段来存储停止词，和服务端对齐
        self.stop_words = tokenizer_info.get("special_tokens_dict")["additional_special_tokens"]
        if self.encoder is None or bpe_data is None or special_tokens_dict is None:
            raise ValueError("format in aixTokenizer was wrong")
        
        self.max_len = max_len if max_len is not None else int(1e12)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.print_info = print_info

        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens_dict["additional_special_tokens"])
        
        self.bos_token = special_tokens_dict["bos_token"]
        self.eos_token = special_tokens_dict["eos_token"]
        self.unk_token = special_tokens_dict["unk_token"]

        self.eos_id = self.encoder.get("<|endoftext|>", 0)
        self.bos_id = self.encoder.get(self.bos_token, 0)
        self.unk_id = self.encoder.get(self.unk_token, 0)
        
        self.pad_tok_id = self.encoder.get("<fim_pad>", 0)
        self.prefix_tok_id = self.encoder["<fim_prefix>"]
        self.middle_tok_id = self.encoder["<fim_middle>"]
        self.suffix_tok_id = self.encoder["<fim_suffix>"]

    def __len__(self):
        return len(self.encoder)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        
        self.special_tokens = dict((tok, self.encoder.get(tok, -1))
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        if self.print_info:
            print("Special tokens {}".format(self.special_tokens))

    def token2id(self, token: str):
        if token not in self.encoder:
            print(f"WARNING: {token} was not in vocabulary, use UNK id {self.encoder.get(self.unk_token, 0)}")
            return self.encoder.get(self.unk_token, 0)
        else:
            return self.encoder.get(token, 0)
    
    def id2token(self, id_: int):
        if id_ >= len(self.encoder):
            print(f"WARNING: {id_} is larger than vocabulary, return None")
            return ""
        else:
            return self.decoder.get(id_, "")
        
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except BaseException:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token)
            else:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)):
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            print(
                "WARNING: Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(
                    len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text: str, file_path="", bos=False, eos=False):
        
        if len(file_path) > 0:
            ids_ = self.convert_tokens_to_ids(self.tokenize(file_path + "\n" + text))
            ids_ = [self.encoder["<filename>"]] + ids_
        else:
            ids_ = self.convert_tokens_to_ids(self.tokenize(text))
        
        if bos:
            ids_ = [self.bos_id] + ids_
        if eos:
            ids_ = ids_ + [self.eos_id]
        
        return ids_
    
    def encode_span(self, pre_context: str, post_context: str, file_path=""):
        assert isinstance(pre_context, str)
        assert isinstance(post_context, str)
        
        if len(file_path) > 0:
            pre_context = file_path + "\n" + pre_context
            res_ = [self.special_tokens["<fim_prefix>"], self.encoder["<filename>"]]
        else:
            res_ = [self.special_tokens["<fim_prefix>"]]
        
        if len(pre_context) > 0:
            res_.extend(self.encode(pre_context))
        res_.append(self.special_tokens["<fim_suffix>"])
        if len(post_context) > 0:
            res_.extend(self.encode(post_context))
        res_.append(self.special_tokens["<fim_middle>"])
        return res_

    def encode_chat(self, contents, env_prompts=None):
        """
            [
                {
                    "content": "Is it possible to imagine a society without law?",
                    "role": "user",
                },
                {
                    "content": "It is difficult to imagine a society...",
                    "role": "assistant",
                },
                {
                    "content": 'It seems like you consider the absence of law equal...',
                    "role": "user",
                },
                ...
            ]
        """
        assert self.encoder.get(f"<|assistant|>") is not None, f"vocab is not Chat model"
        system_msg = "Below is a dialogue between a human and an AI assistant called AixChat, which trained by aiXcoder. The AixChat tries to be helpful, polite, honest, sophisticated, and humble-but-knowledgeable. The AixChat is happy to help with almost anything about programming tasks, and will do its best to understand exactly what is needed, and will respond in details in Chinese. It also tries to avoid giving false or misleading information, and it caveats when it isn’t entirely sure about the right answer. That said, the assistant is practical and really does its best, and doesn’t let caution get too much in the way of being useful."
        prompt = [self.encoder["<|system|>"]] + self.encode("\n" + system_msg) + [self.encoder["<|end|>"]] + self.encode("\n")
        last_is_user = False
        
        if env_prompts is not None:
            for message in env_prompts:
                if message["role"] == "user":
                    prompt += [self.encoder["<|user|>"]] + self.encode("\n" + message["content"]) + [self.encoder["<|end|>"]] + self.encode("\n")
                    last_is_user = True
                else:
                    prompt += [self.encoder["<|assistant|>"]] + self.encode("\n" + message["content"]) + [self.encoder["<|end|>"]] + self.encode("\n")
                    last_is_user = False
        
        if contents is not None:
            for message in contents:
                if message["role"] == "user":
                    prompt += [self.encoder["<|user|>"]] + self.encode("\n" + message["content"]) + [self.encoder["<|end|>"]] + self.encode("\n")
                    last_is_user = True
                else:
                    prompt += [self.encoder["<|assistant|>"]] + self.encode("\n" + message["content"]) + [self.encoder["<|end|>"]] + self.encode("\n")
                    last_is_user = False
        
        if last_is_user:
            prompt += [self.encoder["<|assistant|>"]]
        else:
            prompt += [self.encoder["<|user|>"]] + self.encode("\nPlease continue") + [self.encoder["<|end|>"]] + self.encode("\n") +  [self.encoder["<|assistant|>"]]
        
        return prompt
    
    def decode(self, tokens, offset):
        text = ''.join(self.convert_ids_to_tokens(tokens))
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class SentencePieceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        from sentencepiece import SentencePieceProcessor
        self.model = SentencePieceProcessor(model_file=model_file)
        self._no_prefix_space_tokens = None

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size()

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_id()

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_id()

    @property
    def no_prefix_space_tokens(self):
        """tokens without prefix space."""
        if self._no_prefix_space_tokens is None:
            vocab = self.model.IdToPiece(list(range(self.vocab_size)))
            self._no_prefix_space_tokens = {
                i
                for i, tok in enumerate(vocab) if not tok.startswith('▁')
            }
        return self._no_prefix_space_tokens

    def _maybe_add_prefix_space(self, tokens, decoded):
        """maybe add prefix space for incremental decoding."""
        if len(tokens) and tokens[0] not in self.no_prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        add_bos = False
        add_eos = False
        if s.find('<BOS>') != -1:
            s = s.replace('<BOS>', '')
            add_bos = True
        if s == '<EOS>':
            s = ''
            add_eos = True
        return self.model.Encode(s, add_bos=add_bos, add_eos=add_eos)

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
        Returns:
            str: text of decoding tokens
        """
        if isinstance(t, torch.Tensor):
            t = t.tolist()
        t = t[offset:]
        out_string = self.model.Decode(t)
        if offset:
            out_string = self._maybe_add_prefix_space(t, out_string)
        return out_string

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        import addict
        add_bos = False
        add_eos = False

        input_ids = self.model.Encode(s, add_bos=add_bos, add_eos=add_eos)
        return addict.Addict(input_ids=input_ids)


class HuggingFaceTokenizer:
    """Tokenizer of sentencepiece.

    Args:
        model_dir (str): the directory of the tokenizer model
    """

    def __init__(self, model_dir: str):
        from transformers import (AutoTokenizer, CodeLlamaTokenizerFast,
                                  LlamaTokenizerFast)
        model_file = osp.join(model_dir, 'tokenizer.model')
        backend_tokenizer_file = osp.join(model_dir, 'tokenizer.json')
        model_file_exists = osp.exists(model_file)
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            print('WARNING: Can not find tokenizer.json. '
                  'It may take long time to initialize the tokenizer.')
        self.model = AutoTokenizer.from_pretrained(model_dir,
                                                   trust_remote_code=True)
        self.need_padding = isinstance(self.model, LlamaTokenizerFast) \
            or isinstance(self.model, CodeLlamaTokenizerFast)
        self._no_prefix_space_tokens = None
        # save tokenizer.json to reuse
        if not osp.exists(backend_tokenizer_file) and model_file_exists:
            if hasattr(self.model, 'backend_tokenizer'):
                self.model.backend_tokenizer.save(backend_tokenizer_file)

        if self.model.eos_token_id is None:
            generation_config_file = osp.join(model_dir,
                                              'generation_config.json')
            with open(generation_config_file, 'r') as f:
                cfg = json.load(f)
                self.model.eos_token_id = cfg['eos_token_id']

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_token_id

    @property
    def no_prefix_space_tokens(self):
        """tokens without prefix space."""
        if self._no_prefix_space_tokens is None:
            vocab = self.model.convert_ids_to_tokens(
                list(range(self.vocab_size)))
            self._no_prefix_space_tokens = {
                i
                for i, tok in enumerate(vocab) if not tok.startswith('▁')
            }
        return self._no_prefix_space_tokens

    def _maybe_add_prefix_space(self, tokens, decoded):
        """maybe add prefix space for incremental decoding."""
        if self.need_padding and len(
                tokens) and tokens[0] not in self.no_prefix_space_tokens:
            return ' ' + decoded
        else:
            return decoded

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        if s.find('<BOS>') != -1:
            s = s.replace('<BOS>', '<s>')
        if s == '<EOS>':
            s = '</s>'
        if len(s) == 0:
            add_special_tokens = True
        return self.model.encode(s, add_special_tokens=add_special_tokens)

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
        Returns:
            str: text of decoding tokens
        """
        skip_special_tokens = True
        t = t[offset:]
        out_string = self.model.decode(t,
                                       skip_special_tokens=skip_special_tokens)
        if offset:
            out_string = self._maybe_add_prefix_space(t, out_string)
        return out_string

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        add_special_tokens = False
        return self.model(s, add_special_tokens=add_special_tokens)


class Tokenizer:
    """Tokenize prompts or de-tokenize tokens into texts.

    Args:
        model_file (str): the path of the tokenizer model
    """

    def __init__(self, model_file: str):
        if model_file.endswith('.model'):
            model_folder = osp.split(model_file)[0]
        else:
            model_folder = model_file
            model_file = osp.join(model_folder, 'tokenizer.model')
        tokenizer_config_file = osp.join(model_folder, 'tokenizer_config.json')

        model_file_exists = osp.exists(model_file)
        config_exists = osp.exists(tokenizer_config_file)
        use_hf_model = config_exists or not model_file_exists

        if not use_hf_model:
            self.model = SentencePieceTokenizer(model_file)
        else:
            self.model = HuggingFaceTokenizer(model_folder)

    @property
    def vocab_size(self):
        """vocabulary size."""
        return self.model.vocab_size

    @property
    def bos_token_id(self):
        """begine of the sentence token id."""
        return self.model.bos_token_id

    @property
    def eos_token_id(self):
        """end of the sentence token id."""
        return self.model.eos_token_id

    def encode(self, s: str):
        """Tokenize a prompt.

        Args:
            s (str): a prompt
        Returns:
            list[int]: token ids
        """
        return self.model.encode(s)

    def decode(self, t: Sequence[int], offset: Optional[int] = None):
        """De-tokenize.

        Args:
            t (List[int]): a list of token ids
            offset (int): for incrementally decoding. Default to None, which
                means not applied.
        Returns:
            str: text of decoding tokens
        """
        return self.model.decode(t, offset)

    def __call__(self, s: Union[str, Sequence[str]]):
        """Tokenize prompts.

        Args:
            s (str): prompts
        Returns:
            list[int]: token ids
        """
        return self.model(s)