import logging
from typing import Dict, List
from src.helps import ConfigurationError
from pathlib import Path
from src.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

logger = logging.getLogger(__name__)

class BasicTokenizer(object):
    SPACE = chr(32)           # ' ': half-width white space (ASCII)
    SPACE_ESCAPE = chr(9601)  # '▁': sentencepiece default
    SPECIALS = [BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]
    def __init__(self, level:str="word", lowercase:bool=False, normalize: bool=False, max_length: int=-1, 
                 min_length:int=1, filter_or_truncate: str="truncate") -> None:
        self.level = level
        self.lowercase = lowercase
        self.normalize = normalize
        self.max_length = max_length
        self.min_length = min_length
        self.filter_or_truncate = filter_or_truncate
        assert self.filter_or_truncate in ["filter", "truncate"], "Invalid filter_or_truncate!"

    def pre_process(self, sentence:str) -> str:
        """
        Pre-process setence. 
        Lowercase, Normalize.
        """
        if self.normalize:
            sentence = sentence.strip()
        if self.lowercase:
            sentence = sentence.lower()
        return sentence
    
    def post_process(self, sentence: List[str], generate_unk: bool=True) -> str:
        """
        Post-process sentence tokens.
        result a sentence (a string).
        """
        sentence = self.remove_special(sentence, generate_unk=generate_unk)
        sentence = self.SPACE.join(sentence)
        return sentence

    def remove_special(self, sentence: List[str], generate_unk: bool=True) -> List[str]:
        specials = self.SPECIALS[:-1] if generate_unk else self.SPECIALS
        return [token for token in sentence if token not in specials]

    def filter_or_truncate_by_length(self, sentence_token:List[str]) -> List[str]:
        if self.filter_or_truncate == "filter":
            if len(sentence_token) < self.min_length or len(sentence_token) > self.max_length:
                logger.warning("{} is filtered".format(sentence_token))
                sentence_token = None
            else:
                sentence_token = sentence_token
        elif self.filter_or_truncate == "truncate":
            sentence_token = sentence_token[:self.max_length]
        else:
            raise NotImplementedError("Invalid filter_or_truncate.")

        return sentence_token

    def __call__(self, sentence: str) -> List[str]:
        """
        Tokenize single sentence.
        """
        # sentence_token = sentence.split(self.SPACE) can't handle continuous space
        sentence_token = sentence.split()
        sentence_token = self.filter_or_truncate_by_length(sentence_token)

        return sentence_token

    def __repr__(self):
        return (f"{self.__class__.__name__}(level={self.level}, "
                f"lowercase={self.lowercase}, normalize={self.normalize}, "
                f"filter_by_length=({self.min_length}, {self.max_length}))")


class UnigramTokenizer(BasicTokenizer):
    def __init__(self, level:str="word", lowercase:bool=False, normalize: bool=False, max_length: int=-1,
                 min_length:int=1, filter_or_truncate: str="truncate") -> None:
        super().__init__(level, lowercase, normalize, max_length, min_length, filter_or_truncate)
        assert self.level == "unigram"
        files = "data/script_java"
        self.trained_tokenizer = self.train_tokenizer(files)

    def prepare_tokenizer_trainer(self):
        """
        Prepares the tokenizer and trainer with unknown & special tokens.
        """
        SPECIALS = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, ]
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=UNK_TOKEN, special_tokens=SPECIALS)
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer, trainer

    def train_tokenizer(self, files):
        """
        Takes the files and trains the tokenizer.
        """
        tokenizer, trainer = self.prepare_tokenizer_trainer()
        if os.path.exists("./tokenizer-trained.json"):
            tokenizer = Tokenizer.from_file("./tokenizer-trained.json")
        else:
            files = [os.path.join(files, file) for file in os.listdir(files)]
            tokenizer.train(files, trainer)  # training the tokenzier
            tokenizer.save("./tokenizer-trained.json")
        return tokenizer
    def decode(self, output: List[List[int]], generate_unk: bool=True) -> List[str]:
        """
        Post-process sentence tokens.
        result a sentence (a string).
        """
        return self.trained_tokenizer.decode_batch(output)
    def get_vocab(self):
        return self.trained_tokenizer.get_vocab()

    def __call__(self, sentence: str) -> List[str]:
        sentence_token = self.trained_tokenizer.encode(sentence).tokens
        sentence_token = self.filter_or_truncate_by_length(sentence_token)
        return sentence_token

import sentencepiece as sp
class SentencePieceTokenizer(BasicTokenizer):
    # TODO
    def __init__(self, level: str = "bpe", lowercase: bool = False, normalize: bool = False, 
                 max_length: int = -1, min_length: int = 1) -> None:
        super().__init__(level, lowercase, normalize, max_length, min_length)
        assert self.level == "bpe"
        self.model_file = Path(None)
        assert self.model_file.is_file(), "spm model file not found."
        self.spm = sp.SentencePieceProcessor()
        self.spm.load(self.model_file)
    

class SubwordNMTTokenizer(BasicTokenizer):
    # TODO
    def __init__(self, level: str = "word", lowercase: bool = False, normalize: bool = False, 
                 max_length: int = -1, min_length: int = 1) -> None:
        super().__init__(level, lowercase, normalize, max_length, min_length)


def build_tokenizer(data_cfg: Dict) -> Dict[str,BasicTokenizer]:
    """
    Build tokenizer: a dict.
    """
    src_language = data_cfg["src"]["language"]
    trg_language = data_cfg["trg"]["language"]
    tokenizer = {
        src_language: build_language_tokenizer(data_cfg["src"]),
        trg_language: build_language_tokenizer(data_cfg["trg"]),
    }
    logger.info("%s tokenizer: %s", src_language, tokenizer[src_language])
    logger.info("%s tokenizer: %s", trg_language, tokenizer[trg_language])
    return tokenizer

def build_language_tokenizer(cfg: Dict):
    """
    Build language tokenizer.
    """
    tokenizer = None 
    if cfg["level"] == "word":
        tokenizer = BasicTokenizer(level=cfg["level"],lowercase=cfg["lowercase"],
                                   normalize=cfg["normalize"], max_length=cfg["max_length"],
                                   min_length=cfg["min_length"], filter_or_truncate=cfg["filter_or_truncate"])
    elif cfg["level"] == "bpe":
        tokenizer_type = cfg.get("tokenizer_type", "sentencepiece")
        if tokenizer_type == "sentencepiece":
            tokenizer = SentencePieceTokenizer()
        elif tokenizer_type == "subword-nmt":
            tokenizer  = SubwordNMTTokenizer()
        else:
            raise ConfigurationError("Unkonwn tokenizer type.")
    elif cfg["level"] == "unigram":
        tokenizer = UnigramTokenizer(level=cfg["level"],lowercase=cfg["lowercase"],
                                   normalize=cfg["normalize"], max_length=cfg["max_length"],
                                   min_length=cfg["min_length"], filter_or_truncate=cfg["filter_or_truncate"])
    else:
        raise ConfigurationError("Unknown tokenizer level.")
    
    return tokenizer
