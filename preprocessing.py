from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import xml.etree.ElementTree as ET
import re

@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    sentence_pairs = []
    alignments = []

    with open(filename) as file:
        with open('my_file.txt', 'w') as f:
            for line in file:
                s = re.sub('\&', '&amp;', line)
                f.write(s)

    tree = ET.parse('my_file.txt')
    root = tree.getroot()
    s1, s2 = [], []
    p1, p2 = [], []
    for i in range(len(root)):
        for j in range(len(root[i])):
            if root[i][j].tag == 'english':
                s1 = root[i][j].text.split()

            if root[i][j].tag == 'czech':
                s2 = root[i][j].text.split()

            if root[i][j].tag == 'sure':
                s = root[i][j].text
                if s is None:
                    s = ''
                s = s.split()
                p1 = [(int(i.split('-')[0]), int(i.split('-')[1])) for i in s]

            if root[i][j].tag == 'possible':
                s = root[i][j].text
                if s is None:
                    s = ''
                s = s.split()
                p2 = [(int(i.split('-')[0]), int(i.split('-')[1])) for i in s]

        obj2 = LabeledAlignment(p1, p2)
        obj1 = SentencePair(s1, s2)

        sentence_pairs.append(obj1)
        alignments.append(obj2)
    return sentence_pairs, alignments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff -- natural number -- most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language
        
    Tip: 
        Use cutting by freq_cutoff independently in src and target. Moreover in both cases of freq_cutoff (None or not None) - you may get a different size of the dictionary

    """
    tokens_source = []
    count_source = []
    tokens_target = []
    count_target = []
    for obj in sentence_pairs:
        for tok in obj.source:
            if tok in tokens_source:
                ind = tokens_source.index(tok)
                count_source[ind] += 1
            else:
                tokens_source.append(tok)
                count_source.append(0)

        for tok in obj.target:
            if tok in tokens_target:
                ind = tokens_target.index(tok)
                count_target[ind] += 1
            else:
                tokens_target.append(tok)
                count_target.append(0)

    tokens_source = np.array(tokens_source)
    count_source = np.array(count_source)
    tokens_target = np.array(tokens_target)
    count_target = np.array(count_target)

    ind_source = np.argsort(count_source)[::-1]
    ind_target = np.argsort(count_target)[::-1]
    tokens_source = tokens_source[ind_source]
    tokens_target = tokens_target[ind_target]
    if freq_cutoff is not None:
        tokens_source = tokens_source[:freq_cutoff]
        tokens_target = tokens_target[:freq_cutoff]

    return dict(zip(tokens_source, np.arange(len(tokens_source)))), \
           dict(zip(tokens_target, np.arange(len(tokens_target))))


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    answ = []
    for obj in sentence_pairs:
        source_sentence = []
        target_sentence = []
        not_tok = False
        for tok in obj.source:
            if tok in source_dict.keys():
                source_sentence.append(source_dict[tok])
            else:
                not_tok = True
                break
        if not_tok:
            continue

        for tok in obj.target:
            if tok in target_dict.keys():
                target_sentence.append(target_dict[tok])
            else:
                not_tok = True
                break
        if not_tok:
            continue

        answ.append(TokenizedSentencePair(np.array(source_sentence), np.array(target_sentence)))
    return answ
