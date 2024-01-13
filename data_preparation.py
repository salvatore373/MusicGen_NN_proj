import re

import numpy as np
import torch
from datasets import load_from_disk

from Encodec import AudioTokenizer
from T5_encoder import TextToTokenConverter


def convert_dataset_wavs_to_tokens():
    """
    Converts all the songs of the datasets to tokens and saves the tokens sequence
    as a file in the same directory.
    """
    audio_files_path = '/Volumes/SALVATORE R/Università/NN/dataset/music_data'
    dataset = load_from_disk('/Volumes/SALVATORE R/Università/NN/dataset/musiccaps.hf')
    audio_tokenizer = AudioTokenizer()

    for i in range(len(dataset)):
        row = dataset[i]
        caption = row['caption']
        audio_data = row['audio']
        audio_name = audio_data['path']

        file_audio_path = f'{audio_files_path}/{audio_name}'
        tokens = audio_tokenizer.get_tokens_from_file(file_audio_path).t().to(torch.float32)  # n x d
        # Cut all the sequences at the same length for ease of training
        tokens = tokens[:751, :]
        tokens_file_path = re.sub(r'\..+$', '.pt', file_audio_path)
        torch.save(tokens, tokens_file_path)
        # load it with torch.load('file.pt')


def convert_dataset_captions_to_tokens():
    """
    Converts all the songs of the datasets to tokens and saves the tokens sequence
    as a file in the same directory.
    """
    audio_files_path = '/Volumes/SALVATORE R/Università/NN/dataset/music_data'
    dataset = load_from_disk('/Volumes/SALVATORE R/Università/NN/dataset/musiccaps.hf')
    text_tokenizer = TextToTokenConverter()

    for i in range(len(dataset)):
        row = dataset[i]
        caption = row['caption']
        text_tokens = text_tokenizer.convert_text_to_tokens(caption)
        audio_name = row['audio']['path']

        file_audio_path = f'{audio_files_path}/{audio_name}'
        tokens_file_path = re.sub(r'\..+$', '-caption.pt', file_audio_path)
        torch.save(text_tokens, tokens_file_path)
        # load it with torch.load('file.pt')


def load_dataset():
    """
    Loads the full dataset as a list of objects that associate a nome to a caption and a sequence of tokens for
    the associated audio.
    """
    audio_files_path = '/Volumes/SALVATORE R/Università/NN/dataset/music_data'
    dataset = load_from_disk('/Volumes/SALVATORE R/Università/NN/dataset/musiccaps.hf')

    import sys
    is_debug_mode = hasattr(sys, 'gettrace') and sys.gettrace() is not None

    new_ds = np.empty(len(dataset) if not is_debug_mode else 3, dtype=dict)
    for i in range(len(dataset) if not is_debug_mode else 3):
        row = dataset[i]
        audio_name = row['ytid']

        text_tokens_path = f'{audio_files_path}/{audio_name}-caption.pt'
        text_tokens = torch.load(text_tokens_path)
        audio_tokens_path = f'{audio_files_path}/{audio_name}.pt'
        audio_tokens = torch.load(audio_tokens_path)

        new_ds[i] = {'name': audio_name,
                     'text': text_tokens,
                     'audio': audio_tokens}

    return new_ds
