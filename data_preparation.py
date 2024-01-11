import re

import numpy as np
import torch
from datasets import load_from_disk

from Encodec import AudioTokenizer


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
        tokens = audio_tokenizer.get_tokens_from_file(file_audio_path)
        tokens_file_path = re.sub(r'\..+$', '.pt', file_audio_path)
        torch.save(tokens, tokens_file_path)
        # load it with torch.load('file.pt')


def load_dataset():
    """
    Loads the full dataset as a list of objects that associate a nome to a caption and a sequence of tokens for
    the associated audio.
    """
    audio_files_path = '/Volumes/SALVATORE R/Università/NN/dataset/music_data'
    dataset = load_from_disk('/Volumes/SALVATORE R/Università/NN/dataset/musiccaps.hf')

    new_ds = np.empty(len(dataset), dtype=dict)
    for i in range(len(dataset)):
        row = dataset[i]
        caption = row['caption']
        audio_name = row['ytid']

        tokens_path = f'{audio_files_path}/{audio_name}.pt'
        tokens = torch.load(tokens_path)

        new_ds[i] = {audio_name: {'caption': caption, 'tokens': tokens}}

    return new_ds

