import torch

from Encodec import AudioTokenizer
from data_preparation import load_dataset
from transformer import TransformerWithText


class AudioGenerator:
    def __init__(self, model_path: str):
        self.model = TransformerWithText(num_layers=5,
                                         q_val=4,
                                         v_val=4,
                                         h_val=3,
                                         dropout=0.1,
                                         embed_size=4,  # 4,
                                         trg_vocab_size=4,
                                         src_pad_idx=0, )
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # set layers to evaluation mode

        self.audio_tokenizer = AudioTokenizer()

    def generate_audio(self, text, num_tokens: int = 1000, file_path=None):
        """
        Returns the audio returned by the model when it is fed with random noise and the given text.
        It returns the generated tokens and saves them as audio file based on whether file_path is None.
        :param text: The description of the audio to generate
        :param num_tokens: The number of tokens to generate.
        :param file_path: The .wav file where to save the generated audio.
        :return: the generated tokens and saves them as audio file based on whether file_path is None.
        """
        rnd_noise1 = torch.rand((1, 4))
        rnd_noise2 = torch.rand((1, 4))
        mask = self.model.make_mask(1)

        output = torch.empty((num_tokens, 4))
        for i in range(num_tokens):
            last_output = self.model.decoder(
                x=output[i - 1:i] if i >= 1 else rnd_noise1,  # last generated token or random noise
                encoder_output=output[i - 1:i] if i >= 1 else rnd_noise2,
                src_mask=mask,
                trg_mask=mask,
                text_tokens=text,
            )
            output[i] = last_output

        if file_path is not None:
            self.audio_tokenizer.save_tokens_to_audio_file(output.t(), file_path)

        return output


if __name__ == '__main__':
    dataset = load_dataset()
    sample = dataset[-1]
    text_tokens = sample['text']

    generator = AudioGenerator(model_path='/Volumes/SALVATORE R/Università/NN/models/1705162934753')
    generator.generate_audio(text_tokens, file_path='/Volumes/SALVATORE R/Università/NN/outputs/test.wav')
