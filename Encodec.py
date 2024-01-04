from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
import soundfile as sf
import librosa


def get_tokens_from_file(file_path):
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    # Load the audio:
    audio_sample, sample_rate = sf.read(file_path)

    # Convert from multi-channel to mono-channel with the mean:
    if len(audio_sample.shape) > 1:
        # Se l'audio è stereo o multicanale, calcola la media dei canali per ottenere un segnale mono
        audio_sample = audio_sample.mean(axis=1)

    # Resampling:
    if sample_rate != processor.sampling_rate:
        audio_sample = librosa.resample(audio_sample, orig_sr=sample_rate, target_sr=processor.sampling_rate)

    inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

    encoder_outputs = model.encode(input_values=inputs["input_values"], padding_mask=inputs.get("attention_mask", None),
                                   bandwidth=3.0)

    # print("Shape of encoder outputs:", encoder_outputs.audio_codes.shape)
    # print("Total number of representation:", encoder_outputs.audio_codes.shape[2])
    # print("Total number of tokens:", encoder_outputs.audio_codes.shape[3])
    # print(
    #     "x Salvatore: dovrebbero essere quindi 1125 token complessi (nel senso che, avendo due rappresentazioni, "
    #     "un singolo token è la coppia formata dal valore della prima e della seconda rapp.) e 2*1125 token totali")
    # print(
    #     "quindi il numero dei token quanto è? 1125? e la dimensione D che hai scritto su whatsapp è 2? o il "
    #     "contrario? Fammi sapere")

    # Take the tokens with the attribute audio_codes
    tokens = encoder_outputs.audio_codes
    return tokens[0][0]


def save_tokens_to_audio_file(tokens, output_file_path):
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

    audio_values = model.decode(tokens, None, None)[0]
    # audio_values = \
    #     model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs.get("attention_mask", None))[0]
    # or the equivalent with a forward pass
    # audio_values = model(inputs["input_values"], inputs.get("attention_mask", None)).audio_values

    # Convert the tensor to obtain a correct file audio wav
    reconstructed_audio = audio_values.detach().numpy().flatten()
    reconstructed_audio = reconstructed_audio * (2 ** 15)  # Scale the audio

    # Save the audio
    sf.write(output_file_path, reconstructed_audio, processor.sampling_rate)

    print("Finished!!")
