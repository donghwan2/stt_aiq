#!/usr/bin/env python3
"""
Dependencies:
    python 3.7

The librosa requires libsndfile.
    osx) brew install libsndfile
    ubuntu) apt install libsndfile1

Usage:
    $ ./grpc_stream.py --api_key <AIQ api key>
"""

from absl import app
from absl import flags
from google.cloud import speech
from google.cloud.speech_v1.services.speech import transports
import librosa
import numpy as np

import grpc_utils
import utils

flags.DEFINE_string('api_url', 'aiq.skelterlabs.com:443', 'https://aiq.skelterlabs.com/')
flags.DEFINE_string('api_key', None, 'CqCqJg_dEfU7baEOZYFeMxYOap_CkllZ.ee1cba04')
flags.DEFINE_boolean('insecure', None, 'Use plaintext and insecure connection.')
flags.DEFINE_string('audio_path', './idaji_sadoseja_backstory.wav', '/Users/aiden/Dropbox/Mac/Desktop/python-speech-master/stt/idaji_sadoseja_backstory.wav')
flags.DEFINE_boolean(
    'interim_results', False,
    'Stream request should return temporary results that '
    'may be refined at a later time.')
flags.DEFINE_list('speech_context_phrases', None, 'Phrases for speech context')
FLAGS = flags.FLAGS


def generate_requests(audio_path, chunk_size=1024):
    """Generate chunks of 16kHz audio encoded as LINEAR16.

    Args:
        audio_path: Audio file path.
        chunk_size: Size of each chunk in bytes.

    Yields:
        StreamingRecognizeRequest objects.
    """
    content, sample_rate = librosa.load(audio_path, sr=16000)
    del sample_rate
    if content.dtype in (np.float32, np.float64):
        content = (content * np.iinfo(np.int16).max).astype(np.int16)
    content = content.tobytes()

    for from_idx in range(0, len(content), chunk_size):
        yield speech.StreamingRecognizeRequest(
            audio_content=content[from_idx:from_idx + chunk_size])


def main(args):
    del args  # Unused

    channel = grpc_utils.create_channel(
        FLAGS.api_url, api_key=FLAGS.api_key, insecure=FLAGS.insecure)
    transport = transports.SpeechGrpcTransport(channel=channel)
    client = speech.SpeechClient(transport=transport)

    requests = generate_requests(FLAGS.audio_path)

    if FLAGS.speech_context_phrases:
        speech_contexts = [
            speech.SpeechContext(phrases=FLAGS.speech_context_phrases)
        ]
    else:
        speech_contexts = None
    config = speech.RecognitionConfig(
        enable_word_time_offsets=True,
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='ko-KR',
        sample_rate_hertz=16000,
        speech_contexts=speech_contexts,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=FLAGS.interim_results,
    )

    # streaming_recognize() returns a generator of responses.
    responses = client.streaming_recognize(streaming_config, requests)

    total_texts = []
    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        texts = []
        for result in response.results:
            print(f'Finished: {result.is_final}')
            texts_temp = utils.print_recognition_result(result)
            texts.extend(texts_temp)
        print(texts)
        total_texts.extend(texts)
    print(total_texts)

    # 리스트를 텍스트 파일로 저장하기
    with open('이다지.txt','w',encoding='UTF-8') as f:
        for name in total_texts:
            f.write(name+'\n')


if __name__ == '__main__':
    app.run(main)
