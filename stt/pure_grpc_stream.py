#!/usr/bin/env python3
"""
Dependencies:
    python 3.7

The librosa requires libsndfile.
    macOS) brew install libsndfile
    ubuntu) apt install libsndfile1

Usage:
    $ ./pure_grpc_stream.py --api_key <AIQ api key>
"""
from typing import Generator

from absl import app
from absl import flags
import librosa
import numpy as np

from google.speech.v1 import cloud_speech_pb2
from google.speech.v1 import cloud_speech_pb2_grpc
import grpc_utils
import utils

flags.DEFINE_string('api_url', 'aiq.skelterlabs.com:443', 'AIQ portal address.')
flags.DEFINE_string('api_key', None, 'AIQ project api key.')
flags.DEFINE_boolean('insecure', None, 'Use plaintext and insecure connection.')
flags.DEFINE_string('audio_path', './resources/hello.wav', 'Input wav path.')
flags.DEFINE_boolean(
    'interim_results', False,
    'Stream request should return temporary results that '
    'may be refined at a later time.')
flags.DEFINE_list('speech_context_phrases', None, 'Phrases for speech context')
FLAGS = flags.FLAGS


def generate_requests(
    audio_path: str,
    config: cloud_speech_pb2.StreamingRecognitionConfig,
    chunk_size: int = 1024
) -> Generator[cloud_speech_pb2.StreamingRecognizeRequest, None, None]:
    """Generate chunks of 16kHz audio encoded as LINEAR16.

    Args:
        audio_path: Audio file path.
        config: StreamingRecognitionConfig object.
        chunk_size: Size of each chunk in bytes.

    Yields:
        StreamingRecognizeRequest objects.
    """
    assert config is not None, 'StreamingRecognitionConfig should be given'
    # The first request should hold config only.
    yield cloud_speech_pb2.StreamingRecognizeRequest(streaming_config=config)

    content, sample_rate = librosa.load(audio_path, sr=16000)
    del sample_rate
    if content.dtype in (np.float32, np.float64):
        content = (content * np.iinfo(np.int16).max).astype(np.int16)
    content = content.tobytes()

    for from_idx in range(0, len(content), chunk_size):
        yield cloud_speech_pb2.StreamingRecognizeRequest(
            audio_content=content[from_idx:from_idx + chunk_size])


def main(args):
    del args  # Unused

    channel = grpc_utils.create_channel(
        FLAGS.api_url, api_key=FLAGS.api_key, insecure=FLAGS.insecure)
    stub = cloud_speech_pb2_grpc.SpeechStub(channel)

    if FLAGS.speech_context_phrases:
        speech_contexts = [
            cloud_speech_pb2.SpeechContext(phrases=FLAGS.speech_context_phrases)
        ]
    else:
        speech_contexts = None
    config = cloud_speech_pb2.RecognitionConfig(
        enable_word_time_offsets=True,
        encoding=cloud_speech_pb2.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='ko-KR',
        sample_rate_hertz=16000,
        speech_contexts=speech_contexts,
    )
    streaming_config = cloud_speech_pb2.StreamingRecognitionConfig(
        config=config,
        interim_results=FLAGS.interim_results,
    )

    request_generator = generate_requests(FLAGS.audio_path, streaming_config)

    # StreamingRecognize() returns a generator of responses.
    response_generator = stub.StreamingRecognize(request_generator)

    for response in response_generator:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            print(f'Finished: {result.is_final}')
            utils.print_recognition_result(result)


if __name__ == '__main__':
    app.run(main)
