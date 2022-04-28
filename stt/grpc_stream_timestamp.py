#!/usr/bin/env python3
"""
Dependencies:
    - python 3.7

The librosa requires libsndfile.
    osx) brew install libsndfile
    ubuntu) apt install libsndfile1

Usage:
    $ ./grpc_stream_timestamp.py --api_key <AIQ api key>
"""

import datetime

from absl import app
from absl import flags
from google.cloud import speech
from google.cloud.speech_v1.services.speech import transports
import librosa
import numpy as np

import grpc_utils

flags.DEFINE_string('api_url', 'aiq.skelterlabs.com:443', 'AIQ portal address.')
flags.DEFINE_string('api_key', None, 'AIQ project api key.')
flags.DEFINE_boolean('insecure', None, 'Use plaintext and insecure connection.')
flags.DEFINE_string('audio_path', './resources/hello.wav', 'Input wav path.')
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


def time_to_second(time_info):
    """Convert time_info to seconds."""
    if isinstance(time_info, datetime.timedelta):
        return time_info.total_seconds()
    return time_info.seconds + time_info.nanos / 1e9


def main(args):
    del args  # Unused

    channel = grpc_utils.create_channel(
        FLAGS.api_url, api_key=FLAGS.api_key, insecure=FLAGS.insecure)
    transport = transports.SpeechGrpcTransport(channel=channel)
    client = speech.SpeechClient(transport=transport)

    requests = generate_requests(FLAGS.audio_path)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='ko-KR',
        sample_rate_hertz=16000,
        # Below option is required for timestamp
        enable_word_time_offsets=True)

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False,
    )

    # streaming_recognize() returns a generator of responses.
    responses = client.streaming_recognize(streaming_config, requests)

    for response in responses:
        # Once the transcription has settled, the first result will contain the
        # is_final result. The other results will be for subsequent portions of
        # the audio.
        for result in response.results:
            # The alternatives are ordered from most likely to least.
            for i, alternative in enumerate(result.alternatives):
                print(f'Alternatives[{i}]')
                print(f'  Confidence[{i}]: {alternative.confidence}')
                print(f'  Transcript[{i}]: {alternative.transcript}')
                # Print timestamp
                print(f'  Timestamps[{i}]:')
                for word in alternative.words:
                    start_time = time_to_second(word.start_time)
                    end_time = time_to_second(word.end_time)
                    print(
                        f'  - [{start_time:.2f} ~ {end_time:.2f}] {word.word}')


if __name__ == '__main__':
    app.run(main)
