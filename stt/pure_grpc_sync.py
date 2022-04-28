#!/usr/bin/env python3
"""
Dependencies:
    - python 3.6
    - google-cloud-speech==2.0.1
    - librosa==0.7.1
    - numpy==1.17.0

The librosa requires libsndfile.
    osx) brew install libsndfile
    ubuntu) apt install libsndfile1

Before executing this script, you should compile protobuf files:
    $ cd proto
    $ make

Usage:
    $ export AIQ_API_KEY="YOUR API KEY HERE"
    $ ./pure_grpc_sync.py <AUDIO file path>

NOTE:
    - Input audio duration is less than or equal to 60 seconds.
"""

import os
import sys

import librosa
import numpy as np

from google.speech.v1 import cloud_speech_pb2
from google.speech.v1 import cloud_speech_pb2_grpc
import grpc_utils

API_URL = os.environ.get('AIQ_API_URL', 'aiq.skelterlabs.com:443')
API_KEY = os.environ.get('AIQ_API_KEY', '')  # Enter your API key here
INSECURE = os.environ.get('INSECURE')


def make_audio(audio_path):
    """Create recognition audio of 16kHz audio encoded as LINEAR16.

    Args:
        audio_path: Audio file path.

    Returns:
        RecognitionAudio object.
    """
    content, sample_rate = librosa.load(audio_path, sr=16000)
    del sample_rate
    if content.dtype in (np.float32, np.float64):
        content = (content * np.iinfo(np.int16).max).astype(np.int16)
    return cloud_speech_pb2.RecognitionAudio(content=content.tobytes())


def main(args):
    if len(args) != 2:
        print(f'Usage: {args[0]} <Audio file path>', file=sys.stderr)
        sys.exit(1)

    channel = grpc_utils.create_channel(
        API_URL, api_key=API_KEY, insecure=INSECURE)
    stub = cloud_speech_pb2_grpc.SpeechStub(channel)

    audio_path = args[1]
    audio = make_audio(audio_path)

    # pylint: disable=no-member
    config = cloud_speech_pb2.RecognitionConfig(
        encoding=cloud_speech_pb2.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='ko-KR')
    # pylint: enable=no-member
    request = cloud_speech_pb2.RecognizeRequest(config=config, audio=audio)
    response = stub.Recognize(request)

    for result in response.results:
        # The alternatives are ordered from most likely to least.
        for i, alternative in enumerate(result.alternatives):
            print(f'Alternatives[{i}]')
            print(f'  Confidence[{i}]: {alternative.confidence}')
            print(f'  Transcript[{i}]: {alternative.transcript}')


if __name__ == '__main__':
    main(sys.argv)
