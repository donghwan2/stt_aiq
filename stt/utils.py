"""Utilities for our STT APIs."""

import datetime
import sys


def time_to_second(time_info):
    """Convert time_info to seconds."""
    if isinstance(time_info, datetime.timedelta):
        return time_info.total_seconds()
    return time_info.seconds + time_info.nanos / 1e9


def print_recognition_result(result, file=sys.stdout):
    
    """Print google/saojung recognition result."""
    if not result.alternatives:
        return

    alternative = result.alternatives[0]
    if not alternative.words:
        print(alternative.transcript, file=file, flush=True)
    else:
        start_time = time_to_second(alternative.words[0].start_time)
        if hasattr(result, 'result_end_time'):
            end_time = time_to_second(result.result_end_time)
        else:
            end_time = time_to_second(alternative.words[-1].end_time)
        print(
            f'[{start_time:.2f} ~ {end_time:.2f}] '
            f'{alternative.transcript}',
            file=file,
            flush=True)
        texts_temp = []
        for word in alternative.words:
            start_time = time_to_second(word.start_time)
            end_time = time_to_second(word.end_time)
            texts_temp.append(word.word)
            print(
                f'- [{start_time:.2f} ~ {end_time:.2f}] {word.word}',
                file=file,
                flush=True)
        return texts_temp
