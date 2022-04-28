# AIQ.TALK STT Python Example

The AIQ.TALK STT API is mostly compatible with the Google Cloud Speech API,
so you can use
[Google Cloud Speech Python Client](https://github.com/googleapis/python-speech)
to use AIQ.TALK STT API.

This repository contains simple example CLI programs that recognizes the given
`resources/.wav` audio file.

## Before you begin

Our examples are based on python 3.7 runtime.

Before running the examples, make sure you've followed the steps.

```shell
$ pip install -U -r ./requirements.txt
```

Get your AIQ API key from the
[AIQ Console](https://aiq.skelterlabs.com/console).

## Samples

NOTE. We support mono audio only now.

### Synchronously transcribe a local file

Perform synchronous transcription on a local audio file.
Synchronous request supports ~1 minute audio length.

```shell
$ ./grpc_sync.py --api-key=<your API key>
```

### Streaming speech recognition

Perform streaming request on a local audio file.

```shell
$ ./grpc_stream.py --api-key=<your API key>
```
