PROTOC = python -m grpc_tools.protoc

all:
	$(PROTOC) --python_out=../ --grpc_python_out=../ -I. ./google/speech/v1/cloud_speech.proto
