#!/usr/bin/env python
from event_service_utils.streams.redis import RedisStreamFactory
from event_service_utils.img_serialization.redis import RedisImageCache


from service import AttentionCaptionService

from attention_caption_service.conf import (
    REDIS_ADDRESS,
    REDIS_PORT,
    SERVICE_STREAM_KEY,
    SERVICE_CMD_KEY,
    LOGGING_LEVEL,
    TRACER_REPORTING_HOST,
    TRACER_REPORTING_PORT,
)


def run_service():
    tracer_configs = {
        'reporting_host': TRACER_REPORTING_HOST,
        'reporting_port': TRACER_REPORTING_PORT,
    }

    redis_fs_cli_config = {
        'host': REDIS_ADDRESS,
        'port': REDIS_PORT,
        'db': 0,
    }

    file_storage_cli = RedisImageCache()
    file_storage_cli.file_storage_cli_config = redis_fs_cli_config
    file_storage_cli.initialize_file_storage_client()


    stream_factory = RedisStreamFactory(host=REDIS_ADDRESS, port=REDIS_PORT)

    service = AttentionCaptionService(
        service_stream_key=SERVICE_STREAM_KEY,
        service_cmd_key=SERVICE_CMD_KEY,
        file_storage_cli=file_storage_cli,
        #dnn_configs=dnn_configs,
        stream_factory=stream_factory,
        logging_level=LOGGING_LEVEL,
        tracer_configs=tracer_configs
    )
    service.run()


def main():
    try:
        run_service()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
