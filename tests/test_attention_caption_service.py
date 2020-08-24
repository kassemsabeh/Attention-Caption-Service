from unittest.mock import patch

from event_service_utils.tests.base_test_case import MockedServiceStreamTestCase
from event_service_utils.tests.json_msg_helper import prepare_event_msg_tuple

from attention_caption_service.service import AttentionCaptionService

from attention_caption_service.conf import (
    SERVICE_STREAM_KEY,
    SERVICE_CMD_KEY,
)

class TestAttentionCaptionService(MockedServiceStreamTestCase):

    def instantiate_service(self):
        with patch('attention_caption_service.service.AttentionCaptionService.setup_model') as mocked_setup_model:
            self.service = super(TestAttentionCaptionService, self).instantiate_service()
            return self.service
        # service_kwargs = self.service_config.copy()
        # # quickfix by piyush for testcase of window-manager
        # if 'no_stream_factory_flag' not in service_kwargs:
        #     service_kwargs.update({'stream_factory': self.stream_factory})
        # else:
        #     del service_kwargs['no_stream_factory_flag']
        # with patch('event_service_utils.tracing.jaeger.init_tracer') as mockedTracer:
        #     self.service = self.service_cls(**service_kwargs)
        #     if hasattr(self.service, 'tracer') and self.service.tracer:
        #         self.service.tracer.close()
        #     self.service.tracer = mockedTracer
        # return self.service

    GLOBAL_SERVICE_CONFIG = {
        'service_stream_key': SERVICE_STREAM_KEY,
        'service_cmd_key': SERVICE_CMD_KEY,
        #'dnn_configs': {'model_name': 'etc'},
        'file_storage_cli': None,
        'logging_level': 'ERROR',
        'tracer_configs': {'reporting_host': None, 'reporting_port': None},
    }
    SERVICE_CLS = AttentionCaptionService
    MOCKED_STREAMS_DICT = {
        SERVICE_STREAM_KEY: [],
        SERVICE_CMD_KEY: [],
    }

    # @patch('')
    @patch('attention_caption_service.service.AttentionCaptionService.process_action')
    def test_process_cmd_should_call_process_action(self, mocked_process_action):
        action = 'someAction'
        event_data = {
            'id': 1,
            'action': action,
            'some': 'stuff'
        }
        msg_tuple = prepare_event_msg_tuple(event_data)
        mocked_process_action.__name__ = 'process_action'

        self.service.service_cmd.mocked_values = [msg_tuple]
        self.service.process_cmd()
        self.assertTrue(mocked_process_action.called)
        self.service.process_action.assert_called_once_with(action=action, event_data=event_data, json_msg=msg_tuple[1])
