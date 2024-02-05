from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class EncoderRequest(_message.Message):
    __slots__ = ["context", "is_prefix"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    IS_PREFIX_FIELD_NUMBER: _ClassVar[int]
    context: str
    is_prefix: bool
    def __init__(self, context: _Optional[str] = ..., is_prefix: bool = ...) -> None: ...

class EncoderResponse(_message.Message):
    __slots__ = ["outputs"]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, outputs: _Optional[_Iterable[int]] = ...) -> None: ...

class TokenInfo(_message.Message):
    __slots__ = ["value", "context", "type", "id_end"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ID_END_FIELD_NUMBER: _ClassVar[int]
    value: int
    context: bytes
    type: str
    id_end: bool
    def __init__(self, value: _Optional[int] = ..., context: _Optional[bytes] = ..., type: _Optional[str] = ..., id_end: bool = ...) -> None: ...

class DecoderResponse(_message.Message):
    __slots__ = ["token_info", "prompts"]
    class PromptsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Prompts
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Prompts, _Mapping]] = ...) -> None: ...
    TOKEN_INFO_FIELD_NUMBER: _ClassVar[int]
    PROMPTS_FIELD_NUMBER: _ClassVar[int]
    token_info: _containers.RepeatedCompositeFieldContainer[TokenInfo]
    prompts: _containers.MessageMap[str, Prompts]
    def __init__(self, token_info: _Optional[_Iterable[_Union[TokenInfo, _Mapping]]] = ..., prompts: _Optional[_Mapping[str, Prompts]] = ...) -> None: ...

class Prompts(_message.Message):
    __slots__ = ["prompt"]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    prompt: _containers.RepeatedCompositeFieldContainer[Prompt]
    def __init__(self, prompt: _Optional[_Iterable[_Union[Prompt, _Mapping]]] = ...) -> None: ...

class Prompt(_message.Message):
    __slots__ = ["file_path", "code_string", "signature"]
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    CODE_STRING_FIELD_NUMBER: _ClassVar[int]
    SIGNATURE_FIELD_NUMBER: _ClassVar[int]
    file_path: str
    code_string: str
    signature: str
    def __init__(self, file_path: _Optional[str] = ..., code_string: _Optional[str] = ..., signature: _Optional[str] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ["tokens", "id", "sampling_type", "beam_width", "strategy_ids", "debug"]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_TYPE_FIELD_NUMBER: _ClassVar[int]
    BEAM_WIDTH_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_IDS_FIELD_NUMBER: _ClassVar[int]
    DEBUG_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedScalarFieldContainer[int]
    id: str
    sampling_type: str
    beam_width: int
    strategy_ids: _containers.RepeatedScalarFieldContainer[int]
    debug: bool
    def __init__(self, tokens: _Optional[_Iterable[int]] = ..., id: _Optional[str] = ..., sampling_type: _Optional[str] = ..., beam_width: _Optional[int] = ..., strategy_ids: _Optional[_Iterable[int]] = ..., debug: bool = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ["out", "detail", "debug_out"]
    OUT_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    DEBUG_OUT_FIELD_NUMBER: _ClassVar[int]
    out: _containers.RepeatedScalarFieldContainer[int]
    detail: _containers.RepeatedCompositeFieldContainer[PredictDetail]
    debug_out: str
    def __init__(self, out: _Optional[_Iterable[int]] = ..., detail: _Optional[_Iterable[_Union[PredictDetail, _Mapping]]] = ..., debug_out: _Optional[str] = ...) -> None: ...

class PredictDetail(_message.Message):
    __slots__ = ["prob", "candidate"]
    class CandidateEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: float
        def __init__(self, key: _Optional[int] = ..., value: _Optional[float] = ...) -> None: ...
    PROB_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    prob: float
    candidate: _containers.ScalarMap[int, float]
    def __init__(self, prob: _Optional[float] = ..., candidate: _Optional[_Mapping[int, float]] = ...) -> None: ...

class ModelConfig(_message.Message):
    __slots__ = ["name", "checkpoint_hash"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_HASH_FIELD_NUMBER: _ClassVar[int]
    name: str
    checkpoint_hash: str
    def __init__(self, name: _Optional[str] = ..., checkpoint_hash: _Optional[str] = ...) -> None: ...

class MachineConfig(_message.Message):
    __slots__ = ["gpu_name", "gpu_cnt", "is_ampere"]
    GPU_NAME_FIELD_NUMBER: _ClassVar[int]
    GPU_CNT_FIELD_NUMBER: _ClassVar[int]
    IS_AMPERE_FIELD_NUMBER: _ClassVar[int]
    gpu_name: str
    gpu_cnt: int
    is_ampere: bool
    def __init__(self, gpu_name: _Optional[str] = ..., gpu_cnt: _Optional[int] = ..., is_ampere: bool = ...) -> None: ...

class ConfigResponse(_message.Message):
    __slots__ = ["model_config", "machine_config", "connected_cnt"]
    MODEL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    MACHINE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    CONNECTED_CNT_FIELD_NUMBER: _ClassVar[int]
    model_config: ModelConfig
    machine_config: MachineConfig
    connected_cnt: int
    def __init__(self, model_config: _Optional[_Union[ModelConfig, _Mapping]] = ..., machine_config: _Optional[_Union[MachineConfig, _Mapping]] = ..., connected_cnt: _Optional[int] = ...) -> None: ...

class Strategy(_message.Message):
    __slots__ = ["type", "args"]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    type: int
    args: bytes
    def __init__(self, type: _Optional[int] = ..., args: _Optional[bytes] = ...) -> None: ...

class Strategies(_message.Message):
    __slots__ = ["strategies"]
    class StrategiesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: Strategy
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[Strategy, _Mapping]] = ...) -> None: ...
    STRATEGIES_FIELD_NUMBER: _ClassVar[int]
    strategies: _containers.MessageMap[int, Strategy]
    def __init__(self, strategies: _Optional[_Mapping[int, Strategy]] = ...) -> None: ...
