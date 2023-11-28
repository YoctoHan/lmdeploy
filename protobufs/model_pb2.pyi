from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class EvalRequest(_message.Message):
    __slots__ = ["context"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    context: str
    def __init__(self, context: _Optional[str] = ...) -> None: ...

class EvalResponse(_message.Message):
    __slots__ = ["output"]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    output: str
    def __init__(self, output: _Optional[str] = ...) -> None: ...

class EncoderRequest(_message.Message):
    __slots__ = ["context"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    context: str
    def __init__(self, context: _Optional[str] = ...) -> None: ...

class EncoderResponse(_message.Message):
    __slots__ = ["outputs"]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    outputs: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, outputs: _Optional[_Iterable[int]] = ...) -> None: ...

class DecoderResponse(_message.Message):
    __slots__ = ["decoder", "special_tokens", "bos_token", "eos_token", "unk_token", "prompts"]
    class DecoderEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: str
        def __init__(self, key: _Optional[int] = ..., value: _Optional[str] = ...) -> None: ...
    class PromptsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Prompts
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Prompts, _Mapping]] = ...) -> None: ...
    DECODER_FIELD_NUMBER: _ClassVar[int]
    SPECIAL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    BOS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    EOS_TOKEN_FIELD_NUMBER: _ClassVar[int]
    UNK_TOKEN_FIELD_NUMBER: _ClassVar[int]
    PROMPTS_FIELD_NUMBER: _ClassVar[int]
    decoder: _containers.ScalarMap[int, str]
    special_tokens: _containers.RepeatedScalarFieldContainer[str]
    bos_token: str
    eos_token: str
    unk_token: str
    prompts: _containers.MessageMap[str, Prompts]
    def __init__(self, decoder: _Optional[_Mapping[int, str]] = ..., special_tokens: _Optional[_Iterable[str]] = ..., bos_token: _Optional[str] = ..., eos_token: _Optional[str] = ..., unk_token: _Optional[str] = ..., prompts: _Optional[_Mapping[str, Prompts]] = ...) -> None: ...

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
    __slots__ = ["tokens", "id", "sampling_type", "beam_width"]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_TYPE_FIELD_NUMBER: _ClassVar[int]
    BEAM_WIDTH_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedScalarFieldContainer[int]
    id: str
    sampling_type: str
    beam_width: int
    def __init__(self, tokens: _Optional[_Iterable[int]] = ..., id: _Optional[str] = ..., sampling_type: _Optional[str] = ..., beam_width: _Optional[int] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ["out", "detail"]
    OUT_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    out: _containers.RepeatedScalarFieldContainer[int]
    detail: _containers.RepeatedCompositeFieldContainer[PredictDetail]
    def __init__(self, out: _Optional[_Iterable[int]] = ..., detail: _Optional[_Iterable[_Union[PredictDetail, _Mapping]]] = ...) -> None: ...

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

class ConfigResponse(_message.Message):
    __slots__ = ["is_instruct_model", "is_less_content_token", "is_has_not_file_path", "is_post_after_code"]
    IS_INSTRUCT_MODEL_FIELD_NUMBER: _ClassVar[int]
    IS_LESS_CONTENT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    IS_HAS_NOT_FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    IS_POST_AFTER_CODE_FIELD_NUMBER: _ClassVar[int]
    is_instruct_model: bool
    is_less_content_token: bool
    is_has_not_file_path: bool
    is_post_after_code: bool
    def __init__(self, is_instruct_model: bool = ..., is_less_content_token: bool = ..., is_has_not_file_path: bool = ..., is_post_after_code: bool = ...) -> None: ...
