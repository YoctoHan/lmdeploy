from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PunishTokens(_message.Message):
    __slots__ = ["scaling", "comment_tokens", "space_tokens"]
    SCALING_FIELD_NUMBER: _ClassVar[int]
    COMMENT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SPACE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    scaling: float
    comment_tokens: _containers.RepeatedScalarFieldContainer[int]
    space_tokens: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, scaling: _Optional[float] = ..., comment_tokens: _Optional[_Iterable[int]] = ..., space_tokens: _Optional[_Iterable[int]] = ...) -> None: ...
