from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ContextWithTimestamp(_message.Message):
    __slots__ = ["context", "timestamp"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    context: str
    timestamp: int
    def __init__(self, context: _Optional[str] = ..., timestamp: _Optional[int] = ...) -> None: ...

class CodeRequest(_message.Message):
    __slots__ = ["code_string", "later_code", "code_string_offset", "later_code_offset", "code_string_md5", "later_code_md5"]
    CODE_STRING_FIELD_NUMBER: _ClassVar[int]
    LATER_CODE_FIELD_NUMBER: _ClassVar[int]
    CODE_STRING_OFFSET_FIELD_NUMBER: _ClassVar[int]
    LATER_CODE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    CODE_STRING_MD5_FIELD_NUMBER: _ClassVar[int]
    LATER_CODE_MD5_FIELD_NUMBER: _ClassVar[int]
    code_string: str
    later_code: str
    code_string_offset: int
    later_code_offset: int
    code_string_md5: str
    later_code_md5: str
    def __init__(self, code_string: _Optional[str] = ..., later_code: _Optional[str] = ..., code_string_offset: _Optional[int] = ..., later_code_offset: _Optional[int] = ..., code_string_md5: _Optional[str] = ..., later_code_md5: _Optional[str] = ...) -> None: ...

class CodeResponse(_message.Message):
    __slots__ = ["code_string", "later_code", "message", "ok"]
    CODE_STRING_FIELD_NUMBER: _ClassVar[int]
    LATER_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    code_string: str
    later_code: str
    message: str
    ok: bool
    def __init__(self, code_string: _Optional[str] = ..., later_code: _Optional[str] = ..., message: _Optional[str] = ..., ok: bool = ...) -> None: ...

class ReferenceFilesRequest(_message.Message):
    __slots__ = ["required_files", "accept"]
    class RequiredFilesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class AcceptEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: ContextWithTimestamp
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[ContextWithTimestamp, _Mapping]] = ...) -> None: ...
    REQUIRED_FILES_FIELD_NUMBER: _ClassVar[int]
    ACCEPT_FIELD_NUMBER: _ClassVar[int]
    required_files: _containers.ScalarMap[str, str]
    accept: _containers.MessageMap[str, ContextWithTimestamp]
    def __init__(self, required_files: _Optional[_Mapping[str, str]] = ..., accept: _Optional[_Mapping[str, ContextWithTimestamp]] = ...) -> None: ...

class ReferenceFilesReponse(_message.Message):
    __slots__ = ["context", "md5", "name"]
    class ContextEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class Md5Entry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    MD5_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    context: _containers.ScalarMap[str, str]
    md5: _containers.ScalarMap[str, str]
    name: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, context: _Optional[_Mapping[str, str]] = ..., md5: _Optional[_Mapping[str, str]] = ..., name: _Optional[_Iterable[str]] = ...) -> None: ...

class Context(_message.Message):
    __slots__ = ["uuid", "project_root", "file_path"]
    UUID_FIELD_NUMBER: _ClassVar[int]
    PROJECT_ROOT_FIELD_NUMBER: _ClassVar[int]
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    uuid: str
    project_root: str
    file_path: str
    def __init__(self, uuid: _Optional[str] = ..., project_root: _Optional[str] = ..., file_path: _Optional[str] = ...) -> None: ...

class Request(_message.Message):
    __slots__ = ["ctx", "rfr", "cr"]
    CTX_FIELD_NUMBER: _ClassVar[int]
    RFR_FIELD_NUMBER: _ClassVar[int]
    CR_FIELD_NUMBER: _ClassVar[int]
    ctx: Context
    rfr: ReferenceFilesRequest
    cr: CodeRequest
    def __init__(self, ctx: _Optional[_Union[Context, _Mapping]] = ..., rfr: _Optional[_Union[ReferenceFilesRequest, _Mapping]] = ..., cr: _Optional[_Union[CodeRequest, _Mapping]] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ["rfr", "cr"]
    RFR_FIELD_NUMBER: _ClassVar[int]
    CR_FIELD_NUMBER: _ClassVar[int]
    rfr: ReferenceFilesReponse
    cr: CodeResponse
    def __init__(self, rfr: _Optional[_Union[ReferenceFilesReponse, _Mapping]] = ..., cr: _Optional[_Union[CodeResponse, _Mapping]] = ...) -> None: ...
