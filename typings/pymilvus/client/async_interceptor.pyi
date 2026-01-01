from typing import Callable

from grpc.aio import (
    ClientCallDetails,
    StreamStreamClientInterceptor,
    StreamUnaryClientInterceptor,
    UnaryStreamClientInterceptor,
    UnaryUnaryClientInterceptor,
)
from grpc.aio._call import StreamStreamCall as StreamStreamCall
from grpc.aio._call import StreamUnaryCall as StreamUnaryCall
from grpc.aio._call import UnaryStreamCall as UnaryStreamCall
from grpc.aio._call import UnaryUnaryCall as UnaryUnaryCall
from grpc.aio._typing import RequestIterableType as RequestIterableType
from grpc.aio._typing import RequestType as RequestType
from grpc.aio._typing import ResponseIterableType as ResponseIterableType
from grpc.aio._typing import ResponseType as ResponseType

class _GenericAsyncClientInterceptor(
    UnaryUnaryClientInterceptor,
    UnaryStreamClientInterceptor,
    StreamUnaryClientInterceptor,
    StreamStreamClientInterceptor,
):
    def __init__(self, interceptor_function: Callable) -> None: ...
    async def intercept_unary_unary(
        self,
        continuation: Callable[[ClientCallDetails, RequestType], UnaryUnaryCall],
        client_call_details: ClientCallDetails,
        request: RequestType,
    ) -> UnaryUnaryCall | ResponseType: ...
    async def intercept_unary_stream(
        self,
        continuation: Callable[[ClientCallDetails, RequestType], UnaryStreamCall],
        client_call_details: ClientCallDetails,
        request: RequestType,
    ) -> ResponseIterableType | UnaryStreamCall: ...
    async def intercept_stream_unary(
        self,
        continuation: Callable[[ClientCallDetails, RequestType], StreamUnaryCall],
        client_call_details: ClientCallDetails,
        request_iterator: RequestIterableType,
    ) -> StreamUnaryCall: ...
    async def intercept_stream_stream(
        self,
        continuation: Callable[[ClientCallDetails, RequestType], StreamStreamCall],
        client_call_details: ClientCallDetails,
        request_iterator: RequestIterableType,
    ) -> ResponseIterableType | StreamStreamCall: ...

def async_header_adder_interceptor(headers: list[str], values: list[str] | list[bytes]): ...
