import argparse
from fastapi import FastAPI, Form, UploadFile, HTTPException
import uvicorn

from framework.client.grpc_client import GrpcClient
import framework.common.MessageUtil as mu
import framework.protos.node_pb2 as fpn
import framework.protos.message_pb2 as fpm
from typing_extensions import Annotated, Optional
from framework.common.yaml_loader import load_yaml
from contextlib import asynccontextmanager
from argparse import Namespace
import os
from framework.web.api_utils import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, \
    UsageInfo, FunctionCallResponse, ChatMessage, DeltaMessage, ChatCompletionResponseStreamChoice
from loguru import logger
from sse_starlette.sse import EventSourceResponse
import time

service = {}

# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


@asynccontextmanager
async def lifespan(app: FastAPI):
    args = parse_args()
    print(args)
    init_grpc_client(args)
    yield
    service['grpc_client'].close()
    service.clear()


app = FastAPI(lifespan=lifespan)

node = fpn.Node(node_id="web")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/job/upload")
async def upload_job(file: UploadFile):
    if file is None:
        return {"result": "error", "message": "No file exists"}
    contents = await file.read()
    msg = Namespace()
    msg.data = {"config": contents, 'async': True}
    msg.type = fpm.CREATE_JOB
    result = service['grpc_client'].parse_message(msg)
    job_id = result['job_id']
    return {"result": "success", "job_id": job_id}


@app.post("/job")
def create_job(config: Annotated[str, Form()]):
    with open(config, "r") as f:
        data = f.read()
        value = fpm.Value()
        value.string = data
        msg = mu.MessageUtil.create(node, {"config": value}, 1)
        result = service['grpc_client'].open_and_send(msg)
        job_id = result.named_values['job_id'].sint64
        return {"result": "success", "job_id": job_id}


@app.get("/job")
def show_job(id: int):
    msg = Namespace()
    msg.data = id
    msg.type = fpm.QUERY_JOB
    job = service['grpc_client'].parse_message(msg)
    return job


@app.post("/start")
async def start_model(model_id: str, file: Optional[UploadFile] = None):
    msg = Namespace()
    msg.type = fpm.LOAD_MODEL
    if file:
        service['contents'] = await file.read()
    msg.data = {"config": service['contents'], "model_id": model_id}
    service['grpc_client'].parse_message(msg)
    return {"result": "success"}


@app.post("/message")
async def send_message(msg: Annotated[str, Form()]):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": msg}
    ]
    msg = Namespace()
    msg.data = {"config": service['contents'], "messages": messages}
    msg.type = fpm.CREATE_JOB
    result = service['grpc_client'].parse_message(msg)
    job_id = result['job_id']
    return {"result": "success", "job_id": job_id}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_length=request.max_tokens or 1024,
        max_new_tokens=request.max_new_tokens,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
        tools=request.tools,
    )
    logger.debug(f"==== request ====\n{gen_params}")

    # Here is the handling of stream = False
    msg = Namespace()
    msg.data = {"config": service['contents'], 'kwargs': gen_params, 'stream': request.stream}
    msg.type = fpm.CREATE_JOB

    if request.stream:
        predict_stream_generator = _predict_stream(request.model, msg)
        return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
    else:
        result = service['grpc_client'].parse_message(msg)
        response = {}
        if result:
            response = result['response']
            job_id = result['job_id']
            _close_job(job_id)
        return _create_response(request.model, response)


def _close_job(job_id):
    msg = Namespace()
    msg.data = job_id
    msg.type = fpm.CLOSE_JOB
    service['grpc_client'].parse_message(msg)
    logger.info("---end---")


def _predict_stream(model, msg):
    job_id = 0
    try:
        result = service['grpc_client'].parse_message(msg)
        job_id = result['job_id']
        for item in result.get('response'):
            chunk = _create_stream_response(model, item)
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    finally:
        _close_job(job_id)


def _create_response(model: str, response: dict):
    usage = UsageInfo()
    function_call, finish_reason = None, "stop"

    message = ChatMessage(
        role="assistant",
        content=response["text"],
        function_call=function_call if isinstance(function_call, FunctionCallResponse) else None,
    )

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
    )

    return ChatCompletionResponse(
        model=model,
        id="",  # for open_source model, id is empty
        choices=[choice_data],
        object="chat.completion",
        usage=usage
    )


def _create_stream_response(model_id, send_msg: str):
    function_call, finish_reason = None, "stop"
    message = DeltaMessage(
        content=send_msg,
        role="assistant",
        function_call=function_call,
    )
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=message,
        finish_reason=finish_reason
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        id="",
        choices=[choice_data],
        created=int(time.time()),
        object="chat.completion.chunk"
    )
    return chunk


def init_grpc_client(args):
    service['grpc_client'] = GrpcClient("web", 0, args.grpc_host, args.grpc_port, args.compression)
    service['contents'] = read_json_config()


def parse_args():
    parser = argparse.ArgumentParser("WebServer")
    parser.add_argument('--config', default='./web_config.yml')
    args = parser.parse_args()
    config = load_yaml(args.config)
    args.grpc_host = config["grpc_server"]["host"]
    args.grpc_port = config["grpc_server"]["port"]
    args.compression = config["grpc_server"]["compression"]
    return args


def read_json_config():
    config_path = os.path.join(os.path.dirname(__file__), "../../configs/llm_configs/dev_llm_inference.json")
    with open(config_path, "r") as f:
        contents = f.read()
    return contents


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, log_level="info")
