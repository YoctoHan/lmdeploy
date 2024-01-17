# 模型启动需要的包
import dataclasses
import os
import sys
import traceback
import os.path as osp
import numpy as np
import random
import time
import torch

import fire

from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS
# from lmdeploy.turbomind.tokenizer import Tokenizer, GPTTokenizer
from aixm_core.aixmeg.tokenizer import GPTTokenizer

# TurboMind的日志等级
os.environ['TM_LOG_LEVEL'] = 'ERROR'

# 服务端需要的包
import grpc
from concurrent.futures import ThreadPoolExecutor
from protobufs import model_pb2_grpc
from protobufs import model_pb2

def valid_str(string, coding='utf-8'):
    """decode text according to its encoding type."""
    invalid_chars = [b'\xef\xbf\xbd']
    bstr = bytes(string, coding)
    for invalid_char in invalid_chars:
        bstr = bstr.replace(invalid_char, b'')
    ret = bstr.decode(encoding=coding, errors='ignore')
    return ret

@dataclasses.dataclass
class GenParam:
    top_p: float
    top_k: float
    temperature: float
    repetition_penalty: float
    sequence_start: bool = False
    sequence_end: bool = False
    step: int = 0
    request_output_len: int = 8192

def get_gen_param(cap,
                  sampling_param,
                  nth_round,
                  step,
                  request_output_len=8192,
                  **kwargs):
    """return parameters used by token generation."""
    gen_param = GenParam(**dataclasses.asdict(sampling_param),
                         request_output_len=request_output_len)
    # Fix me later. turbomind.py doesn't support None top_k
    if gen_param.top_k is None:
        gen_param.top_k = 1

    if cap == 'chat':
        gen_param.sequence_start = (nth_round == 1)
        gen_param.sequence_end = False
        gen_param.step = step
    else:
        gen_param.sequence_start = True
        gen_param.sequence_end = True
        gen_param.step = 0
    return gen_param

class ModelServicer(model_pb2_grpc.ModelServicer):

    def __init__(self, model_path, vocab_dir):
        self.tokenizer = GPTTokenizer(vocab_dir="/data3/aix2_base_v2/")
        self.tm_model = tm.TurboMind(model_path, eos_id = self.tokenizer.eos_id, tp=1)
        self.generator = self.tm_model.create_instance()
        self.output = None

    def CreatePrediction(self, tokens, user_id, sampling_type, beam_width) -> None:
        """
        The prediction is initiated according to the given parameters and an iterator is generated to store the result. 
        Note that this method does not directly put back the prediction result.

        Args:
            tokens (list): A list of tokens for the input data, used for model prediction.
            user_id (str): An identifier for the user, used for recording or customizing predictions.
            sampling_type (str): Specifies whether to take nuclear sampling.
            beam_width (int): Specifies the width of the result beam to be returned.

        Returns:
            None: This method does not return a value. The results will be streamed using the GetResult method.

        Raises:
            ValueError: If the provided parameters do not match the expected data types or value ranges.
            RuntimeError: If an error occurs during model initialization or prediction.

        Example:
            >>> servicer = ModelServicer()
            >>> servicer.CreatePrediction(["hello world"], "", 'type1', 5)
            >>> results = GetResult()
        """

        # Method implementation...
        session_id = 1
        cap = 'completion'
        # sys_instruct = None
        tp = 1
        stream_output = True
        nth_round = 1
        step = 0
        seed = random.getrandbits(64)
        model_name = self.tm_model.model_name
        model = MODELS.get(model_name)(capability=cap) \
        # if sys_instruct is None else MODELS.get(model_name)(
        #     capability=cap, system=sys_instruct)
        
        gen_param = get_gen_param(cap, model.sampling_param, nth_round,
                                    step)
        # import pdb;pdb.set_trace()
        self.output = self.generator.stream_infer(
                        session_id=session_id,
                        input_ids=[tokens],
                        stream_output=stream_output,
                        **dataclasses.asdict(gen_param),
                        ignore_eos=False,
                        random_seed=seed if nth_round == 1 else None)


    def Encoder(self, req, context):
        tmp = self.tokenizer.tokenize(req.context)
        tmp = self.tokenizer.convert_tokens_to_ids(tokens=tmp)
        return model_pb2.EncoderResponse(outputs=tmp)

    def Decoder(self, req, context):
        try:
            tmp = model_pb2.DecoderResponse()
            for k in self.tokenizer.encoder:
                v = self.tokenizer.encoder[k]
                assert v not in tmp.decoder
                tmp.decoder[v] = k
            # import pdb;pdb.set_trace()
            for v in self.tokenizer.special_tokens_dict["additional_special_tokens"]:
                assert v in self.tokenizer.encoder
                tmp.special_tokens.append(v)
            tmp.bos_token = self.tokenizer.special_tokens_dict['bos_token']
            tmp.eos_token = self.tokenizer.special_tokens_dict['eos_token']
            tmp.unk_token = self.tokenizer.special_tokens_dict['unk_token']
            assert tmp.bos_token in self.tokenizer.encoder
            assert tmp.eos_token in self.tokenizer.encoder
            assert tmp.unk_token in self.tokenizer.encoder
            if self.tokenizer.prompts_data is not None:
                for k in self.tokenizer.prompts_data:
                    prompts_obj = model_pb2.Prompts()
                    for prompt in self.tokenizer.prompts_data[k]:
                        assert len(prompt.keys()) == 3
                        assert 'file_path' in prompt
                        assert 'code_string' in prompt
                        assert 'signature' in prompt
                        prompts_obj.prompt.append(model_pb2.Prompt(
                            file_path=prompt['file_path'],
                            code_string=prompt['code_string'],
                            signature=prompt['signature'],
                        ))
                    tmp.prompts[k].CopyFrom(prompts_obj)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise e
        return tmp

    def GetResult(self):        
        try:
            res, tokens = next(self.output)[0]
        except StopIteration:
            res = torch.tensor(-1)
        return res
        # return model_pb2.PredictResponse(out=res.to_list())

    def Config(self, req, context):
        # self.connected_cnt+=1
        # print("code=5"+self.model.checkpoint_hash(), flush=True, file=sys.stderr)
        return model_pb2.ConfigResponse(is_instruct_model=False,
                                        is_post_after_code=False,
                                        is_less_content_token=False,
                                        is_has_not_file_path=False,
                                        checkpoint_hash="5-yocto",
                                        connected_cnt=0,
                                        )

    def Init(self, req, context):
        return model_pb2.Empty()


def serve():
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_ModelServicer_to_server(ModelServicer(), server)
    server.add_insecure_port('[::]:' + str("12310"))
    server.start()
    server.wait_for_termination()



def main(model_path, vocab_dir):
    
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    model_pb2_grpc.add_ModelServicer_to_server(ModelServicer(model_path, vocab_dir), server)
    server.add_insecure_port('[::]:' + str("12310"))
    server.start()
    print("Launch success")
    server.wait_for_termination()
    exit(0)
    
    # model_path = "./star_coder_workspace"
    # vocab_dir="/data3/StarCoderBase/"
    servicer = ModelServicer(model_path, vocab_dir)
    input_str = """import copy
import os
import queue
import sys
import hashlib
import threading"""

    input_ids = servicer.tokenizer.encode(input_str)
    # import pdb;pdb.set_trace()
    servicer.CreatePrediction(input_ids, "", "", 0)

    past_times = []
    start_time = time.time()
    tensor_minus_one = torch.tensor(-1)

    while(True):
        res = servicer.GetResult()                
        # decode res
        response = servicer.tokenizer.decode(res.tolist(), offset=0)
        response = valid_str(response)

        # print(f'{response}', end='', flush=True)
        print(input_str + response)
        # import pdb;pdb.set_trace()

        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        past_times.append(elapsed_time)
        start_time = end_time

        # Calculate average time
        if len(past_times) == 1024:
            print(f"Average time for past 1024 results: {sum(past_times) / len(past_times) * 1000} ms")
            past_times = []  # Reset the list

        if torch.equal(res.cpu(), tensor_minus_one):
            break


    # serve()


def test(model_path, vocab_dir):
# model_path = "./star_coder_workspace"
    # vocab_dir="/data3/StarCoderBase/"
    servicer = ModelServicer(model_path, vocab_dir)
#     input_str = """import copy
# import os
# import queue
# import sys
# import hashlib
# import threading"""

#     input_ids = servicer.tokenizer.encode(input_str)
    input_ids = np.random.randint(low=0, high=100, size=1024)
    servicer.CreatePrediction(input_ids, "", "", 0)

    start_time = time.time()
    servicer.GetResult()                
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ms = elapsed_time * 1000
    print(elapsed_time_ms)


if __name__ == '__main__':
    fire.Fire(main)
    # fire.Fire(test)