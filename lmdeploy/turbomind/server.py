import traceback
import threading
import dataclasses
import queue
import time
import os
import sys
import grpc
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor
from protobufs import model_pb2_grpc
from protobufs import model_pb2
from lmdeploy.turbomind.model import Model, ModelBase
from lmdeploy.model import MODELS

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
    def __init__(self, model: ModelBase):
        self.created_just_now: bool = False
        self.model: ModelBase = model
        self.com_len: int = 0
        self.prediction_tokens: List[int] = []
        self.sampling_type: str = ""
        self.id: str = ""
        self.beam_width: int = 0
        self.connected_cnt: int = 0
        self.strategy_ids: List[int] = []
        self.debug: bool = False
        self.create_id: int = 0
        self.get_result_id: int = 0
        self.is_use_common_len_cache: int = False
        
        self.generator = self.model.get_generator()
        self.already_return : int = 0
        self.result_len : int = 0
    
        self.result_queue = queue.Queue()
        self.current_thread = None
        self.stop_event = threading.Event()
        self.iterator = None
        
        self.request = [0 for _ in range(256)]

    def Encoder(self, req, context):
        return model_pb2.EncoderResponse(outputs=self.model.encode(context=req.context, is_prefix=req.is_prefix))

    def Decoder(self, req, context):
        return model_pb2.DecoderResponse(token_info=self.model.token_info(), prompts=self.model.prompts())

    def IteratorWorker(self):
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Init iterator, session id = {}.".format(self.create_id))
            
        for item in self.iterator:
            # Check the stop event before putting the item into the queue
            current_result_len = item[0][1]
            if self.stop_event.is_set():
                break
            if current_result_len == self.result_len:
                continue
            self.result_len = current_result_len
            self.result_queue.put(item)
            # print(f'Produced {item}')
            
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Stop iteration, session id = {}.".format(self.create_id))
            
        self.result_queue.put(None)  # Indicate the generator is finished
        self.request[self.create_id] = 0

    def ClearQueue(self):
        # Stop previous generator thread if it exists
        if self.current_thread is not None:
            if os.getenv('AIXCODER_DEBUG') == 'ON':
                print("[AIXCODER_DEBUG]Free thread, session id = {}.".format(self.create_id - 1))
            self.stop_event.set()
            self.current_thread.join()  # Wait for the thread to finish
        if not self.result_queue.empty():
            if os.getenv('AIXCODER_DEBUG') == 'ON':
                print("[AIXCODER_DEBUG]Clear results queue, session id = {}.".format(self.create_id - 1))
            try:
                while True:
                    self.result_queue.get_nowait()
            except queue.Empty:
                pass
        
        # Clear the event for the new thread
        self.stop_event.clear()

    def CreatePrediction(self, req, context):
        self.create_id += 1
        self.request[self.create_id % 256] = 1
        
        self.prediction_tokens: List[int] = req.tokens
        self.sampling_type: str = req.sampling_type
        self.id: str = req.id
        self.beam_width: int = req.beam_width
        # self.strategy_ids: List[int] = [strategy_id for strategy_id in req.strategy_ids]
        self.debug: bool = req.debug
        self.already_return = 0
        self.result_len = 0
        self.com_len = 0
        
        cap = 'completion'
        nth_round = 1
        step = 0
        seed = random.getrandbits(64)
        model_name = self.model.name()
        sampling_param = MODELS.get(model_name)(capability=cap).sampling_param
        
        gen_param = get_gen_param(cap, sampling_param, nth_round,
                            step)
        
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Init generator, session id = {}.".format(self.create_id))
            print("[AIXCODER_DEBUG]Input tokens:", req.tokens)
            
        self.ClearQueue()
        # import pdb;pdb.set_trace()
        if self.request[self.create_id - 1]:
            self.generator.interuption()
            self.request[self.create_id - 1] = 0
            
        self.iterator = self.generator.stream_infer(
                session_id=self.create_id,
                input_ids=[req.tokens],
                stream_output=True,
                **dataclasses.asdict(gen_param),
                ignore_eos=False,
                random_seed=seed if nth_round == 1 else None)
        
        self.current_thread = threading.Thread(target=self.IteratorWorker)
        self.current_thread.start()
        
        print(f"create id={self.create_id}, len={len(req.tokens)}, com_len={self.com_len}, debug={req.debug}",
              flush=True, file=sys.stderr)
        return model_pb2.Empty()

    def GetResult(self, req, context):
        self.get_result_id += 1
        debug_str = ""
        out = []
        details = []
        
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Get results, session id = {}, result id = {}.".format(self.create_id, self.get_result_id))
        
        res = self.result_queue.get()
        if res != None:
            res, _ = res[0]
            res = res.cpu().tolist()[self.already_return:]
            tokens = len(res)
            
            if os.getenv('AIXCODER_DEBUG') == 'ON':
                print("[AIXCODER_DEBUG]Res = {}, token = \"{}\", already_return = {}.".format(res,
                                                                                              self.model.get_tokenizer().decode(res),
                                                                                              self.already_return))
            
            self.already_return += tokens
            res_p = [1.0] * tokens # [n,]
            res_s = [[x] for x in res] # [n, 1]
            res_ps = [[x] for x in res_p] # [n, 1]
            
            debug_str = ""

            self.com_len = len(self.prediction_tokens)
            self.prediction_tokens.extend(res)
            details = [] # [n,]
            for index in range(tokens): # j = 0~n-1
                details.append(model_pb2.PredictDetail(prob=res_p[index],
                                                candidate={kv[0]:kv[1] for kv in zip(res_s[index], res_ps[index])}))
            out = res
        return model_pb2.PredictResponse(out=out, detail=details, debug_out=debug_str)


    def Config(self, req, context):
        self.connected_cnt += 1
        print("code=E" + self.model.checkpoint_hash(), flush=True, file=sys.stderr)
        return model_pb2.ConfigResponse(
            model_config=model_pb2.ModelConfig(
                name=self.model.name(),
                checkpoint_hash="E" + self.model.checkpoint_hash(),
            ),
            machine_config=model_pb2.MachineConfig(
                gpu_name="",  # todo
                gpu_cnt=1,  # todo
                is_ampere=self.model.is_ampere(),
            ),
            connected_cnt=self.connected_cnt,
        )

    def Init(self, req, context):
        # self.model.init_strategies(strategies=req)
        return model_pb2.Empty()

    def serve(self, *, port: int):
        server = grpc.server(ThreadPoolExecutor(max_workers=10))
        model_pb2_grpc.add_ModelServicer_to_server(self, server)
        server.add_insecure_port('[::]:' + str(port))
        server.start()
        server.wait_for_termination()


