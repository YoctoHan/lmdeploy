import os
import fire
import time

import numpy as np

from typing import List
from aixm_core.aixmeg.utils import is_ampere
from lmdeploy.turbomind.server import Model, ModelServicer
from lmdeploy.turbomind.predictor import Predictor
from lmdeploy.turbomind import test_cases

os.environ['TM_LOG_LEVEL'] = 'ERROR'
os.environ['AIXCODER_DEBUG'] = 'ON'

class PredictionRequest:
    def __init__(self, index):
        self.tokens = test_cases[index]
        self.sampling_type = None
        self.id = "YoctoHan"
        self.beam_width = 1
        self.debug = False

class MyCLI:
    def main(self, *args, **kwargs):
        predictor = Predictor(kwargs)
        
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Init model.")

        model = Model(
            predictor=predictor
        )
        
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Init model success.")
        
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Init servicer.")
            
        servicer = ModelServicer(model)
        
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Init servicer success.")
            
        if os.getenv('AIXCODER_DEBUG') == 'ON':
            print("[AIXCODER_DEBUG]Launch service.")
            
        servicer.serve(port=kwargs["port"])
        return

    def test(self, *args, **kwargs):
        predictor = Predictor(kwargs)
        
        model = Model(
            predictor=predictor
        )
        
        servicer = ModelServicer(model)
        
        for index in range(len(test_cases)):
            request = PredictionRequest(index)
            servicer.CreatePrediction(req = request, context = None)
            while True:
                res = servicer.GetResult(req = request, context = None)
                if len(res.out) == 0:            
                    if os.getenv('AIXCODER_DEBUG') == 'ON':
                        print("[AIXCODER_DEBUG]Iterator is empty, session over.")
                    break
                if os.getenv('AIXCODER_DEBUG') == 'ON':
                    print("[AIXCODER_DEBUG]Results:")
                    print(predictor.tokenizer.decode(request.tokens))
        servicer.ClearQueue()

    def test_inference_time(self, *args, **kwargs):
        predictor = Predictor(kwargs)
        
        model = Model(
            predictor=predictor
        )
        
        servicer = ModelServicer(model)
    
        request = PredictionRequest(0)
        
        count = 0
        start = time.perf_counter()
        pre_len = 960
        run_len = 64

        request.tokens = np.array([np.random.randint(0,20000, size=pre_len)], dtype='int32')[0].tolist()
        servicer.CreatePrediction(req = request, context = None)
        while True:
            res = servicer.GetResult(req = request, context = None)
            # import pdb;pdb.set_trace()
            if len(res.out) == 0:            
                if os.getenv('AIXCODER_DEBUG') == 'ON':
                    print("[AIXCODER_DEBUG]Iterator is empty, session over.")
                break       
            
            count += len(res.out)
        
            if count >= run_len:
                print(f"avg time: {(time.perf_counter()-start) * 1000 / (run_len+1):.2f}ms")
                break
        servicer.ClearQueue()

if __name__ == "__main__":
    fire.Fire(MyCLI)