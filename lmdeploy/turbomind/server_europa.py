import os
import fire

from typing import List
from aixm_core.aixmeg.utils import is_ampere
from lmdeploy.turbomind.server import Model, ModelServicer
from lmdeploy.turbomind.predictor import Predictor
from lmdeploy.turbomind import test_cases

os.environ['TM_LOG_LEVEL'] = 'INFO'
os.environ['AIXCODER_DEBUG'] = 'OFF'

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
            predictor=predictor,
            is_instruct_model=kwargs["is_instruct_model"],
            attention_head_type=kwargs["attention_head_type"],
            is_ampere=is_ampere(),
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
        # 测试功能实现
        predictor = Predictor(kwargs)
        
        model = Model(
            predictor=predictor,
            is_instruct_model=kwargs["is_instruct_model"],
            attention_head_type=kwargs["attention_head_type"],
            is_ampere=is_ampere(),
        )
        
        servicer = ModelServicer(model)
        
        for index in range(len(test_cases)):
            request = PredictionRequest(index)
            servicer.CreatePrediction(req = request, context = None)
            while True:
                res = servicer.GetResult(req = request, context = None)
                # import pdb;pdb.set_trace()
                if len(res.out) == 0:            
                    if os.getenv('AIXCODER_DEBUG') == 'ON':
                        print("[AIXCODER_DEBUG]Iterator is empty, session over.")
                    break
                if os.getenv('AIXCODER_DEBUG') == 'ON':
                    print("[AIXCODER_DEBUG]Results:")
                    print(predictor.tokenizer.decode(request.tokens))
        servicer.ClearQueue()

if __name__ == "__main__":
    fire.Fire(MyCLI)