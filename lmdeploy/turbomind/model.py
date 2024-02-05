from typing import List, Tuple, Optional, Dict

from protobufs import model_pb2

from aixm_core.aixmeg.utils import is_ampere
from lmdeploy.turbomind.predictor import Predictor



class ModelBase(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def is_ampere() -> bool:
        return is_ampere()

    def name(self) -> str:
        raise NotImplementedError

    def checkpoint_hash(self) -> str:
        raise NotImplementedError

    def max_len(self) -> int:
        raise NotImplementedError

    def init_strategies(self, *, strategies: model_pb2.Strategies) -> None:
        return None

    def token_info(self) -> List[model_pb2.TokenInfo]:
        raise NotImplementedError

    def prompts(self) -> Dict[str, model_pb2.Prompts]:
        return dict()

    def encode(self, *, context: str, is_prefix: bool) -> List[int]:
        raise NotImplementedError

    def predict(self, *,
                token_ids: List[int],
                common_len: int,
                voting_type: int,
                sampling_type: str,
                uid: str,
                strategy_ids: List[int],
                created_just_now: bool,
                ) -> Tuple[
        List[int],
        List[float],
        List[List[int]],
        List[List[float]],
    ]:
        raise NotImplementedError

class Model(ModelBase):
    def __init__(self, *,
                 predictor: Predictor,
                 ):
        self.__predictor = predictor
        self.__predictor_instance = self.__predictor.predictor.create_instance()

    def get_tokenizer(self) -> str:
        return self.__predictor.tokenizer

    # def get_model_name(self) -> str:
    #     return self.__predictor.predictor.model_name
    
    def name(self) -> str:
        return self.__predictor.predictor.model_name
    
    def get_generator(self):
        return self.__predictor_instance

    def checkpoint_hash(self) -> str:
        return self.__predictor.checkpoint_head_hash

    def prompts(self) -> Dict[str, model_pb2.Prompts]:
        tmp = dict()
        if self.__predictor.tokenizer.prompts_data is not None:
            for k in self.__predictor.tokenizer.prompts_data:
                prompts_obj = model_pb2.Prompts()
                for prompt in self.__predictor.tokenizer.prompts_data[k]:
                    assert len(prompt.keys()) == 3
                    assert 'file_path' in prompt
                    assert 'code_string' in prompt
                    assert 'signature' in prompt
                    prompts_obj.prompt.append(model_pb2.Prompt(
                        file_path=prompt['file_path'],
                        code_string=prompt['code_string'],
                        signature=prompt['signature'],
                    ))
                tmp[k] = prompts_obj
        return tmp
    
    def encode(self, *, context: str, is_prefix: bool) -> List[int]:
        tk_server = self.__predictor.tokenizer
        tmp = tk_server.tokenize(context)
        tmp = tk_server.convert_tokens_to_ids(tokens=tmp)
        return tmp
    
    def predict(self, token_ids: List[int], common_len: int, voting_type: int = 0, sampling_type: str = "no",
                uid: str = "only_my_rail_gun", strategy_ids: Optional[List[int]] = None, created_just_now: bool = False) -> Tuple[
        List[int], List[float], List[List[int]], List[List[float]]]:
        return self.__predictor.predict(
            token_ids=token_ids,
            common_len=common_len,
            voting_type=voting_type,
            sampling_type=sampling_type,
            uid=uid,
            strategy_ids=strategy_ids,
            created_just_now=created_just_now,
        )

    def token_info(self) -> List[model_pb2.TokenInfo]:
        special = self.__predictor.tokenizer.special_tokens_dict["additional_special_tokens"]
        tmp = []
        for k, v in self.__predictor.tokenizer.encoder.items():
            bs = bytearray([self.__predictor.tokenizer.byte_decoder[c] for c in k])
            tmp.append(model_pb2.TokenInfo(value=v, context=bytes(bs), type="bytes", id_end=k in special))
        return tmp