from typing import List, Tuple, Optional, Dict

from protobufs import model_pb2

from lmdeploy.turbomind.predictor import Predictor


class Model(object):
    def __init__(self, *,
                 predictor: Predictor,
                 is_instruct_model: bool,
                 attention_head_type: str,
                 is_ampere: bool,
                 ):
        self.__predictor = predictor
        self.__predictor_instance = self.__predictor.predictor.create_instance()
        self.__is_instruct_model:bool = is_instruct_model
        self.__is_post_after_code:bool = attention_head_type == "multiquery"
        self.__is_less_content_token:bool = not is_ampere
        self.__is_has_not_file_path:bool = attention_head_type == "multiquery"

    def get_tokenizer(self) -> str:
        return self.__predictor.tokenizer

    def get_model_name(self) -> str:
        return self.__predictor.predictor.model_name
    
    def get_generator(self):
        return self.__predictor_instance

    def is_instruct_model(self) -> bool:
        return self.__is_instruct_model

    def is_post_after_code(self) -> bool:
        return self.__is_post_after_code

    def is_less_content_token(self) -> bool:
        return self.__is_less_content_token

    def is_has_not_file_path(self) -> bool:
        return self.__is_has_not_file_path

    def checkpoint_hash(self) -> str:
        return self.__predictor.checkpoint_head_hash

    def init_strategies(self, *, strategies: model_pb2.Strategies) -> None:
        self.__predictor.init_strategies(strategies=strategies)

    def encoder(self) -> Dict[int, str]:
        return self.__predictor.tokenizer.encoder

    def special_tokens(self) -> List[str]:
        return self.__predictor.tokenizer.special_tokens_dict["additional_special_tokens"]

    def bos_token(self) -> str:
        return self.__predictor.tokenizer.special_tokens_dict["bos_token"]

    def eos_token(self) -> str:
        return self.__predictor.tokenizer.special_tokens_dict["eos_token"]

    def unk_token(self) -> str:
        return self.__predictor.tokenizer.special_tokens_dict["unk_token"]

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

    def encode(self, *, context: str) -> List[int]:
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
