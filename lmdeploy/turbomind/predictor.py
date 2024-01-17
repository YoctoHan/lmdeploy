import sys
import os
import time
import pathlib
import torch
import traceback
import numpy as np
import hashlib


from lmdeploy import turbomind as tm
from typing import List, Tuple, Optional, Dict
from aixm_core.aixmeg.tokenizer import GPTTokenizer
from aixm_core.aixmeg import get_args, print_rank_0
from aixm_core.aixmeg.initialize import initialize_aixmeg
from aixm_core.aixmeg.model import GPTModel
from aixm_core.aixmeg.model.ladder_net import LST
from protobufs import model_pb2, strategy_args_pb2
from aixm_core.aixmeg.utils import get_model_for_infer, is_ampere, LRUKVCache, TrainConfig

# try:
#     from aixm_core.quantization import quantize_aix2_cpm
# except BaseException:
#     traceback.print_exc()
#     quantize_aix2_cpm = None

# try:
#     from aixm_core.quantization import quantize_aix2_bnb
# except BaseException:
#     quantize_aix2_bnb = None
    

# try:
#     from functools import lru_cache
# except ImportError:
#     # Just a dummy decorator to get the checks to run on python2
#     # because honestly I don't want to support a byte-level unicode BPE
#     # tokenizer on python 2 right now.
#     def lru_cache():
#         return lambda func: func


def softmax(
        *,
        x
):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def add_code_generation_args(parser):
    """Code generation arguments."""
    group = parser.add_argument_group(title="code generation")

    group.add_argument("--input", type=str, required=False, default="")
    group.add_argument("--output", type=str, required=False, default="")
    group.add_argument("--outputCopyToStdout", action="store_true", default=False)

    group.add_argument(
        "--padded_vocab_size",
        type=int,
        default=10,
        help="Start id for whitespace encoding",
    )
    # max-position-embeddings, seq-length
    group.add_argument("--element_wise_gate", type=str, default="no")
    group.add_argument("--use_output_gate", type=str, default="no")
    group.add_argument("--stem_model_dim", type=int, default=6144)
    group.add_argument("--n_sub_layers", type=int, default=6)
    group.add_argument("--quant_type", type=int, default=-1)
    group.add_argument("--reduction_scale", type=int, default=1)
    group.add_argument("--model_dir", type=str, default="")
    group.add_argument("--second_model_dir", type=str, default="")
    group.add_argument("--model_name", type=str, default="AixEuropaBaseV2")
    group.add_argument("--ladder_type", type=str, default="SwiGLU")
    group.add_argument("--ladder_load_path", type=str, default=os.environ.get("LADDER_PATH", ""))

    return parser

PRINT_TIME= False
AIXDEBUG_LADDER = os.environ.get("AIXDEBUG_LADDER", "") != ""



class PredictStrategy(object):
    def predict_strategy_eval(self, *, input_ids: List[int], probs: np.ndarray, strategy_state: dict) -> None:
        raise NotImplementedError("Unimplemented PredictStrategy in this version")


class PenalizationStrategy(PredictStrategy):
    def __init__(self, *, args: bytes, tokenizer: GPTTokenizer):
        # self.strategy_args = strategy_args_pb2.PunishTokens()
        # self.strategy_args.ParseFromString(args)
        self.strategy_args = strategy_args_pb2.PunishTokens.FromString(args)
        if len(self.strategy_args.comment_tokens) < 1 or len(self.strategy_args.comment_tokens) > 1234:
            raise ValueError(f"Penalization tokens length should be in [1, 1234], but got {len(self.strategy_args.comment_tokens)}")  # must have one comment token
        for token in self.strategy_args.comment_tokens:
            if token < 0 or token >= len(tokenizer.encoder):
                raise ValueError(f"Penalization token {token} is invalid")  # comment
        if len(self.strategy_args.space_tokens) > 1234:
            raise ValueError(f"Penalization tokens length should be in [0, 1234], but got {len(self.strategy_args.space_tokens)}")  # do not have to have space token
        for token in self.strategy_args.space_tokens:
            if token < 0 or token >= len(tokenizer.encoder):
                raise ValueError(f"Penalization token {token} is invalid")  # space
        self.already_skip_space = "Penalization:already_skip_space"

    def predict_strategy_eval(self, *, input_ids: List[int], probs: np.ndarray, strategy_state: dict) -> None:
        if self.already_skip_space not in strategy_state:
            strategy_state[self.already_skip_space] = False
            return
        if not strategy_state[self.already_skip_space]:
            for input_id in input_ids:
                if input_id not in self.strategy_args.space_tokens:
                    strategy_state[self.already_skip_space] = True
                    return
            probs[self.strategy_args.comment_tokens] *= self.strategy_args.scaling


class EOSPreferenceStrategy(PredictStrategy):
    def __init__(self, *, tokenizer: GPTTokenizer):
        self.eos_id = tokenizer.eos_id

    def predict_strategy_eval(self, *, input_ids: List[int], probs: np.ndarray, strategy_state: dict) -> None:
        delta = 1.0 - np.sum(probs)
        if delta > 0:
            probs[self.eos_id] += delta


class L1NormalizationStrategy(PredictStrategy):
    def predict_strategy_eval(self, *, input_ids: List[int], probs: np.ndarray, strategy_state: dict) -> None:
        probs /= np.sum(probs)


class Predictor(object):

    def __init__(self, args):

        self.is_ampere = is_ampere()
        self.args = args
        self.checkpoint_head_hash: str = ""
        
        # build predictor
        self.tokenizer = self.create_tokenizer()
        self.predictor = self.create_predictor()

    def create_tokenizer(self, path=None):
        tokenizer = GPTTokenizer(vocab_dir=self.args["vocab_dir"] if path is None else path, is_ampere=self.is_ampere)
        return tokenizer

    def create_predictor(self):
        tm_model = tm.TurboMind(model_path=self.args["model_dir"], eos_id = self.tokenizer.eos_id, tp=1)
        return tm_model
    
    def load_checkpoint(self, model, path, second_path=None, is_ladder=False):
        
        assert isinstance(model, list)

        if not (path is not None and os.path.exists(path)):
            raise ValueError

        iteration = 0
        if is_ladder and self.args.tensor_model_parallel_size == 1:

            checkpoint_name = os.path.join(path,"CodeLST.pt")
            assert os.path.isfile(checkpoint_name)
        elif is_ladder:
            checkpoints = sorted(pathlib.Path(path).glob("CodeLST_states_*.pt"))
            assert len(checkpoints) == self.args.tensor_model_parallel_size
            checkpoint_name = checkpoints[self.args.rank]
        elif self.args.tensor_model_parallel_size == 1 and self.args.rank < self.args.tensor_model_parallel_size:
            checkpoint_name = os.path.join(path, f"{self.args.model_name}")
            assert os.path.isfile(checkpoint_name)
        elif self.args.tensor_model_parallel_size == 1 and self.args.rank >= self.args.tensor_model_parallel_size:
            checkpoint_name = os.path.join(second_path, f"{self.args.model_name}")
            assert os.path.isfile(checkpoint_name)
        elif self.args.rank < self.args.tensor_model_parallel_size:
            checkpoints = sorted(pathlib.Path(path).glob(f"{self.args.model_name}_states_*"))
            assert len(checkpoints) == self.args.tensor_model_parallel_size
            checkpoint_name = checkpoints[self.args.rank]
        elif self.args.rank >= self.args.tensor_model_parallel_size:
            checkpoints = sorted(pathlib.Path(second_path).glob(f"{self.args.model_name}_states_*"))
            assert len(checkpoints) == self.args.tensor_model_parallel_size
            checkpoint_name = checkpoints[self.args.rank-self.args.tensor_model_parallel_size]
        else:
            raise ValueError

        # Load the checkpoint.
        print(f"rank_{self.args.rank} load: {checkpoint_name}", flush=True, file=sys.stderr)
        state_dict = torch.load(checkpoint_name, map_location="cpu")
        with open(checkpoint_name, 'rb') as f:
            self.checkpoint_head_hash = hashlib.sha256(f.read(1024*1024)).hexdigest()  # 计算hash

        # Set iteration.
        iteration = state_dict.get("iteration", 0)

        if "model" in state_dict:
            state_dict = state_dict["model"]
        if "module" in state_dict:
            state_dict = state_dict["module"]

        # Model.
        model[0].load_state_dict(state_dict, strict=True)

        print_rank_0(
            f"successfully loaded checkpoint from {path} "
            f"at iteration {iteration}"
        )

        return iteration

    def get_voting(self, stem_probs, ladder_probs, v_type=0):

        s_id = np.argmax(stem_probs)
        l_id = np.argmax(ladder_probs)
        need_print = s_id != l_id

        force_stem = 0
        if l_id >= len(self.tokenizer):
            force_stem = 1
        force_stem = self.sync_sess_id(force_stem)
        
        # "switch"
        if v_type == 0:
            if np.max(stem_probs) >= np.max(ladder_probs) or force_stem > 0:
                if AIXDEBUG_LADDER and need_print:
                    print_rank_0(f"switch mode, and selecting model one")
                r = stem_probs
            else:
                if AIXDEBUG_LADDER and need_print:
                    print_rank_0(f"switch mode, and selecting model Two")
                r = ladder_probs
        # "mixture"
        elif v_type == 1:
            r = (stem_probs+ladder_probs) / 2
        # "stem"
        elif v_type == 2:
            r = stem_probs
        # "ladder"
        elif v_type == 3:
            r = ladder_probs

        if AIXDEBUG_LADDER and need_print:
            top_k_idx = np.argpartition(stem_probs, kth=-3)[-3:]
            top_k_idx = top_k_idx[np.argsort(-stem_probs[top_k_idx])].tolist()
            top_k_probs = stem_probs[top_k_idx].tolist()
            print_rank_0(f"stem_probs: {', '.join([str(top_k_idx[i]) + ' ' + self.tokenizer.decode([top_k_idx[i]]) + ' ' + str(top_k_probs[i]) for i in range(len(top_k_idx))])}".replace("\n", "<ENTER>"))

            top_k_idx = np.argpartition(ladder_probs, kth=-3)[-3:]
            top_k_idx = top_k_idx[np.argsort(-ladder_probs[top_k_idx])].tolist()
            top_k_probs = ladder_probs[top_k_idx].tolist()
            print_rank_0(f"ladder_probs: {', '.join([str(top_k_idx[i]) + ' ' + self.tokenizer.decode([top_k_idx[i]]) + ' ' + str(top_k_probs[i]) for i in range(len(top_k_idx))])}")
        return r

    def predict_batch(self, data):
        """
        data: [
            token_ids,
            common_len,
            voting_type,
            uid
        ]
        """
        
        # common_len = int(data[1].cpu().numpy())
        # voting_type = int(data[2].cpu().numpy())
        common_len = int(data[1].cpu().item())
        voting_type = int(data[2].cpu().item())
        uid = data[3]

        if common_len == 0 and self.kv_cache is not None:
            cache = self.kv_cache.get(uid=uid, token_ids=data[0])
            if cache is not None and len(cache) == 3 and cache[-1] is not None:
                data[0] = cache[0]
                common_len = cache[1]
                self.cached_memory[:,:, : common_len,:,:,:] = cache[-1].cuda()
            elif cache is not None and len(cache) == 3 and cache[-1] is None:
                data[0] = cache[0]
                common_len = cache[1]
        input_seq_len = data[0].shape[-1]
        
        with torch.no_grad():
            tokens_ids = data[0].clone().detach().cuda()
            position_ids = self.position_ids[:, common_len:common_len + input_seq_len]
            attention_mask = self.attention_mask[common_len:common_len + input_seq_len, :common_len + input_seq_len]
            caches = None if common_len < 1 else self.cached_memory[:, :, :common_len, :, :, :]
            
            # print_rank_0(f"{tokens_ids.shape=}")
            # print_rank_0(f"{position_ids.shape=}")
            # print_rank_0(f"{attention_mask.shape=}")
            # if caches is not None:
            #     print_rank_0(f"{caches.shape=}")
            
            if self.ladder_predictor is None and not self.is_multi_backbone:
                logits, caches = self.predictor(
                    tokens_ids,                 # shape: [bsz, 1024]
                    position_ids,               # shape: [bsz, 1024]
                    attention_mask,             # shape: [1024, 1024], [query_len, key_len]
                    inference_params=caches,    # cached key and value in last step
                    get_key_value=True,         # return cached key and value
                    infer_logits=True
                )
                logits = logits.view(1, -1).contiguous().to(torch.float32)
                cacha_len = caches.shape[2]
                self.cached_memory[:,:, common_len: common_len + cacha_len,:,:,:] = caches
                if self.kv_cache is not None:
                    self.kv_cache.update(uid=uid, token_ids=tokens_ids.cpu().numpy()[0], last_common_len=common_len, kv_cache=caches.cpu())
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                return [np.squeeze(probs).astype(np.float32)]
            elif self.ladder_predictor is None and self.is_multi_backbone:
                logits, caches = self.predictor(
                    tokens_ids,                 # shape: [bsz, 1024]
                    position_ids,               # shape: [bsz, 1024]
                    attention_mask,             # shape: [1024, 1024], [query_len, key_len]
                    inference_params=caches,    # cached key and value in last step
                    get_key_value=True,         # return cached key and value
                    infer_logits=True
                )
                logits = logits.contiguous().view(1, -1).to(torch.float32)
                cacha_len = caches.shape[2]
                self.cached_memory[:,:, common_len: common_len + cacha_len,:,:,:] = caches
                probs = torch.softmax(logits, dim=-1)

                if self.args.rank != self.args.world_size - 1:
                    second_probs = torch.zeros_like(probs, dtype=torch.float32)
                else:
                    second_probs = probs
                
                torch.distributed.broadcast(
                    second_probs,
                    self.args.world_size - 1,
                )

                probs = self.get_voting(probs.squeeze().cpu().numpy(), second_probs.squeeze().cpu().numpy())

                return [np.squeeze(probs).astype(np.float32)]
            else:
                stem_logits, caches, hidden_states, embed = self.predictor(
                tokens_ids,                     # shape: [bsz, 1024]
                    position_ids,               # shape: [bsz, 1024]
                    attention_mask,             # shape: [1024, 1024], [query_len, key_len]
                    inference_params=caches,    # cached key and value in last step
                    get_key_value=True,         # return cached key and value
                    get_hidden=True
                )
                # print_rank_0(f"stem_logits: {stem_logits.dtype}, {stem_logits.shape}")
                # print_rank_0(f"caches: {caches.dtype}, {caches.shape}")
                # print_rank_0(f"hidden_states: {hidden_states[0].dtype}, {hidden_states[0].shape}")
                # print_rank_0(f"embed: {embed.dtype}, {embed.shape}")
                stem_logits = stem_logits[:, -1].view(1, -1).contiguous().to(torch.float32)
                stem_probs = torch.softmax(stem_logits, dim=-1).cpu().numpy()

                cacha_len = caches.shape[2]
                self.cached_memory[:,:, common_len: common_len + cacha_len,:,:,:] = caches

                if self.args.ladder_type == "SwiGLU":
                    # [seq_len, bsz, hsz] -> [1, bsz, hsz]
                    hidden_states = [h[-1:, :, :] for h in hidden_states]
                    ladder_logits = self.ladder_model(hidden_states, embed, None, None, False)
                    ladder_logits = ladder_logits.transpose(0, 1).contiguous()
                elif self.args.ladder_type == "attention":
                    caches = None if common_len < 1 else self.cached_ladder_memory[:, :, :common_len, :, :, :]
                    ladder_logits, caches = self.ladder_model(
                        hidden_states=hidden_states, embed_table=embed, 
                        attention_mask=attention_mask, inference_params=caches, 
                        get_key_value=True, infer_mode=True)
                    cacha_len = caches.shape[2]
                    self.cached_ladder_memory[:,:, common_len: common_len + cacha_len,:,:,:] = caches
                    ladder_logits = ladder_logits.transpose(0, 1).contiguous()
                else:
                    raise ValueError
                ladder_probs = torch.softmax(ladder_logits, dim=-1).cpu().numpy()
                return [self.get_voting(np.squeeze(stem_probs), np.squeeze(ladder_probs), voting_type).astype(np.float32)]

    def weighted_nucleus_sampling(self,
            top_k_probs: List[float],
            top_k_idx: List[int],
    ) -> Tuple[int, float]:
        top_k = 1
        accumulate_probs = 0
        while True:
            probs = top_k_probs[top_k - 1]
            accumulate_probs += probs
            if accumulate_probs < 0.8 and top_k <= 3 and probs > 0.1:
                top_k += 1
            elif top_k > 1:
                top_k -= 1
                break
            else:
                break

        if top_k > 1:
            weights = softmax(x=top_k_probs[:top_k])
            candidates = top_k_idx[:top_k]
            prediction_id = self.np_rand.choice(candidates,
                                                size=1,
                                                p=weights).tolist()[0]
            k = candidates.index(prediction_id)
            prediction_probs = top_k_probs[k]
            print_rank_0(
                f"candidate: {top_k}, {top_k_probs[:top_k]}, {top_k_idx[:top_k]}, select {prediction_id}"
            )
            return prediction_id, prediction_probs
        return top_k_idx[0], top_k_probs[0]

    def init_strategies(self, *, strategies: model_pb2.Strategies) -> None:
        for strategy_id, strategy in strategies.strategies.items():
            if strategy.type == 1:
                self.strategy_map[strategy_id] = PenalizationStrategy(args=strategy.args, tokenizer=self.tokenizer)
            elif strategy.type == 2:
                self.strategy_map[strategy_id] = EOSPreferenceStrategy(tokenizer=self.tokenizer)
            elif strategy.type == 3:
                self.strategy_map[strategy_id] = L1NormalizationStrategy()
            else:
                raise AssertionError(f"grpc strategy {strategy.type} is not supported on this version")


    def predict(self, token_ids: List[int], common_len: int, voting_type: int = 0, sampling_type: str = "no", uid: str = "only_my_rail_gun",
                strategy_ids: Optional[List[int]] = None, created_just_now: bool = False) -> Tuple[List[int], List[float], List[List[int]], List[List[float]]]:
        if created_just_now:
            self.strategy_state = {}
        common_len_input = common_len
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        try:
            # voting type, work for load ladder model: (0, switch), (1, mixture), (2, stem), (3, ladder)
            voting_type_nda = np.array([voting_type], dtype=np.int64)
            common_len_nda = np.array([common_len]).astype("int64")
            token_ids_nda = np.array([token_ids], dtype=np.int64)



            if voting_type_nda.item() not in {0, 1, 2, 3}:
                raise ValueError(f"voting_type_nda is wrong: {voting_type_nda} not in 0-3")

            max_pad_len = max(token_ids_nda.shape[-1], 128)
            max_pad_len = self.sync_sess_id(max_pad_len)
            token_ids_nda, tokens_id_len = self.pad_batch(token_ids_nda, max_seq_len=max_pad_len)

            context_tensor = torch.tensor(token_ids_nda, dtype=torch.int64, device='cuda')
            context_tensor_length = torch.tensor(tokens_id_len, dtype=torch.int64, device='cuda')
            context_common_len = torch.tensor(common_len_nda, dtype=torch.int64, device='cuda')
            context_voting_type = torch.tensor(voting_type_nda, dtype=torch.int64, device='cuda')

            torch.distributed.broadcast(
                context_tensor,
                0,
            )
            torch.distributed.broadcast(
                context_tensor_length,
                0,
            )
            torch.distributed.broadcast(
                context_common_len,
                0,
            )
            torch.distributed.broadcast(
                context_voting_type,
                0,
            )
            
            uid = self.sync_model_dir(uid)
            tokens_id_len = context_tensor_length.min().item()
            batch = [context_tensor[:, :tokens_id_len], context_common_len, context_voting_type, uid]
            
            
            _start = time.perf_counter()
            out = self.predict_batch(batch)
            _end = time.perf_counter()
            if PRINT_TIME:
                print_rank_0(f"\nself.predict_batch costs {(_end - _start) * 1000:.2f}ms")
            # 预测结果
            # shape: [bsz, vocab_size] => [vocab_size]
            out = out[0]
            for strategy_id in strategy_ids or []:
                if strategy_id in self.strategy_map:
                    self.strategy_map[strategy_id].predict_strategy_eval(input_ids=token_ids, probs=out, strategy_state=self.strategy_state)
            top_k_idx = np.argpartition(out, kth=-5)[-5:]
            top_k_idx = top_k_idx[np.argsort(-out[top_k_idx])].tolist()
            top_k_probs = out[top_k_idx].tolist()
            predict_id, predict_prob = top_k_idx[0], top_k_probs[0]
            # 预测结果处理
            if sampling_type == "nucleus" and common_len_input != 0:
                predict_id, predict_prob = self.weighted_nucleus_sampling(top_k_probs=top_k_probs, top_k_idx=top_k_idx)
            return [predict_id], [predict_prob], [top_k_idx], [top_k_probs]
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(e)


def test():
    
    # model_dir = "/data3/aix2_Base"
    # attn_type = "multiquery"
    model_dir = "/data3/aix2_base_v2/"
    attn_type = "groupedquery"
    
    tokenizer = GPTTokenizer(
        vocab_dir=model_dir
        )
    

    tokens = tokenizer.encode("""import dataclasses
import os
import os.path as osp
import random
import time

import fire

from lmdeploy import turbomind as tm
from lmdeploy.model import MODELS
from lmdeploy.turbomind.tokenizer import Tokenizer, GPTTokenizer

os.environ['TM_LOG_LEVEL'] = 'INFO'""", post_context="")
    
    
    aix_config = {
        "num_layers": 40, "hidden_size": 6144, "num_attention_heads": 48,
        "bf16": True, "fp16": False, "padded_vocab_size": 49152, "quant_type": 1,
        "micro_batch_size": 1, "attention_head_type": attn_type,
        "use_cpu_initialization": True, "use_flash_attn": True, "max_position_embeddings": 8192
        
    }
    
    if is_ampere():
        aix_config["bf16"] = True
        aix_config["fp16"] = False
        aix_config["use_flash_attn"] = True
    else:
        aix_config["bf16"] = False
        aix_config["fp16"] = True
        aix_config["use_flash_attn"] = False
    
    if aix_config.get("quant_type", 0) == 8:
        aix_config["bf16"] = False
        aix_config["fp16"] = True
    
    initialize_aixmeg(
        extra_args_provider=add_code_generation_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
        },
        aix_config=aix_config
    )
    args = get_args()
    args.model_dir = model_dir
    # args.ladder_load_path = "/data/server_megatron/saved_ladder_model/iter_6500"

    sess = Predictor(args=args)

    last_step_size = 0
    if torch.distributed.get_rank() == 0:
        terminate_runs = sess.sync_type_info(0)
        if terminate_runs > 0:
            return
        output_vals = sess.predict(
            np.array([tokens[:-1]], dtype='int32'),
            np.array([0], dtype='int32')
            )

        terminate_runs = sess.sync_type_info(0)
        if terminate_runs > 0:
            return
        output_vals = sess.predict(
            np.array([tokens[-1:]], dtype='int32'), 
            np.array([len(tokens)-1], dtype='int32')
            )
        
        top_k_idx = output_vals[2][0]
        top_k_probs = output_vals[3][0]
        print_rank_0(f"step_cache: {', '.join([str(top_k_idx[i]) + ' @@' + tokenizer.decode([top_k_idx[i]]) + '@@ ' + str(top_k_probs[i]) for i in range(len(top_k_idx))])}".replace("\n", "<ENTER>").replace("@@", "\'"))

        """ 
            aix2_base_v2, grouped-query, flash-attan:
                step_cache: 17228 '<ENTER>   ' 0.940068781375885, 42013 '<ENTER>' 0.03216738626360893, 3812 '<ENTER><ENTER>   ' 0.013409360311925411, 3827 '<ENTER>  ' 0.0019318015547469258, 23891 '<ENTER> ' 0.0019318015547469258
                step:       17228 '<ENTER>   ' 0.943885326385498, 42013 '<ENTER>' 0.028502866625785828, 3812 '<ENTER><ENTER>   ' 0.013463800773024559, 23891 '<ENTER> ' 0.0019396443385630846, 3827 '<ENTER>  ' 0.0018221273785457015
            aix2_base_v2, grouped-query, core-attan:
                step_cache: 17228 '<ENTER>   ' 0.9453542828559875, 42013 '<ENTER>' 0.028547227382659912, 3812 '<ENTER><ENTER>   ' 0.011900254525244236, 23891 '<ENTER> ' 0.0019426631042733788, 45061 '<ENTER>    <ENTER>   ' 0.0018249631393700838
                step: 17228 '<ENTER>   ' 0.9387193322181702, 42013 '<ENTER>' 0.0321212075650692, 3812 '<ENTER><ENTER>   ' 0.013390111736953259, 23891 '<ENTER> ' 0.0021858755499124527, 3827 '<ENTER>  ' 0.0020534403156489134
        """
        
        terminate_runs = sess.sync_type_info(0)
        if terminate_runs > 0:
            return
        output_vals = sess.predict(
            np.array([tokens], dtype='int32'),
            np.array([0], dtype='int32')
            )

        top_k_idx = output_vals[2][0]
        top_k_probs = output_vals[3][0]
        
        print_rank_0(f"step: {', '.join([str(top_k_idx[i]) + ' @@' + tokenizer.decode([top_k_idx[i]]) + '@@ ' + str(top_k_probs[i]) for i in range(len(top_k_idx))])}".replace("\n", "<ENTER>").replace("@@", "\'"))

        terminate_runs = sess.sync_type_info(1)
        if terminate_runs > 0:
            return
    else:
        while True:
            terminate_runs = sess.sync_type_info(0)
            if terminate_runs > 0:
                return
            tokens = [0] * 4
            output_vals = sess.predict(
                np.array([tokens[:-1]], dtype='int32'),
                np.array([0], dtype='int32')
                )



def test_inference_time():
    
    model_dir = "/data3/aix2_base_v2/"
    attn_type = "groupedquery"

    # model_dir = "/data3/aix2_Base"
    # attn_type = "multiquery"
    
    aix_config = {
        "num_layers": 40, "hidden_size": 6144, "num_attention_heads": 48,
        "bf16": True, "fp16": False, "padded_vocab_size": 49152,  
        "micro_batch_size": 1, "attention_head_type": attn_type, "quant_type": -1,
        "use_cpu_initialization": True, "use_flash_attn": True, "max_position_embeddings": 8192
        
    }
    
    if is_ampere():
        aix_config["bf16"] = True
        aix_config["fp16"] = False
        aix_config["use_flash_attn"] = True
    else:
        aix_config["bf16"] = False
        aix_config["fp16"] = True
        aix_config["use_flash_attn"] = False
    
    if aix_config.get("quant_type", 0) == 8:
        aix_config["bf16"] = False
        aix_config["fp16"] = True
    
    initialize_aixmeg(
        extra_args_provider=add_code_generation_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
        },
        aix_config=aix_config
    )
    args = get_args()
    args.model_dir = model_dir

    sess = Predictor(args=args)

    if torch.distributed.get_rank() == 0:

        count_ = 100
        count = 0
        start = time.perf_counter()
        pre_len = 1024
        run_len = 0
        while True:
            
            terminate_runs = sess.sync_type_info(0)
            if terminate_runs > 0:
                return
            # print_rank_0("\n\n\n")
            output_vals = sess.predict(
                np.array([np.random.randint(0,20000, size=pre_len)], dtype='int32'),
                np.array([0], dtype='int32'),
                )
            
            for i in range(run_len):
                terminate_runs = sess.sync_type_info(0)
                if terminate_runs > 0:
                    return
                output_vals = sess.predict(
                    np.array([np.random.randint(0,20000, size=1)], dtype='int32'),
                    np.array([pre_len+i], dtype='int32')
                    )
            count += 1
            
            if count > count_:
                print_rank_0(f"avg time: {(time.perf_counter()-start) * 1000 / (run_len+1) / count:.2f}ms")
                terminate_runs = sess.sync_type_info(1)
                if terminate_runs > 0:
                    return
    else:
        while True:
            terminate_runs = sess.sync_type_info(0)
            if terminate_runs > 0:
                return
            tokens = [0] * 4
            output_vals = sess.predict(
                np.array([tokens[:-1]], dtype='int32'),
                np.array([0], dtype='int32')
                )

if __name__ == "__main__":
    test()
    # test_inference_time()
