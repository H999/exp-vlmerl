from operator import attrgetter, itemgetter
from torch import Tensor
from tensordict import TensorDict
from torch.nn.functional import kl_div
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import io
import torch
import numpy as np

from model_geneticCNN import Individual


def js_div(p: Tensor, w: Union[Literal["mean", "asc", "desc"], Tensor]="mean", reduction: Literal["sum", "mean", "per", "none"]="sum", norm=False, eps=1e-10):
    "compute the Jensen-Shannon divergence between multiple probability distributions"
    d = list(range(1, p.ndim))
    p = minmaxnorm(p, d, True).clamp(min=eps)
    p = p / p.sum(dim=d, keepdim=True)

    if w=="mean":
        pi = torch.full((p.size(0),), fill_value= 1 / p.size(0))
    elif w in ["asc", "desc"]:
        pi = normalized_rank(torch.arange(p.size(0)))
        if w == "desc":
            pi = pi.flip(0)
    elif isinstance(w, Tensor):
        if w.size(0) != p.size(0):
            raise ValueError(f"w size ({w.size(0)}, ...) different with p size ({p.size(0)}, ...)")
        if w.sum() != 1 and not norm:
            raise ValueError(f"total sum of w must = 1 not {w.sum()} if {norm=}")
        pi = w
    else:
        raise ValueError(f"Unsupport {w=}")
    pi = pi.to(p.device)
    jsd = kl_div(prod_rewards(p, pi, norm=norm).log(), p, reduction='none')

    if reduction == "none":
        return jsd
    else:
        jsd = jsd.sum(d) * pi / torch.tensor(p.size(0)).log()
        if reduction == "per":
            return jsd
        elif reduction == "mean":
            return jsd.mean()
        elif reduction == "sum":
            return jsd.sum()
        else:
            raise ValueError(f"Unsupport {reduction=}")


def normalized_rank(rewards: Tensor, tau: float = 2., f: Literal["z-score", "zero-centered", "geometric-decay", "temp-linear", "min-max"] = "z-score", use_sftm: bool = True) -> Tensor:
    "rank-based reward normalization"
    ranks = rewards.argsort().argsort().float()
    if f == "z-score":
        norm_ranks = tau * (ranks - ranks.mean()) / ranks.std()
    elif f == "zero-centered":
        norm_ranks = tau * (ranks / (ranks.numel() - 1.) - 0.5)
    elif f == "geometric-decay":
        norm_ranks = tau**(ranks - ranks.numel())
    elif f == "temp-linear":
        norm_ranks = tau * ranks
    elif f == "min-max":
        norm_ranks = minmaxnorm(ranks)
    else:
        raise ValueError(f"Unknow {f=}")
    return norm_ranks.softmax(0) if use_sftm else norm_ranks


def minmaxnorm(arr: Union[Tensor, np.ndarray], dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> Union[Tensor, np.ndarray]:
    "norm arr from min -> max into  0 -> 1"
    if dim is None:
        min_val = arr.min()
        max_val = arr.max()
    else:
        min_val = arr.amin(dim=dim, keepdim=keepdim)
        max_val = arr.amax(dim=dim, keepdim=keepdim)
    return (arr - min_val) / (max_val - min_val + 1e-10)


def roll(ts: Union[TensorDict, Tensor], sh: int = 1):
    "replacement for torch.roll to use roll in dim 0 with Tensor & TensorDict"
    ts[:] = ts[(torch.arange(ts.size(0)) - sh) % ts.size(0)]


def scale_eta(ts: Tensor) -> Tensor:
    "shift the exponent of tensor follow largest number inside tensor (e.g.: [2e-10, 2e-5] => [2e-6, 2e-1])"
    if not ts.ndim or ts.max() >= 1.:
        return torch.ones_like(ts)
    return ts.abs() / 10.**ts.abs().max().log10().nan_to_num(0.,0.,0.).ceil()


def close_together(input: Union[Tensor, TensorDict], other: Union[Tensor, TensorDict], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False, return_mask: bool = False, reduce_mode: Literal["all", "any"] = "all") -> bool:
    "replacement for torch.allclose to use with Tensor & TensorDict"
    equal_nan_mask = (input.isnan() & other.isnan()) & equal_nan
    abs_diff = (input - other).abs()
    tolerance = atol + rtol * other.abs()
    comparison_mask = (abs_diff <= tolerance) | equal_nan_mask
    if return_mask:
        return comparison_mask
    return getattr(comparison_mask, reduce_mode)()


def get_inds_attr(inds: Union[List[Individual], Individual], attrs: Union[List[str], str], index: Union[List[int], int] = None, to_tensor: bool = False) -> Union[List[Any], Tensor]:
    "attrs get by operator.attrgetter"
    pbest_handle = ''
    if attrs.startswith("pbest"):
        index = 0
        pbest_handle = attrs.removeprefix("pbest").removeprefix(".")
        attrs = "pbest"
    inds = [inds] if isinstance(inds, Individual) else inds
    attrs= [attrs] if isinstance(attrs, str) else attrs
    res = map(attrgetter(*attrs), inds)
    if index is not None:
        index = [index] if isinstance(index, int) else index
        res = map(itemgetter(*index), res)
    if pbest_handle != '':
        res = get_inds_attr(res, pbest_handle)
    res = list(res)
    return torch.tensor(res) if to_tensor else res


def update_gen(inds: Union[List[Individual], Individual], new_gens: Union[List[Tuple[str]], Tuple[str], Tensor]) -> None:
    if isinstance(inds, Individual):
        inds = [inds]
    if isinstance(new_gens[0], str):
        new_gens = [new_gens]
    elif isinstance(new_gens, Tensor):
        new_gens = revert_gen(new_gens, inds[0].Stages.num_stages)
    [ind.Stages.update_stages_state(tuple(new_gen)) for ind, new_gen in zip(inds, new_gens)]


def flat_gen(gens: Union[Individual, List[Individual], List[Tuple[str]], Tuple[str]]) -> Tensor:
    "convert list of string(s) gen(s) into tensor for computing"
    if isinstance(gens, Individual) or isinstance(gens[0], Individual):
        return flat_gen(get_inds_attr(gens, "Stages.gen"))
    elif isinstance(gens[0], list) or isinstance(gens[0], tuple):
        return torch.stack(list(map(flat_gen, gens)))
    elif isinstance(gens[0], str):
        return torch.tensor(np.fromiter("".join(gens), dtype=np.float32))
    raise ValueError(f"Unknow type {type(gens[0])}")


def revert_gen(flattened_gens: Tensor, num_stages: Tuple[int]) -> List[Tuple[str]]:
    "revert tensor of gen(s) into list of string(s)"
    split_points = np.array(num_stages)
    split_points = (split_points * (split_points - 1) / 2).cumsum()
    if flattened_gens.size(-1) != split_points[-1]:
        raise ValueError(f"size of gen(s) ({flattened_gens.size(-1)}) with num_stage(s) different ({split_points[-1]})")
    return list(map(lambda x: tuple(map(lambda y: ''.join(y.where(y.any(),
        y.scatter(0, torch.randint(y.size(0), [1]).to(y.device), 1)
    ).cpu().numpy().astype(str)), x)), zip(*(
        flattened_gens if flattened_gens.ndim > 1 else flattened_gens.unsqueeze(0)
    ).int().hsplit(split_points[:-1].astype(int).tolist()))))


def prod_rewards(tensor: Union[Tensor, Individual, List[Individual]], rewards: Tensor = None, norm = False, tau: float = 2., f: Literal["z-score", "zero-centered", "geometric-decay", "temp-linear", "min-max"] = "z-score", use_sftm: bool = True) -> Tensor:
    """calc dist of tensor

    if pass list of Individual will use flat_gen to compute dist of gen
        if no pass rewards will use get_fitness then auto norm it as rewards
    """
    if isinstance(tensor, Individual):
        tensor = [tensor]
    if isinstance(tensor[0], Individual):
        if rewards is None:
            rewards = get_inds_attr(tensor, "fitness", to_tensor=True)
            norm = True
        tensor = flat_gen(tensor)
    if norm:
        rewards = normalized_rank(rewards, tau, f, use_sftm)
    return (tensor * rewards.to(tensor.device).view(-1,*[1]*(tensor.ndim-1))).sum(0)


def tsd_iter_along_axis0(tsd: TensorDict, fn: Union[Callable[[Tensor], Tensor], Callable[[str, Tensor], Tensor]], tsd_set: Optional[TensorDict] = None, use_key=False, use_set=True) -> None:
    """
    If not provide tsd_set: Applies a callable to all keys stored in the :term:`TensorDict <tsd>` 
                            and re-writes them in-place along axis 0 of batch size
    If     provide tsd_set: Applies a callable to all keys stored in the :term:`TensorDict <tsd>`
                            and writes them in-place of :term:`TensorDict <tsd_set>` (assume them have same keys)
                            **careful with batch size**
    """
    if tsd_set is None:
        tsd_set = tsd
    use_key_fn = (lambda k, v: fn(k, v)) if use_key else (lambda _, v: fn(v))
    apply_fn = (lambda k, v: tsd_set.set_(k, use_key_fn(k, v))) if use_set else use_key_fn
    [apply_fn(k, v) for k, v in tsd.items(include_nested=True, leaves_only=True)]


def serialize_model(model: torch.nn.Module, make_tensor=True, device="cpu", get_len=False) -> Union[Tensor, int]:
    """ Serialize model state_dict to byte buffer as a tensor"""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    serialized_state_dict = buffer.getvalue()
    if get_len: return len(serialized_state_dict)
    if make_tensor:
        serialized_state_dict = torch.tensor(bytearray(serialized_state_dict), dtype=torch.uint8, device=torch.device(device))
    return serialized_state_dict


def deserialize_model(buffer: Tensor, device="cpu") -> dict[str, Any]:
    """ Deserialize tensor byte buffer back to state_dict """
    state_dict_bytes = buffer.cpu().numpy().tobytes()
    buffer = io.BytesIO(state_dict_bytes)
    return torch.load(buffer, map_location=torch.device(device))
