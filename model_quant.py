import os
import json
import argparse
import warnings
from functools import partial

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics.downstream import MR_GPTQ_TASKS, TASKS, check_downstream_setup, evaluate_downstream
from src.metrics.perplexity import compute_perplexity
from src.transforms.transforms import TRANSFORMS
from src.quantization.quant_ops import NVFP_GROUPSIZE, MXFP_GROUPSIZE
from src.quantization.quant_args import ScalePrecision
from src.quantization.qconfig import prepare_quantization_config
from src.quantization import rtn_quantization, gptq_quantization
from src.utils.common_utils import clear_device_cache, fix_seed
from src.utils.data_utils import get_data, get_wikitext2, get_c4_eval

try:
    import wandb
except ImportError:
    wandb = None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def positive_int(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got '{value}'") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got '{value}'")
    return number


def auto_or_int(value):
    return value if value == "auto" else positive_int(value)


def export_quantized_model(model, quantized_state_dict, non_quantized_state_dict, args):
    config = model.config
    # Prepare directory to save model
    os.makedirs(args.save_path, exist_ok=True)

    blocks = model.model.layers

    # State dict to save
    model_state_dict = {}

    for block_idx, block in enumerate(blocks):
        prefix = f"model.layers.{block_idx}."
        for k, v in block.state_dict().items():
            layer_name, param_name = k.rsplit(".", 1)
            if f"{prefix}{layer_name}" in quantized_state_dict and param_name == "weight":
                for k_compr, v_compr in quantized_state_dict[f"{prefix}{layer_name}"].items():
                    model_state_dict[f"{prefix}{layer_name}.{k_compr}"] = v_compr.cpu()
            elif f"{prefix}{k}" in non_quantized_state_dict:
                model_state_dict[f"{prefix}{k}"] = non_quantized_state_dict[f"{prefix}{k}"].cpu()
            else:
                model_state_dict[f"{prefix}{k}"] = v.cpu()

    # Add non_quantized_state_dict block parameters (dict is non-empty for blockwise_qat)
    model_state_dict.update(non_quantized_state_dict)

    # Process all remaining blocks
    tie_word_embeddings = getattr(model.config, "tie_word_embeddings", False)

    for k, v in model.state_dict().items():
        if not (k.startswith("model.layers") or (k == "lm_head.weight" and tie_word_embeddings)):
            model_state_dict[k] = v.cpu()

    # Split checkpoint into shards
    current_shard_size = 0
    current_shard = {}
    shards = []

    for k, v in model_state_dict.items():
        tensor_size = v.numel() * v.element_size()
        if current_shard_size + tensor_size > args.max_shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_shard_size = 0

        if tensor_size > args.max_shard_size:
            shards.append({k: v})
            continue
        
        current_shard[k] = v
        current_shard_size += tensor_size

    # Dump last shard if it is not empty
    if len(current_shard) > 0:
        shards.append(current_shard)

    safetensors_index = {}
    num_shards = len(shards)
    max_digits = len(str(max(num_shards, 1)))

    # Save shards
    for shard_idx, shard in enumerate(shards):
        current_shard_path = f"model-{str(shard_idx+1).zfill(max_digits)}-of-{str(num_shards).zfill(max_digits)}.safetensors"
        save_file(shard, os.path.join(args.save_path, current_shard_path))
        for k in shard:
            safetensors_index[k] = current_shard_path

    # Save safetensors index
    with open(os.path.join(args.save_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": safetensors_index}, f)

    # Add quantization metadata
    config.quantization_config = prepare_quantization_config(
        args.hadamard_group_size, 
        args.format,
        pseudoquantization=(args.export_quantized_model == "pseudoquant")
    )
    # Save configs
    config.save_pretrained(args.save_path)
    model.generation_config.save_pretrained(args.save_path)

    
def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to quantized model.",
    )
    # Data params
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="The name or path to the calibration dataset.",
    )
    parser.add_argument(
        "--sequence_length", 
        default=2048, 
        type=int, 
        help="Length of calibration sequences."
    )
    parser.add_argument(
        "--num_sequences", 
        default=1024, 
        type=int, 
        help="Number of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        "--format",
        type=str,
        default="int",
        choices=["int", "fp", "nvfp", "mxfp"],
        help="Quantization format.",
    )
    parser.add_argument(
        "--scale_precision",
        type=str,
        default="fp16",
        choices=["fp16", "e8m0", "e4m3"],
        help="Scale precision.",
    )
    parser.add_argument(
        "--w_granularity",
        type=str,
        default="group",
        choices=["tensor", "channel", "group"],
        help="Weight quantization granularity.",
    )
    parser.add_argument(
        "--w_bits",
        type=int,
        required=True,
        help="Weight quantization bitwidth.",
    )
    parser.add_argument(
        "--w_group_size",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--w_observer",
        type=str,
        default="minmax",
        choices=["minmax", "mse"],
        help="Weight observer.",
    )
    parser.add_argument(
        "--a_bits",
        type=int,
        default=16,
        help="Activation quantization bitwidth.",
    )
    parser.add_argument(
        "--a_granularity",
        type=str,
        default="group",
        choices=["tensor", "channel", "group"],
        help="Activation quantization granularity.",
    )
    parser.add_argument(
        "--a_group_size",
        type=int,
        default=None,
        help="How many activation columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--a_observer",
        type=str,
        default="minmax",
        choices=["minmax"],
        help="Activation observer.",
    )
    parser.add_argument(
        "--export_quantized_model",
        type=str,
        default="",
        choices=["", "realquant", "pseudoquant"],
        help="Whether export quantized model in realquant or pseudoquant format.",
    )
    # GPTQ params
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="Run GPTQ quantization.",
    )
    parser.add_argument(
        "--quantization_order",
        type=str,
        default="default",
        choices=["default", "activation"],
        help="Weigth quantization order in GPTQ.",
    )
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    # Transform params
    parser.add_argument(
        "--transform_class",
        type=str,
        default="identity",
        choices=TRANSFORMS.keys(),
        help="The transform class."
    )
    parser.add_argument(
        "--hadamard_group_size",
        type=int,
        default=128,
        help="Hadamard group size"
    )
    # Logging params
    parser.add_argument(
        "--log_wandb",
        action="store_true",
        help="Whether to log to wandb."
    )
    # Misc params
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed.")
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--amp", action="store_true", help="whether to enable fp16 autocasting.")
    parser.add_argument("--compile", action="store_true", help="whether to use torch.compile.")
    parser.add_argument("--fuse_global_scale", action="store_true", help="whether to fuse global scale in qkv and gate_up.")
    # Eval params
    parser.add_argument("--eval_perplexity", action="store_true", help="whether to eval perplexity after quantization.")
    parser.add_argument("--eval_openllm", action="store_true", help="Run the MR-GPTQ downstream task suite after quantization.")
    parser.add_argument("--ppl_limit", type=positive_int, help="Cap the number of PPL windows per dataset for smoke tests.")
    parser.add_argument("--limit", type=positive_int, help="Cap examples per downstream task for smoke tests.")
    # LM eval params
    parser.add_argument(
        "--lm_eval_seed", type=int,
        help="Seed Python, NumPy, Torch and few-shot sampling. Defaults to the lm-eval protocol.",
    )
    parser.add_argument(
        "--lm_eval_batch_size",
        type=auto_or_int,
        default="auto",
        help="Cap the downstream batch size. 'auto' uses the MR-GPTQ per-task batch sizes.",
    )
    parser.add_argument(
        "--lm_eval_tasks",
        nargs="+",
        choices=TASKS,
        default=list(MR_GPTQ_TASKS),
        help="Downstream tasks to evaluate. Defaults to the ten MR-GPTQ tasks."
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Whether to disable thinking mode for Qwen3.",
    )
    # Save params
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save quantized model",
    )
    parser.add_argument(
        "--max_shard_size", 
        type=int, 
        default=5 * 1024 * 1024 * 1024, 
        help="Maximum shard size in bytes."
    )
    # Parse arguments
    args = parser.parse_args()
    if args.eval_perplexity and args.sequence_length < 2:
        parser.error("--eval_perplexity requires --sequence_length >= 2.")
    # Check and fix group_size (if needed)
    if args.format == "nvfp":
        if args.w_group_size != NVFP_GROUPSIZE:
            args.w_group_size = NVFP_GROUPSIZE
            print(f"Changed weight group_size to {NVFP_GROUPSIZE} for nvfp format.")
        if args.a_group_size != NVFP_GROUPSIZE:
            args.a_group_size = NVFP_GROUPSIZE
            print(f"Changed activation group_size to {NVFP_GROUPSIZE} for nvfp format.")
        if args.scale_precision != "e4m3":
            args.scale_precision = "e4m3"
            print(f"Changed scale_precision to e4m3 for nvfp format.")
    elif args.format == "mxfp":
        if args.w_group_size != MXFP_GROUPSIZE:
            args.w_group_size = MXFP_GROUPSIZE
            print(f"Changed weight group_size to {MXFP_GROUPSIZE} for mxfp format.")
        if args.a_group_size != MXFP_GROUPSIZE:
            args.a_group_size = MXFP_GROUPSIZE
            print(f"Changed activation group_size to {MXFP_GROUPSIZE} for mxfp format.")
        if args.scale_precision != "e8m0":
            args.scale_precision = "e8m0"
            print(f"Changed scale precision to e8m0 for mxfp format.")
    # Check logging
    if args.log_wandb:
        assert wandb is not None, "wandb is not installed. Please install wandb `pip install wandb`."
    # Check real_quant config
    if args.export_quantized_model:
        assert args.save_path is not None, "`save_path` must be specified when exporting quantized model."
        assert args.format in ["nvfp", "mxfp"], "`export_quantization` is only supported for nvfp and mxfp formats."
        assert args.w_bits == 4, "`export_quantization` is only supported for 4 bit weights."
        assert args.a_bits == 4, "`export_quantization` is only supported for 4 bit activations."
    return args


def main():
    args = parse_args()
    if args.eval_openllm:
        check_downstream_setup(args.lm_eval_tasks)
    # Fix seed
    fix_seed(args.seed)
    # Set device
    device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    # Get dtype
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Init logger
    if args.log_wandb:
        wandb.init(config=args)
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=args.dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if args.eval_perplexity or args.eval_openllm else "eager",
    )
    if not args.cpu_offload_modules:
        model = model.to(device)
    model.config.use_cache = False
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Only the optional chat variants need a chat template.
    if args.eval_openllm:
        chat_tasks = [task for task in args.lm_eval_tasks if TASKS[task].chat]
        if chat_tasks and not getattr(tokenizer, "chat_template", None):
            raise ValueError(f"{', '.join(chat_tasks)} require a tokenizer chat template.")
        if args.disable_thinking:
            if model.config.model_type == "qwen3":
                tokenizer.apply_chat_template = partial(
                    tokenizer.apply_chat_template, 
                    enable_thinking=False
                )
            else:
                warnings.warn("`disable_thinking` has no effect on non-Qwen3 models.")

    quantize_anything = args.w_bits < 16 or args.a_bits < 16

    if quantize_anything:
        # MXFP RTN uses per-group scales and does not consume calibration data.
        calibration_data = []
        if args.gptq or args.scale_precision == ScalePrecision.E4M3:
            calibration_data = get_data(
                args.dataset_name_or_path, tokenizer, args.sequence_length, args.num_sequences, args.seed
            )
        if args.gptq:
            quantized_state_dict, non_quantized_state_dict = gptq_quantization(model, calibration_data, args, device)
        else:
            quantized_state_dict, non_quantized_state_dict = rtn_quantization(model, calibration_data, args, device)
        del calibration_data

        if args.export_quantized_model:
            export_quantized_model(model, quantized_state_dict, non_quantized_state_dict, args) 
            tokenizer.save_pretrained(args.save_path)

    if args.eval_perplexity or args.eval_openllm:
        model = model.to(device).eval()
        clear_device_cache(garbage_collection=True)

    if args.compile:
        model = torch.compile(model)

    summary = {"model": args.model_name_or_path, "wikitext2_ppl": None, "c4_ppl": None, "results": {}}
    if args.eval_perplexity:
        for dataset, label, loader in (
            ("wikitext2_ppl", "WikiText-2", get_wikitext2),
            ("c4_ppl", "C4", get_c4_eval),
        ):
            eval_data = loader(tokenizer, args.sequence_length)[:args.ppl_limit]
            if not eval_data:
                raise ValueError(f"{label} has no complete {args.sequence_length}-token evaluation windows.")
            ppl = compute_perplexity(model, eval_data)
            summary[dataset] = ppl
            print(f"{label} perplexity: {ppl:.4f} ({len(eval_data)} windows)")
            del eval_data
            if args.log_wandb:
                wandb.log({f"eval/{dataset}": ppl})
        clear_device_cache(garbage_collection=True)

    if args.eval_openllm:
        summary["results"] = evaluate_downstream(
            model, tokenizer, args.lm_eval_tasks, args.lm_eval_batch_size, args.limit, seed=args.lm_eval_seed
        )
        if args.log_wandb:
            wandb.log({"eval/openllm": summary["results"]})

    if args.eval_perplexity or args.eval_openllm:
        print(f"\n[RESULT_SUMMARY] {json.dumps(summary)}")


if __name__ == "__main__":
    main()
