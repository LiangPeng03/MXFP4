import os
from dataclasses import dataclass

from ..utils.common_utils import clear_device_cache


@dataclass(frozen=True)
class TaskConfig:
    fewshot: int | None
    batch_size: int
    metric: str
    limit: int | None = None
    chat: bool = False


# Protocol from MR-GPTQ's compare.sh, compare_mds.sh and bnv_mds.sh.
MR_GPTQ_TASKS = {
    "winogrande": TaskConfig(5, 64, "acc"),
    "hellaswag": TaskConfig(10, 8, "acc_norm", limit=4000),
    "piqa": TaskConfig(0, 64, "acc_norm"),
    "arc_challenge": TaskConfig(25, 8, "acc_norm"),
    "boolq": TaskConfig(0, 8, "acc"),
    "gsm8k": TaskConfig(5, 16, "exact_match"),
    "mmlu": TaskConfig(5, 4, "acc", limit=200),
    "arc_easy": TaskConfig(25, 8, "acc_norm", limit=4000),
    "mbpp": TaskConfig(3, 16, "pass_at_1"),
    "lambada_openai": TaskConfig(0, 16, "acc"),
}

TASKS = {
    **MR_GPTQ_TASKS,
    "gsm8k_llama": TaskConfig(None, 8, "exact_match", chat=True),
    "mmlu_cot_llama": TaskConfig(None, 8, "exact_match", chat=True),
}
GENERATION_TASKS = {"gsm8k", "mbpp", "gsm8k_llama", "mmlu_cot_llama"}
CONTEXT_LENGTH = 2048


def check_downstream_setup(tasks):
    if "mbpp" in tasks and os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise ValueError("MBPP executes generated Python. Set HF_ALLOW_CODE_EVAL=1 in the evaluation environment.")
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as exc:
        raise RuntimeError("Downstream evaluation requires lm-eval[hf]==0.4.11 and its dependencies.") from exc


def evaluate_downstream(model, tokenizer, tasks, batch_size, limit):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    summary = {}
    original_context = model.config.max_position_embeddings
    original_cache = model.config.use_cache
    generation_config = model.generation_config
    original_generation_cache = generation_config.use_cache
    context_length = min(original_context, CONTEXT_LENGTH)
    model.config.max_position_embeddings = context_length
    try:
        for task_name in tasks:
            task = TASKS[task_name]
            task_batch_size = task.batch_size if batch_size == "auto" else min(batch_size, task.batch_size)
            task_limit = task.limit
            if limit is not None:
                task_limit = limit if task_limit is None else min(limit, task_limit)

            model.config.use_cache = task_name in GENERATION_TASKS
            generation_config.use_cache = model.config.use_cache
            print(f"Evaluating {task_name} (few-shot={task.fewshot}, batch={task_batch_size}, limit={task_limit})")
            lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=task_batch_size, max_length=context_length)
            evaluation = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task_name],
                num_fewshot=task.fewshot,
                limit=task_limit,
                apply_chat_template=task.chat,
                confirm_run_unsafe_code=(task_name == "mbpp"),
                log_samples=False,
            )
            column = "groups" if task_name in evaluation.get("groups", {}) else "results"
            print(make_table(evaluation, column))
            metrics = evaluation[column][task_name]
            score = next((value for key, value in metrics.items() if key.partition(",")[0] == task.metric), None)
            if score is None:
                raise ValueError(f"{task_name} did not report {task.metric}. Available metrics: {list(metrics)}")
            summary[task_name] = float(score)
            del evaluation, lm
            clear_device_cache(garbage_collection=True)
    finally:
        model.config.max_position_embeddings = original_context
        model.config.use_cache = original_cache
        generation_config.use_cache = original_generation_cache

    return summary
