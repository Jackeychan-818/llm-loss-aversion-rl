#!/usr/bin/env python3
"""Evaluate a local Hugging Face model or PEFT adapter on GSM8K.

The prompt follows the standard eight-shot chain-of-thought examples used by
EleutherAI's ``gsm8k_cot`` task.  Evaluation is deterministic (greedy decoding)
and resumable.  Only test-split questions are scored.

Examples
--------
Validate a downloaded CSV without loading a model::

    python eval/run_gsm8k_local.py \
        --data_file /path/to/gsm8k_dataset/all/full.csv \
        --validate_data_only

Run the base model::

    python eval/run_gsm8k_local.py \
        --model_path models/Qwen2.5-7B-Instruct \
        --data_file /path/to/gsm8k_dataset/all/full.csv \
        --output_dir results/gsm8k/base

Run a LoRA checkpoint::

    python eval/run_gsm8k_local.py \
        --model_path models/Qwen2.5-7B-Instruct \
        --adapter_path checkpoints/grpo_qwen_delta/checkpoint-8000 \
        --data_file /path/to/gsm8k_dataset/all/full.csv \
        --output_dir results/gsm8k/qwen_delta_8000

If ``--data_file`` is omitted, the official ``openai/gsm8k`` test split is
loaded through Hugging Face Datasets.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_TEST_SIZE = 1319
PROTOCOL_VERSION = "gsm8k_cot_8shot_chat_v1"


# These are the canonical eight demonstrations in lm-evaluation-harness's
# gsm8k_cot task.  Keeping them in code makes the prompt fixed and auditable.
FEWSHOT_EXAMPLES: Sequence[Tuple[str, str]] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in "
        "the grove today. After they are done, there will be 21 trees. How "
        "many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some "
        "more were planted.\nSo there must have been 21 - 15 = 6. The answer "
        "is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how "
        "many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The "
        "answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how "
        "many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The "
        "answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has "
        "12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. How many computers "
        "are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is "
        "29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the end "
        "of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he "
        "had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. "
        "The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money "
        "does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = "
        "15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The "
        "answer is 8.",
    ),
)


NUMBER_PATTERN = r"-?\$?\d[\d,]*(?:\.\d+)?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen/PEFT on GSM8K")
    parser.add_argument("--model_path", default="models/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument(
        "--data_file",
        default=None,
        help="CSV/JSONL containing GSM8K. If omitted, download openai/gsm8k.",
    )
    parser.add_argument("--output_dir", default="results/gsm8k/model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--validate_data_only",
        action="store_true",
        help="Validate the test records and exit before importing torch.",
    )
    parser.add_argument(
        "--allow_nonstandard_size",
        action="store_true",
        help="Allow a source whose complete test split is not 1,319 records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing predictions in output_dir before running.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def normalize_answer(value: object) -> str:
    """Normalize GSM8K's integer answer representation without changing value."""
    text = str(value).strip().replace(",", "").replace("$", "")
    return text.rstrip(".")


def answer_from_solution(solution: str) -> Optional[str]:
    if "####" not in solution:
        return None
    return normalize_answer(solution.rsplit("####", 1)[-1].strip().splitlines()[0])


def target_from_row(row: Dict[str, object]) -> str:
    direct = row.get("final_answer")
    if direct is not None and str(direct).strip():
        return normalize_answer(direct)

    answer = row.get("answer")
    if answer is not None and str(answer).strip():
        marked = answer_from_solution(str(answer))
        if marked is not None:
            return marked

    solution = row.get("solution")
    if solution is not None and str(solution).strip():
        marked = answer_from_solution(str(solution))
        if marked is not None:
            return marked

    raise ValueError("Record has no final_answer and no #### answer marker")


def record_from_row(row: Dict[str, object], index: int) -> Dict[str, str]:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("Record has an empty question")
    record_id = str(row.get("id") or "test_{:05d}".format(index + 1)).strip()
    return {
        "id": record_id,
        "question": question,
        "target": target_from_row(row),
    }


def read_csv_records(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "question" not in reader.fieldnames:
            raise ValueError("CSV must contain a question column")
        for row in reader:
            split = str(row.get("split", "test")).strip().lower()
            if split and split != "test":
                continue
            records.append(record_from_row(row, len(records)))
    return records


def read_jsonl_records(path: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("JSONL line {} is not an object".format(line_number))
            split = str(row.get("split", "test")).strip().lower()
            if split and split != "test":
                continue
            records.append(record_from_row(row, len(records)))
    return records


def load_records(data_file: Optional[str]) -> Tuple[List[Dict[str, str]], str]:
    if data_file:
        path = resolve(data_file)
        if not path.exists():
            raise FileNotFoundError("GSM8K data file not found: {}".format(path))
        suffix = path.suffix.lower()
        if suffix == ".csv":
            records = read_csv_records(path)
        elif suffix in (".jsonl", ".json"):
            records = read_jsonl_records(path)
        else:
            raise ValueError("Expected a .csv or .jsonl GSM8K file")
        source = str(path.resolve())
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "datasets is required when --data_file is omitted"
            ) from exc
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        records = [record_from_row(dict(row), i) for i, row in enumerate(dataset)]
        source = "huggingface:openai/gsm8k:main:test"
    return records, source


def validate_records(records: Sequence[Dict[str, str]], allow_nonstandard: bool) -> None:
    if not records:
        raise ValueError("No test records were loaded")
    if not allow_nonstandard and len(records) != EXPECTED_TEST_SIZE:
        raise ValueError(
            "Expected {} GSM8K test records, found {}. Use "
            "--allow_nonstandard_size only for deliberate custom data.".format(
                EXPECTED_TEST_SIZE, len(records)
            )
        )

    ids = [row["id"] for row in records]
    questions = [" ".join(row["question"].split()) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate test record IDs detected")
    if len(questions) != len(set(questions)):
        raise ValueError("Duplicate test questions detected")
    if any(not re.fullmatch(r"-?\d+", row["target"]) for row in records):
        raise ValueError("GSM8K targets must normalize to integer strings")


def extract_prediction(response: str) -> Tuple[Optional[str], str]:
    """Extract an answer using strict GSM8K form first, then safe fallbacks."""
    patterns = (
        ("answer_is", r"(?i)the\s+answer\s+is\s*({})".format(NUMBER_PATTERN)),
        ("hash_marker", r"####\s*({})".format(NUMBER_PATTERN)),
        ("boxed", r"\\boxed\s*\{{\s*({})\s*\}}".format(NUMBER_PATTERN)),
    )
    for method, pattern in patterns:
        matches = re.findall(pattern, response)
        if matches:
            return normalize_answer(matches[-1]), method

    matches = re.findall(NUMBER_PATTERN, response)
    if matches:
        return normalize_answer(matches[-1]), "last_number"
    return None, "unparsed"


def fewshot_messages(question: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for example_question, example_answer in FEWSHOT_EXAMPLES:
        messages.append({"role": "user", "content": "Q: {}\nA:".format(example_question)})
        messages.append({"role": "assistant", "content": example_answer})
    messages.append({"role": "user", "content": "Q: {}\nA:".format(question)})
    return messages


def build_prompt(tokenizer, question: str) -> str:
    messages = fewshot_messages(question)
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts: List[str] = []
    for message in messages:
        if message["role"] == "user":
            parts.append(message["content"])
        else:
            parts.append(message["content"] + "\n")
    return "\n".join(parts)


def dtype_from_name(torch_module, name: str):
    if name == "bf16":
        return torch_module.bfloat16
    if name == "fp16":
        return torch_module.float16
    return torch_module.float32


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_metadata(args: argparse.Namespace, source: str) -> Dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "source": source,
        "model_path": str(resolve(args.model_path)),
        "adapter_path": str(resolve(args.adapter_path)) if args.adapter_path else None,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "seed": args.seed,
        "decoding": "greedy",
    }


def load_existing_predictions(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid predictions JSONL at line {}: {}".format(line_number, path)
                ) from exc
            rows.append(row)
    return rows


def validate_resume_metadata(path: Path, expected: Dict[str, object]) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        found = json.load(handle)
    if found != expected:
        raise ValueError(
            "Existing run metadata does not match this invocation. Use a new "
            "--output_dir or pass --overwrite."
        )


def batches(items: Sequence[Dict[str, str]], batch_size: int) -> Iterable[Sequence[Dict[str, str]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def summarize(rows: Sequence[Dict[str, object]], total_available: int) -> Dict[str, object]:
    correct = sum(bool(row.get("correct")) for row in rows)
    parsed = sum(row.get("predicted_answer") is not None for row in rows)
    count = len(rows)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "evaluated": count,
        "total_available": total_available,
        "correct": correct,
        "accuracy": correct / count if count else None,
        "parsed": parsed,
        "parse_rate": parsed / count if count else None,
        "complete": count == total_available,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")

    records, source = load_records(args.data_file)
    validate_records(records, args.allow_nonstandard_size)
    print("GSM8K source: {}".format(source))
    print("Validated test records: {}".format(len(records)))
    print("Protocol: {}".format(PROTOCOL_VERSION))
    if args.validate_data_only:
        print("Data validation complete; model was not loaded.")
        return

    selected = records[: args.limit] if args.limit is not None else records
    output_dir = resolve(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    metadata_path = output_dir / "run_metadata.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for path in (predictions_path, summary_path, metadata_path):
            if path.exists():
                path.unlink()

    metadata = expected_metadata(args, source)
    validate_resume_metadata(metadata_path, metadata)
    if not metadata_path.exists():
        dump_json(metadata_path, metadata)

    existing = load_existing_predictions(predictions_path)
    existing_by_id = {str(row["id"]): row for row in existing}
    if len(existing_by_id) != len(existing):
        raise ValueError("Duplicate IDs found in existing predictions")
    remaining = [row for row in selected if row["id"] not in existing_by_id]

    print("Model path: {}".format(resolve(args.model_path)))
    print(
        "Adapter path: {}".format(
            resolve(args.adapter_path) if args.adapter_path else "(base model)"
        )
    )
    print("Output directory: {}".format(output_dir))
    print("Selected: {}; already complete: {}; remaining: {}".format(
        len(selected), len(selected) - len(remaining), len(remaining)
    ))

    if not remaining:
        ordered = [existing_by_id[row["id"]] for row in selected]
        dump_json(summary_path, summarize(ordered, len(selected)))
        print(json.dumps(summarize(ordered, len(selected)), indent=2))
        return

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "torch and transformers are required for model evaluation"
        ) from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_path = resolve(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError("Model path not found: {}".format(model_path))
    adapter_path = resolve(args.adapter_path) if args.adapter_path else None
    if adapter_path is not None and not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            "LoRA adapter_config.json not found under {}".format(adapter_path)
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=dtype_from_name(torch, args.dtype),
        device_map=args.device_map,
        trust_remote_code=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        print("Loading LoRA adapter from {}...".format(adapter_path))
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    device = next(model.parameters()).device
    try:
        from tqdm import tqdm

        iterator = tqdm(
            batches(remaining, args.batch_size),
            total=(len(remaining) + args.batch_size - 1) // args.batch_size,
            desc="GSM8K",
        )
    except ImportError:
        iterator = batches(remaining, args.batch_size)

    with predictions_path.open("a", encoding="utf-8") as output_handle:
        for batch in iterator:
            prompts = [build_prompt(tokenizer, row["question"]) for row in batch]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_length = encoded["input_ids"].shape[1]

            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            responses = tokenizer.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True
            )

            for source_row, response in zip(batch, responses):
                predicted, extraction = extract_prediction(response)
                result: Dict[str, object] = {
                    "id": source_row["id"],
                    "question": source_row["question"],
                    "target": source_row["target"],
                    "predicted_answer": predicted,
                    "correct": predicted == source_row["target"],
                    "extraction": extraction,
                    "response": response,
                    "protocol_version": PROTOCOL_VERSION,
                }
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                existing_by_id[source_row["id"]] = result
            output_handle.flush()
            os.fsync(output_handle.fileno())

            ordered_so_far = [
                existing_by_id[row["id"]]
                for row in selected
                if row["id"] in existing_by_id
            ]
            dump_json(summary_path, summarize(ordered_so_far, len(selected)))

    ordered = [existing_by_id[row["id"]] for row in selected]
    final_summary = summarize(ordered, len(selected))
    final_summary["data_sha256"] = (
        file_sha256(resolve(args.data_file)) if args.data_file else None
    )
    dump_json(summary_path, final_summary)
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
