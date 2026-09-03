"""This file can be used to generate new completions for attack runs using a
specified generation_config from conf/sampling.yaml.

The script filters runs based on the filter_by configuration and generates
completions from scratch using the generation_config, replacing all existing
completions and removing scores.

Set `keep_existing_samples: true` to instead append the newly generated
completions/scores to the existing ones for each step (nothing is thrown
away). Existing and new completions stay index-aligned with their score
lists.

Set `only_where_sampled: true` to restrict generation to steps that already
had at least one completion in the original file (e.g. steps hit by a
sampling_schedule). Steps that originally collected 0 samples are left
untouched (still 0) instead of being sampled for the first time.

Set `exclude_idx` to drop matched paths for given dataset_params.idx values
(e.g. a behaviour another concurrent job is already handling), and/or
`path_offset`/`path_limit` to process a slice of the (sorted, filtered)
matched paths instead of all of them, e.g. to split one filter_by across
multiple concurrent jobs without them working on the same runs.

Example:

filter_by:
  model:
    - google/gemma-3-1b-it
  attack:
    - gcg
    - pgd

generation_config:
  temperature: 0.0
  top_p: 1.0
  top_k: 0
  max_new_tokens: 256
  num_return_sequences: 1
"""
import os
from datetime import datetime

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # determinism
import copy
import json
import logging
import sys

import hydra
import torch
from judgezoo import Judge
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from adversariallm.errors import print_exceptions
from adversariallm.io_utils import (
    CompactJSONEncoder,
    RunConfig,
    filter_config,
    free_vram,
    get_filtered_and_grouped_paths,
    get_mongodb_connection,
    load_model_and_tokenizer,
)
from adversariallm.lm_utils import generate_ragged_batched

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_tf32 = True


def eval_list_expressions_in_dict(d):
    if isinstance(d, dict):
        return {k: eval_list_expressions_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [eval_list_expressions_in_dict(v) for v in d]
    elif isinstance(d, str):
        try:
            result = eval(d, {"__builtins__": {}}, {"list": list, "range": range})
            if isinstance(result, list):
                return result
        except Exception:
            pass
    return d


def generate_ragged_batched_vllm(
    model,
    tokenizer,
    token_list,
    initial_batch_size,
    num_return_sequences,
    max_new_tokens,
    temperature,
    top_p,
    top_k
) :
    from vllm import SamplingParams, TokensPrompt
    sampling_params = SamplingParams(
        n=num_return_sequences,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k != 0 else -1,
        max_tokens=max_new_tokens,
        skip_special_tokens=False
    )
    prompts = [TokensPrompt(prompt_token_ids=tokens.tolist()) for tokens in token_list]
    generations = model.generate(prompts=prompts, sampling_params=sampling_params)
    out = []
    for generation in generations:
        instance = []
        for output in generation.outputs:
            instance.append(output.text)
        out.append(instance)
    return out # (n_steps, n_to_generate)


@hydra.main(config_path="./conf", config_name="sampling", version_base="1.3")
@print_exceptions
def main(cfg: DictConfig) -> None:
    date_time_string = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    logging.info("-------------------")
    logging.info(f"Commencing run `{date_time_string}`")
    logging.info("-------------------")

    keep_existing_samples = cfg.get("keep_existing_samples", False)
    if keep_existing_samples:
        logging.info("keep_existing_samples=True: appending new completions/scores instead of overwriting")
    only_where_sampled = cfg.get("only_where_sampled", False)
    if only_where_sampled:
        logging.info("only_where_sampled=True: skipping steps that had 0 samples in the original file")

    filter_by = OmegaConf.to_container(cfg.filter_by)
    filter_by = eval_list_expressions_in_dict(filter_by)
    print(filter_by)
    # get all paths
    paths = sorted(get_filtered_and_grouped_paths(filter_by, None)[("all", )])
    if not paths:
        logging.info("No paths found, exiting")
        return
    logging.info(f"Found {len(paths)} paths")

    exclude_idx = cfg.get("exclude_idx", None)
    if exclude_idx:
        exclude_idx = set(exclude_idx)
        filtered_paths = []
        for p in paths:
            with open(p) as f:
                run_idx = json.load(f)["config"]["dataset_params"]["idx"]
            if not any(i in exclude_idx for i in run_idx):
                filtered_paths.append(p)
        paths = filtered_paths
        logging.info(f"exclude_idx={sorted(exclude_idx)}: {len(paths)} paths remain")

    path_offset = cfg.get("path_offset", 0)
    path_limit = cfg.get("path_limit", None)
    if path_offset or path_limit is not None:
        paths = paths[path_offset: path_offset + path_limit if path_limit is not None else None]
        logging.info(f"path_offset={path_offset} path_limit={path_limit}: processing {len(paths)} paths")

    n = 0
    pbar = tqdm(paths, file=sys.stdout)
    backend = "hf"
    gen_function = {
        "vllm": generate_ragged_batched_vllm,
        "hf": generate_ragged_batched
    }

    model, tokenizer, judge, last_judge, last_model_params = None, None, None, None, None
    for path in pbar:
        try:
            with open(path, "r") as f:
                attack_run = json.load(f)

            # Use the generation config from the YAML file instead of the original
            new_gen_config = OmegaConf.to_container(cfg.generation_config)

            # Update the attack run config to use the new generation config
            attack_run["config"]["attack_params"]["generation_config"] = new_gen_config
            defense_params = attack_run["config"].get("defense_params")
            run_config = RunConfig(
                model=attack_run["config"]["model"],
                dataset=attack_run["config"]["dataset"],
                attack=attack_run["config"]["attack"],
                defense=attack_run["config"].get("defense"),
                model_params=OmegaConf.structured(attack_run["config"]["model_params"]),
                dataset_params=OmegaConf.structured(attack_run["config"]["dataset_params"]),
                attack_params=OmegaConf.structured(attack_run["config"]["attack_params"]),
                defense_params=OmegaConf.structured(defense_params) if defense_params is not None else None,
            )

            run_config = filter_config(run_config, -1)
            if run_config is None:
                continue
            model_params = run_config.model_params
            if model_params != last_model_params:
                pbar.set_description(f"Loading new model and tokenizer {model_params}")
                last_model_params = model_params
                del model, tokenizer
                free_vram()
                model, tokenizer = load_model_and_tokenizer(model_params)
                backend = "hf"

            # Generate new completions from scratch using the new generation config
            for subrun in attack_run["runs"]:
                if only_where_sampled:
                    sampled_steps = [
                        step for step in subrun["steps"]
                        if len(step.get("model_completions", [])) > 0
                    ]
                else:
                    sampled_steps = subrun["steps"]

                tokens = []
                classifiers = set()
                for step in sampled_steps:
                    tokens.append(torch.tensor(step["model_input_tokens"]))
                    # Check which judges were used in the original file
                    if "scores" in step:
                        classifiers.update(list(step["scores"].keys()))

                if not sampled_steps:
                    logging.info("No sampled steps found (only_where_sampled=True), skipping subrun.")
                    continue
                logging.info(f"Generating {len(sampled_steps)}x{new_gen_config['num_return_sequences']} completions (out of {len(subrun['steps'])} steps).")

                new_completions = gen_function[backend](
                    model,
                    tokenizer,
                    tokens,
                    initial_batch_size=None,
                    num_return_sequences=new_gen_config["num_return_sequences"],
                    max_new_tokens=new_gen_config["max_new_tokens"],
                    temperature=new_gen_config["temperature"],
                    top_p=new_gen_config["top_p"],
                    top_k=new_gen_config["top_k"],
                    verbose=True
                )  # (n_steps, n_to_generate)

                # Judge the new completions using the same judges as the original file
                prompt = subrun["original_prompt"]
                n_to_generate = new_gen_config["num_return_sequences"]
                for classifier in classifiers:
                    if classifier != last_judge:
                        last_judge = classifier
                        del judge
                        free_vram()
                        judge = Judge.from_name(classifier)
                        print(judge.classifier.device)
                    modified_prompts = []
                    for completions in new_completions:
                        for completion in completions:
                            modified_prompt = copy.deepcopy(prompt)
                            if modified_prompt[-1]["role"] == "assistant":
                                modified_prompt[-1]["content"] += completion
                            else:
                                modified_prompt.append({"role": "assistant", "content": completion})
                            modified_prompts.append(modified_prompt)

                    results = judge(modified_prompts, verbose=True)
                    if all(r is None for r in results):
                        continue
                    i = 0
                    for step in sampled_steps:
                        # Initialize scores dict if it doesn't exist
                        if "scores" not in step:
                            step["scores"] = {}
                        if classifier not in step["scores"]:
                            step["scores"][classifier] = {}
                        for k, v in results.items():
                            existing = step["scores"][classifier].get(k, [])
                            new_vals = v[i:i+n_to_generate]
                            step["scores"][classifier][k] = (existing + new_vals) if keep_existing_samples else new_vals
                        i += n_to_generate
                if len(classifiers) > 1:
                    del judge
                    free_vram()

                # Replace (or, with keep_existing_samples, extend) model_completions
                for step, completions in zip(sampled_steps, new_completions):
                    existing_completions = step.get("model_completions", []) if keep_existing_samples else []
                    step["model_completions"] = existing_completions + completions
                    n += n_to_generate

                pbar.set_description(f"{len(sampled_steps) * n_to_generate} | {n} total")

            log_dir = os.path.join(cfg.save_dir, date_time_string)
            i = 0
            while os.path.exists(os.path.join(log_dir, str(i), f"run.json")):
                i += 1
            log_file = os.path.join(log_dir, str(i), f"run.json")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            json.dump(attack_run, open(log_file, "w"), indent=2, cls=CompactJSONEncoder)
            logging.info(f"Saved to {log_file}")

            db = get_mongodb_connection()
            collection = db.runs

            # Find all entries that match the original log_file path
            matching_entries = list(collection.find({"log_file": path}))

            # Create new entries with updated log_file and generation_config
            new_entries = []
            for entry in matching_entries:
                new_entry = entry.copy()
                # Remove the _id field so MongoDB will generate a new one
                if "_id" in new_entry:
                    del new_entry["_id"]
                # Update the log_file to the new path
                new_entry["log_file"] = log_file
                # Update the generation_config in the config to match the new one used
                new_entry["config"]["attack_params"]["generation_config"] = new_gen_config
                new_entries.append(new_entry)

            # Insert the new entries if any were found
            if new_entries:
                collection.insert_many(new_entries)
        except Exception as e:
            logging.error(f"Error in {path}. Original exception: {e}")
            raise Exception(f"Error in {path}. Original exception: {e}") from e


if __name__ == "__main__":
    main()
