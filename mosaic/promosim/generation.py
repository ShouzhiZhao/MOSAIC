"""Generate independent PromoSim replays for a fixed movie and seed intervention."""

import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
import yaml
from tqdm import tqdm
from yacs.config import CfgNode

from mosaic import paths
from mosaic.artifacts import atomic_write, environment_metadata, sha256_file, write_json
from mosaic.catalog import load_movies, resolve_movie
from mosaic.records import validate_replay

from .integrity import assert_initial_state, capture_recommender, seed_replay
from .simulator import PromoSimulator, parse_args
from .utils import utils
from .utils.event import update_event
from .utils.message import Message


def _resolve_data_paths(config):
    """Resolve every PromoSim input/output path against its data directory."""
    for key in (
        "item_path",
        "user_path",
        "relationship_path",
        "interaction_path",
        "ckpt_path",
        "index_name",
    ):
        value = Path(config[key])
        if not value.is_absolute():
            config[key] = str(paths.PROMOSIM_DATA_DIR / value)


def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return CfgNode(config_dict)


def init_seed_agents(sim, seed_agents, target_movie):
    messages = []
    for agent_id in seed_agents:
        message = []
        agent = sim.agents[agent_id]
        name = agent.name
        contacts = sim.data.get_all_contacts(agent_id)
        if len(contacts) == 0:
            sim.logger.info(f"{name} (ID: {agent_id}) has no acquaintance.")
            message.append(
                Message(
                    agent_id=agent_id,
                    action="SOCIAL",
                    content=f"{name} has no acquaintance.",
                )
            )
            sim.round_msg.append(
                Message(
                    agent_id=agent_id,
                    action="SOCIAL",
                    content=f"{name} has no acquaintance.",
                )
            )

        sim.social_stat.post_num += 1
        sim.logger.info(f"{name} (ID: {agent_id}) is posting.")
        observation = f"I've just watched <{target_movie}> and it was amazing! I really want to recommend it to my friends."
        sim.logger.info(name + " posted: " + observation)
        if agent.event.action_type == "idle":
            agent.event = update_event(
                original_event=agent.event,
                start_time=sim.now,
                duration=0.1,
                target_agent=None,
                action_type="posting",
            )
        message.append(
            Message(
                agent_id=agent_id,
                action="POST",
                content=name + " posts: " + observation,
            )
        )
        sim.round_msg.append(
            Message(
                agent_id=agent_id,
                action="POST",
                content=name + " posts: " + observation,
            )
        )
        item_names = utils.extract_item_names(observation, "SOCIAL")
        message.extend(sim.deliver_post(agent_id, observation, item_names))

        sim.logger.info(f"{contacts} get this post.")
        messages.append(message)
    return messages


def run_simulation(
    seed_size,
    target_movie_id,
    seed_agents,
    args,
    *,
    rng_seed: int | None = None,
    replay_start_time: datetime | None = None,
    reference_state_digest: str | None = None,
    integrity_callback: Callable[[dict], None] | None = None,
    config_overrides: dict | None = None,
    initialization_state: dict | None = None,
):
    if rng_seed is not None:
        seed_replay(rng_seed)
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    # create config
    config = CfgNode(new_allowed=True)
    output_file = os.path.join("output/data_generator", args.output_file)
    config = utils.add_variable_to_config(config, "output_file", output_file)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config = utils.add_variable_to_config(config, "memory_backend", args.memory_backend)
    config.merge_from_file(args.config_file)
    for key, value in (config_overrides or {}).items():
        config[key] = value
    _resolve_data_paths(config)
    utils.configure_api_from_environment(config)
    logger.info("\n%s", utils.redacted_config(config))
    os.environ["OPENAI_API_KEY"] = config["api_keys"][0]
    st = time.time()
    if config["simulator_restore_file_name"]:
        if rng_seed is not None or integrity_callback is not None:
            raise ValueError("policy replay cannot restore mutable simulator state")
        restore_path = os.path.join(config["simulator_dir"], config["simulator_restore_file_name"])
        sim = PromoSimulator.restore(restore_path, config, logger)
        logger.info(f"Successfully Restore simulator from the file <{restore_path}>\n")
        logger.info(f"Start from the round {sim.round_cnt + 1}\n")
    else:
        sim = PromoSimulator(config, logger)
        if replay_start_time is not None:
            sim.now = replay_start_time
        sim.load_simulator(initialization_state or None)
    if rng_seed is not None:
        seed_replay(rng_seed)
    if sim.config["social_random_k"] > 0:
        sim.clear_social()
        sim.add_social(sim.config["social_random_k"])
    print("Time for loading simulator: ", time.time() - st)
    sim.play()

    if (
        seed_size != len(seed_agents)
        or seed_size != len(set(seed_agents))
        or any(
            type(node) not in (int, np.int64, np.int32) or node not in sim.agents
            for node in seed_agents
        )
    ):
        raise ValueError("Seed IDs must be distinct, in range, and match seed_size")
    if target_movie_id not in sim.data.items:
        raise ValueError("Target movie ID is absent from the simulation catalog")
    if initialization_state is not None and not initialization_state:
        initialization_state.update(
            {
                "interests": {str(i): a.interest for i, a in sim.agents.items()},
                "recommender": capture_recommender(sim.recsys.model),
            }
        )
    # Initialization may consume different numbers of draws when it is reused.
    if rng_seed is not None:
        seed_replay(rng_seed)

    if rng_seed is not None or integrity_callback is not None:
        integrity = assert_initial_state(sim, reference_state_digest)
        integrity["rng_seed"] = rng_seed
        integrity["effective_config"] = yaml.safe_load(utils.redacted_config(config).dump())
        if integrity_callback is not None:
            integrity_callback(integrity)

    # Capture the exact pre-intervention descriptions embedded by BGE-M3.
    # They are saved once in the aggregate condition record below.
    user_descriptions, item_descriptions = sim.get_initial_content_descriptions()

    # Target Movie & Seed Agents
    target_movie = sim.data.items[target_movie_id]["name"]
    watch_count = seed_size
    watched_agents = set(seed_agents)
    newly_watched_agents = set(seed_agents)
    print(f"\n[Data Generator] Target Movie: <{target_movie}>")
    print(f"[Data Generator] Seeding {seed_size} agents: {seed_agents}")

    n_users = sim.data.get_user_num()

    # Get item embedding for target movie
    target_movie_emb = sim.recsys.model.item_content_emb[target_movie_id]
    target_movie_emb = target_movie_emb.cpu().detach().numpy()

    def calculate_array(user_content_emb, newly_watched_agents):
        sim_array = np.zeros(n_users)
        for user_id in range(n_users):
            user_emb = user_content_emb[user_id]
            similarity = np.dot(user_emb.flatten(), target_movie_emb.flatten()) / (
                np.linalg.norm(user_emb) * np.linalg.norm(target_movie_emb) + 1e-9
            )
            sim_array[user_id] = similarity
        X_array = np.zeros(n_users, dtype=bool)
        for user_id in newly_watched_agents:
            X_array[user_id] = True
        return X_array, sim_array

    sim_scores = np.zeros((config["round"] + 1, n_users))
    X_flags = np.zeros((config["round"] + 1, n_users), dtype=bool)
    X_flags[0], sim_scores[0] = calculate_array(
        sim.recsys.model.user_content_emb.cpu().detach().numpy(), newly_watched_agents
    )

    # Seed users post at the beginning of round 0
    messages = init_seed_agents(sim, seed_agents, target_movie)

    for i in tqdm(range(sim.round_cnt + 1, config["round"] + 1)):
        round_st = time.time()
        sim.round_cnt = sim.round_cnt + 1
        sim.logger.info(f"Round {sim.round_cnt}")
        sim.active_agents.clear()
        message = sim.round()
        messages.append(message)
        newly_watched_agents = set()  # Track for this round
        for msgs in message:
            for m in msgs:
                if m.event_type == "watch" and m.item_id == target_movie_id:
                    newly_watched_agents.add(m.agent_id)
                    watched_agents.add(m.agent_id)
        X_flags[sim.round_cnt], sim_scores[sim.round_cnt] = calculate_array(
            sim.recsys.model.user_content_emb.cpu().detach().numpy(), newly_watched_agents
        )
        watch_count = len(watched_agents)
        newly_watch_count = len(newly_watched_agents)
        print(
            f"[Data Generator] Round {sim.round_cnt}: watchers of <{target_movie}> = {watch_count}"
        )
        print(
            f"[Data Generator] Round {sim.round_cnt}: newly watchers of <{target_movie}> = {newly_watch_count}"
        )

        print("Time for round: ", time.time() - round_st)
        print("Time for all: ", time.time() - st)
        sim.recsys.save_interaction()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n[Data Generator] Simulation Finished.")

    # Explicitly release memory
    del sim
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        X_flags,
        sim_scores,
        user_descriptions,
        item_descriptions[target_movie_id],
    )


@contextmanager
def generation_lock(directory):
    # Linux advisory locks are released by the OS even after a killed process.
    import fcntl

    with (directory / ".generation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another generator is using {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PromoSim generation requires CUDA; CPU fallback is disabled")
    args = parse_args()
    config = load_config(args.config_file)
    _resolve_data_paths(config)
    utils.configure_api_from_environment(config)
    num_agents = int(config["agent_num"])
    if num_agents < 10 or config["round"] < 1:
        raise ValueError("Generation requires at least ten agents and positive rounds")
    movie = resolve_movie(
        int(os.environ.get("TARGET_MOVIE_ID", "0")), load_movies(Path(config["item_path"]))
    )
    replications = getattr(args, "replications", 10)
    base_seed = getattr(args, "seed", 2026)
    if replications < 1:
        raise ValueError("replications must be positive")
    output_dir = paths.get_simulation_output_dir(movie.key)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "format": "mosaic-condition-v2",
        "movie_id": movie.id,
        "movie": movie.key,
        "rounds": int(config["round"]),
        "num_nodes": num_agents,
        "replications": replications,
        "base_seed": base_seed,
        "config": yaml.safe_load(utils.redacted_config(config).dump()),
        "inputs": {
            key: sha256_file(Path(config[key]))
            for key in ("item_path", "user_path", "relationship_path")
        },
    }
    with generation_lock(output_dir):
        _generate_condition(args, output_dir, provenance, movie)


def _generate_condition(args, output_dir, provenance, movie):
    from .integrity import paired_rng_seed

    root = output_dir / "temp"
    root.mkdir(exist_ok=True)
    resume = getattr(args, "resume_condition", None)
    pending = []
    for path in sorted(root.iterdir()):
        manifest = path / "condition.json"
        if path.is_dir() and manifest.is_file() and not (path / "complete.json").exists():
            saved = json.loads(manifest.read_text())
            if all(saved.get(k) == v for k, v in provenance.items()):
                pending.append(path)
    if resume:
        if Path(resume).name != resume:
            raise ValueError("resume-condition must be a condition ID, not a path")
        directory = root / resume
    elif pending:
        directory = pending[0]
    else:
        rng = np.random.default_rng(provenance["base_seed"])
        for _ in range(100000):
            budget = int(rng.integers(0, 11))
            seeds = sorted(
                int(i) for i in rng.choice(provenance["num_nodes"], budget, replace=False)
            )
            condition = hashlib.sha256(json.dumps([movie.id, seeds]).encode()).hexdigest()[:24]
            directory = root / condition
            if not directory.exists():
                directory.mkdir()
                write_json(
                    directory / "condition.json",
                    {**provenance, "seeds": seeds, "environment": environment_metadata()},
                )
                break
        else:
            raise RuntimeError("Could not sample an unused seed condition")
    manifest = directory / "condition.json"
    if not manifest.is_file():
        raise ValueError(
            "Resume requires a versioned condition.json; legacy archives remain read-only"
        )
    saved = json.loads(manifest.read_text())
    if any(saved.get(k) != v for k, v in provenance.items()):
        raise ValueError("Resume configuration or input hashes differ from the original condition")
    seeds = saved["seeds"]
    if (
        not isinstance(seeds, list)
        or any(type(node) is not int or not 0 <= node < provenance["num_nodes"] for node in seeds)
        or len(set(seeds)) != len(seeds)
        or len(seeds) > 10
    ):
        raise ValueError("Invalid seed IDs in condition manifest")
    expected = np.zeros(provenance["num_nodes"], dtype=bool)
    expected[seeds] = True
    descriptions_path = directory / "profiles.json"
    recorded_hashes = {}
    ledger_path = directory / "attempts.jsonl"
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # A killed writer can leave an incomplete final ledger line.
            if entry.get("status") == "completed":
                recorded_hashes[entry["replication"]] = entry["sha256"]
    for replication in range(provenance["replications"]):
        target = directory / f"{replication}.npz"
        rng_seed = paired_rng_seed(provenance["base_seed"], directory.name, len(seeds), replication)
        if target.exists():
            if (
                replication in recorded_hashes
                and sha256_file(target) != recorded_hashes[replication]
            ):
                raise ValueError(f"Replay checksum differs from the completion ledger: {target}")
            with np.load(target, allow_pickle=False) as replay:
                x, _ = validate_replay(
                    replay["X"],
                    replay["sim"],
                    rounds=provenance["rounds"],
                    num_nodes=provenance["num_nodes"],
                    source=str(target),
                )
                if (
                    not np.array_equal(x[0], expected)
                    or int(replay["movie_id"]) != movie.id
                    or int(replay["rng_seed"]) != rng_seed
                    or str(replay["format"]) != "mosaic-replay-v2"
                ):
                    raise ValueError(f"Replay identity differs from condition manifest: {target}")
            continue
        run_args = SimpleNamespace(**vars(args))
        run_args.log_file = str(directory / f"replay-{replication}.log")
        audit = {"replication": replication, "rng_seed": rng_seed, "started": time.time()}
        with (directory / "attempts.jsonl").open("a") as ledger:
            ledger.write(json.dumps({**audit, "status": "started"}) + "\n")
        try:
            x, sim, users, item = run_simulation(
                seed_size=len(seeds),
                target_movie_id=movie.id,
                seed_agents=seeds,
                args=run_args,
                rng_seed=rng_seed,
                replay_start_time=datetime(2023, 1, 1, 8),
                config_overrides={
                    "interaction_path": str(directory / "interaction.csv"),
                    "simulator_restore_file_name": "",
                },
            )
            x, sim = validate_replay(
                x,
                sim,
                rounds=provenance["rounds"],
                num_nodes=provenance["num_nodes"],
                source=str(target),
            )
            if not np.array_equal(x[0], expected):
                raise ValueError("Replay seed vector differs from requested seeds")
            if not descriptions_path.exists():
                write_json(descriptions_path, {"users": users, "item": item})
            atomic_write(
                target,
                lambda f: np.savez_compressed(
                    f, X=x, sim=sim, movie_id=movie.id, rng_seed=rng_seed, format="mosaic-replay-v2"
                ),
            )
        except Exception as exc:
            with (directory / "attempts.jsonl").open("a") as ledger:
                ledger.write(
                    json.dumps({**audit, "status": "failed", "error_type": type(exc).__name__})
                    + "\n"
                )
            raise
        with (directory / "attempts.jsonl").open("a") as ledger:
            ledger.write(
                json.dumps({**audit, "status": "completed", "sha256": sha256_file(target)}) + "\n"
            )
    arrays = []
    similarities = []
    for replication in range(provenance["replications"]):
        with np.load(directory / f"{replication}.npz", allow_pickle=False) as replay:
            arrays.append(replay["X"])
            similarities.append(replay["sim"])
    descriptions = json.loads(descriptions_path.read_text())
    aggregate = output_dir / f"{directory.name}.npz"
    atomic_write(
        aggregate,
        lambda f: np.savez_compressed(
            f,
            X=np.stack(arrays),
            sim=np.stack(similarities),
            movie_id=movie.id,
            seed_agents=np.asarray(seeds, dtype=np.int64),
            user_descriptions=np.asarray(descriptions["users"], dtype=str),
            item_description=np.asarray(descriptions["item"], dtype=str),
        ),
    )
    write_json(
        directory / "complete.json",
        {"replications": provenance["replications"], "aggregate_sha256": sha256_file(aggregate)},
    )
    print(f"Completed {movie.key}: {aggregate}")


if __name__ == "__main__":
    main()
