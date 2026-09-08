import base64
import itertools
import json
import logging
import math
import os
import random
import re
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from yacs.config import CfgNode

from ..llm import CustomLLM, SingletonLocalLLM


# logger
def set_logger(log_file, name="default"):
    """
    Set logger.
    Args:
        log_file (str): log file path
        name (str): logger name
    """

    # Set random seed for reproducibility
    # seed = 42
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.handlers = []
    logger.propagate = False

    if log_file in (None, "", "-"):
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    if os.path.isabs(log_file):
        handler = logging.FileHandler(log_file, mode="w")
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    log_folder = os.path.join(output_folder, "log")
    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, log_file)
    handler = logging.FileHandler(log_file, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# json
def load_json(json_file: str, encoding: str = "utf-8") -> Dict:
    with open(json_file, "r", encoding=encoding) as fi:
        data = json.load(fi)
    return data


def save_json(
    json_file: str,
    obj: Any,
    encoding: str = "utf-8",
    ensure_ascii: bool = False,
    indent: Optional[int] = None,
    **kwargs,
) -> None:
    with open(json_file, "w", encoding=encoding) as fo:
        json.dump(obj, fo, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def bytes_to_json(data: bytes) -> Dict:
    return json.loads(data)


def dict_to_json(data: Dict) -> str:
    return json.dumps(data)


# cfg
def load_cfg(cfg_file: str, new_allowed: bool = True) -> CfgNode:
    """
    Load config from file.
    Args:
        cfg_file (str): config file path
        new_allowed (bool): whether to allow new keys in config
    """
    with open(cfg_file, "r") as fi:
        cfg = CfgNode.load_cfg(fi)
    cfg.set_new_allowed(new_allowed)
    return cfg


def add_variable_to_config(cfg: CfgNode, name: str, value: Any) -> CfgNode:
    """
    Add variable to config.
    Args:
        cfg (CfgNode): config
        name (str): variable name
        value (Any): variable value
    """
    cfg.defrost()
    cfg[name] = value
    cfg.freeze()
    return cfg


def merge_cfg_from_list(cfg: CfgNode, cfg_list: list) -> CfgNode:
    """
    Merge config from list.
    Args:
        cfg (CfgNode): config
        cfg_list (list): a list of config, it should be a list like
        `["key1", "value1", "key2", "value2"]`
    """
    cfg.defrost()
    cfg.merge_from_list(cfg_list)
    cfg.freeze()
    return cfg


def configure_api_from_environment(cfg):
    """Keep API credentials out of configuration files and config logs."""
    cfg.defrost()
    for variable, key in (("PROMOSIM_API_KEY", "api_keys"), ("PROMOSIM_API_BASE", "api_bases")):
        if os.environ.get(variable):
            cfg[key] = [os.environ[variable]]
    if os.environ.get("PROMOSIM_MODEL"):
        cfg["llm"] = os.environ["PROMOSIM_MODEL"]


def redacted_config(cfg):
    result = cfg.clone()
    result.defrost()
    if "api_keys" in result:
        result["api_keys"] = ["<redacted>"] * len(result["api_keys"])
    return result


def extract_item_names(observation: str, action: str = "RECOMMENDER") -> List[str]:
    """
    Extract item names from observation
    Args:
        observation: observation from the environment
        action: action type, RECOMMENDER or SOCIAL
    """
    item_names = []
    if observation.find("<") != -1:
        matches = re.findall(r"<(.*?)>", observation)
        # Filter out empty or slash-only matches which indicate malformed tags like <></>
        valid_matches = [m for m in matches if m.strip() and m.strip() != "/"]

        if valid_matches:
            item_names = valid_matches
        else:
            # Fallback for cases like <>Name></>
            # Remove <, > and trim / from ends
            cleaned = observation.replace("<", "").replace(">", "").strip().strip("/")
            if cleaned:
                item_names = [cleaned]
            else:
                item_names = []
    elif observation.find(";") != -1:
        item_names = observation.split(";")
        item_names = [item.strip(" '\"") for item in item_names]
    elif action == "RECOMMENDER":
        matches = re.findall(r'"([^"]+)"', observation)
        for match in matches:
            item_names.append(match)
    elif action == "SOCIAL":
        matches = re.findall(r'[<"]([^<>"]+)[">]', observation)
        for match in matches:
            item_names.append(match)
    return item_names


def layout_img(background, img, place: Tuple[int, int]):
    """
    Place the image on a specific position on the background
    Args:
        background: background image
        img: the specified image
        place: [top, left]
    """
    back_h, back_w, _ = background.shape
    height, width, _ = img.shape
    for i, j in itertools.product(range(height), range(width)):
        if img[i, j, 3]:
            background[place[0] + i, place[1] + j] = img[i, j, :3]


def get_avatar1(idx):
    """
    Retrieve the avatar for the specified index and encode it as a byte stream suitable for display in a text box.
    Args:
        idx (int): The index of the avatar, used to determine the path to the avatar image.
    """
    import cv2

    img = cv2.imread(f"./asset/img/v_1/{idx}.png")
    base64_str = cv2.imencode(".png", img)[1].tostring()
    avatar = "data:image/png;base64," + base64.b64encode(base64_str).decode("utf-8")
    msg = f'<img src="{avatar}" style="width: 100%; height: 100%; margin-right: 50px;">'
    return msg


def get_avatar2(idx):
    """
    Retrieve the avatar for the specified index and encode it as a Base64 data URI.
    Args:
        idx (int): The index of the avatar, used to determine the path to the avatar image.
    """
    import cv2

    img = cv2.imread(f"./asset/img/v_1/{idx}.png")
    base64_str = cv2.imencode(".png", img)[1].tostring()
    return "data:image/png;base64," + base64.b64encode(base64_str).decode("utf-8")


def html_format(orig_content: str):
    """
    Convert the original content to HTML format.
    Args:
        orig_content (str): The original content.
    """
    new_content = orig_content.replace("<", "")
    new_content = new_content.replace(">", "")
    for name in [
        "Eve",
        "Tommie",
        "Jake",
        "Lily",
        "Alice",
        "Sophia",
        "Rachel",
        "Lei",
        "Max",
        "Emma",
        "Ella",
        "Sen",
        "James",
        "Ben",
        "Isabella",
        "Mia",
        "Henry",
        "Charlotte",
        "Olivia",
        "Michael",
    ]:
        html_span = "<span style='color: red;'>" + name + "</span>"
        new_content = new_content.replace(name, html_span)
    new_content = new_content.replace("['", '<span style="color: #06A279;">[\'')
    new_content = new_content.replace("']", "']</span>")
    return new_content


# border: 0;
def chat_format(msg: Dict):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    html_text = "<br>"
    avatar = get_avatar2(msg["agent_id"])
    html_text += '<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += '<div style="background-color: #FAE1D1; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f"{msg['content']}"
    html_text += "</div></div>"
    return html_text


def rec_format(msg: Dict):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    html_text = "<br>"
    avatar = get_avatar2(msg["agent_id"])
    html_text += '<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += '<div style="background-color: #D9E8F5; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f"{msg['content']}"
    html_text += "</div></div>"
    return html_text


def social_format(msg: Dict):
    """
    Convert the message to HTML format.
    Args:
        msg (Dict): The message.
    """
    html_text = "<br>"
    avatar = get_avatar2(msg["agent_id"])
    html_text += '<div style="display: flex; align-items: center; margin-bottom: 10px;">'
    html_text += f'<img src="{avatar}" style="width: 10%; height: 10%; border: solid white; background-color: white; border-radius: 25px; margin-right: 10px;">'
    html_text += '<div style="background-color: #DFEED5; color: black; padding: 10px; border-radius: 10px; max-width: 80%;">'
    html_text += f"{msg['content']}"
    html_text += "</div></div>"
    return html_text


def round_format(round: int, agent_name: str):
    """
    Convert the round information to HTML format.
    Args:
        round (int): The round number.
        agent_name (str): The agent name.
    """
    round_info = ""
    round_info += '<div style="display: flex; font-family: 微软雅黑, sans-serif; font-size: 20px; color: #000000; font-weight: bold;">'
    round_info += f"&nbsp;&nbsp; Round: {round}  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  Actor: {agent_name}  &nbsp;&nbsp;"
    round_info += "</div>"
    return round_info


def ensure_dir(dir_path):
    """
    Make sure the directory exists, if it does not exist, create it
    Args:
        dir_path (str): The directory path.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_id(dir_name):
    ensure_dir(dir_name)
    existed_id = set()
    for f in os.listdir(dir_name):
        existed_id.add(f.split("-")[0])
    id = random.randint(1, 999999999)
    while id in existed_id:
        id = random.randint(1, 999999999)
    return id


def get_llm(config, logger, api_key, api_base):
    """
    Get the large language model.
    Args:
        config (CfgNode): The config.
        logger (Logger): The logger.
        api_key (str): The API key.
    """
    if config["llm"].find("gpt") != -1:
        LLM = ChatOpenAI(
            max_tokens=config["max_token"],
            temperature=config["temperature"],
            model_kwargs={"top_p": config.get("top_p", 0.8)},
            openai_api_key=api_key,
            openai_api_base=api_base,
            model=config["llm"],
            max_retries=config["max_retries"],
        )
    elif "http" not in api_base:
        LLM = SingletonLocalLLM.get_instance(
            config=config, api_key=api_key, logger=logger, api_base=api_base
        )
    else:
        LLM = CustomLLM(
            model=config["llm"],
            max_token=config["max_token"],
            max_retries=config["max_retries"],
            logger=logger,
            request_timeout=config.get("llm_request_timeout", 60),
            max_concurrency=config.get("llm_max_concurrency", 40),
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.8),
            top_k=config.get("top_k", 20),
            enable_thinking=config.get("enable_thinking", False),
            URL=api_base,
            api_key=api_key,
        )
    return LLM


def is_chatting(agent, agent2):
    """Determine if agent1 and agent2 is chatting"""
    name = agent.name
    agent_name2 = agent2.name
    return (
        (agent2.event.target_agent)
        and (agent.event.target_agent)
        and (name in agent2.event.target_agent)
        and (agent_name2 in agent.event.target_agent)
    )


def get_feature_description(feature):
    """Get description of given features."""
    descriptions = {
        "Watcher": "Choose movies, enjoy watching, and provide feedback and ratings to the recommendation system.",
        "Explorer": "Search for movies heard of before and expand movie experiences.",
        "Critic": "Demanding high standards for movies and the recommendation system, may criticize both the recommendation system and the movies.",
        "Chatter": "Engage in private conversations, trust friends' recommendations.",
        "Poster": "Enjoy publicly posting on social media and sharing content and insights with more people.",
    }
    features = feature.split(";")
    descriptions_list = [descriptions[feature] for feature in features if feature in descriptions]
    return ".".join(descriptions_list)


def count_files_in_directory(target_directory: str):
    """Count the number of files in the target directory"""
    return len(os.listdir(target_directory))


def get_avatar_url(id: int, gender: str, type: str = "origin", role=False):
    if role:
        target = "/asset/img/avatar/role/" + gender + "/"
        return target + str(id % 10) + ".png"
    target = "/asset/img/avatar/" + type + "/" + gender + "/"
    return target + str(id % 10) + ".png"


def calculate_entropy(movie_types):
    type_freq = {}
    for movie_type in movie_types:
        if movie_type in type_freq:
            type_freq[movie_type] += 1
        else:
            type_freq[movie_type] = 1

    total_movies = len(movie_types)

    entropy = 0
    for key in type_freq:
        prob = type_freq[key] / total_movies
        entropy -= prob * math.log2(prob)

    return entropy


def get_entropy(inters, data):
    genres = data.get_genres_by_id(inters)
    entropy = calculate_entropy(genres)
    return entropy


class HuggingFaceEmbeddingPool(Embeddings):
    def __init__(self, model_name: str):
        if not torch.cuda.is_available():
            raise RuntimeError("PromoSim embedding model requires CUDA")
        self.device = "cuda"
        self._model = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=dict(device=self.device)
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.embed_documents(texts)
        t = torch.tensor(vecs, dtype=torch.float32, device=self.device)
        norms = torch.norm(t, dim=1, keepdim=True)
        norms_safe = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
        normed = t / norms_safe
        return normed.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        v = self._model.embed_query(text)
        t = torch.tensor(v, dtype=torch.float32, device=self.device)
        n = torch.norm(t)
        return (t / n).cpu().tolist() if n.item() > 1e-12 else t.cpu().tolist()


@lru_cache(maxsize=1)
def get_embedding_model():
    from mosaic import paths

    local_model = paths.PROJECT_ROOT / "models" / "BAAI" / "bge-m3"
    model_name = (
        os.environ.get("PROMOSIM_EMBEDDING_MODEL")
        or os.environ.get("MOSAIC_TEXT_ENCODER")
        or (str(local_model) if local_model.is_dir() else "BAAI/bge-m3")
    )
    return 1024, HuggingFaceEmbeddingPool(model_name=model_name)
