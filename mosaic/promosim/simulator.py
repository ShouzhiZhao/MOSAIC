import argparse
import concurrent.futures
import heapq
import json
import logging
import math
import os
import re
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import List

import dill
import faiss
import numpy as np
import torch
from langchain.docstore import InMemoryDocstore
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_experimental.generative_agents import (
    GenerativeAgentMemory,
)
from openai import APIError
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from yacs.config import CfgNode

from .agents import BehavioralAgent, InteractiveAgent
from .agents.memory_backend import BehavioralMemory, BehavioralMemoryRetriever
from .recommender.data.data import Data
from .recommender.model import Random
from .recommender.recommender import Recommender
from .utils import interval, message, utils
from .utils.concurrency import broadcast_map
from .utils.event import reset_event, update_event
from .utils.message import Message

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
logging.basicConfig(level=logging.ERROR)

lock = threading.Lock()


class Simulator:
    """
    Simulator class for running the simulation.
    """

    def __init__(self, config: CfgNode, logger: logging.Logger):
        if not torch.cuda.is_available():
            raise RuntimeError("PromoSim requires CUDA; CPU fallback is disabled")
        self.config = config
        self.logger = logger
        self.round_cnt = 0
        self.round_msg: List[Message] = []
        self.active_agents: List[int] = []  # active agents in current round
        self.active_agent_threshold = config["active_agent_threshold"]
        self.active_method = config["active_method"]
        self.file_name_path: List[str] = []
        self.play_event = threading.Event()
        self.working_agents: List[BehavioralAgent] = []  # busy agents
        self.now = datetime.now().replace(hour=8, minute=0, second=0)
        self.interval = interval.parse_interval(config["interval"])
        self.round_entropy = []
        self.rec_stat = message.RecommenderStat(
            tot_user_num=0,
            cur_user_num=0,
            tot_item_num=0,
            inter_num=0,
            rec_model=config["rec_model"],
            pop_items=[],
        )
        self.social_stat = message.SocialStat(
            tot_user_num=0,
            cur_user_num=0,
            tot_link_num=0,
            chat_num=0,
            cur_chat_num=0,
            post_num=0,
            pop_items=[],
            network_density=0,
        )
        # Reuse the singleton embedding model to save GPU memory
        _, pool = utils.get_embedding_model()
        self.model_st = pool._model.client

    def generate_content_embeddings(self, update_user=True, update_item=True):
        """
        Generate content embeddings for users and items and set them in the recommender model.
        """
        if not hasattr(self.recsys.model, "set_content_embeddings"):
            return

        # Only proceed if the model is configured to use content
        if not getattr(self.recsys.model, "use_content", False):
            return

        started = time.perf_counter()
        descriptions = self.get_initial_content_descriptions()
        previous = dict(getattr(self, "_content_texts", {}))
        batch_size = int(self.config.get("content_embedding_batch_size", 16))
        if batch_size < 1:
            raise ValueError("content_embedding_batch_size must be positive")
        updated = []
        counts = []
        for kind, texts, requested in zip(
            ("user", "item"), descriptions, (update_user, update_item)
        ):
            cached = getattr(self.recsys.model, f"{kind}_content_emb", None)
            if not requested and cached is not None:
                updated.append(cached)
                counts.append(0)
                continue
            old_texts = previous.get(kind)
            if cached is None or old_texts is None or len(old_texts) != len(texts):
                changed = list(range(len(texts)))
            else:
                changed = [i for i, text in enumerate(texts) if text != old_texts[i]]
            if changed:
                encoded = self.model_st.encode(
                    [texts[i] for i in changed],
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device="cuda",
                )
                if cached is None or len(cached) != len(texts):
                    cached = encoded
                else:
                    cached = cached.clone()
                    cached[changed] = encoded.to(cached.device)
            updated.append(cached)
            counts.append(len(changed))
            previous[kind] = list(texts)
        if any(counts):
            self.recsys.model.set_content_embeddings(*updated)
        self._content_texts = previous
        self.logger.info(
            "Content embeddings seconds=%.3f encoded_users=%d encoded_items=%d",
            time.perf_counter() - started,
            *counts,
        )

    def get_initial_content_descriptions(self):
        """Return the exact user/item strings encoded by frozen BGE-M3.

        On construction the memory fields are empty. Saving these strings in
        the aggregate PromoSim record makes the CASO content feature auditable.
        """
        user_texts = []
        for user_id in range(self.config["agent_num"]):
            agent = self.agents[user_id]
            user = self.data.users[user_id]
            traits_value = user.get("traits", [])
            traits = ",".join(traits_value) if isinstance(traits_value, list) else str(traits_value)
            interest = agent.interest if isinstance(agent.interest, str) else str(agent.interest)
            profile = (
                f"agent profile: name {agent.name}, age {user.get('age', '')}, "
                f"gender {user.get('gender', '')}, traits {traits}, "
                f"interests {interest}, status {user.get('status', '')}"
            )
            memory = (
                "short-term memory: "
                + agent.memory.get_short_memory()
                + "\nlong-term memory: "
                + agent.memory.get_long_memory()
            )
            user_texts.append(profile + "\n" + memory)
        return user_texts, self.data.get_item_strings()

    def get_file_name_path(self):
        return self.file_name_path

    def load_simulator(self, initialization_state=None):
        """Load and initiate the simulator."""
        self.round_cnt = 0
        self._initial_interests = (initialization_state or {}).get("interests")
        self.data = Data(self.config)
        self.all_interests = self.data.get_all_interests()
        self.agents = self.agent_creation()

        # Sort agents by id to ensure deterministic order
        self.agents = dict(sorted(self.agents.items()))

        self.recsys = Recommender(self.config, self.logger, self.data)
        if initialization_state:
            from .integrity import restore_recommender

            restore_recommender(self.recsys.model, initialization_state["recommender"])
            users, items = self.get_initial_content_descriptions()
            self._content_texts = {"user": users, "item": items}
        else:
            self.generate_content_embeddings()
        self.logger.info("Simulator loaded.")

    def save(self, save_dir_name):
        """Save the simulator status of current round"""
        utils.ensure_dir(save_dir_name)
        ID = utils.generate_id(self.config["simulator_dir"])
        file_name = f"{ID}-Round[{self.round_cnt}]-AgentNum[{self.config['agent_num']}]-{datetime.now().strftime('%Y-%m-%d-%H_%M_%S')}"
        self.file_name_path.append(file_name)
        save_file_name = os.path.join(save_dir_name, file_name + ".pkl")
        with open(save_file_name, "wb") as f:
            dill.dump(self.__dict__, f)
        self.logger.info("Current simulator Save in: \n" + str(save_file_name) + "\n")
        self.logger.info("Simulator File Path (root -> node): \n" + str(self.file_name_path) + "\n")
        utils.ensure_dir(self.config["ckpt_path"])
        cpkt_path = os.path.join(self.config["ckpt_path"], file_name + ".pth")
        if not isinstance(self.recsys.model, Random):
            self.recsys.save_model(cpkt_path)
            self.logger.info("Current Recommender Model Save in: \n" + str(cpkt_path) + "\n")
        else:
            self.logger.info("Random model does not need to be saved.\n")

    @classmethod
    def restore(cls, restore_file_name, config, logger):
        """Restore the simulator status from the specific file"""
        with open(restore_file_name + ".pkl", "rb") as f:
            obj = cls.__new__(cls)
            obj.__dict__ = dill.load(f)
            obj.config, obj.logger = config, logger
            return obj

    def relevance_score_fn(self, score: float) -> float:
        d = math.sqrt(max(score, 0.0))
        return max(0.0, min(1.0, 1.0 - d / 2.0))

    def create_new_memory_retriever(self):
        """Create a new vector store retriever unique to the agent."""
        # Define your embedding model
        embedding_size, embeddings_model = utils.get_embedding_model()
        # Initialize the vectorstore as empty
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(
            embeddings_model,
            index,
            InMemoryDocstore({}),
            {},
            relevance_score_fn=self.relevance_score_fn,
            normalize_L2=True,
        )

        # If choose BehavioralMemory, you must use BehavioralMemoryRetriever rather than TimeWeightedVectorStoreRetriever.
        RetrieverClass = (
            BehavioralMemoryRetriever
            if self.config["memory_backend"] == "behavioral"
            else TimeWeightedVectorStoreRetriever
        )

        return RetrieverClass(
            vectorstore=vectorstore, other_score_keys=["importance"], now=self.now, k=5
        )

    def check_active(self, index: int):
        # If agent's previous action is completed, reset the event
        agent = self.agents[index]
        if isinstance(agent, InteractiveAgent):
            return True

        if self.active_agent_threshold and len(self.active_agents) >= self.active_agent_threshold:
            return False
        # If the movie does not end, the agent continues watching the movie.
        if agent.event.action_type == "watching":
            self.round_msg.append(
                Message(
                    agent_id=agent.id,
                    action="WATCH",
                    content=f"{agent.name} is watching movie.",
                )
            )
            return False

        active_prob = agent.get_active_prob(self.active_method)
        if np.random.random() > active_prob:
            agent.no_action_round += 1
            return False
        self.active_agents.append(index)
        return True

    def pause(self):
        self.play_event.clear()

    def play(self):
        self.play_event.set()

    def global_message(self, message: str):
        for i, agent in self.agents.items():
            agent.memory.add_memory(message, self.now)

    def update_stat(self):
        self.rec_stat.tot_user_num = len(self.agents)
        self.social_stat.tot_user_num = len(self.agents)
        self.rec_stat.cur_user_num = 0
        self.social_stat.cur_user_num = 0
        self.social_stat.cur_chat_num = 0
        for agent in self.working_agents:
            if agent.event.action_type == "watching":
                self.rec_stat.cur_user_num += 1
            elif agent.event.action_type == "chatting":
                self.social_stat.cur_user_num += 1
                self.social_stat.cur_chat_num += len(agent.event.target_agent)
        self.rec_stat.pop_items = self.data.get_inter_popular_items()
        self.social_stat.pop_items = self.data.get_mention_popular_items()
        self.rec_stat.tot_item_num = self.data.get_item_num()
        self.rec_stat.inter_num = self.recsys.get_inter_num()
        self.social_stat.tot_link_num = self.data.get_relationship_num() / 2
        self.social_stat.cur_chat_num /= 2
        self.social_stat.network_density = self.recsys.data.get_network_density()
        # chat_num and post_num update in the one_step function

    def one_step(self, agent_id):
        """Run one step of an agent."""
        self.play_event.wait()
        if not self.check_active(agent_id):
            return [Message(agent_id=agent_id, action="NO_ACTION", content="No action.")]
        agent: BehavioralAgent = self.agents[agent_id]
        name = agent.name
        user_id = agent.user_id
        message = []
        # while True:
        try:
            choice, observation = agent.take_action(self.now)
        except APIError:
            raise
        except Exception:
            traceback.print_exc()
            choice, observation = "NOTHING", "NOTHING"
        with lock:
            heapq.heappush(self.working_agents, agent)
        if "RECOMMENDER" in choice:
            ids = []
            self.logger.info(f"{name} enters the recommender system.")
            message.append(
                Message(
                    agent_id=agent_id,
                    action="RECOMMENDER",
                    content=f"{name} enters the recommender system.",
                )
            )
            self.round_msg.append(
                Message(
                    agent_id=agent_id,
                    action="RECOMMENDER",
                    content=f"{name} enters the recommender system.",
                )
            )
            leave = False
            with torch.no_grad():
                item_ids, rec_items = self.recsys.get_full_sort_items(agent_id)
            page = 0
            cnt = 0
            searched_name = None
            while not leave:
                rec_score = agent.generate_rec_list_rating(observation, self.now)
                self.logger.info(
                    f"{name} [{user_id}] is recommended {item_ids[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]} {rec_items[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]}, score={rec_score}."
                )
                message.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} [{user_id}] is recommended {item_ids[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]} {rec_items[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]}, score={rec_score}.",
                    )
                )
                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="RECOMMENDER",
                        content=f"{name} [{user_id}] is recommended {item_ids[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]} {rec_items[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]}, score={rec_score}.",
                    )
                )

                observation = f"{name} is browsing the recommender system."
                if searched_name is not None:
                    observation = (
                        observation
                        + f" {name} has searched for {searched_name} in recommender system and recommender system returns item list:{rec_items[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]} as search results."
                    )
                else:
                    observation = (
                        observation
                        + f" {name} [{user_id}] is recommended {item_ids[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]} {rec_items[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]}."
                    )
                choice, action = agent.take_recommender_action(observation, self.now)
                self.recsys.update_history_by_id(
                    agent_id,
                    item_ids[page * self.recsys.page_size : (page + 1) * self.recsys.page_size],
                )
                ids.extend(
                    item_ids[page * self.recsys.page_size : (page + 1) * self.recsys.page_size]
                )

                if "BUY" in choice and (
                    agent.event.action_type == "idle" or agent.event.action_type == "posting"
                ):
                    if not isinstance(action, int):
                        action = 1
                    item_names = rec_items[page * self.recsys.page_size + action - 1]
                    item_id = item_ids[page * self.recsys.page_size + action - 1]
                    duration = 2
                    agent.event = update_event(
                        original_event=agent.event,
                        start_time=self.now,
                        duration=duration,
                        target_agent=None,
                        action_type="watching",
                    )

                    self.logger.info(f"{name} [{user_id}] watches {item_names} [{item_id}]")
                    message.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} watches {item_names}.",
                            item_id=int(item_id),
                            event_type="watch",
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} watches {item_names}.",
                            item_id=int(item_id),
                            event_type="watch",
                        )
                    )
                    agent.update_watched_history(self.data.get_item_names([item_id]))
                    self.recsys.update_positive_by_id(agent_id, item_id)

                    for i in range(self.recsys.page_size):
                        if i == action - 1:
                            self.recsys.add_train_data(
                                agent_id, item_ids[page * self.recsys.page_size + i], 1
                            )
                        else:
                            self.recsys.add_train_data(
                                agent_id, item_ids[page * self.recsys.page_size + i], 0
                            )

                    item_descriptions = self.data.get_item_description_by_id([item_id])
                    if len(item_descriptions) == 0:
                        self.logger.info(f"{name} leaves the recommender system.")
                        message.append(
                            Message(
                                agent_id=agent_id,
                                action="RECOMMENDER",
                                content=f"{name} leaves the recommender system.",
                            )
                        )
                        self.round_msg.append(
                            Message(
                                agent_id=agent_id,
                                action="RECOMMENDER",
                                content=f"{name} leaves the recommender system.",
                            )
                        )
                        leave = True
                        continue

                    observation = (
                        f"{name} has just finished watching {item_names};;{item_descriptions[0]}."
                    )
                    feelings = agent.generate_feeling(
                        observation, self.now + timedelta(hours=duration)
                    )
                    self.logger.info(f"{name} feels: {feelings}")

                    message.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} feels: {feelings}",
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} feels: {feelings}",
                        )
                    )
                    searched_name = None
                    leave = True

                elif "NEXT" in choice:
                    self.logger.info(f"{name} looks next page.")
                    for i in range(self.recsys.page_size):
                        self.recsys.add_train_data(
                            agent_id, item_ids[page * self.recsys.page_size + i], 0
                        )
                    message.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} looks next page.",
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} looks next page.",
                        )
                    )
                    if (page + 1) * self.recsys.page_size < len(rec_items):
                        page = page + 1
                    else:
                        self.logger.info("No more items.")
                        self.logger.info(f"{name} leaves the recommender system.")
                        message.append(
                            Message(
                                agent_id=agent_id,
                                action="RECOMMENDER",
                                content=f"No more items. {name} leaves the recommender system.",
                            )
                        )
                        self.round_msg.append(
                            Message(
                                agent_id=agent_id,
                                action="RECOMMENDER",
                                content=f"No more items. {name} leaves the recommender system.",
                            )
                        )
                        leave = True
                elif "SEARCH" in choice:
                    observation = f"{name} is searching in recommender system."
                    item_name = agent.search_item(observation, self.now)
                    self.logger.info(f"{name} searches {item_name}.")
                    message.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} searches {item_name}.",
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} searches {item_name}.",
                        )
                    )
                    item_names = utils.extract_item_names(item_name)
                    if item_names == []:
                        agent.memory.add_memory(
                            "There are no items related in the system.", now=self.now
                        )
                        self.logger.info("There are no related items in the system.")
                        message.append(
                            Message(
                                agent_id=agent_id,
                                action="RECOMMENDER",
                                content="There are no related products in the system.",
                            )
                        )
                        self.round_msg.append(
                            Message(
                                agent_id=agent_id,
                                action="RECOMMENDER",
                                content="There are no related products in the system.",
                            )
                        )
                        leave = True
                        continue
                    item_name = item_names[0]
                    search_items, item_ids = self.recsys.get_search_items(item_name)
                    self.logger.info(f"Search Engine returned {search_items} {item_ids}.")
                    message.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"Search Engine returned {search_items} {item_ids}.",
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"Search Engine returned {search_items} {item_ids}.",
                        )
                    )
                    if len(search_items) != 0:
                        rec_items = search_items
                        # item_ids = self.data.get_item_ids_exact(search_items)
                        page = 0
                        searched_name = item_name
                    else:
                        agent.memory.add_memory(
                            f"There are no items related to {item_name} in the system.",
                            now=self.now,
                        )
                        self.logger.info("There are no related items in the system.")
                        message.append(
                            Message(
                                agent_id=agent_id,
                                action="RECOMMENDER",
                                content="There are no related products in the system.",
                            )
                        )
                        self.round_msg.append(
                            Message(
                                agent_id=agent_id,
                                action="RECOMMENDER",
                                content="There are no related products in the system.",
                            )
                        )
                else:
                    self.logger.info(f"{name} leaves the recommender system.")
                    message.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} leaves the recommender system.",
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} leaves the recommender system.",
                        )
                    )
                    leave = True
                cnt += 1
                if cnt == 5:
                    self.logger.info(f"{name} leaves the recommender system.")
                    message.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} leaves the recommender system.",
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=agent_id,
                            action="RECOMMENDER",
                            content=f"{name} leaves the recommender system.",
                        )
                    )
                    leave = True
            entropy = utils.get_entropy(ids, self.data)
            self.round_entropy.append(entropy)
            self.recsys.round_record[agent_id].append(ids)

        elif "SOCIAL" in choice:
            contacts = self.data.get_all_contacts(agent_id)
            if len(contacts) == 0:
                self.logger.info(f"{name} has no acquaintance.")
                message.append(
                    Message(
                        agent_id=agent_id,
                        action="SOCIAL",
                        content=f"{name} has no acquaintance.",
                    )
                )
                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="SOCIAL",
                        content=f"{name} has no acquaintance.",
                    )
                )
            else:
                self.social_stat.cur_user_num += 1
                self.logger.info(f"{name} is going to social media.")
                message.append(
                    Message(
                        agent_id=agent_id,
                        action="SOCIAL",
                        content=f"{name} is going to social media.",
                    )
                )
                self.round_msg.append(
                    Message(
                        agent_id=agent_id,
                        action="SOCIAL",
                        content=f"{name} is going to social media.",
                    )
                )
                observation = f"{name} is going to social media. {name} and {contacts} are acquaintances. {name} can chat with acquaintances, or post to all acquaintances. What will {name} do?"
                choice, action, duration = agent.take_social_action(observation, self.now)
                if "CHAT" in choice:
                    agent_name2 = action.strip(" \t\n'\"")
                    agent_id2 = self.data.get_user_ids([agent_name2])[0]
                    if agent_id2 < len(self.agents):
                        agent2 = self.agents[agent_id2]
                        # If agent2 is watching moives, he cannot be interupted.
                        if agent2.event.action_type == "watching":
                            agent.memory.add_memory(
                                f"{agent.name} wants to chat with {agent_name2}, but {agent_name2} is watching. So {agent.name} does nothing.",
                                now=self.now,
                            )
                            self.logger.info(
                                f"{name} wants to chat with {agent_name2}, but {agent_name2} is watching. So {name} does nothing."
                            )
                            message.append(
                                Message(
                                    agent_id=agent_id,
                                    action="LEAVE",
                                    content=f"{name} wants to chat with {agent_name2}, but {agent_name2} is watching. So {name} does nothing.",
                                )
                            )
                            self.round_msg.append(
                                Message(
                                    agent_id=agent_id,
                                    action="LEAVE",
                                    content=f"{name} wants to chat with {agent_name2}, but {agent_name2} is watching. So {name} does nothing.",
                                )
                            )
                            return message

                        #  If agent2 is chatting with agent1, skipping this round
                        if utils.is_chatting(agent, agent2):
                            self.logger.info(f"{name} is chatting with {agent_name2}")
                            message.append(
                                Message(
                                    agent_id=agent_id,
                                    action="CHAT",
                                    content=f"{name} is chatting with {agent_name2}",
                                )
                            )
                            self.round_msg.append(
                                Message(
                                    agent_id=agent_id,
                                    action="CHAT",
                                    content=f"{name} is chatting with {agent_name2}.",
                                )
                            )
                            return message
                        agent.event = update_event(
                            original_event=agent.event,
                            start_time=self.now,
                            duration=duration,
                            target_agent=agent_name2,
                            action_type="chatting",
                        )
                        agent2.event = update_event(
                            original_event=agent2.event,
                            start_time=self.now,
                            duration=duration,
                            target_agent=name,
                            action_type="chatting",
                        )
                        self.logger.info(f"{name} is chatting with {agent_name2}.")
                        self.social_stat.chat_num += 1
                        message.append(
                            Message(
                                agent_id=agent_id,
                                action="CHAT",
                                content=f"{name} is chatting with {agent_name2}.",
                            )
                        )
                        self.round_msg.append(
                            Message(
                                agent_id=agent_id,
                                action="CHAT",
                                content=f"{name} is chatting with {agent_name2}.",
                            )
                        )
                        # If the system has a role, and it is her term now.
                        if self.config["play_role"] and self.data.role_id == agent_id:
                            conversation = ""
                            observation = f"{name} is going to chat with {agent2.name}."
                            # Obtain the response from the role.
                            contin, result, role_dia = agent.generate_role_dialogue(
                                agent2, observation
                            )
                            conversation += role_dia + result
                            self.logger.info(role_dia)
                            self.logger.info(result)
                            # If both of them do not stop, an extra round will be held.
                            while contin:
                                contin, result, role_dia = agent.generate_role_dialogue(
                                    agent2, observation, conversation
                                )
                                conversation += role_dia + result
                                self.logger.info(role_dia)
                                self.logger.info(result)
                        else:
                            observation = f"{name} is going to chat with {agent2.name}."
                            # If an agent wants to chat with the role.
                            if self.config["play_role"] and agent_id2 == self.data.role_id:
                                conversation = ""
                                observation = f"{name} is going to chat with {agent2.name}."
                                # Obtain the response from the agent(LLM).
                                contin, result = agent.generate_dialogue_response(observation)
                                agent_dia = "%s %s" % (agent.name, result)
                                self.logger.info(agent_dia)
                                # Obtain the response from the role.
                                role_contin, role_dia = agent2.generate_dialogue_response(
                                    observation
                                )
                                self.logger.info(role_dia)
                                contin = contin and role_contin
                                conversation += agent_dia + role_dia
                                # If both of them do not stop, an extra round will be held.
                                while contin:
                                    observation = f"{name} is going to chat with {agent2.name}."
                                    contin, result = agent.generate_dialogue_response(observation)
                                    agent_dia = "%s %s" % (agent.name, result)
                                    self.logger.info(agent_dia)
                                    (
                                        role_contin,
                                        role_dia,
                                    ) = agent2.generate_dialogue_response(observation)
                                    self.logger.info(role_dia)
                                    contin = contin and role_contin
                                    conversation += agent_dia + role_dia
                            else:
                                # Otherwise, two agents(LLM) will generate dialogues.
                                conversation = agent.generate_dialogue(agent2, observation)
                            self.logger.info(conversation)

                        msgs = []
                        matches = re.findall(r"\[([^]]+)\]:\s*(.*)", conversation)
                        for match in matches:
                            speaker = match[0]
                            content = match[1]
                            if speaker == agent.name:
                                id = agent_id
                                id2 = agent_id2
                            else:
                                id = agent_id2
                                id2 = agent_id
                            item_names = utils.extract_item_names(content, "SOCIAL")
                            self.data.add_mention_cnt(item_names)
                            if item_names != []:
                                self.agents[id2].update_heared_history(item_names)
                            msgs.append(
                                Message(
                                    agent_id=id,
                                    action="CHAT",
                                    content=f"{speaker} says:{content}",
                                )
                            )
                            self.round_msg.append(
                                Message(
                                    agent_id=id,
                                    action="CHAT",
                                    content=f"{speaker} says:{content}",
                                )
                            )
                        message.extend(msgs)

                else:
                    self.social_stat.post_num += 1
                    self.logger.info(f"{name} is posting.")
                    observation = f"{name} want to post for all acquaintances."
                    observation = agent.publish_posting(observation, self.now)
                    item_names = utils.extract_item_names(observation, "SOCIAL")
                    self.logger.info(agent.name + " posted: " + observation)
                    if agent.event.action_type == "idle":
                        agent.event = update_event(
                            original_event=agent.event,
                            start_time=self.now,
                            duration=0.1,
                            target_agent=None,
                            action_type="posting",
                        )
                    message.append(
                        Message(
                            agent_id=agent_id,
                            action="POST",
                            content=agent.name + " posts: " + observation,
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=agent_id,
                            action="POST",
                            content=agent.name + " posts: " + observation,
                        )
                    )
                    message.extend(self.deliver_post(agent_id, observation, item_names))

                    self.logger.info(f"{contacts} get this post.")

        else:
            self.logger.info(f"{name} does nothing.")
            message.append(
                Message(agent_id=agent_id, action="LEAVE", content=f"{name} does nothing.")
            )
            self.round_msg.append(
                Message(agent_id=agent_id, action="LEAVE", content=f"{name} does nothing.")
            )

        return message

    def deliver_post(self, sender_id, observation, item_names):
        """Complete a broadcast before returning, preserving recipient order."""
        sender = self.agents[sender_id]
        contacts = set(self.data.get_all_contacts(sender_id))
        recipients = [agent for agent in self.agents.values() if agent.name in contacts]
        workers = int(self.config.get("broadcast_workers", 16))
        # The behavioral backend synchronizes all memory mutation/retrieval.
        # Other backends retain their serial delivery behavior.
        if self.config["execution_mode"] != "parallel" or any(
            not isinstance(agent.memory, BehavioralMemory) for agent in recipients
        ):
            workers = 1

        def deliver(agent):
            agent.memory.add_memory(sender.name + " posts: " + observation, now=self.now)
            agent.update_heared_history(item_names)
            return Message(
                agent_id=agent.id,
                action="POST",
                content=agent.name + " observes that" + sender.name + " posts: " + observation,
            )

        started = time.perf_counter()
        messages = broadcast_map(deliver, recipients, workers)
        self.round_msg.extend(messages)
        self.logger.info(
            "Broadcast sender=%d recipients=%d seconds=%.3f",
            sender_id,
            len(recipients),
            time.perf_counter() - started,
        )
        return messages

    def update_working_agents(self):
        with lock:
            agent: BehavioralAgent = None
            while len(self.working_agents) > 0:
                agent = heapq.heappop(self.working_agents)
                if agent.event.end_time <= self.now:
                    agent.event = reset_event(self.now)
                else:
                    break
            if agent is not None and agent.event.end_time > self.now:
                heapq.heappush(self.working_agents, agent)

    def round(self):
        """
        Run one step for all agents.
        """
        messages = []
        futures = []
        actions_started = time.perf_counter()
        # The user's role takes one step first.
        if self.config["play_role"]:
            role_msg = self.one_step(self.data.role_id)
            messages.extend(role_msg)
        if self.config["execution_mode"] == "parallel":
            workers = self.config.get("parallel_workers", None)
            executor_args = {}
            if workers is not None and int(workers) > 0:
                executor_args["max_workers"] = int(workers)
            with concurrent.futures.ThreadPoolExecutor(**executor_args) as executor:
                for i in range(self.config["agent_num"]):
                    futures.append(executor.submit(self.one_step, i))
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Completed agents",
                ):
                    msgs = future.result()
                    messages.append(msgs)
        else:
            for i in tqdm(range(self.config["agent_num"])):
                msgs = self.one_step(i)
                messages.append(msgs)
        self.logger.info("Round actions seconds=%.3f", time.perf_counter() - actions_started)
        self.now = interval.add_interval(self.now, self.interval)

        for i, agent in self.agents.items():
            agent.memory.update_now(self.now)

        self.update_working_agents()
        if self.config.get("predict_social_each_round", False):
            messages.append(self.predict_social())
        self.generate_content_embeddings()
        if self.config.get("rec_train", False):
            training_started = time.perf_counter()
            self.recsys.train()
            self.recsys.clear_train_data()
            self.logger.info(
                "Round recommendation training seconds=%.3f", time.perf_counter() - training_started
            )
        if self.config.get("simulator_dir", False):
            self.save(os.path.join(self.config["simulator_dir"]))
        return messages

    def convert_agent_to_role(self, agent_id):
        self.agents[agent_id] = InteractiveAgent.from_behavioral_agent(self.agents[agent_id])

    def create_agent(self, id, api_key, api_base) -> BehavioralAgent:
        """
        Create an agent with the given id.
        :param id: the id of the agent.
        :param api_key: the API key of the agent.
        :param api_base: the API base of the agent.
        """
        LLM = utils.get_llm(
            config=self.config, logger=self.logger, api_key=api_key, api_base=api_base
        )
        MemoryClass = (
            BehavioralMemory
            if (
                self.config.get("memory_backend", self.config.get("agent_memory", "behavioral"))
                == "behavioral"
            )
            else GenerativeAgentMemory
        )

        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            now=self.now,
            verbose=False,
            reflection_threshold=10,
        )
        agent = BehavioralAgent(
            id=id,
            name=self.data.users[id]["name"],
            age=self.data.users[id]["age"],
            gender=self.data.users[id]["gender"],
            traits=self.data.users[id]["traits"],
            status=self.data.users[id]["status"],
            interest=self.data.users[id]["interest"],
            relationships=self.data.get_relationship_names(id),
            feature=utils.get_feature_description(self.data.users[id]["feature"]),
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=reset_event(self.now),
            avatar_url=utils.get_avatar_url(
                id=id, gender=self.data.users[id]["gender"], type="origin"
            ),
            idle_url=utils.get_avatar_url(id=id, gender=self.data.users[id]["gender"], type="idle"),
            watching_url=utils.get_avatar_url(
                id=id, gender=self.data.users[id]["gender"], type="watching"
            ),
            chatting_url=utils.get_avatar_url(
                id=id, gender=self.data.users[id]["gender"], type="chatting"
            ),
            posting_url=utils.get_avatar_url(
                id=id, gender=self.data.users[id]["gender"], type="posting"
            ),
        )
        agent.user_id = self.data.users[id]["user_id"]
        agent.choose = self.data.users[id]["group"] == "choose"
        if getattr(self, "_initial_interests", None) is not None:
            agent.interest = self._initial_interests[str(id)]
        elif self.config.generate_interests:
            new_interest = agent.select_interest(self.config.all_interests, agent.choose)
            if self.config.get("rate_initial_tags", True):
                self.logger.info(
                    f"{agent.name} [{agent.user_id}] rates the tags: {agent.rate_tags(self.config.all_interests)}"
                )
            if not agent.choose:
                new_interest = ", ".join(
                    (
                        t
                        for t in self.config.all_interests
                        if t not in {i.strip() for i in new_interest.split(",")}
                    )
                )
            self.logger.info(
                f"{agent.name} [{agent.user_id}] in group {'choose' if agent.choose else 'reject'}, interest in profile is {agent.interest}, has selected new interests: {new_interest}"
            )
            # agent.interest = f'interested types: {agent.interest}, interested genres: {new_interest}'
            agent.interest = new_interest
        # observations = self.data.users[i]["observations"].strip(".").split(".")
        # for observation in observations:
        #     agent.memory.add_memory(observation, now=self.now)
        return agent

    def create_user_role(self, id, api_key, api_base):
        """
        @ Zeyu Zhang
        Create a user controllable agent.
        :param id: the id of role.
        :param api_key: the API key of the role.
        :param api_base: the API base of the role.
        :return: an object of `InteractiveAgent`.
        """
        name, gender, age, traits, status, interest, feature = (
            "Tommy",
            "male",
            23,
            "happy",
            "nice",
            "sci-fic",
            "Watcher",
        )
        relationships = {0: "friend", 1: "friend"}
        event = reset_event(self.now)
        avatar_url = utils.get_avatar_url(id=id, gender=gender, type="origin", role=True)
        idle_url = utils.get_avatar_url(id=id, gender=gender, type="idle", role=True)
        watching_url = utils.get_avatar_url(id=id, gender=gender, type="watching", role=True)
        chatting_url = utils.get_avatar_url(id=id, gender=gender, type="chatting", role=True)
        posting_url = utils.get_avatar_url(id=id, gender=gender, type="posting", role=True)
        LLM = utils.get_llm(
            config=self.config, logger=self.logger, api_key=api_key, api_base=api_base
        )
        MemoryClass = (
            BehavioralMemory
            if (
                self.config.get("memory_backend", self.config.get("agent_memory", "behavioral"))
                == "behavioral"
            )
            else GenerativeAgentMemory
        )
        agent_memory = MemoryClass(
            llm=LLM,
            memory_retriever=self.create_new_memory_retriever(),
            verbose=False,
            reflection_threshold=10,
            now=self.now,
        )
        agent = InteractiveAgent(
            id=id,
            name=name,
            age=age,
            gender=gender,
            traits=traits,
            status=status,
            interest=interest,
            relationships=relationships,
            feature=feature,
            memory_retriever=self.create_new_memory_retriever(),
            llm=LLM,
            memory=agent_memory,
            event=event,
            avatar_url=avatar_url,
            idle_url=idle_url,
            watching_url=watching_url,
            chatting_url=chatting_url,
            posting_url=posting_url,
        )

        self.data.load_role(id, name, gender, age, traits, status, interest, feature, relationships)

        return agent

    def agent_creation(self):
        """
        Create agents in parallel
        """
        agents = {}
        api_keys = list(self.config["api_keys"])
        api_bases = list(self.config["api_bases"])
        agent_num = int(self.config["agent_num"])
        # Add ONE user controllable user into the simulator if the flag is true.
        # We block the main thread when the user is creating the role.
        if self.config["play_role"]:
            role_id = self.data.get_user_num()
            api_key = api_keys[role_id % len(api_keys)]
            api_base = api_bases[role_id % len(api_bases)]
            agent = self.create_user_role(role_id, api_key, api_base)
            agents[role_id] = agent
            self.data.role_id = role_id
        if self.active_method == "random":
            active_probs = [self.config["active_prob"]] * agent_num
        else:
            active_probs = np.random.pareto(self.config["active_prob"] * 10, agent_num)
            active_probs = active_probs / active_probs.max()

        if self.config["execution_mode"] == "parallel":
            futures = []
            start_time = time.time()
            workers = self.config.get("parallel_workers", None)
            executor_args = {}
            if workers is not None and int(workers) > 0:
                executor_args["max_workers"] = int(workers)
            with concurrent.futures.ThreadPoolExecutor(**executor_args) as executor:
                for i in range(agent_num):
                    api_key = api_keys[i % len(api_keys)]
                    api_base = api_bases[i % len(api_bases)]
                    futures.append(executor.submit(self.create_agent, i, api_key, api_base))
                for future in tqdm(concurrent.futures.as_completed(futures)):
                    agent = future.result()
                    agent.active_prob = active_probs[agent.id]
                    agents[agent.id] = agent
            end_time = time.time()
            self.logger.info(f"Time for creating {agent_num} agents: {end_time - start_time}")
        else:
            for i in tqdm(range(agent_num)):
                api_key = api_keys[i % len(api_keys)]
                api_base = api_bases[i % len(api_bases)]
                agent = self.create_agent(i, api_key, api_base)
                agent.active_prob = active_probs[agent.id]
                agents[agent.id] = agent

        return agents

    def reset(self):
        # Reset the system
        self.pause()
        self.round_cnt = 0
        log_string = ""
        self.load_simulator()
        log_string = "The system is reset, and the historic records are removed."
        self.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
        return log_string

    def start(self):
        self.play()
        messages = []
        for i in range(self.round_cnt + 1, self.config["round"] + 1):
            self.round_cnt = self.round_cnt + 1
            self.logger.info(f"Round {self.round_cnt}")
            message = self.round()
            messages.append(message)
            with open(self.config["output_file"], "w") as file:
                json.dump(messages, file, default=lambda o: o.__dict__, indent=4)
            self.recsys.save_interaction()
            if self.config["simulator_dir"] is not None:
                self.save(os.path.join(self.config["simulator_dir"]))

    def clear_social(self):
        for i in self.agents:
            agent = self.agents[i]
            agent.relationships = {}
            self.data.users[agent.id]["contact"] = {}

    def add_relation(self, user_1, user_2, relationship):
        if "contact" not in self.data.users[user_1]:
            self.data.users[user_1]["contact"] = {}
        self.data.users[user_1]["contact"][user_2] = relationship
        self.agents[user_1].relationships[self.agents[user_2].name] = relationship

        if "contact" not in self.data.users[user_2]:
            self.data.users[user_2]["contact"] = {}
        self.data.users[user_2]["contact"][user_1] = relationship
        self.agents[user_2].relationships[self.agents[user_1].name] = relationship

        self.data.tot_relationship_num += 2

    def add_social(self, num):
        """
        Add social relationship.
        """
        homo = False
        if num < 0:
            homo = True
            num = -num

        for i in range(len(self.agents)):
            agent = self.agents[i]

            for j in range(i + 1, len(self.agents)):
                if len(agent.relationships) == num:
                    break
                if homo:
                    if self.agents[j].interest == agent.interest:
                        self.add_relation(i, j, "friend")
                else:
                    if self.agents[j].interest != agent.interest:
                        self.add_relation(i, j, "friend")

    def predict_social(self):
        nodes = self.config["agent_num"]
        threshold = float(self.config.get("predict_social_threshold", 0.85))
        node_texts = []
        message = []

        for i in range(nodes):
            agent = self.agents[i]
            name = agent.name
            age = str(self.data.users[i].get("age", ""))
            gender = str(self.data.users[i].get("gender", ""))
            traits = (
                ",".join(self.data.users[i].get("traits", []))
                if isinstance(self.data.users[i].get("traits", []), list)
                else str(self.data.users[i].get("traits", ""))
            )
            interest = agent.interest if isinstance(agent.interest, str) else str(agent.interest)
            status = str(self.data.users[i].get("status", ""))
            profile_desc = (
                "agent profile: name "
                + name
                + ", age "
                + age
                + ", gender "
                + gender
                + ", traits "
                + traits
                + ", interests "
                + interest
                + ", status "
                + status
            )
            memory_desc = (
                "short-term memory: "
                + agent.memory.get_short_memory()
                + "\nlong-term memory: "
                + agent.memory.get_long_memory()
            )
            desc = profile_desc + "\n" + memory_desc
            node_texts.append(desc)

        embeddings = self.model_st.encode(
            node_texts, batch_size=1, show_progress_bar=False, device="cuda"
        )
        sim_matrix = cosine_similarity(np.asarray(embeddings))

        # Clear CUDA cache after encoding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i in range(nodes):
            for j in range(i + 1, nodes):
                if self.data.users[i].get("contact", {}).get(j, None) is not None:
                    continue
                sim = sim_matrix[i, j]
                if sim >= threshold:
                    self.logger.info(
                        f"Add relation between {self.agents[i].name} and {self.agents[j].name} with similarity {sim}."
                    )
                    message.append(
                        Message(
                            agent_id=i,
                            action="ADD_RELATIONSHIP",
                            content=f"{self.agents[i].name} and {self.agents[j].name} with similarity {sim}.",
                        )
                    )
                    self.round_msg.append(
                        Message(
                            agent_id=i,
                            action="ADD_RELATIONSHIP",
                            content=f"{self.agents[i].name} and {self.agents[j].name} with similarity {sim}.",
                        )
                    )
                    self.add_relation(i, j, "friend")
        return message

    def load_round_record(self):
        self.recsys.round_record = {}

        for i in range(len(self.agents)):
            self.recsys.round_record[i] = []
            for r in range(self.round_cnt):
                self.recsys.round_record[i].append(self.recsys.record[i][r * 5 : (r + 1) * 5])


# Paper-aligned public name; ``Simulator`` remains for archived imports.
PromoSimulator = Simulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Path to config file")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Path to output file")
    parser.add_argument("-l", "--log_file", type=str, default="log.log", help="Path to log file")
    parser.add_argument(
        "-n", "--log_name", type=str, default=str(os.getpid()), help="Name of logger"
    )
    parser.add_argument(
        "-p",
        "--play_role",
        type=bool,
        default=False,
        help="Add a user controllable role",
    )
    parser.add_argument(
        "-m", "--memory_backend", type=str, default="behavioral", help="memory backend"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--resume-condition")
    args = parser.parse_args()
    return args


def reset_system(behavioral, logger):
    # Reset the system
    reset = input("Do you want to reset the system? [y/n]: ")
    all_agents = [v for k, v in behavioral.agents.items()]
    log_string = ""
    if reset in "y":
        for agent in all_agents:
            agent.reset_agent()
        log_string = "The system is reset, and the historic records are removed."
    else:
        log_string = "The system keeps unchanged."
    behavioral.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
    return log_string


def modify_attr(behavioral, logger):
    all_agents = [v for k, v in behavioral.agents.items()]
    agents_name = [agent.name for agent in all_agents]

    # Modify agent attribute
    modify = input("Do you want to modify agent's attribute? [y/n]: ")
    log_string = ""
    if modify in "y":
        while True:
            modify_name = input(
                "Please type the agent name, select from: "
                + str(sorted([agent.name for agent in all_agents], reverse=False))
                + " : "
            )
            if modify_name not in agents_name:
                logger.info("Please type the correct agent name.")
            else:
                break
        target = [agent for agent in all_agents if agent.name == modify_name][0]
        target.modify_agent()
        log_string += "The attributes of {modify_name} are: \n"
        log_string += "The age of {} is : {}\n".format(
            modify_name,
            [agent.age for agent in all_agents if agent.name == modify_name],
        )
        log_string += "The gender of {} is : {}\n".format(
            modify_name,
            [agent.gender for agent in all_agents if agent.name == modify_name],
        )
        log_string += "The traits of {} is : {}\n".format(
            modify_name,
            [agent.traits for agent in all_agents if agent.name == modify_name],
        )
        log_string += "The status of {} is : {}\n".format(
            modify_name,
            [agent.status for agent in all_agents if agent.name == modify_name],
        )
    else:
        log_string = "The attributes of agent keep unchanged."

    behavioral.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
    return log_string


def inter_agent(behavioral, logger):
    all_agents = [v for k, v in behavioral.agents.items()]
    agents_name = [agent.name for agent in all_agents]
    log_string = ""
    # Interact with an agent
    interact = input("Do you want to interact with agent? [y/n]: ")
    while interact in "y":
        while True:
            interact_name = input(
                "Please type agent name, select from: "
                + str(sorted([agent.name for agent in all_agents], reverse=False))
                + " : "
            )
            if interact_name not in agents_name:
                logger.info("Please type the correct agent name.")
            else:
                log_string += "interact with " + interact_name + "\n"
                break
        target = [agent for agent in all_agents if agent.name == interact_name][0]
        observation, response = target.interact_agent()
        logger.info(response)
        log_string += "Observation: " + observation + "\n"
        log_string += "Response:" + response + "\n"
        cont = input("Do you want to keep interacting with agent? [y/n]")
        if cont in "y":
            continue
        else:
            break
    if interact in "n":
        log_string += "Do not interact with agent."

    behavioral.round_msg.append(Message(agent_id=-1, action="System", content=log_string))
    return log_string


def system_status(behavioral, logger):
    # Reset the system
    log = reset_system(behavioral, logger)
    logger.info(log)
    # Modify the agent attribute
    log = modify_attr(behavioral, logger)
    logger.info(log)
    # Interact with agent
    log = inter_agent(behavioral, logger)
    logger.info(log)


def main():
    args = parse_args()
    logger = utils.set_logger(args.log_file, args.log_name)
    logger.info(f"os.getpid()={os.getpid()}")
    # create config
    config = CfgNode(new_allowed=True)
    output_file = os.path.join("output/message", args.output_file)
    config = utils.add_variable_to_config(config, "output_file", output_file)
    config = utils.add_variable_to_config(config, "log_file", args.log_file)
    config = utils.add_variable_to_config(config, "log_name", args.log_name)
    config = utils.add_variable_to_config(config, "play_role", args.play_role)
    config = utils.add_variable_to_config(config, "memory_backend", args.memory_backend)
    config.merge_from_file(args.config_file)
    utils.configure_api_from_environment(config)
    logger.info("\n%s", utils.redacted_config(config))
    os.environ["OPENAI_API_KEY"] = config["api_keys"][0]
    st = time.time()
    if config["simulator_restore_file_name"]:
        restore_path = os.path.join(config["simulator_dir"], config["simulator_restore_file_name"])
        behavioral = Simulator.restore(restore_path, config, logger)
        logger.info(f"Successfully Restore simulator from the file <{restore_path}>\n")
        logger.info(f"Start from the round {behavioral.round_cnt + 1}\n")
    else:
        behavioral = Simulator(config, logger)
        behavioral.load_simulator()
    if behavioral.config["social_random_k"] > 0:
        behavioral.clear_social()
        behavioral.add_social(behavioral.config["social_random_k"])
    print("Time for loading simulator: ", time.time() - st)
    messages = []
    behavioral.play()

    for i in tqdm(range(behavioral.round_cnt + 1, config["round"] + 1)):
        round_st = time.time()
        behavioral.round_cnt = behavioral.round_cnt + 1
        behavioral.logger.info(f"Round {behavioral.round_cnt}")
        behavioral.active_agents.clear()
        message = behavioral.round()
        messages.append(message)
        with open(config["output_file"], "w") as file:
            json.dump(messages, file, default=lambda o: o.__dict__, indent=4)
        print("Time for round: ", time.time() - round_st)
        print("Time for all: ", time.time() - st)
        behavioral.recsys.save_interaction()


if __name__ == "__main__":
    main()
