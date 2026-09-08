from typing import List

from pydantic import BaseModel


class Message(BaseModel):
    """
    Message class for communication between backend and frontend.
    """

    content: str
    agent_id: int
    action: str
    item_id: int | None = None
    event_type: str | None = None

    @classmethod
    def from_dict(cls, message_dict):
        return cls(**message_dict)


class RecommenderStat(BaseModel):
    tot_user_num: int
    cur_user_num: int
    tot_item_num: int
    inter_num: int
    rec_model: str
    pop_items: List[str]


class SocialStat(BaseModel):
    tot_user_num: int
    cur_user_num: int
    tot_link_num: int
    chat_num: int
    cur_chat_num: int
    post_num: int
    pop_items: List[str]
    network_density: float
