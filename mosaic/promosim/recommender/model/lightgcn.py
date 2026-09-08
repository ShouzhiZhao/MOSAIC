import torch
import torch.nn as nn

from .base import BaseModel


class LightGCN(BaseModel, nn.Module):
    def __init__(self, config, n_users, n_items):
        BaseModel.__init__(self, config, n_users, n_items)
        nn.Module.__init__(self)
        self.config = config
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = config["embedding_size"]
        self.n_layers = config.get("n_layers", 3)
        self.use_content = config.get("use_content_embedding", False)
        self.text_embedding_size = config.get("text_embedding_size", 1024)

        if self.use_content:
            self.user_mlp = nn.Sequential(
                nn.Linear(self.text_embedding_size, self.text_embedding_size // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(self.text_embedding_size // 2, self.text_embedding_size // 4),
                nn.LeakyReLU(0.01),
                nn.Linear(self.text_embedding_size // 4, self.embedding_size),
            )
            self.item_mlp = nn.Sequential(
                nn.Linear(self.text_embedding_size, self.text_embedding_size // 2),
                nn.LeakyReLU(0.01),
                nn.Linear(self.text_embedding_size // 2, self.text_embedding_size // 4),
                nn.LeakyReLU(0.01),
                nn.Linear(self.text_embedding_size // 4, self.embedding_size),
            )
        else:
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        self._init_params()

    def _init_params(self):
        """
        Initialize model parameters.
        """
        if self.use_content:
            # Initialize embeddings to zero if using content embeddings
            for mlp in [self.user_mlp, self.item_mlp]:
                for layer in mlp:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            # Placeholders for content embeddings
            self.user_content_emb = None
            self.item_content_emb = None
        else:
            # Initialize embeddings with normal distribution if not using content embeddings
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            nn.init.normal_(self.item_embedding.weight, std=0.1)
        self.graph = None

    def set_content_embeddings(self, user_emb, item_emb):
        """
        Set pre-computed content embeddings
        user_emb: tensor of shape (n_users, text_embedding_size)
        item_emb: tensor of shape (n_items, text_embedding_size)
        """
        # Ensure embeddings are on the correct device and detached from any previous graph
        if user_emb is not None:
            self.user_content_emb = user_emb.clone().detach().to(self.user_mlp[0].weight.device)
            self.user_content_emb.requires_grad = False
        if item_emb is not None:
            self.item_content_emb = item_emb.clone().detach().to(self.item_mlp[0].weight.device)
            self.item_content_emb.requires_grad = False

    def set_graph(self, graph):
        """
        Set the adjacency matrix (sparse tensor) for graph convolution.
        graph: torch.sparse.FloatTensor
        """
        self.graph = graph

    def computer(self):
        """
        Propagate embeddings through the graph.
        """
        if self.use_content:
            if self.user_content_emb is None or self.item_content_emb is None:
                raise ValueError("Content embeddings not set!")
            users_emb = self.user_mlp(self.user_content_emb)
            items_emb = self.item_mlp(self.item_content_emb)
        else:
            users_emb = self.user_embedding.weight
            items_emb = self.item_embedding.weight

        if self.graph is None:
            # Fallback to MF behavior if no graph is provided
            return users_emb, items_emb

        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def forward(self, user, item, neg_item=None):
        all_users, all_items = self.computer()
        user_emb = all_users[user]
        item_emb = all_items[item]

        # If neg_item is provided, it means we are in BPR training mode
        if neg_item is not None:
            neg_item_emb = all_items[neg_item]
            pos_scores = (user_emb * item_emb).sum(1)
            neg_scores = (user_emb * neg_item_emb).sum(1)
            return pos_scores, neg_scores

        return (user_emb * item_emb).sum(1)

    def get_full_sort_items(self, user, items):
        all_users, all_items = self.computer()
        user_emb = all_users[user]
        items_emb = all_items[items]

        if len(user_emb.shape) == 1:
            scores = torch.matmul(items_emb, user_emb)
        else:
            scores = (user_emb * items_emb).sum(-1)

        return self._sort_full_items(scores, items)

    def _sort_full_items(self, predicted_ratings, items):
        _, sorted_indices = torch.sort(predicted_ratings, descending=True)
        return items[sorted_indices].tolist()
