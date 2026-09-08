import torch
import torch.nn.functional as F

from .lightgcn import LightGCN


class SimGCL(LightGCN):
    def __init__(self, config, n_users, n_items):
        super(SimGCL, self).__init__(config, n_users, n_items)
        self.eps = config.get("eps", 0.1)
        self.lamda = config.get("lamda", 0.5)
        self.tau = config.get("tau", 0.2)

    def forward(self, user, item, neg_item=None):
        # Normal LightGCN forward
        return super(SimGCL, self).forward(user, item, neg_item)

    def calculate_cl_loss(self, user, item):
        """
        Calculate Contrastive Learning Loss.
        We apply perturbation to embeddings and propagate them.
        To save computation, we only compute CL loss for the unique users and items in the current batch.
        """
        # Get unique users and items in the batch to reduce computation
        unique_users = torch.unique(user)
        unique_items = torch.unique(item)

        # Perturbation 1
        users_emb1, items_emb1 = self.computer(perturbed=True)
        # Perturbation 2
        users_emb2, items_emb2 = self.computer(perturbed=True)

        # Select embeddings for current batch
        u_view1 = users_emb1[unique_users]
        u_view2 = users_emb2[unique_users]

        i_view1 = items_emb1[unique_items]
        i_view2 = items_emb2[unique_items]

        # Calculate InfoNCE loss
        user_cl_loss = self.info_nce_loss(u_view1, u_view2)
        item_cl_loss = self.info_nce_loss(i_view1, i_view2)

        return self.lamda * (user_cl_loss + item_cl_loss)

    def computer(self, perturbed=False):
        """
        Propagate embeddings through the graph.
        If perturbed is True, add noise to initial embeddings.
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
            return users_emb, items_emb

        all_emb = torch.cat([users_emb, items_emb])

        embs = []

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb)
                all_emb = (
                    all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=1) * self.eps
                )
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def info_nce_loss(self, view1, view2):
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)

        pos_score = (view1 * view2).sum(dim=1)
        pos_score = torch.exp(pos_score / self.tau)

        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.tau).sum(dim=1)

        cl_loss = -torch.log(pos_score / ttl_score).mean()
        return cl_loss
