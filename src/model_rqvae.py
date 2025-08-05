"""
该文件提供了使用RQ-VAE（Residual-Quantized Variational Autoencoder）将高维多模态Embedding转换为离散的Semantic ID的框架代码。
这有助于将连续的特征向量量化为稀疏ID，以便于在推荐模型中作为一种新的稀疏特征使用。

主要包含以下几个部分：
1.  **注释掉的 `MmEmbDataset` 类**:
    -   这是一个数据集类的示例，用于加载多模态Embedding数据，为RQ-VAE的训练做准备。
    -   它会从指定路径加载`creative_emb`，并将其转换为PyTorch的Dataset格式。

2.  **`kmeans` 函数**:
    -   一个简单的K-Means聚类函数，使用`sklearn.cluster.KMeans`实现。
    -   用于对数据进行聚类，返回聚类中心（codebook）和每个数据点的标签。
    -   注意：scikit-learn的KMeans只在CPU上运行。

3.  **`BalancedKmeans` 类**:
    -   实现了平衡K-Means算法，旨在确保每个聚类分配到的样本数量大致相等。
    -   `_assign_clusters`: 分配聚类时，会优先将样本分配给尚未满员的聚类，以实现负载均衡。
    -   `_update_codebook`: 根据新的聚类分配结果，更新聚类中心。
    -   `fit`: 迭代执行分配和更新步骤，直到codebook收敛或达到最大迭代次数。

4.  **`VectorQuantizer` 类**:
    -   实现了向量量化层。
    -   `_tile`: 将输入张量在特定维度上重复，以便于与codebook中的所有码本向量进行比较。
    -   `forward`:
        -   接收输入向量。
        -   计算输入向量与codebook中所有码本的距离。
        -   找到距离最近的码本，并返回其索引（即量化后的ID）和对应的量化向量。

5.  **`ResidualVectorQuantizer` (RVQ) 类**:
    -   实现了残差向量量化。这是RQ-VAE的核心思想。
    -   它包含多个级联的`VectorQuantizer`。
    -   `forward`:
        -   输入向量首先被第一个量化器量化。
        -   计算残差（原始向量 - 量化向量）。
        -   残差被送入下一个量化器进行量化。
        -   这个过程重复进行，每一级的量化器都试图对上一级的残差进行编码。
        -   最终返回每一级量化得到的ID序列和总的量化向量（所有级量化向量之和）。

6.  **`RQVAE` 类**:
    -   定义了完整的RQ-VAE模型，包含编码器、解码器和残差向量量化器。
    -   `_init_weights`: 初始化模型权重。
    -   `encode`: 使用编码器将输入数据压缩成潜在表示。
    -   `decode`: 使用解码器将量化后的向量重构回原始数据空间。
    -   `forward`:
        -   执行完整的“编码 -> 量化 -> 解码”流程。
        -   计算重构损失（原始数据与重构数据之间的MSE）和codebook损失（用于更新码本）。
    -   `get_codes`: 一个推理方法，只执行“编码 -> 量化”过程，返回输入数据对应的Semantic ID序列。

**使用流程建议**:
1.  使用 `MmEmbDataset` (或类似的) 加载多模态Embedding。
2.  实例化并训练 `RQVAE` 模型。
3.  训练完成后，使用 `rqvae.get_codes()` 方法将所有多模态Embedding转换为Semantic ID。
4.  将这些Semantic ID作为新的稀疏特征，整合到 `BaselineModel` 中进行训练。
"""

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

# class MmEmbDataset(torch.utils.data.Dataset):
#     """
#     Build Dataset for RQ-VAE Training

#     Args:
#         data_dir = os.environ.get('TRAIN_DATA_PATH')
#         feature_id = MM emb ID
#     """

#     def __init__(self, data_dir, feature_id):
#         super().__init__()
#         self.data_dir = Path(data_dir)
#         self.mm_emb_id = [feature_id]
#         self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_id)

#         self.mm_emb = self.mm_emb_dict[self.mm_emb_id[0]]
#         self.tid_list, self.emb_list = list(self.mm_emb.keys()), list(self.mm_emb.values())
#         self.emb_list = [torch.tensor(emb, dtype=torch.float32) for emb in self.emb_list]

#         assert len(self.tid_list) == len(self.emb_list)
#         self.item_cnt = len(self.tid_list)

#     def __getitem__(self, index):
#         tid = torch.tensor(self.tid_list[index], dtype=torch.long)
#         emb = self.emb_list[index]
#         return tid, emb

#     def __len__(self):
#         return self.item_cnt

#     @staticmethod
#     def collate_fn(batch):
#         tid, emb = zip(*batch)

#         tid_batch, emb_batch = torch.stack(tid, dim=0), torch.stack(emb, dim=0)
#         return tid_batch, emb_batch


## Kmeans
def kmeans(data, n_clusters, kmeans_iters):
    """
    auto init: n_init = 10 if n_clusters <= 10 else 1
    """
    km = KMeans(n_clusters=n_clusters, max_iter=kmeans_iters, n_init="auto")

    # sklearn only support cpu
    data_cpu = data.detach().cpu()
    np_data = data_cpu.numpy()
    km.fit(np_data)
    return torch.tensor(km.cluster_centers_), torch.tensor(km.labels_)


## Balanced Kmeans
class BalancedKmeans(torch.nn.Module):
    def __init__(self, num_clusters: int, kmeans_iters: int, tolerance: float, device: str):
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_iters = kmeans_iters
        self.tolerance = tolerance
        self.device = device
        self._codebook = None

    def _compute_distances(self, data):
        return torch.cdist(data, self._codebook)

    def _assign_clusters(self, dist):
        samples_cnt = dist.shape[0]
        samples_labels = torch.zeros(samples_cnt, dtype=torch.long, device=self.device)
        clusters_cnt = torch.zeros(self.num_clusters, dtype=torch.long, device=self.device)

        sorted_indices = torch.argsort(dist, dim=-1)
        for i in range(samples_cnt):
            for j in range(self.num_clusters):
                cluster_idx = sorted_indices[i, j]
                if clusters_cnt[cluster_idx] < samples_cnt // self.num_clusters:
                    samples_labels[i] = cluster_idx
                    clusters_cnt[cluster_idx] += 1
                    break

        return samples_labels

    def _update_codebook(self, data, samples_labels):
        _new_codebook = []
        for i in range(self.num_clusters):
            cluster_data = data[samples_labels == i]
            if len(cluster_data) > 0:
                _new_codebook.append(cluster_data.mean(dim=0))
            else:
                _new_codebook.append(self._codebook[i])
        return torch.stack(_new_codebook)

    def fit(self, data):
        num_emb, codebook_emb_dim = data.shape
        data = data.to(self.device)

        # initialize codebook
        indices = torch.randperm(num_emb)[: self.num_clusters]
        self._codebook = data[indices].clone()

        for _ in range(self.kmeans_iters):
            dist = self._compute_distances(data)
            samples_labels = self._assign_clusters(dist)
            _new_codebook = self._update_codebook(data, samples_labels)
            if torch.norm(_new_codebook - self._codebook) < self.tolerance:
                break

            self._codebook = _new_codebook

        return self._codebook, samples_labels

    def predict(self, data):
        data = data.to(self.device)
        dist = self._compute_distances(data)
        samples_labels = self._assign_clusters(dist)
        return samples_labels


## Base RQVAE
class RQEncoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: list, latent_dim: int):
        super().__init__()

        self.stages = torch.nn.ModuleList()
        in_dim = input_dim

        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim

        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, latent_dim), torch.nn.ReLU()))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


class RQDecoder(torch.nn.Module):
    def __init__(self, latent_dim: int, hidden_channels: list, output_dim: int):
        super().__init__()

        self.stages = torch.nn.ModuleList()
        in_dim = latent_dim

        for out_dim in hidden_channels:
            stage = torch.nn.Sequential(torch.nn.Linear(in_dim, out_dim), torch.nn.ReLU())
            self.stages.append(stage)
            in_dim = out_dim

        self.stages.append(torch.nn.Sequential(torch.nn.Linear(in_dim, output_dim), torch.nn.ReLU()))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


## Generate semantic id
class VQEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        num_clusters,
        codebook_emb_dim: int,
        kmeans_method: str,
        kmeans_iters: int,
        distances_method: str,
        device: str,
    ):
        super(VQEmbedding, self).__init__(num_clusters, codebook_emb_dim)

        self.num_clusters = num_clusters
        self.codebook_emb_dim = codebook_emb_dim
        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.device = device

    def _create_codebook(self, data):
        if self.kmeans_method == 'kmeans':
            _codebook, _ = kmeans(data, self.num_clusters, self.kmeans_iters)
        elif self.kmeans_method == 'bkmeans':
            BKmeans = BalancedKmeans(
                num_clusters=self.num_clusters, kmeans_iters=self.kmeans_iters, tolerance=1e-4, device=self.device
            )
            _codebook, _ = BKmeans.fit(data)
        else:
            _codebook = torch.randn(self.num_clusters, self.codebook_emb_dim)
        _codebook = _codebook.to(self.device)
        assert _codebook.shape == (self.num_clusters, self.codebook_emb_dim)
        self.codebook = torch.nn.Parameter(_codebook)

    @torch.no_grad()
    def _compute_distances(self, data):
        _codebook_t = self.codebook.t()
        assert _codebook_t.shape == (self.codebook_emb_dim, self.num_clusters)
        assert data.shape[-1] == self.codebook_emb_dim

        if self.distances_method == 'cosine':
            data_norm = F.normalize(data, p=2, dim=-1)
            _codebook_t_norm = F.normalize(_codebook_t, p=2, dim=0)
            distances = 1 - torch.mm(data_norm, _codebook_t_norm)
        # l2
        else:
            data_norm_sq = data.pow(2).sum(dim=-1, keepdim=True)
            _codebook_t_norm_sq = _codebook_t.pow(2).sum(dim=0, keepdim=True)
            distances = torch.addmm(data_norm_sq + _codebook_t_norm_sq, data, _codebook_t, beta=1.0, alpha=-2.0)
        return distances

    @torch.no_grad()
    def _create_semantic_id(self, data):
        distances = self._compute_distances(data)
        _semantic_id = torch.argmin(distances, dim=-1)
        return _semantic_id

    def _update_emb(self, _semantic_id):
        update_emb = super().forward(_semantic_id)
        return update_emb

    def forward(self, data):
        self._create_codebook(data)
        _semantic_id = self._create_semantic_id(data)
        update_emb = self._update_emb(_semantic_id)

        return update_emb, _semantic_id


## Residual Quantizer
class RQ(torch.nn.Module):
    """
    Args:
        num_codebooks, codebook_size, codebook_emb_dim -> Build codebook
        if_shared_codebook -> If use same codebook
        kmeans_method, kmeans_iters -> Initialize codebook
        distances_method -> Generate semantic_id

        loss_beta -> Calculate RQ-VAE loss
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: list,
        codebook_emb_dim,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        assert len(self.codebook_size) == self.num_codebooks
        self.codebook_emb_dim = codebook_emb_dim
        self.shared_codebook = shared_codebook

        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.loss_beta = loss_beta
        self.device = device

        if self.shared_codebook:
            self.vqmodules = torch.nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[0],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for _ in range(self.num_codebooks)
                ]
            )

        else:
            self.vqmodules = torch.nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[idx],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for idx in range(self.num_codebooks)
                ]
            )

    def quantize(self, data):
        """
        Exa:
            i-th quantize: input[i]( i.e. res[i-1] ) = VQ[i] + res[i]
            vq_emb_list: [vq1, vq1+vq2, ...]
            res_emb_list: [res1, res2, ...]
            semantic_id_list: [vq1_sid, vq2_sid, ...]

        Returns:
            vq_emb_list[0] -> [batch_size, codebook_emb_dim]
            semantic_id_list -> [batch_size, num_codebooks]
        """
        res_emb = data.detach().clone()

        vq_emb_list, res_emb_list = [], []
        semantic_id_list = []
        vq_emb_aggre = torch.zeros_like(data)

        for i in range(self.num_codebooks):
            vq_emb, _semantic_id = self.vqmodules[i](res_emb)

            res_emb -= vq_emb
            vq_emb_aggre += vq_emb

            res_emb_list.append(res_emb)
            vq_emb_list.append(vq_emb_aggre)
            semantic_id_list.append(_semantic_id.unsqueeze(dim=-1))

        semantic_id_list = torch.cat(semantic_id_list, dim=-1)
        return vq_emb_list, res_emb_list, semantic_id_list

    def _rqvae_loss(self, vq_emb_list, res_emb_list):
        rqvae_loss_list = []
        for idx, quant in enumerate(vq_emb_list):
            # stop gradient
            loss1 = (res_emb_list[idx].detach() - quant).pow(2.0).mean()
            loss2 = (res_emb_list[idx] - quant.detach()).pow(2.0).mean()
            partial_loss = loss1 + self.loss_beta * loss2
            rqvae_loss_list.append(partial_loss)

        rqvae_loss = torch.sum(torch.stack(rqvae_loss_list))
        return rqvae_loss

    def forward(self, data):
        vq_emb_list, res_emb_list, semantic_id_list = self.quantize(data)
        rqvae_loss = self._rqvae_loss(vq_emb_list, res_emb_list)

        return vq_emb_list, semantic_id_list, rqvae_loss


class RQVAE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_channels: list,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: list,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
    ):
        super().__init__()
        self.encoder = RQEncoder(input_dim, hidden_channels, latent_dim).to(device)
        self.decoder = RQDecoder(latent_dim, hidden_channels[::-1], input_dim).to(device)
        self.rq = RQ(
            num_codebooks,
            codebook_size,
            latent_dim,
            shared_codebook,
            kmeans_method,
            kmeans_iters,
            distances_method,
            loss_beta,
            device,
        ).to(device)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z_vq):
        if isinstance(z_vq, list):
            z_vq = z_vq[-1]
        return self.decoder(z_vq)

    def compute_loss(self, x_hat, x_gt, rqvae_loss):
        recon_loss = F.mse_loss(x_hat, x_gt, reduction="mean")
        total_loss = recon_loss + rqvae_loss
        return recon_loss, rqvae_loss, total_loss

    def _get_codebook(self, x_gt):
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        return semantic_id_list

    def forward(self, x_gt):
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        x_hat = self.decode(vq_emb_list)
        recon_loss, rqvae_loss, total_loss = self.compute_loss(x_hat, x_gt, rqvae_loss)
        return x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss
