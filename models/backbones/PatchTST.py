import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    # ==========================================================================
    # Decomposed sub-APIs for ContinualPromptTSF Prefix-Tuning
    # ==========================================================================
    # The three methods below replace the monolithic encode_local /
    # forecast_from_local pair.  They expose the three stages of the PatchTST
    # forward pass as independently callable sub-modules so that the framework
    # can interleave prompt injection between patch embedding and encoding.
    #
    # Stage 1 → patch_embedding   : tokenise the input                [B,C,D,P]
    # Stage 2 → transformer_encoder: encode a sequence of any length  [B,C,D,*]
    # Stage 3 → prediction_head   : project encoded tokens to horizon [B,pred,C]
    # +helper → get_norm_stats    : return normalisation statistics
    # --------------------------------------------------------------------------

    def get_norm_stats(self, x_enc: torch.Tensor):
        """
        Compute and return instance-normalisation statistics WITHOUT modifying
        the input tensor.

        These statistics are needed for denormalisation at the prediction head
        stage (Stage 3).  We compute them from x_enc without side-effects so
        they can be cached between Stage 1 and Stage 3.

        Args:
            x_enc : [B, seq_len, C]  — raw (un-normalised) input.

        Returns:
            means : [B, 1, C]  — per-channel mean (detached).
            stdev : [B, 1, C]  — per-channel standard deviation (detached).
        """
        means = x_enc.mean(1, keepdim=True).detach()           # [B, 1, C]
        x_enc_centered = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()                                             # [B, 1, C]
        return means, stdev

    def patch_embedding(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Stage 1 — Normalise the input and produce patch embeddings.

        This stage tokenises the raw time series into a sequence of overlapping
        patch embeddings using the learned PatchEmbedding layer, but does NOT
        run the Transformer Encoder.  The output is the raw token sequence that
        the framework will augment with a prompt prefix before encoding.

        Args:
            x_enc : [B, seq_len, C]  — raw (un-normalised) input window.

        Returns:
            H_patches : [B, C, D, P]
                        B = batch size
                        C = number of sensor channels (n_vars)
                        D = d_model  (embedding dimension)
                        P = patch_num (number of patches after striding)

        Internal layout note
        --------------------
        PatchEmbedding internally works on the flattened CI layout [B*C, P, D].
        We reshape back to [B, C, D, P] (permuting D and P) to match the
        channel-separated convention used throughout the framework.
        """
        # --- Instance normalisation ----------------------------------------
        means, stdev = self.get_norm_stats(x_enc)
        x_norm = (x_enc - means) / stdev                      # [B, seq_len, C]

        # --- Permute to [B, C, seq_len] for the PatchEmbedding layer -------
        x_norm = x_norm.permute(0, 2, 1)                      # [B, C, L]

        # --- Patch embedding (CI layout: [B*C, P, D]) ----------------------
        # self.patch_embedding is the nn.Module attribute (same name).
        # Python resolves `self.patch_embedding(x_norm)` as a method call on
        # the nn.Module stored at that attribute — this is intentional.
        enc_out, n_vars = super(Model, self).patch_embedding(x_norm) \
            if False else self._patch_embed_ci(x_norm)        # [B*C, P, D]

        # --- Reshape to [B, C, P, D] then permute to [B, C, D, P] ---------
        B = x_enc.shape[0]
        enc_out = enc_out.view(B, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        # enc_out : [B, C, P, D]  →  H_patches : [B, C, D, P]
        H_patches = enc_out.permute(0, 1, 3, 2)               # [B, C, D, P]
        return H_patches

    def _patch_embed_ci(self, x_ci: torch.Tensor):
        """
        Internal helper: run the PatchEmbedding nn.Module on CI-layout input.

        Args:
            x_ci : [B, C, L]  — channel-first, normalised time series.

        Returns:
            (enc_out, n_vars)
            enc_out : [B*C, P, D]
            n_vars  : int  — number of channels C
        """
        # Delegate to the nn.Module stored as self.patch_embedding.
        # Access the nn.Module directly via __dict__ to avoid the name clash
        # with this method.
        patch_embed_module = self.__dict__['_modules']['patch_embedding']
        enc_out, n_vars = patch_embed_module(x_ci)            # [B*C, P, D]
        return enc_out, n_vars

    def transformer_encoder(self, H: torch.Tensor) -> torch.Tensor:
        """
        Stage 2 — Run the Transformer Encoder on a token sequence of length S.

        This method accepts any sequence length S along the last axis (Patch
        axis) of H.  When called from the framework with a prefix-augmented
        sequence, S = P + 1; when called without a prefix, S = P.  The Encoder
        architecture is position-independent (no positional embedding that
        hard-codes length), so it handles both cases transparently.

        Args:
            H : [B, C, D, S]
                B = batch, C = channels, D = d_model, S = sequence length.
                S = P   for standard encoding.
                S = P+1 when a prefix token has been prepended by the framework.

        Returns:
            H_encoded : [B, C, D, S]  — same shape as input.

        Internal layout note
        --------------------
        self.encoder (Transformer) operates on [B*C, S, D] (sequence-first).
        We permute/reshape before and after to match that convention.
        """
        B, C, D, S = H.shape

        # Permute to sequence-first layout expected by the encoder:
        # [B, C, D, S] → [B, C, S, D] → [B*C, S, D]
        H_seq = H.permute(0, 1, 3, 2)                         # [B, C, S, D]
        H_flat = H_seq.reshape(B * C, S, D)                   # [B*C, S, D]

        # Run the frozen Transformer Encoder
        H_enc_flat, _ = self.encoder(H_flat)                  # [B*C, S, D]

        # Reshape back to [B, C, S, D] → permute to [B, C, D, S]
        H_enc = H_enc_flat.view(B, C, S, D)                   # [B, C, S, D]
        H_encoded = H_enc.permute(0, 1, 3, 2)                 # [B, C, D, S]
        return H_encoded

    def prediction_head(
        self,
        H_real: torch.Tensor,
        means: torch.Tensor,
        stdev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Stage 3 — Apply the FlattenHead and denormalise to produce the forecast.

        Args:
            H_real : [B, C, D, P]  — encoded patch tokens (prefix ALREADY
                                     truncated by the framework; shape must
                                     match the head's expected nf = D * P).
            means  : [B, 1, C]     — normalisation means from get_norm_stats.
            stdev  : [B, 1, C]     — normalisation stdev from get_norm_stats.

        Returns:
            dec_out : [B, pred_len, C]  — denormalised forecast.
        """
        # FlattenHead: [B, C, D, P] → flatten(D,P) → Linear → [B, C, pred_len]
        dec_out = self.head(H_real)                           # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1)                    # [B, pred_len, C]

        # Instance denormalisation
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        )
        return dec_out                                        # [B, pred_len, C]

    # ==========================================================================
    # Legacy compatibility wrappers (kept for non-framework callers)
    # ==========================================================================

    def encode_local(self, x_enc):
        """
        [LEGACY] Monolithic encoder: normalise → patch embed → transformer.

        Retained for backward compatibility with any callers outside the
        ContinualPromptTSF framework.  New code should use the decomposed
        sub-APIs (get_norm_stats / patch_embedding / transformer_encoder).

        Returns:
            H_local : [B, C, D, P]
            means   : [B, 1, C]
            stdev   : [B, 1, C]
        """
        means, stdev = self.get_norm_stats(x_enc)             # [B, 1, C] each

        # Normalise and tokenise
        x_norm = (x_enc - means) / stdev                      # [B, seq_len, C]
        x_norm = x_norm.permute(0, 2, 1)                      # [B, C, L]
        enc_out, n_vars = self._patch_embed_ci(x_norm)        # [B*C, P, D]

        # Encode
        enc_out, _ = self.encoder(enc_out)                    # [B*C, P, D]

        # Reshape to [B, C, D, P]
        B = x_enc.shape[0]
        enc_out = enc_out.view(B, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        H_local = enc_out.permute(0, 1, 3, 2)                 # [B, C, D, P]
        return H_local, means, stdev

    def forecast_from_local(self, H_local, means, stdev):
        """
        [LEGACY] Apply the prediction head and denormalise.

        Retained for backward compatibility.  New code should use
        prediction_head(H_real, means, stdev) directly.
        """
        return self.prediction_head(H_local, means, stdev)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
# =========================================================================
    # ContinualPromptTSF (Prefix-Tuning) API Contract Extension - 终极对齐版
    # =========================================================================

    def get_norm_stats(self, x_enc):
        """提取归一化统计量"""
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        return means, stdev

    def encode_local(self, x_enc):
        """
        API 1: 提取局部 Patch 表征，并返回统计量 (替代 extract_patch_embeddings)
        彻底消除双重归一化与维度置换的隐患。
        Output: H_patches [B, C, D, P], means, stdev
        """
        means, stdev = self.get_norm_stats(x_enc)
        x_norm = (x_enc - means) / stdev
        
        # 转换为底层 PatchEmbedding 需要的 [B, C, L]
        x_norm = x_norm.permute(0, 2, 1)
        
        # 绕过类方法，直接调用内部子模块，彻底避开 Method Shadowing
        patch_embed_module = self.__dict__['_modules'].get('patch_embedding')
        enc_out, n_vars = patch_embed_module(x_norm)  # -> [B*C, P, D]
        
        B, C = x_enc.shape[0], x_enc.shape[2]
        # 恢复物理维度 [B, C, P, D]
        enc_out = torch.reshape(enc_out, (B, C, enc_out.shape[-2], enc_out.shape[-1]))
        
        # 转换为 Prefix-Tuning 标准契约 [B, C, D, P]
        H_patches = enc_out.permute(0, 1, 3, 2)
        
        return H_patches, means, stdev

    def forward_transformer_encoder(self, H_prefix):
        """
        API 2: 接收包含 Prefix Token 的动态序列进行编码
        Input: [B, C, D, P_mod], Output: [B, C, D, P_mod]
        """
        B, C, D, P_mod = H_prefix.shape
        enc_in = H_prefix.permute(0, 1, 3, 2).reshape(B * C, P_mod, D)
        enc_out, _ = self.encoder(enc_in)
        enc_out = torch.reshape(enc_out, (B, C, P_mod, D))
        return enc_out.permute(0, 1, 3, 2)

    def apply_prediction_head(self, H_real, means, stdev):
        """
        API 3: 预测头与反归一化
        Input: [B, C, D, P], Output: [B, pred_len, C]
        """
        dec_out = self.head(H_real)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out