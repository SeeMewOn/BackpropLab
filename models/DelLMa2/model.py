
import torch
import torch.nn as nn



class DelLMa2(nn.Module):
	def __init__(self, vocab_size=16000, d_model=512, context_size=1024, n_layers=6, n_head=8):
		super().__init__()
		self.context_size = context_size

		# 1. Embedding слой
		self.input_embedding = nn.Embedding(vocab_size, d_model)

		# 2. Positional Embedding (обучаемый)
		self.positional_encoding = nn.Parameter(torch.randn(1, context_size, d_model) * 0.02)

		# 3. Transformer Blocks (используем стандартный слой Torch для скорости)
		# Внутри nn.TransformerEncoderLayer уже есть MHA, Dropout, LayerNorm и FeedForward
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=n_head,
			dim_feedforward=4 * d_model,
			dropout=0.1,
			activation='gelu',
			batch_first=True,
			norm_first=True,
		)
		self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)

		# 4. Финальная нормализация и линейный слой
		self.ln_f = nn.LayerNorm(d_model)
		self.dense = nn.Linear(d_model, vocab_size)

		# --- Weight Tying ---
		# В Torch это делается простой привязкой ссылок на веса
		self.dense.weight = self.input_embedding.weight

		# Инициализация весов
		self.apply(self._init_weights)

	def _init_weights(self, module):    # TODO сделать лучше
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
			if isinstance(module, nn.Linear) and module.bias is not None:
				module.bias.data.zero_()

	def forward(self, X):
		B, L = X.shape

		# В Torch маска прибавляется к аттеншну: 0 - можно смотреть, -inf - нельзя
		mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(X.device) # TODO Вроде как тут маска не обязательна

		# Токены + Позиции
		out = self.input_embedding(X) + self.positional_encoding[:, :L, :]

		# Трансформер
		out = self.transformer_blocks(out, mask=mask, is_causal=True)

		# Выход
		out = self.ln_f(out)
		logits = self.dense(out)

		return logits
