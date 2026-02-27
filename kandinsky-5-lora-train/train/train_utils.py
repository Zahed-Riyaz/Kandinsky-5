import torch
import numpy as np
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._functional_collectives import AsyncCollectiveTensor


class Incremental_Timesteps():
    def __init__(self, F, T):
        self.F = F
        self.T = T

        mat = np.zeros((T, F))
        # Инициализируем последний столбец так, чтобы число путей на каждом временном шаге было равно 1.
        for t in range(T):
            mat[t, F - 1] = 1
        # Считаем видеокадры от конца к началу
        for f in range(F - 2, -1, -1):
            mat[T - 1, f] = 1
            for t in range(T - 2, -1, -1):
                mat[t, f] = mat[t + 1, f] + mat[t, f + 1]
        self.mat_s = mat

        mat = np.zeros((T, F))
        # Инициализируем первый столбец, количество путей на каждом временном шаге равно 1
        for t in range(T):
            mat[t, 0] = 1
        # Считаем видеокадры от начала до конца
        for f in range(1, F):
            mat[0, f] = 1
            for t in range(1, T):
                mat[t, f] = mat[t - 1, f] + mat[t, f - 1]
        self.mat_e = mat

    def sample_step_sequence(self):
        preT = 0
        timesteps = np.zeros(self.F)

        for f in range(self.F):
            candidate_weights = self.mat_s[preT:, f]
            sum_weight = np.sum(candidate_weights)
            prob_sequence = candidate_weights / sum_weight
            cur_step = np.random.choice(range(preT, self.T), p=prob_sequence)
            timesteps[f] = cur_step
            preT = cur_step
        return timesteps

    def sample_stepseq_from_mid(self):
        timesteps = np.zeros(self.F)

        # Случайным образом выберите видеокадр и равномерно отсортируйте временные шаги этого видеокадра.
        curf = np.random.randint(self.F)
        timesteps[curf] = np.random.randint(self.T)

        # Генерация временного шага предыдущего видеокадра, шум предыдущего видеокадра постепенно уменьшается
        for f in range(curf - 1, -1, -1):
            # print(f, timesteps[f+1], self.mat_e)
            candidate_weights = self.mat_e[:int(timesteps[f+1]) + 1, f]
            sum_weight = np.sum(candidate_weights)
            prob_sequence = candidate_weights / sum_weight
            cur_step = np.random.choice(range(0, int(timesteps[f+1]) + 1), 
                                        p=prob_sequence)
            timesteps[f] = int(cur_step)

        # Временной шаг следующего сгенерированного видеокадра
        for f in range(curf + 1, self.F):
            candidate_weights = self.mat_s[int(timesteps[f-1]):, f]
            sum_weight = np.sum(candidate_weights)
            prob_sequence = candidate_weights / sum_weight
            cur_step = np.random.choice(range(int(timesteps[f-1]), self.T), 
                                        p=prob_sequence)
            timesteps[f] = int(cur_step)

        return timesteps


class DeviceMeshs:

    def __init__(self, device_meshs, world_size):
        for mesh_type, mesh_params in device_meshs.items():
            meshs = {}
            if 'fsdp_mesh' in mesh_params:
                local_shards = mesh_params['fsdp_mesh']
                if local_shards == -1:
                    mesh = (world_size,)
                    mesh_dim_names = None
                else:
                    mesh = (world_size // local_shards, local_shards)
                    mesh_dim_names = ("dp", "fsdp")
                meshs['model'] = init_device_mesh("cuda", mesh_shape=mesh, mesh_dim_names=mesh_dim_names)
            if 'tp_mesh' in mesh_params:
                tp_shards = mesh_params['tp_mesh']
                mesh = (world_size // tp_shards, tp_shards)
                mesh_dim_names = ("dp", "tp")
                meshs['data'] = init_device_mesh("cuda", mesh_shape=mesh, mesh_dim_names=mesh_dim_names)
            setattr(self, mesh_type, meshs)

    def __enter__(self):
        for meshs in self.__dict__.values():
            for mesh_name in meshs:
                meshs[mesh_name] = meshs[mesh_name].__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for meshs in self.__dict__.values():
            for mesh_name in meshs:
                meshs[mesh_name] = meshs[mesh_name].__exit__(
                    exc_type, exc_value, exc_traceback
                )


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_local_distributed_tensor(tensor):
    if isinstance(tensor, torch.distributed.tensor.DTensor):
        tensor = tensor.to_local()
        if isinstance(tensor, AsyncCollectiveTensor):
            tensor = tensor.wait()
    return tensor


def _preprocess_batch(model, batch, device, text_embedder=None):
    batch_embeds, cu_seqlens = {}, {}
    batch_embeds['visual'] = torch.cat(batch['visual']).to(device)
    cu_seqlens['visual'] = torch.cumsum(
        torch.tensor([0, *[embeds.shape[0] for embeds in batch['visual']]]), dim=0
    ).to(device=device, dtype=torch.int32)
    cu_seqlens['visual_rope'] = cu_seqlens['visual'].clone()
    if 'text_embeds' not in batch:
        text_embeds, text_cu_seqlens = text_embedder.encode(batch['text'], images=batch.get('image'), type_of_content=batch['type_of_content'])
        batch_embeds.update(text_embeds)
        cu_seqlens['text_embeds'] = text_cu_seqlens
    else:
        batch_embeds['text_embeds'] = torch.cat(batch['text_embeds']).to(device)
        batch_embeds['pooled_embed'] = torch.cat(batch['pooled_embed']).to(device)
        cu_seqlens['text_embeds'] = torch.cumsum(
            torch.tensor([0, *[embeds.shape[0] for embeds in batch['text_embeds']]]), dim=0
        ).to(device=device, dtype=torch.int32)

    if model.attention.causal and batch['type_of_content'] == 'video':    
        # с какого кадра новое видео/текст начинается
        visual_cu_seqlens = cu_seqlens['visual']
        text_cu_seqlens = cu_seqlens['text_embeds']
        # длина видео в кадрах
        visual_seqlens = torch.diff(visual_cu_seqlens)
        text_seqlens = torch.diff(text_cu_seqlens)

        # batch_embeds['text_embeds'] делим на len(text_seqlens) 
        # - на количество текстовых эмбеддингов
        batch_text_embeds = torch.split(batch_embeds['text_embeds'], text_seqlens.tolist(), dim=0)

        if model.attention.chunk:
            chunk_num_repeats = torch.ceil(visual_seqlens / model.attention.chunk_len).int()
            # текстовые эмбеддинги дублируем по количеству чанков 
            # и конкатенируем это все
            batch_embeds['text_embeds'] = torch.cat([
                text_embeds.repeat(num_repeats, 1)
                for text_embeds, num_repeats in zip(batch_text_embeds, chunk_num_repeats)
            ], dim=0)

            # pooled_embed эмбеддинги дублируем по количеству чанков 
            # (изначально равны количеству видео)
            batch_embeds['pooled_embed'] = torch.repeat_interleave(
                batch_embeds['pooled_embed'], repeats=chunk_num_repeats, dim=0
            )
        else:
            # текстовые эмбеддинги дублируем по количеству кадров в видео 
            # и конкатенируем это все
            batch_embeds['text_embeds'] = torch.cat([
                text_embeds.repeat(num_repeats, 1)
                for text_embeds, num_repeats in zip(batch_text_embeds, visual_seqlens)
            ], dim=0)

            # pooled_embed эмбеддинги дублируем по количеству кадров 
            # (изначально равны количеству видео)
            batch_embeds['pooled_embed'] = torch.repeat_interleave(
                batch_embeds['pooled_embed'], repeats=visual_seqlens, dim=0
            )

        if model.attention.chunk:
            # указываем позиции, с которых начинается новый чанк
            cu_seqlens['visual'] = torch.cat([
                torch.cat([torch.arange(visual_cu_seqlens[i-1], visual_cu_seqlens[i], model.attention.chunk_len, dtype=torch.int32) 
                           for i in range(1, visual_cu_seqlens.shape[0])], dim=0).to(device),
                visual_cu_seqlens[-1:]
            ], dim=0)
            # для каждого чанка текстовый эмбеддинг, указываем длину на каждом чанке
            cu_seqlens['text_embeds'] = torch.cat([
                text_cu_seqlens[:1],
                torch.cumsum(torch.repeat_interleave(text_seqlens, repeats=chunk_num_repeats), dim=0, dtype=torch.int32)
            ], dim=0)
        else:
            # указываем позиции, с которых начинается новый кадр - т.е. каждый кадр получается
            cu_seqlens['visual'] = torch.arange(visual_cu_seqlens[-1] + 1, dtype=torch.int32)
            # для каждого кадра текстовый эмбеддинг, указываем длину на каждом кадре
            cu_seqlens['text_embeds'] = torch.cat([
                text_cu_seqlens[:1],
                torch.cumsum(torch.repeat_interleave(text_seqlens, repeats=visual_seqlens), dim=0)
            ], dim=0)

    return batch_embeds, cu_seqlens


def preprocess_batch(model, batch, device, text_embedder=None):
    batch_embeds, cu_seqlens = {}, {}
    attention_mask = None

    batch_embeds['visual'] = torch.cat(batch['visual']).to(device)
    cu_seqlens['visual'] = torch.cumsum(
        torch.tensor([0, *[embeds.shape[0] for embeds in batch['visual']]]), dim=0
    ).to(device=device, dtype=torch.int32)
    cu_seqlens['visual_rope'] = cu_seqlens['visual'].clone()

    if 'text_embeds' not in batch:
        text_embeds, text_cu_seqlens, attention_mask = text_embedder.encode(
            batch['text'],
            images=batch.get('image'),
            type_of_content='video'
        )
        batch_embeds.update(text_embeds)
        cu_seqlens['text_embeds'] = text_cu_seqlens
        # visual: torch.Size([1, 64, 64, 16])
        # text_embeds: torch.Size([1, 512, 3584])
        # pooled_embed: torch.Size([1, 768])
        # attention_mask: torch.Size([1, 512])
    else:
        batch_embeds['text_embeds'] = torch.cat(
            batch['text_embeds']
        ).to(device)
        batch_embeds['pooled_embed'] = torch.cat(
            batch['pooled_embed']
        ).to(device)
        cu_seqlens['text_embeds'] = torch.cumsum(
            torch.tensor(
                [0, *[embeds.shape[0] for embeds in batch['text_embeds']]]
            ), dim=0
        ).to(device=device, dtype=torch.int32)
        # Создаем базовый attention_mask если его нет в батче
        if 'attention_mask' in batch:
            attention_mask = torch.cat(batch['attention_mask']).to(device)
            # print(f'attention_mask in batch: {attention_mask.shape}')
        else:
            # Создаем маску из единиц той же длины, что и text_embeds
            total_text_tokens = batch_embeds['text_embeds'].shape[1] # 0
            attention_mask = torch.ones(
                total_text_tokens, dtype=torch.bool, device=device
            ).unsqueeze(0)
            attention_mask = torch.cat(
                [attention_mask for _ in range(batch_embeds['text_embeds'].shape[0])]
            )
            # print(f'attention_mask not in batch: {attention_mask.shape}')

    if model.attention.causal and batch['type_of_content'] == 'video':
        visual_cu_seqlens = cu_seqlens['visual']
        text_cu_seqlens = cu_seqlens['text_embeds']
        visual_seqlens = torch.diff(visual_cu_seqlens)
        text_seqlens = torch.diff(text_cu_seqlens)

        batch_text_embeds = torch.split(
            batch_embeds['text_embeds'], text_seqlens.tolist(), dim=0
        )

        # ОБРАБОТКА ATTENTION_MASK ДЛЯ ВИДЕО
        if attention_mask is not None:
            batch_attention_masks = torch.split(
                attention_mask, text_seqlens.tolist(), dim=0
            )

        if model.attention.chunk:
            chunk_num_repeats = torch.ceil(
                visual_seqlens / model.attention.chunk_len
            ).int()

            batch_embeds['text_embeds'] = torch.cat([
                text_embeds.repeat(num_repeats, 1)
                for text_embeds, num_repeats in zip(
                    batch_text_embeds, chunk_num_repeats
                )
            ], dim=0)

            batch_embeds['pooled_embed'] = torch.repeat_interleave(
                batch_embeds['pooled_embed'], repeats=chunk_num_repeats, dim=0
            )

            # ОБРАБОТКА ATTENTION_MASK ДЛЯ CHUNK MODE
            if attention_mask is not None:
                attention_mask = torch.cat([
                    mask.repeat(num_repeats)
                    for mask, num_repeats in zip(
                        batch_attention_masks, chunk_num_repeats
                    )
                ], dim=0)

        else:
            batch_embeds['text_embeds'] = torch.cat([
                text_embeds.repeat(num_repeats, 1)
                for text_embeds, num_repeats in zip(
                    batch_text_embeds, visual_seqlens
                )
            ], dim=0)

            batch_embeds['pooled_embed'] = torch.repeat_interleave(
                batch_embeds['pooled_embed'], repeats=visual_seqlens, dim=0
            )

            # ОБРАБОТКА ATTENTION_MASK ДЛЯ FRAME MODE
            if attention_mask is not None:
                attention_mask = torch.cat([
                    mask.repeat(num_repeats)
                    for mask, num_repeats in zip(
                        batch_attention_masks, visual_seqlens
                    )
                ], dim=0)

        if model.attention.chunk:
            cu_seqlens['visual'] = torch.cat(
                [
                    torch.cat(
                        [
                            torch.arange(
                                visual_cu_seqlens[i-1],
                                visual_cu_seqlens[i],
                                model.attention.chunk_len,
                                dtype=torch.int32
                            ) for i in range(1, visual_cu_seqlens.shape[0])
                        ], dim=0
                    ).to(device),
                    visual_cu_seqlens[-1:]
                ], dim=0
            )

            cu_seqlens['text_embeds'] = torch.cat([
                text_cu_seqlens[:1],
                torch.cumsum(
                    torch.repeat_interleave(
                        text_seqlens,
                        repeats=chunk_num_repeats
                    ), dim=0, dtype=torch.int32
                )
            ], dim=0)
        else:
            cu_seqlens['visual'] = torch.arange(
                visual_cu_seqlens[-1] + 1,
                dtype=torch.int32
            )
            cu_seqlens['text_embeds'] = torch.cat([
                text_cu_seqlens[:1],
                torch.cumsum(
                    torch.repeat_interleave(
                        text_seqlens, repeats=visual_seqlens
                    ), dim=0
                )
            ], dim=0)

    return batch_embeds, cu_seqlens, attention_mask


def get_rope_pos(model, batch_embeds, cu_seqlens):
    assert model.patch_size[0] == 1

    _, height, width = batch_embeds["visual"].shape[:3]
    height, width = (
        height // model.patch_size[1],
        width // model.patch_size[2]
    )
    visual_rope_pos = [
        torch.cat(
            [torch.arange(end) for end in torch.diff(cu_seqlens["visual_rope"])]
        ),
        torch.arange(height), torch.arange(width)
    ]
    text_rope_pos = torch.cat(
        [torch.arange(end) for end in torch.diff(cu_seqlens["text_embeds"])]
    )
    return visual_rope_pos, text_rope_pos


def get_autoreg_timesteps(model, cu_seqlens):
    visual_seqlens = torch.diff(cu_seqlens)
    chunk_num_repeats = torch.ceil(
        visual_seqlens / model.attention.chunk_len
    ).int()
    timesteps = []
    for i, num_frames in enumerate(chunk_num_repeats):
        train_step_sampler = Incremental_Timesteps(num_frames.cpu(), 1000)
        timesteps.append(
            torch.tensor(
                train_step_sampler.sample_stepseq_from_mid()
            ).to(chunk_num_repeats.device)
        )
    timesteps = torch.cat(timesteps, dim=0) / 1000
    return timesteps


def add_visual_cond(x_t, visual, visual_cu_seqlens, visual_cond_prob):
    weights = torch.tensor([1. - visual_cond_prob, visual_cond_prob])
    is_visual_cond = torch.multinomial(
        weights,
        visual_cu_seqlens.shape[0] - 1,
        replacement=True
    ).bool()
    visual_cond_idx = visual_cu_seqlens[:-1][is_visual_cond]

    visual_cond = torch.zeros_like(visual)
    visual_cond_mask = torch.zeros(
        [*visual.shape[:-1], 1],
        dtype=visual.dtype,
        device=visual.device
    )
    if visual_cu_seqlens.shape[0] - 1 < visual.shape[0]:
        visual_cond[visual_cond_idx] = visual[visual_cond_idx]
        visual_cond_mask[visual_cond_idx] = 1
    return torch.cat([x_t, visual_cond, visual_cond_mask], dim=-1)


def get_sparse_params(model, visual, visual_cu_seqlens):
    assert model.patch_size[0] == 1
    T, H, W, C = visual.shape
    T, H, W = (
        T // model.patch_size[0],
        H // model.patch_size[1],
        W // model.patch_size[2],
    )
    if model.attention.type == "nabla":
        sparse_params = {
            "attention_type": model.attention.type,
            "to_fractal": True,
            'P': model.attention.P,
            'wT': model.attention.wT,
            'wW': model.attention.wW,
            'wH': model.attention.wH,
            'add_sta': model.attention.add_sta,
            'visual_shape': (T, H, W),
            'visual_seqlens': visual_cu_seqlens,
            'method': getattr(model.attention, 'method', 'topcdf')
        }
    elif model.attention.type == "nabla_framewise_causal":
        sparse_params = {
            "attention_type": model.attention.type,
            "to_fractal": True,
            'P': model.attention.P,
            'wT': model.attention.wT,
            'wW': model.attention.wW,
            'wH': model.attention.wH,
            'add_sta': model.attention.add_sta,
            'mf': model.attention.mf,
            'visual_shape': (T, H, W),
            'visual_seqlens': visual_cu_seqlens
        }
    else:
        sparse_params = None

    return sparse_params
