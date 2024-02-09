#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# modified from https://github.com/haotian-liu/LLaVA/blob/7ace501183c4bdec6052ec1a30039cdc3242a67c/llava/model/llava.py

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    CLIPImageProcessor,
    CLIPVisionModel,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaModel,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
REGISTER_NAME = "llava"


class LlavaConfig(LlamaConfig):
    model_type = REGISTER_NAME


class LlavaLlamaModel(LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            self.vision_tower = [
                CLIPVisionModel.from_pretrained(config.mm_vision_tower)
            ]
            # self.vision_tower = CLIPVisionModel.from_pretrained(config.mm_vision_tower)

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(
        self,
        vision_tower,
        mm_vision_select_layer,
        pretrain_mm_mlp_adapter=None,
        fsdp=None,
    ):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, "vision_tower"):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, "mm_projector"):
            self.mm_projector = nn.Linear(
                vision_config.hidden_size, self.config.hidden_size
            )

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )
            self.mm_projector.load_state_dict(
                {k.split(".")[-1]: v for k, v in mm_projector_weights.items()}
            )

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, "orig_embeds_params", None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = self.get_vision_tower()
        if (
            vision_tower is not None
            and (input_ids.shape[1] != 1 or self.training)
            and images is not None
        ):
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(
                            image.unsqueeze(0), output_hidden_states=True
                        )
                        select_hidden_state_layer = getattr(
                            self.config, "mm_vision_select_layer", -1
                        )
                        select_hidden_state = image_forward_out.hidden_states[
                            select_hidden_state_layer
                        ]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(
                        images.to(vision_tower.dtype), output_hidden_states=True
                    )
                    select_hidden_state_layer = getattr(
                        self.config, "mm_vision_select_layer", -1
                    )
                    select_hidden_state = image_forward_outs.hidden_states[
                        select_hidden_state_layer
                    ]
                    image_features = select_hidden_state[:, 1:].to(images.dtype)
            if type(images) is list:
                image_features = [
                    self.mm_projector(image_feature)[0]
                    for image_feature in image_features
                ]
            else:
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(
                256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            dummy_image_features = self.mm_projector(dummy_image_features)

            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = (
                        cur_input_embeds + (0.0 * dummy_image_features).sum()
                    )
                    new_input_embeds.append(cur_input_embeds)
                    cur_image_idx += 1
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (
                        cur_input_ids == vision_tower.config.im_end_token
                    ).sum():
                        raise ValueError(
                            "The number of image start tokens and image end tokens should be the same."
                        )
                    image_start_tokens = torch.where(
                        cur_input_ids == vision_tower.config.im_start_token
                    )[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(
                            device=cur_input_embeds.device
                        )
                        num_patches = cur_image_features.shape[0]
                        if (
                            cur_input_ids[image_start_token_pos + num_patches + 1]
                            != vision_tower.config.im_end_token
                        ):
                            raise ValueError(
                                "The image end token should follow the image start token."
                            )
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[:image_start_token_pos].detach(),
                                    cur_input_embeds[
                                        image_start_token_pos : image_start_token_pos
                                        + 1
                                    ],
                                    cur_image_features,
                                    cur_input_embeds[
                                        image_start_token_pos
                                        + num_patches
                                        + 1 : image_start_token_pos
                                        + num_patches
                                        + 2
                                    ],
                                    cur_input_embeds[
                                        image_start_token_pos + num_patches + 2 :
                                    ].detach(),
                                ),
                                dim=0,
                            )
                        else:
                            cur_new_input_embeds = torch.cat(
                                (
                                    cur_input_embeds[: image_start_token_pos + 1],
                                    cur_image_features,
                                    cur_input_embeds[
                                        image_start_token_pos + num_patches + 1 :
                                    ],
                                ),
                                dim=0,
                            )
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (
                        cur_input_ids == vision_tower.config.im_patch_token
                    ).sum() != num_patches:
                        raise ValueError(
                            "The number of image patch tokens should be the same as the number of image patches."
                        )
                    masked_indices = torch.where(
                        cur_input_ids == vision_tower.config.im_patch_token
                    )[0]
                    mask_index_start = masked_indices[0]
                    if (
                        masked_indices
                        != torch.arange(
                            mask_index_start,
                            mask_index_start + num_patches,
                            device=masked_indices.device,
                            dtype=masked_indices.dtype,
                        )
                    ).any():
                        raise ValueError(
                            "The image patch tokens should be consecutive."
                        )
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start].detach(),
                                cur_image_features,
                                cur_input_embeds[
                                    mask_index_start + num_patches :
                                ].detach(),
                            ),
                            dim=0,
                        )
                    else:
                        cur_new_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:mask_index_start],
                                cur_image_features,
                                cur_input_embeds[mask_index_start + num_patches :],
                            ),
                            dim=0,
                        )
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_image_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlavaLlamaModel, self).forward(
            input_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class EditMapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.llm2hid = nn.Linear(4096, 512)
        self.query = nn.Parameter(torch.randn(1, 77, 512))
        self.mapper = nn.Transformer(
            batch_first=True,
            norm_first=True,
            d_model=512,
            nhead=4,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=2048,
            dropout=0.0,
        )
        self.hid2feat = nn.Linear(512, 768)

    def forward(self, llm, emb):
        hid = self.llm2hid(llm + emb)
        hid = self.mapper(hid, self.query.repeat(llm.shape[0], 1, 1))
        feat = self.hid2feat(hid)

        return feat


class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.edit_head = EditMapper()
        """
        self.scheduler, self.vae, self.unet = [
            diffusers.DDPMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            ),
            diffusers.AutoencoderKL.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="vae"
            ),
            diffusers.UNet2DConditionModel.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="unet"
            ),
        ]
        self.vae.requires_grad_(False)
        self.unet.register_to_config(in_channels=8)
        with torch.no_grad():
            conv = torch.nn.Conv2d(
                8,
                self.unet.conv_in.out_channels,
                self.unet.conv_in.kernel_size,
                self.unet.conv_in.stride,
                self.unet.conv_in.padding,
            )
            conv.weight.zero_()
            conv.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = conv
        """
        # Initialize weights and apply final processing
        self.post_init()

    def get_vision_tower(self):
        return self.model.get_vision_tower()

    def get_vision_tower(self):
        vision_tower = self.model.vision_tower
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        p2p_inp=None,
        p2p_ans=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if labels is not None:
            llm = []
            for i in range(labels.shape[0]):
                try:
                    p = labels[i].data.cpu().tolist().index(32003) - 1
                except:
                    p = len(labels[i]) - 9
                p = min(len(hidden_states[i]) - 9, p)
                llm.append(hidden_states[i][p : p + 8].unsqueeze(0))
            llm = torch.cat(llm, dim=0)
            hid_edit = self.edit_head(
                llm,
                self.model.embed_tokens.weight[-8:]
                .unsqueeze(dim=0)
                .repeat(labels.shape[0], 1, 1),
            )

            B, DROP = labels.shape[0], 0.05

            hid_null = self.edit_head(
                torch.zeros(B, 8, 4096, device=labels.device),
                self.model.embed_tokens.weight[-8:]
                .unsqueeze(dim=0)
                .repeat(labels.shape[0], 1, 1),
            )

            with torch.no_grad():
                lat_ans, lat_inp = (
                    self.vae.encode(p2p_ans).latent_dist.sample()
                    * self.vae.config.scaling_factor,
                    self.vae.encode(p2p_inp).latent_dist.mode(),
                )
                lat_ans, lat_inp = [
                    torch.from_numpy(lat_ans.data.cpu().float().numpy()).to(
                        lat_ans.device
                    ),
                    torch.from_numpy(lat_inp.data.cpu().float().numpy()).to(
                        lat_inp.device
                    ),
                ]

            noise = torch.randn_like(lat_ans)
            ts = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (B,), device=noise.device
            ).long()
            lat_noise = self.scheduler.add_noise(lat_ans, noise, ts)

            prob = torch.rand(B, device=lat_ans.device)
            mask = (prob < (DROP * 2)).reshape(B, 1, 1)
            hid_edit = torch.where(mask, hid_null, hid_edit)
            mask = (
                1.0
                - (
                    (prob >= DROP).to(lat_inp.dtype)
                    * (prob < (DROP * 3)).to(lat_inp.dtype)
                )
            ).reshape(B, 1, 1, 1)
            lat_inp *= mask

            out = self.unet(torch.cat([lat_noise, lat_inp], dim=1), ts, hid_edit).sample

            loss_ce, loss_edit = loss, nn.functional.mse_loss(
                out, noise, reduction="mean"
            )
            if int(os.environ["LOCAL_RANK"]) == 0:
                print("loss_ce:", loss_ce, "/", "loss_edit:", loss_edit)
            loss = loss_ce + loss_edit * 0.5

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self,
        mm_use_im_start_end,
        tokenizer,
        device,
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    ):
        vision_config = self.get_vision_tower().config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))
            (
                vision_config.im_start_token,
                vision_config.im_end_token,
            ) = tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]
            )

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)
                ]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_IMAGE_PATCH_TOKEN]
        )[0]


AutoConfig.register(REGISTER_NAME, LlavaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM, exist_ok=True)
