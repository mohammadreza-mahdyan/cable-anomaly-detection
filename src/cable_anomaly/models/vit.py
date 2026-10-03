import timm
import torch
import torch.nn as nn

def create_vit(model_name="vit_small_patch16_224", train_from_block=6):
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks[train_from_block:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.norm.parameters():
        p.requires_grad = True
    return model

def trainable_cls(model, x, frozen_blocks=6):
    x = model.patch_embed(x)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)

    with torch.no_grad():
        for block in model.blocks[:frozen_blocks]:
            x = block(x)

    for block in model.blocks[frozen_blocks:]:
        x = block(x)

    x = model.norm(x)
    return x[:, 0]
