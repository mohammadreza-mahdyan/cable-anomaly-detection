import random
import torch
import torch.nn.functional as F
from ..data.contexts import extract_grid_contexts
from ..models.vit import trainable_cls

@torch.no_grad()
def extract_features(model, contexts, middle_block=6, batch_size=4):
    middle, final = [], []
    for i in range(0, len(contexts), batch_size):
        x = contexts[i:i+batch_size]
        x = model.patch_embed(x)
        x = model._pos_embed(x)
        x = model.patch_drop(x)
        x = model.norm_pre(x)

        for j, block in enumerate(model.blocks):
            x = block(x)
            if j == middle_block:
                m = model.norm(x)[:, 0]
                middle.append(m.cpu())

        f = model.norm(x)[:, 0]
        final.append(f.cpu())

    return (
        F.normalize(torch.cat(middle), dim=-1),
        F.normalize(torch.cat(final), dim=-1),
    )

@torch.no_grad()
def score_query(
    query_middle,
    query_final,
    ref_middle,
    ref_final,
    ref_num=16,
    top_k=10,
):
    """
    query_*:
        [num_query_contexts, dim]

    ref_*:
        [num_reference_images, num_reference_contexts, dim]
    """

    num_refs = ref_middle.shape[0]

    ids = random.sample(
        range(num_refs),
        min(ref_num, num_refs),
    )

    scores = []

    for idx in ids:
        rm = ref_middle[idx].float()   # [256, 384]
        rf = ref_final[idx].float()    # [256, 384]

        qm = query_middle.float()      # [256, 384]
        qf = query_final.float()       # [256, 384]

        # [query_contexts, reference_contexts]
        sm = qm @ rm.T
        sf = qf @ rf.T

        # For every query context:
        # find its closest reference context.
        local_m = sm.max(dim=1).values.mean()
        local_f = sf.max(dim=1).values.mean()

        score = 0.5 * local_m + 0.5 * local_f
        scores.append(score)

    scores = torch.stack(scores)

    # Keep the best reference images.
    return scores.topk(
        min(top_k, len(scores))
    ).values.mean().item()

def anomaly_score(model, image, ref_middle, ref_final, context_size=96):
    contexts = extract_grid_contexts(image, context_size=context_size)
    qm, qf = extract_features(model, contexts)
    sim = score_query(qm, qf, ref_middle, ref_final)
    return 1.0 - sim
