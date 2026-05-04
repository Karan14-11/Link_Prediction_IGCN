import torch


def temporal_smoothness(memory_old, memory_new):
    return ((memory_old - memory_new) ** 2).mean()




def degree_regularizer(embeddings, degrees):
    target = torch.log1p(degrees.float())
    return ((embeddings.norm(dim=1) - target) ** 2).mean()