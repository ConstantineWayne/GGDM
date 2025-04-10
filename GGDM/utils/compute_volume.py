import torch
import numpy as np

def volume_computation3(grad1, grad2, grad3,batch_size=64):

    g11 = torch.einsum('bi,bi->b', grad1, grad1) #[bsz]

    g12 = torch.einsum('bi,bi->b',grad1,grad2)

    g13 = torch.einsum('bi,bj->bi', grad1, grad3)
    g13 = torch.sum(g13, dim=-1)  # (bsz)



    g22 = torch.einsum('bi,bi->b', grad2, grad2)
    g23 = torch.einsum('bi,bj->bi', grad2, grad3)
    g23 = torch.sum(g23,dim=-1)
    g33 = torch.einsum('bi,bi->b', grad3, grad3)

    G = torch.stack([
        torch.stack([g11, g12, g13], dim=-1),
        torch.stack([g12, g22, g23], dim=-1),
        torch.stack([g13, g23, g33], dim=-1)
    ], dim=-2)


    gram_det = torch.det(G.float())

    res = torch.sqrt(torch.abs(gram_det))

    return res


