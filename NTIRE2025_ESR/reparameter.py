''' converter '''
import torch
from models.team29_MAAN import MAAN
from models.team29_MAANRep import MAAN_rep


path = "model_zoo/team29_MAAN.pth"
state_dict = torch.load(path)


model1 = MAAN().eval()
model1.load_state_dict(state_dict['params'], strict=True)


model2 = MAAN_rep().eval()


for blocks in model1.layers:
    for block in blocks.blocks:
        if block.mlp.merge_kernel == False:
            block.mlp.merge_mlp()


reparam_state_dict = dict()
reparam_state_dict['params'] = model1.state_dict()
torch.save(reparam_state_dict, "model_zoo/team29_MAANRep.pth")
