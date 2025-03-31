import re
from prettytable import PrettyTable
import torch
import numpy as np
import os
import torch.nn.functional as F
import logging
from utils.iotools import read_image
from torchvision import transforms

from utils.simple_tokenizer import SimpleTokenizer

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity.data.cpu(), dim=1, descending=True)
        indices = indices.to(similarity.device)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1) # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader # gallery
        self.txt_loader = txt_loader # query
        self.tokenizer = SimpleTokenizer()
        self.logger = logging.getLogger("IRRA.eval")

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qids, gids, qfeats, gfeats = [], [], [], []
        captions_dict = {}
        # text
        for pid, caption in self.txt_loader:
            
            # for img_pid, img_pth in zip(self.img_loader.dataset.img_pids, self.img_loader.dataset.img_paths):
            #     if img_pid == pid.item() and img_pid not in captions_dict:
            #         captions_dict[img_pth] = []
            #         break

            # caption = caption.to(device)
            # tokens = caption.cpu().tolist() if caption.is_cuda else caption.tolist()
            # tokens = [token for token in tokens if token != 0]  # Remove padding tokens (0)

            # decoded_captions = []
            # for token_list in tokens:  # Handle cases where tokens might be nested lists
            #     if isinstance(token_list, list):
            #         decoded_caption = self.tokenizer.decode(token_list)
            #     else:
            #         decoded_caption = self.tokenizer.decode([token_list])
            #     clean_txt = re.sub(r"<\|endoftext\|>|<\|startoftext\|>", "", decoded_caption)
            #     decoded_captions.append(clean_txt)

            # # Add decoded captions to captions_dict
            # for clean_txt in decoded_captions:
            #     if pid.item() not in captions_dict:
            #         captions_dict[pid.item()] = []
            #     captions_dict[pid.item()].append(clean_txt)

                
                
            caption = caption.to(device)
            with torch.no_grad():
                text_feat = model.encode_text(caption)
            qids.append(pid.view(-1)) # flatten 
            qfeats.append(text_feat.data.cpu())
        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)

        # image
        for pid, img in self.img_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_image(img)
            gids.append(pid.view(-1)) # flatten 
            gfeats.append(img_feat.data.cpu())
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)

        return qfeats.cuda(), gfeats.cuda(), qids, gids
    
    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = self._compute_embedding(model)

        qfeats = F.normalize(qfeats, p=2, dim=1) # text features
        gfeats = F.normalize(gfeats, p=2, dim=1) # image features

        similarity = qfeats @ gfeats.t()

        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(similarity=similarity, q_pids=qids, g_pids=gids, max_rank=10, get_mAP=True)
        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])

        if i2t_metric:
            i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(similarity=similarity.t(), q_pids=gids, g_pids=qids, max_rank=10, get_mAP=True)
            i2t_cmc, i2t_mAP, i2t_mINP = i2t_cmc.numpy(), i2t_mAP.numpy(), i2t_mINP.numpy()
            table.add_row(['i2t', i2t_cmc[0], i2t_cmc[4], i2t_cmc[9], i2t_mAP, i2t_mINP])
        # table.float_format = '.4'
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        self.logger.info('\n' + str(table))
        
        return t2i_cmc[0]
    
    def generate_caption(self, model, image_path, tokenizer):
        """
        Generate a caption for a single image.
        Args:
            model: The trained model.
            image_path: Path to the image.
            tokenizer: The tokenizer used for decoding.
        Returns:
            Decoded caption as a string.
        """
        
        
        