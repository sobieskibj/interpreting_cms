import clip
import torch
import torch.nn.functional as F


class AlreadyLoss(torch.nn.Module):


    def __init__(self, model_id: str, lambda_clip: float, lambda_rec: float):

        super().__init__()

        model, preprocess = clip.load(model_id)
        self.model = model
        self.preprocess = preprocess
        self.lambda_clip = lambda_clip
        self.lambda_rec = lambda_rec


    def forward(self, img_edit, img_source, text_target, text_source):
        # preprocessing
        img_edit, img_source = self.preprocess(img_edit), self.preprocess(img_source)

        # tokenize texts
        text_target_tokens, text_source_tokens = clip.tokenize([text_target, text_source])

        # encode images and texts
        img_edit_emb, img_source_emb = self.model.encode_image(img_edit), self.model.encode_image(img_source)
        import pdb; pdb.set_trace()
        text_target_emb, text_source_emb = self.model.encode_text(text_target_tokens, text_source_tokens)
        
        # get embeddings deltas
        delta_img_emb = img_edit_emb - img_source_emb
        delta_text_emb = text_target_emb - text_source_emb

        # compute loss components
        import pdb; pdb.set_trace()
        loss_dir = 1 - (delta_img_emb * delta_text_emb) / (delta_img_emb.norm() * delta_text_emb.norm())
        loss_rec = F.mse_loss(img_edit, img_source)

        return self.lambda_clip * loss_dir + self.lambda_rec * loss_rec
