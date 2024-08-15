import clip
import torch
import torch.nn.functional as F
import torchvision.transforms as TT

class AlreadyLoss(torch.nn.Module):


    def __init__(self, model_id: str, lambda_clip: float, lambda_rec: float):

        super().__init__()

        model, _ = clip.load(model_id)
        self.model = model
        self.preprocess = TT.Compose([
            TT.Resize(224, TT.InterpolationMode.BICUBIC),
            TT.CenterCrop(224),
            TT.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.lambda_clip = lambda_clip
        self.lambda_rec = lambda_rec


    def forward(self, img_edit, img_source, text_target, text_source):
        # preprocessing
        batch_size = img_source.shape[0]
        img_edit_source = self.preprocess(torch.cat([img_edit, img_source]))

        # tokenize texts
        text_target_tokens, text_source_tokens = clip.tokenize([text_target, text_source])

        # encode images and texts
        img_edit_emb, img_source_emb = self.model.encode_image(img_edit_source).split(batch_size)
        text_target_emb, text_source_emb = self.model.encode_text(torch.stack([text_target_tokens, text_source_tokens]))
        
        # get embeddings deltas
        delta_img_emb = img_edit_emb - img_source_emb
        delta_text_emb = text_target_emb - text_source_emb

        # compute loss components
        loss_dir = (1 - (delta_img_emb @ delta_text_emb) / (delta_img_emb.norm(dim=1) * delta_text_emb.norm())).mean()
        loss_rec = F.mse_loss(img_edit, img_source)

        return self.lambda_clip * loss_dir + self.lambda_rec * loss_rec