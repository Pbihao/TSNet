from torch.optim import Adam


def get_optimizer(model):
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    opt = Adam(
        [
            {'params': model.decoder.parameters()},
            {'params': model.support_qkv.parameters()},
            {'params': model.query_qkv.parameters()},
            {'params': model.conv_q.parameters()},
        ],
        lr=1e-5, betas=(0.9, 0.999), weight_decay=5e-4
    )

    return opt
