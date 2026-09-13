import torch


def configure_optimizer(
    model,
    weight_decay,
    learning_rate,
    device_type,
):
    # Separate parameters into those that should receive
    # weight decay and those that should not.
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    use_fused = (
        device_type == "cuda"
        and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    )

    extra_args = {}

    if use_fused:
        extra_args["fused"] = True

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        **extra_args,
    )

    return optimizer