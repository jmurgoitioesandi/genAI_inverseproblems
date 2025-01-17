import torch


def anneal_dsm_score_estimation(
    scorenet,
    samples,
    y_samples,
    sigmas,
    config,
    labels=None,
    hook=None,
):
    """
    Inputs:
        scorenet: score network
        samples: samples from the true data distribution
        sigmas: a list of noise levels used to train the score network
        labels: labels of the samples
        config: configuration dict. Includes:
            anneal_power: the power of the sigma_value used to weigh each noise scale
        hook: a function that takes the loss and labels as inputs

    Outputs:
        loss: the loss for training the score network

    """
    anneal_power = config.training.anneal_power
    if labels is None:
        labels = torch.randint(
            0, len(sigmas), (samples.shape[0],), device=samples.device
        )
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = -1 / (used_sigmas**2) * noise
    scores = scorenet(perturbed_samples, y_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = (
        1
        / 2.0
        * ((scores - target) ** 2).sum(dim=-1)
        * used_sigmas.squeeze() ** anneal_power
    )

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)
