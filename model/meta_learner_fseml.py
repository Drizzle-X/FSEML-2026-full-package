import logging

import torch
from torch import optim
from torch.nn import functional as F

from model.fseml_model import FSEMLModel, FSSAEConfig
from model.meta_learner_base import BaseContinualMetaLearner
from model.replay_factory import ReplayFactory

logger = logging.getLogger("experiment")


class MetaLearnerFSEML(BaseContinualMetaLearner):
    """
    Paper-aligned SRN(FSSAE) + CPN meta-learner.

    The public surface mirrors the legacy meta learner closely so that the
    training script only needs a very small FSEML-specific branch.

    Replay is intentionally restricted to the meta-training phase. The
    evaluation script loads only ``maml.net`` and should not reuse replay
    state during meta-test fine-tuning, which keeps the protocol aligned with
    the paper's fairness setting.
    """

    branch_name = "FSEML"

    def __init__(self, args):
        super().__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.meta_iteration = 0

        config = FSSAEConfig(
            dataset=getattr(args, "dataset", "omniglot"),
            in_channels=getattr(args, "in_channels", 3),
            image_size=getattr(args, "image_size", 28),
            num_classes=getattr(args, "num_classes", 1000),
            encoder_channels=getattr(args, "encoder_channels", 32),
            latent_dim=getattr(args, "latent_dim", 256),
            adapter_dim=getattr(args, "adapter_dim", 512),
            cpn_hidden_dim=getattr(args, "cpn_hidden_dim", 2304),
            normalization=getattr(args, "normalization", "batch"),
            sparsity_target=getattr(args, "sparsity_target", 0.05),
            sparsity_weight=getattr(args, "sparsity_weight", 5e-3),
            reconstruction_weight=getattr(args, "reconstruction_weight", 0.1),
            fisher_reconstruction_weight=getattr(args, "fisher_reconstruction_weight", 0.1),
            weight_decay_weight=getattr(args, "weight_decay_weight", 0.0),
            fda_weight=getattr(args, "fda_weight", 0.1),
        )
        self.net = FSEMLModel(config)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.meta_lr)
        self.replay = ReplayFactory.build(args)
        self.active_classes = None

    def set_active_classes(self, classes):
        self.active_classes = tuple(sorted(int(label) for label in classes))

    def classification_loss(self, logits, labels):
        """Compute task-free loss over all classes observed so far."""
        if not self.active_classes:
            return self.net.classification_loss(logits, labels)



        active = set(self.active_classes)
        active.update(int(label) for label in labels.detach().view(-1).cpu().tolist())
        allowed = torch.tensor(sorted(active), device=logits.device, dtype=torch.long)
        target_map = torch.full(
            (self.net.config.num_classes,), -1, device=logits.device, dtype=torch.long
        )
        target_map[allowed] = torch.arange(len(allowed), device=logits.device)
        mapped_labels = target_map[labels]
        if (mapped_labels < 0).any():
            raise ValueError("Training labels must belong to the active seen-class set.")
        return F.cross_entropy(logits.index_select(1, allowed), mapped_labels)

    def reset_classifer(self, class_to_reset, initialization="kaiming", clear_optimizer_state=False):
        weight = self.net.cpn.output.weight
        bias = self.net.cpn.output.bias
        with torch.no_grad():
            if initialization == "zero":
                weight[class_to_reset].zero_()
            elif initialization == "kaiming":
                torch.nn.init.kaiming_normal_(weight[class_to_reset].unsqueeze(0))
            else:
                raise ValueError("classifier initialization must be 'zero' or 'kaiming'.")
            bias[class_to_reset].zero_()

        if clear_optimizer_state:


            for parameter in (weight, bias):
                state = self.optimizer.state.get(parameter, {})
                for value in state.values():
                    if torch.is_tensor(value) and value.shape == parameter.shape:
                        value[class_to_reset].zero_()

    def inner_update(self, x, fast_weights, y):
        srn_outputs = self.net.forward_srn(x)
        latent = srn_outputs["latent"].detach()

        if fast_weights is None:
            fast_weights = [param for _, param in self.net.cpn_named_parameters()]

        logits = self.net.forward_cpn(latent, fast_weights=fast_weights)
        loss = self.classification_loss(logits, y)
        grad = torch.autograd.grad(loss, fast_weights, allow_unused=False)

        return [param - self.update_lr * grad_param for param, grad_param in zip(fast_weights, grad)]

    def augment_query_with_replay(self, x, y):
        return self.replay.augment_query(self.net, x, y, self.meta_iteration)

    def meta_loss(self, x, fast_weights, y):
        outputs = self.net.forward_features(x, fast_weights=fast_weights)
        srn_losses = self.net.srn.loss_terms(
            x=x,
            reconstruction=outputs["reconstruction"],
            latent=outputs["latent"],
            labels=y,
        )
        classification = self.classification_loss(outputs["logits"], y)
        total = classification + self.net.config.fda_weight * srn_losses["total"]
        return total, outputs["logits"], srn_losses["total"], classification

    def forward(
        self,
        x_traj,
        y_traj,
        x_rand,
        y_rand,
        *,
        zero_meta_grad=True,
        step_meta_optimizer=True,
        meta_loss_scale=1.0,
    ):
        x_traj, y_traj, x_rand, y_rand = self.maybe_apply_label_patch_augmentation(
            x_traj,
            y_traj,
            x_rand,
            y_rand,
        )

        fast_weights = self.inner_update(x_traj[0], None, y_traj[0])
        for k in range(1, self.update_step):
            fast_weights = self.inner_update(x_traj[k], fast_weights, y_traj[k])

        real_query_x = x_rand[0]
        real_query_y = y_rand[0]
        mixed_query_x, mixed_query_y, replay_count = self.augment_query_with_replay(real_query_x, real_query_y)

        if replay_count > 0 and hasattr(self.replay, "status"):
            replay_status = self.replay.status()
            replay_event = replay_status["replay_events"]
            if replay_event <= 5 or replay_event % 100 == 0:
                logger.info(
                    "Replay event %d at meta iteration %d: replayed=%d, buffer=%d/%d, total_replayed=%d",
                    replay_event,
                    self.meta_iteration,
                    replay_count,
                    replay_status["buffer_length"],
                    replay_status["buffer_capacity"],
                    replay_status["total_replayed_samples"],
                )

        meta_loss, logits, loss_auto, loss_pre = self.meta_loss(mixed_query_x, fast_weights, mixed_query_y)
        if replay_count > 0 and hasattr(self.replay, "record_analysis"):
            self.replay.record_analysis(
                self.net,
                logits,
                len(real_query_y),
                fast_weights=fast_weights,
                candidate_labels=mixed_query_y,
            )

        with torch.no_grad():
            if self.active_classes:
                allowed = torch.tensor(
                    self.active_classes, device=logits.device, dtype=torch.long
                )
                active_logits = logits[: len(real_query_y)].index_select(1, allowed)
                pred_q = allowed[F.softmax(active_logits, dim=1).argmax(dim=1)]
            else:
                pred_q = F.softmax(logits[: len(real_query_y)], dim=1).argmax(dim=1)
            classification_accuracy = torch.eq(pred_q, real_query_y).sum().item()

        if zero_meta_grad:
            self.optimizer.zero_grad()
        (meta_loss * meta_loss_scale).backward()
        if step_meta_optimizer:
            self.optimizer.step()

        self.replay.store_batch(self.net, real_query_x.detach(), real_query_y.detach())

        classification_accuracy /= len(real_query_y)
        self.meta_iteration += 1

        return classification_accuracy, meta_loss, loss_auto, loss_pre
