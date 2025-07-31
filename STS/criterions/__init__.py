from .sts_loss import STSLoss
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .min_cka import MIN_CKA
criterion_list = {
    "sts_loss": STSLoss,
    "dual_space_kd_with_cross_model_attention": DualSpaceKDWithCMA,
    "min_cka": MIN_CKA
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")
