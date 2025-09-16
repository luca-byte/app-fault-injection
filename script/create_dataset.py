import torch
import logging
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from pytorchfi.FI_Weights_classification import FI_manager
from scripts.forward_hook import ForwardHook
from scripts.networks.lenet import LeNet
from scripts.networks.model_io import load_model
from torch.backends import cudnn
from torchdistill.common.main_util import set_seed
import torchvision
import torchvision.transforms as T

seed = 42
device = torch.device("cuda")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DATASET")


@torch.no_grad()
def evaluate(model, data, step, hook, wdir="./data"):
    logger.info(f"Evaluating model at step {step}")
    soft = nn.Softmax(dim=1)

    X, y = data
    X, y = X.to(device), y.to(device)
    outputs = model(X)

    probs = soft(outputs)
    _, preds = torch.max(probs, dim=1)

    if step != "G":
        hook.close()

    ofm = hook.pop_ofm()[0]

    if step != "G":
        torch.save(
            {"ofm": ofm},
            f"{wdir}/ofm_{step}.pt",
        )

    return probs, preds


def compute_severity(
    golden_probs, golden_preds, fi_probs, fi_preds, step, wdir="./data", alpha=0.85
):
    logger.info(f"Computing severity for step {step}")

    sev_class = (golden_preds != fi_preds).float().mean().item()
    sev_prob = ((golden_probs - fi_probs) ** 2).mean().item()

    severity = alpha * sev_class + (1 - alpha) * sev_prob

    torch.save(
        {
            "severity_class": sev_class,
            "severity_prob": sev_prob,
            "severity": severity,
        },
        f"{wdir}/severity_{step}.pt",
    )


def subsample(dataset: Dataset, window_size: int = 300):
    """
    Subsample the dataset to create a DataLoader with a specified window size.
    """
    sampler = RandomSampler(dataset, num_samples=window_size)
    loader = DataLoader(
        dataset,
        batch_size=window_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    return loader


@torch.no_grad()
def main(dnn: nn.Module, dataset: Dataset, layers: list[int], window_size: int = 300):
    # 1. Initialize PyTorch
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.allow_tf32 = True
    set_seed(seed)
    logging.getLogger("pytorchfi").disabled = True

    dnn = dnn.to(device).eval()
    shape = list(dataset[0][0].shape)

    wdir = "./fi"

    # 2. Setup PyTorchFI
    fi_setup = FI_manager(wdir, "ckpt.json", "report.csv")

    # 3. Fault Injection
    fi_setup.FI_framework.create_fault_injection_model(
        device,
        dnn,
        batch_size=window_size,
        input_shape=shape,
        layer_types=[torch.nn.Conv2d],
    )

    fi_setup.generate_fault_list(
        flist_mode="sbfm",
        f_list_file="fault_list.csv",
        layer=layers[0],
        num_faults=30000,
    )
    fi_setup.load_check_point()
    golden_hook = ForwardHook(dnn, layer_indices=layers)

    loader = DataLoader(
        dataset,
        batch_size=window_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    liter = iter(loader)

    for fault, k in fi_setup.iter_fault_list():
        # 3.1 Subsample the dataset
        try:
            data = next(liter)
        except StopIteration:
            liter = iter(loader)
            data = next(liter)

        # 3.2 Golden run
        golden_probs, golden_preds = evaluate(dnn, data, "G", golden_hook, wdir)

        # 3.3 Fault injection run
        fi_setup.FI_framework.bit_flip_weight_inj(fault)
        fi_hook = ForwardHook(fi_setup.FI_framework.faulty_model, layer_indices=layers)
        fi_probs, fi_preds = evaluate(
            fi_setup.FI_framework.faulty_model, data, k, fi_hook, wdir
        )

        # 3.4 Compute severity
        compute_severity(
            golden_probs,
            golden_preds,
            fi_probs,
            fi_preds,
            step=k,
            wdir=wdir,
        )

        # 4 Save Indices

        fi_setup.update_check_point()
    fi_setup.terminate_fsim()


pickle_dir = "./pickle"
name = "LeNet_MNIST"
model = load_model(LeNet(), f"{pickle_dir}/{name}.pth")
transformer = T.ToTensor()
dataset = torchvision.datasets.MNIST(
    "./dataset/mnist", transform=transformer, download=True, train=True
)

if __name__ == "__main__":
    main(model, dataset, layers=[0], window_size=300)
