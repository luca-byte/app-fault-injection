import torchvision.transforms as transforms
import torch.nn as nn
import torch
import os
import torchvision
from torchdistill.common.main_util import is_main_process, set_seed
from torchdistill.misc.log import setup_log_file, MetricLogger
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall, MulticlassPrecision
from torchdistill.common.constant import def_logger
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import math

from pytorchfi.FI_Weights_classification import FI_manager
from pytorchfi.FI_Weights_classification import DatasetSampling
import logging


from scripts.forward_hook import ForwardHook
from scripts.preprocessing import Preprocessing
from scripts.classifier import ModelTrainer
import pandas as pd



logger = def_logger.getChild(__name__)

class TinyVGG(nn.Module):
    def __init__(self, input_shape=3, hidden_units=64, image_dimension=32, output_shape=10):
        linear_in = image_dimension**2 //16 * hidden_units

        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=linear_in, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x
    
print("CUDA Available: ", torch.cuda.is_available())

transformer = transforms.ToTensor()
layer = 0
index = 0

log = f"CIFAR-TV/L{layer}-{index}/log/LeNet.log"
wdir = f"CIFAR-TV/L{layer}-{index}"
seed = None
device = torch.device("cuda")
dataset = torchvision.datasets.CIFAR10(
    "~/dataset/cifar", transform=transformer, download=True, train=False
)
model_path = "CIFAR-TV/TinyVGG_CIFAR10.pth"

batch_size = 1
shuffle = False
num_workers = 16
shape = [1, 28, 28]

n = None

block = layer
pre_path = "CIFAR-TV/CIFAR-0.json"

feat_ex = Preprocessing.load(pre_path, use_scaler=True)

ext_clf = ModelTrainer.load("CIFAR-TV/CIFAR-0.pth", 128, 10)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.inference_mode()
def evaluate(model_wo_ddp, data_loader, device, device_ids, distributed, no_dp_eval=False,
             log_freq=1000, title=None, header='Test:', fsim_enabled=False, Fsim_setup:FI_manager = None, handles=None):
    
    model = model_wo_ddp.to(device)
    
    if title is not None:
        logger.info(title)

    model.eval()

    metric_logger = MetricLogger(delimiter='  ')
    im=0

    val_distr = torch.tensor([], requires_grad=False)
    val_targ = torch.tensor([], requires_grad=False)
    for image, target in metric_logger.log_every(data_loader, log_freq, header):
        if isinstance(image, torch.Tensor):
            image = image.to(device, non_blocking=True)

        if isinstance(target, torch.Tensor):
            target = target.to(device, non_blocking=True)

        if fsim_enabled==True:
            output = model(image)
            Fsim_setup.FI_report.update_classification_report(im,output,target,topk=(1,10))
        else:
            output = model(image)

        cpu_target = target.to('cpu')
        val_targ = torch.cat((cpu_target, val_targ), dim = -1)

        soft = torch.nn.Softmax(dim=1)
        cpu_output = output.to('cpu')
        distr = soft(cpu_output)
        val_distr = torch.cat((distr, val_distr), dim = 0)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # FIXME need to take into account that the datasets
        # could have been padded in distributed setup
        batch_size = len(image)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        im+=1
    
        if fsim_enabled==False:
            val_targ = val_targ.type(torch.int64)
            f1_1 = MulticlassF1Score(task='multiclass', num_classes=10, average='macro')
            rec_1 = MulticlassRecall(average='macro', num_classes=10)
            prec_1 = MulticlassPrecision(average='macro', num_classes=10)

            best_f1 = f1_1(val_distr, val_targ)
            best_rec = rec_1(val_distr, val_targ)
            best_prec = prec_1(val_distr, val_targ)

            f1_k = MulticlassF1Score(task='multiclass', num_classes=10, average='macro', top_k=5)
            rec_k = MulticlassRecall(num_classes=10, average='macro', top_k=5)
            prec_k = MulticlassPrecision(num_classes=10, average='macro', top_k=5)
            k_f1 = f1_k(val_distr, val_targ)
            k_rec = rec_k(val_distr, val_targ)
            k_prec = prec_k(val_distr, val_targ)
            Fsim_setup.FI_report.set_f1_values(best_f1=best_f1, k_f1=k_f1, header=header, best_prec= best_prec, best_rec = best_rec, k_prec= k_prec, k_rec = k_rec)
            counter = 0
            if handles is not None:
                for handle in handles: 
                    counter += handle.to_zeroes_counter
                    # logger.info(f'handle.to_zeroes_counter: {handle.to_zeroes_counter}')
                    handle.to_zeroes_counter = 0
                Fsim_setup.FI_report.set_zeroes_counter(counter.item()/20, header=header)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    top1_accuracy = metric_logger.acc1.global_avg
    top5_accuracy = metric_logger.acc5.global_avg
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}'.format(top1_accuracy, top5_accuracy))
    return metric_logger.acc1.global_avg

@torch.no_grad()
def main():
    logging.getLogger("FM-X").disabled = True

    log_file_path = log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, device_ids = False, None
    cudnn.enabled = True
    cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.allow_tf32 = True

    set_seed(seed)

    test_data_loader = DataLoader(
        dataset=dataset, batch_size=128, shuffle=True, pin_memory=True
    )

    dnn = TinyVGG()
    dnn.load_state_dict(torch.load(model_path))
    dnn.eval()

    

    subsampler = DatasetSampling(test_data_loader.dataset, 30)
    index_dataset = subsampler.listindex()
    data_subset = Subset(test_data_loader.dataset, index_dataset)
    dataloader = DataLoader(
        data_subset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
    )

    # 1. Setup FI
    fi_setup = FI_manager(wdir, "ckpt_FI.json", "fsim_report.csv")

    # 2. Golden Run
    golden_hook = ForwardHook(dnn, layer_index=[block])
    fi_setup.open_golden_results("Golden_results")
    evaluate(
        dnn,
        dataloader,
        device,
        device_ids,
        distributed,
        no_dp_eval=False,
        log_freq=1000,
        title=f"[DNN under test: {type(dnn)}]",
        header="Golden",
        fsim_enabled=True,
        Fsim_setup=fi_setup,
    )
    fi_setup.close_golden_results()
    golden_hook.close()
    y = torch.cat([target for _, target in dataloader], dim=0)
    golden_hook.add_classes(y)
    df = golden_hook.to_dataframe()
    z = df["Prediction"]
    feats = feat_ex.transform(df)

    os.makedirs(f"{wdir}/stats", exist_ok=True)
    golden_df_stats = pd.concat([feats, pd.Series(z, name='Prediction')], axis=1)
    golden_df_stats.to_csv(f"{wdir}/stats/golden_stats.csv")

    w = ModelTrainer.predict(feats.values, ext_clf)

    os.makedirs(f"{wdir}/external", exist_ok=True)
    golden_df_external = pd.DataFrame({'Prediction': z, 'External': w})
    golden_df_external.to_csv(f"{wdir}/external/golden_external.csv")
    


    # 3. Faults
    fi_setup.FI_framework.create_fault_injection_model(
        device,
        dnn,
        batch_size=1,
        input_shape=shape,
        layer_types=[torch.nn.Conv2d, torch.nn.Linear],
    )
    logging.getLogger("pytorchfi").disabled = True
    fi_setup.generate_fault_list(
        flist_mode="sbfm",
        f_list_file="fault_list.csv",
        layer=layer,
        num_faults=n,
    )
    fi_setup.load_check_point()

    num = len(fi_setup._fault_list)

    # 5. Execute the fault injection campaign
    for fault, k in fi_setup.iter_fault_list():
        logger.info(f"\n\nInjection: {k}/{num}")
        # 5.1 inject the fault in the model
        fi_setup.FI_framework.bit_flip_weight_inj(fault)
        fi_setup.open_faulty_results(f"F_{k}_results")
        hook = ForwardHook(fi_setup.FI_framework.faulty_model, layer_index=[block])
        try:
            # 5.2 run the inference with the faulty model
            evaluate(
                fi_setup.FI_framework.faulty_model,
                dataloader,
                device,
                device_ids,
                distributed,
                no_dp_eval=False,
                log_freq=1000,
                title=f"[DNN under test: {type(dnn)}]",
                header="FSIM",
                fsim_enabled=True,
                Fsim_setup=fi_setup,
            )
            hook.close()
            hook.add_classes(y)
            df = hook.to_dataframe()
            z = df["Prediction"]
            feats = feat_ex.transform(df)

            os.makedirs(f"{wdir}/stats", exist_ok=True)
            df_stats = pd.concat([feats, pd.Series(z, name='Prediction')], axis=1)
            df_stats.to_csv(f"{wdir}/stats/stats-{k}.csv")

            w = ModelTrainer.predict(feats.values, ext_clf)
            df_external = pd.DataFrame({'Prediction': z, 'External': w})
            df_external.to_csv(f"{wdir}/external/external-{k}.csv")
        except Exception as Error:
            msg = f"Exception error: {Error}"
            logger.info(msg)
        # 5.3 Report the results of the fault injection campaign
        fi_setup.parse_results()
        # break
    fi_setup.terminate_fsim()


if __name__ == "__main__":
    main()
