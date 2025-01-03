import argparse
import datetime
import json
import os
import time

import torch
from torch import nn
from utils import *
import torchvision
import torchvision.transforms as trsf
from torch.backends import cudnn
from torchmetrics.classification import MulticlassF1Score, MulticlassRecall, MulticlassPrecision
from torchdistill.common import yaml_util
from torchdistill.common.constant import def_logger
from torchdistill.common.main_util import is_main_process, set_seed
# from torchdistill.eval.classification import compute_accuracy
from torchdistill.misc.log import setup_log_file, MetricLogger

from pytorchfi.FI_Weights_classification import FI_manager 
from pytorchfi.FI_Weights_classification import DatasetSampling 

from torch.utils.data import DataLoader, Subset

logger = def_logger.getChild(__name__)
import logging

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    parser.add_argument('--config', required=True, help='yaml file path')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--log', help='log file path')
    parser.add_argument('--seed', type=int, help='seed in random number generator')
    parser.add_argument('-no_dp_eval', action='store_true',
                        help='perform evaluation without DistributedDataParallel/DataParallel')
    parser.add_argument('-log_config', action='store_true', help='log config')
    parser.add_argument('--fsim_config', help='Yaml file path fsim config')
    return parser

def get_transforms(split='train', input_size=(128, 128)):

    transform = trsf.Compose([
        trsf.Compose([
        trsf.Resize((70, 70)),        
        trsf.CenterCrop((64, 64)),            
        trsf.ToTensor(),                
        trsf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    ])
    return transform


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
    logger.info(' * Acc@1 {:.4f}\tAcc@5 {:.4f}\n'.format(top1_accuracy, top5_accuracy))
    return metric_logger.acc1.global_avg



def main(args):
    log_file_path = args.log
    if is_main_process() and log_file_path is not None:
        setup_log_file(os.path.expanduser(log_file_path))

    distributed, device_ids = False, None
    logger.info(args)
    cudnn.enabled=True
    # cudnn.benchmark = True
    cudnn.deterministic = True
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    cudnn.allow_tf32 = True
    
    set_seed(args.seed)

    
    config = yaml_util.load_yaml_file(os.path.expanduser(args.config))

    device = torch.device(args.device)
    logger.info(config['datasets'])

    transformer = get_transforms('test')
    val_set = torchvision.datasets.CIFAR10('~/dataset/cifar10', transform=transformer, download=True)
    test_data_loader = DataLoader(dataset=val_set, batch_size = 128, shuffle=True, pin_memory=True)
    
    ckpt_file_path = '/home/bepi/Desktop/Ph.D_/projects/nvbitFI/code/checkpoint/mnasnet0_5.pth'
    dnn = torchvision.models.mnasnet0_5()
    dnn.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Add dropout for regularization (optional)
            nn.Linear(1280, 10)  # Adjust the input size to match the MNASNet0.5 output features
        )
    dnn.load_state_dict(torch.load(ckpt_file_path)['state_dict'])
    # dnn = rec_dnn_exploration_mf(dnn)
    test_config = config['test']


    log_freq = test_config.get('log_freq', 1000)
    no_dp_eval = args.no_dp_eval
    

    test_batch_size=1
    test_shuffle=config['test']['test_data_loader']['random_sample']
    test_num_workers=config['test']['test_data_loader']['num_workers']
    subsampler = DatasetSampling(test_data_loader.dataset,5)
    index_dataset=subsampler.listindex()
    data_subset=Subset(test_data_loader.dataset, index_dataset)
    dataloader = DataLoader(data_subset,batch_size=test_batch_size, shuffle=test_shuffle,pin_memory=True,num_workers=test_num_workers)

    if args.fsim_config:
        fsim_config_descriptor = yaml_util.load_yaml_file(os.path.expanduser(args.fsim_config))
        conf_fault_dict=fsim_config_descriptor['fault_info']['weights']
        cwd=os.getcwd() 
        dnn.eval() 
        full_log_path=cwd
        # 1. create the fault injection setup
        FI_setup=FI_manager(full_log_path,"ckpt_FI.json","fsim_report.csv")

        # 2. Run a fault free scenario to generate the golden model
        FI_setup.open_golden_results("Golden_results")
        evaluate(dnn, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                log_freq=log_freq, title='[DNN under test: {}]'.format(type(dnn)), header='Golden', fsim_enabled=True, Fsim_setup=FI_setup) 
        FI_setup.close_golden_results()

        # 3. Prepare the Model for fault injections
        FI_setup.FI_framework.create_fault_injection_model(device,dnn,
                                            batch_size=1,
                                            input_shape=[3,32,32],
                                            layer_types=[torch.nn.Conv2d,torch.nn.Linear])

        logging.getLogger('pytorchfi').disabled = True
        FI_setup.generate_fault_list(flist_mode='static_ber_fixed_layr',f_list_file='fault_list.csv',layer=conf_fault_dict['layer'][0])    
        FI_setup.load_check_point()

        # 5. Execute the fault injection campaign
        for fault,k in FI_setup.iter_fault_list():
            # 5.1 inject the fault in the model
            FI_setup.FI_framework.bit_flip_weight_inj(fault)
            FI_setup.open_faulty_results(f"F_{k}_results")
            try:   
                # 5.2 run the inference with the faulty model 
                evaluate(FI_setup.FI_framework.faulty_model, dataloader, device, device_ids, distributed, no_dp_eval=no_dp_eval,
                    log_freq=log_freq, title='[DNN under test: {}]'.format(type(dnn)), header='FSIM', fsim_enabled=True,Fsim_setup=FI_setup)        
            except Exception as Error:
                msg=f"Exception error: {Error}"
                logger.info(msg)
            # 5.3 Report the results of the fault injection campaign            
            FI_setup.parse_results()
            # break
        FI_setup.terminate_fsim()


if __name__ == '__main__':
    argparser = get_argparser()
    main(argparser.parse_args())
