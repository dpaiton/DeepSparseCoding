import numpy as np
import torch

import DeepSparseCoding.utils.data_processing as dp


def compute_conv_output_shape(in_length, kernel_length, stride, padding=0, dilation=1):
    out_shape = ((in_length + 2 * padding - dilation * (kernel_length - 1) - 1) / stride) + 1
    return np.floor(out_shape).astype(np.int)


def compute_deconv_output_shape(in_length, kernel_length, stride, padding=0, output_padding=0, dilation=1):
    out_shape = (in_length - 1) * stride - 2 * padding + dilation * (kernel_length - 1) + output_padding + 1
    return np.floor(out_shape).astype(np.int)


def get_module_encodings(module, data, allow_grads=False):
    if allow_grads:
        return module.get_encodings(data)
    else:
        return module.get_encodings(data).detach()


def train_single_model(model, loss):
    model.optimizer.zero_grad() # clear gradietns of all optimized variables
    loss.backward() # backward pass
    model.optimizer.step()
    if(hasattr(model.params, 'renormalize_weights') and model.params.renormalize_weights):
        with torch.no_grad(): # tell autograd to not record this operation
            model.weight.div_(dp.get_weights_l2_norm(model.weight))


def train_epoch(epoch, model, loader):
    model.train()
    epoch_size = len(loader.dataset)
    num_batches = epoch_size // model.params.batch_size
    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(model.params.device), target.to(model.params.device)
        inputs = []
        if(model.params.model_type.lower() == 'ensemble'):
            inputs.append(model[0].preprocess_data(data)) # First model preprocesses the input
            for submodule_idx, submodule in enumerate(model):
                loss = model.get_total_loss((inputs[-1], target), submodule_idx)
                train_single_model(submodule, loss)
                encodings = get_module_encodings(submodule, inputs[-1],
                    model.params.allow_parent_grads)
                inputs.append(encodings)
        else:
            inputs.append(model.preprocess_data(data))
            loss = model.get_total_loss((inputs[-1], target))
            train_single_model(model, loss)
        if model.params.train_logs_per_epoch is not None:
            if(batch_idx % int(num_batches/model.params.train_logs_per_epoch) == 0.):
                batch_step = int((epoch - 1) * model.params.batches_per_epoch + batch_idx)
                model.print_update(
                    input_data=inputs[0], input_labels=target, batch_step=batch_step)
    if(model.params.model_type.lower() == 'ensemble'):
        for submodule in model:
            submodule.scheduler.step()
    else:
        model.scheduler.step()


def test_single_model(model, data, target, epoch):
    output = model(data)
    #test_loss = torch.nn.functional.nll_loss(output, target, reduction='sum').item()
    test_loss = torch.nn.CrossEntropyLoss()(output, target)
    pred = output.max(1, keepdim=True)[1]
    correct = pred.eq(target.view_as(pred)).sum().item()
    return (test_loss, correct)


def test_epoch(epoch, model, loader, log_to_file=True):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in loader:
            data, target = data.to(model.params.device), target.to(model.params.device)
            if(model.params.model_type.lower() == 'ensemble'):
                inputs = [model[0].preprocess_data(data)]
                for submodule in model:
                    if(submodule.params.model_type == 'mlp'):
                        batch_test_loss, batch_correct = test_single_model(
                            submodule, inputs[-1], target, epoch)
                        test_loss += batch_test_loss
                        correct += batch_correct
                    inputs.append(submodule.get_encodings(inputs[-1]))
            else:
                inputs = [model.preprocess_data(data)]
                batch_test_loss, batch_correct = test_single_model(
                    model, inputs[0], target, epoch)
                test_loss += batch_test_loss
                correct += batch_correct
        test_loss /= len(loader.dataset)
        test_accuracy = 100. * correct / len(loader.dataset)
        stat_dict = {
            'test_epoch':epoch,
            'test_loss':test_loss.item(),
            'test_correct':correct,
            'test_total':len(loader.dataset),
            'test_accuracy':test_accuracy}
        if log_to_file:
            model.logger.log_stats(stat_dict)
        else:
            return stat_dict

def get_inputs_and_outputs(epoch, model, loader, num_batches=1):
    with torch.no_grad():
        model.eval()
        outputs = []
        targets = []
        inputs = []
        batch = 0
        for data, target in loader:
            if batch >= num_batches:
                pass
            batch += 1
            data, target = data.to(model.params.device), target.to(model.params.device)
            output = model(data)
            inputs.append(data)
            targets.append(target)
            outputs.append(output)
        return (inputs, targets, outputs)
