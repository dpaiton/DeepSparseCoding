import torch


def train_single_model(model, loss):
    model.optimizer.zero_grad() # clear gradietns of all optimized variables
    loss.backward() # backward pass
    model.optimizer.step()
    if(hasattr(model.params, 'renormalize_weights') and model.params.renormalize_weights):
        with torch.no_grad(): # tell autograd to not record this operation
            model.w.div_(torch.norm(model.w, dim=0, keepdim=True))


def train_epoch(epoch, model, loader):
    model.train()
    epoch_size = len(loader.dataset)
    num_batches = epoch_size / model.params.batch_size
    correct = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(model.params.device), target.to(model.params.device)
        inputs = []
        if(model.params.model_type.lower() == 'ensemble'): # TODO: Move this to train_model
            inputs.append(model[0].preprocess_data(data)) # First model preprocesses the input
            for submodule_idx, submodule in enumerate(model):
                loss = model.get_total_loss((inputs[-1], target), submodule_idx)
                train_single_model(submodule, loss)
                # TODO: include optional parameter to allow gradients to propagate through the entire ensemble.
                inputs.append(submodule.get_encodings(inputs[-1]).detach()) # must detach to prevent gradient leaking
        else:
            inputs.append(model.preprocess_data(data))
            loss = model.get_total_loss((inputs[-1], target))
            train_single_model(model, loss)
        if model.params.train_logs_per_epoch is not None:
            if(batch_idx % int(num_batches/model.params.train_logs_per_epoch) == 0.):
                batch_step = epoch * model.params.batches_per_epoch + batch_idx
                model.print_update(
                    input_data=inputs[0], input_labels=target, batch_step=batch_step)
    if(model.params.model_type.lower() == 'ensemble'):
        for submodule in model:
            submodule.scheduler.step(epoch)
    else:
        model.scheduler.step(epoch)


def test_single_model(model, data, target, epoch, return_pics):
    output = model(data)
    if return_pics:
        return output
    else:
        test_loss = torch.nn.functional.nll_loss(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        return (test_loss, correct)


def test_epoch(epoch, model, loader, log_to_file=True, return_pics=False):
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
                if return_pics:
                    return (data, test_single_model(model, inputs[0], target, epoch, return_pics))
                else:
                    batch_test_loss, batch_correct = test_single_model(
                        model, inputs[0], target, epoch, return_pics)
                test_loss += batch_test_loss
                correct += batch_correct
        test_loss /= len(loader.dataset)
        test_accuracy = 100. * correct / len(loader.dataset)
        stat_dict = {
            'test_epoch':epoch,
            'test_loss':test_loss,
            'test_correct':correct,
            'test_total':len(loader.dataset),
            'test_accuracy':test_accuracy}
        if log_to_file:
            js_str = model.js_dumpstring(stat_dict)
            model.log_info('<stats>'+js_str+'</stats>')
        else:
            return stat_dict
