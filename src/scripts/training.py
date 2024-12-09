""" 
This script contains the training and validation steps.
There is also a compute_encodings function, which is used to compute the encoding
of source and target domains for domain adaptation plotting and evaluation.

"""

from .constants import *
from .losses_da import mmd_distance

# def generate_iterator(loader, same_suite = True):
#     """Generate iterator for training and validation steps

#     Args:
#         loader (dict): dictionary with the dataloaders
#         same_suite (bool, optional): Defaults to True.

#     Returns:
#         iterator: iterator over the dataloaders
#     """

#     if same_suite:
#         iterator = zip(*loader.values())  
#     else:
#         swapped_loader = dict(reversed(list(loader.items())))
#         iterator = zip(*swapped_loader.values())

#     return iterator

def generate_iterator(loader):
    """Generate iterator for training and validation steps

    Args:
        loader (dict): dictionary with the dataloaders
        same_suite (bool, optional): Defaults to True.

    Returns:
        iterator: iterator over the dataloaders
    """
    return zip(*list(loader.values()))


def compute_loss(y_out, y_true, err_out, encoding_1, encoding_2, hparams, disc, y_out_2=None, disc_cond=None):
    """Compute loss with MMD.

    Args:
        y_out (torch.tensor): predicted values
        y_true (torch.tensor): true values
        err_out (torch.tensor): predicted errors
        encoding_1 (torch.tensor): encoding of one domain
        encoding_2 (torch.tensor): encoding of the other domain
        hparams (HyperParameters): hyperparameters

    Returns:
        torch.tensor: MSE loss
        torch.tensor: LFI loss
        torch.tensor: MMD loss
    """
    # Compute loss of mean, std and MMD
    loss_mse = torch.mean(torch.sum((y_out - y_true)**2., axis=1) , axis=0)
    loss_lfi = torch.mean(torch.sum(((y_out - y_true)**2. - err_out**2.)**2., axis=1) , axis=0)
            # Initialize MMD Loss to zero
    loss_mmd = torch.tensor(0.0, device=device)
    if hparams.domain_adapt == 'MMD':
        loss_mmd = mmd_distance(encoding_1, encoding_2)

        # Regulate the loss with the MMD loss such that mmd_loss is a fraction of hparams.da_loss_fraction of the total loss
        # Idea is log(K * loss_mmd) / (log(loss_mse) + log(loss_lfi) + log(K * loss_mmd)) = hparams.da_loss_fraction

        K = torch.exp((hparams.da_loss_fraction * (torch.log(loss_mse.detach()) + torch.log(loss_lfi.detach()))) / (1 - hparams.da_loss_fraction)) / loss_mmd.detach()
        
        loss_mmd = K * loss_mmd
    elif hparams.domain_adapt == 'ADV':
        # loss_mmd here is loss from domain disciminator
        d_softmax, loss_mmd = disc(torch.cat([encoding_1, encoding_2], dim=0), 
                                   torch.cat((torch.zeros(len(encoding_1)), torch.ones(len(encoding_2))), dim=0).to(device=encoding_1.device, dtype=int))
        if hparams.da_cond_loss_fraction > 0:
            d_softmax_cond, loss_mmd_cond = disc_cond(torch.cat([encoding_1+cos_encoder(y_out, hparams), encoding_2+cos_encoder(y_out_2, hparams)], dim=0), 
                                torch.cat((torch.zeros(len(encoding_1)), torch.ones(len(encoding_2))), dim=0).to(device=encoding_1.device, dtype=int))
            loss_mmd = (1 - hparams.da_cond_loss_fraction) * loss_mmd + hparams.da_cond_loss_fraction * loss_mmd_cond
        K = torch.exp((hparams.da_loss_fraction * (torch.log(loss_mse.detach()) + torch.log(loss_lfi.detach()))) / (1 - hparams.da_loss_fraction)) / loss_mmd.detach()
        loss_mmd = K * loss_mmd
    return loss_mse, loss_lfi, loss_mmd

# Training step
def train(loader, model, hparams, optimizer, scheduler, disc, disc_cond=None):
    """Training step.
    
    Args:
        loader (dict): dictionary with the dataloaders
        model (GNN): model to train
        hparams (HyperParameters): hyperparameters
        optimizer (torch.optim): optimizer
        scheduler (torch.optim.lr_scheduler): scheduler

    Returns:
        float: training loss
        float: MMD loss
    """
    model.train()

    iterator = generate_iterator(loader)

    points = 0

    train_loss_mse = 0
    train_loss_lfi = 0
    train_loss_mmd = 0
    
    for data in iterator:
        data_src_list, data_tgt = data[:-1], data[-1]
        y_true_tgt, out_tgt, encoding_tgt = get_res(data_tgt, model, device)
        loss = 0
        for data_src in data_src_list:
            y_true_src, out_src, encoding_src = get_res(data_src, model, device)
            bs = len(data_src)
            y_true = y_true_src
            out = out_src
            encoding_1, encoding_2 = encoding_src, encoding_tgt
            points += bs
        
            # If cosmo parameters are predicted, perform likelihood-free inference to predict also the standard deviation
            y_out, err_out = out[:,:hparams.pred_params], out[:,hparams.pred_params:2*hparams.pred_params]     # Take mean and standard deviation of the output

            loss_mse, loss_lfi, loss_mmd = compute_loss(y_out, y_true, err_out, encoding_1, encoding_2, hparams, disc, y_out_2=out_tgt[:,:hparams.pred_params], disc_cond=disc_cond)

            train_loss_mse += loss_mse * bs
            train_loss_lfi += loss_lfi * bs
            train_loss_mmd += loss_mmd * bs

            if hparams.weight_da == 0.0:
                loss += torch.log(loss_mse) + torch.log(loss_lfi)
            else:
                loss += torch.log(loss_mse) + torch.log(loss_lfi) + torch.log(hparams.weight_da * loss_mmd)
        
        loss /= len(data_src_list)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        scheduler.step()
        optimizer.zero_grad()  # Clear gradients.
    
    train_loss_mse = train_loss_mse/points
    train_loss_lfi = train_loss_lfi/points
    train_loss_mmd = train_loss_mmd/points

    if hparams.weight_da == 0.0:
        train_loss = (torch.log(train_loss_mse) + torch.log(train_loss_lfi)).item()
        train_loss_mmd_only = 0.0
    elif hparams.domain_adapt == 'ADV':
        train_loss = (torch.log(train_loss_mse) + torch.log(train_loss_lfi) - torch.log(hparams.weight_da * train_loss_mmd)).item() #
        train_loss_mmd_only = -torch.log(hparams.weight_da * train_loss_mmd).item()
    else:
        train_loss = (torch.log(train_loss_mse) + torch.log(train_loss_lfi) + torch.log(train_loss_mmd)).item()
        train_loss_mmd_only = torch.log(train_loss_mmd).item()

    return train_loss, train_loss_mmd_only

# --------------- #

# Testing/validation step
def evaluate(loader, model, hparams, disc, same_suite = True, disc_cond=None):
    """Testing/validation step.
    Saves true values and predictions in Outputs/ folder.

    Args:
        loader (dict): dictionary with the dataloaders
        model (GNN): model to train
        hparams (HyperParameters): hyperparameters
        same_suite (bool, optional): if true, evaluate on same suite, else on opposite suite. Defaults to True.

    Returns:
        float: validation loss
        float: MMD loss
        float: mean absolute error
    """
    model.eval() 

    points = 0

    trueparams = np.zeros((1,hparams.pred_params))
    outparams = np.zeros((1,hparams.pred_params))
    outerrparams = np.zeros((1,hparams.pred_params))

    errs = []
    valid_loss_mse = 0
    valid_loss_lfi = 0
    valid_loss_mmd = 0

    iterator = generate_iterator(loader)
    
    for data in iterator:  # Iterate in batches over the training/test dataset.
        data_src_list, data_tgt = data[:-1], data[-1]
        with torch.no_grad():
            y_true_tgt, out_tgt, encoding_tgt = get_res(data_tgt, model, device)
            for data_src in data_src_list:
                y_true_src, out_src, encoding_src = get_res(data_src, model, device)
                encoding_1, encoding_2 = encoding_src, encoding_tgt # order does NOT matter for MMD, matter for ADV
                if same_suite:
                    bs = len(data_src)
                    y_true = y_true_src
                    out, out_2 = out_src, out_tgt
                else:
                    bs = len(data_tgt)
                    y_true = y_true_tgt
                    out, out_2 = out_tgt, out_src

                points += bs

                # If cosmo parameters are predicted, perform likelihood-free inference to predict also the standard deviation
                y_out, err_out = out[:,:hparams.pred_params], out[:,hparams.pred_params:2*hparams.pred_params]     # Take mean and standard deviation of the output

                loss_mse, loss_lfi, loss_mmd = compute_loss(y_out, y_true, err_out, encoding_1, encoding_2, hparams, disc, y_out_2=out_2[:,:hparams.pred_params], disc_cond=disc_cond)

                valid_loss_mse += loss_mse * bs
                valid_loss_lfi += loss_lfi * bs
                valid_loss_mmd += loss_mmd * bs
                
                err = (y_out - y_true)
                errs.append( np.abs(err.detach().cpu().numpy()).mean() ) # mean over all points in batch

                # Append true values and predictions
                trueparams = np.append(trueparams, y_true.detach().cpu().numpy(), 0)
                outparams = np.append(outparams, y_out.detach().cpu().numpy(), 0)
                outerrparams  = np.append(outerrparams, err_out.detach().cpu().numpy(), 0)

    # Save true values and predictions
    if same_suite:
        np.save("Outputs/"+hparams.name_model()+"_trues_samesuite.npy",trueparams)
        np.save("Outputs/"+hparams.name_model()+"_outputs_samesuite.npy",outparams)
        np.save("Outputs/"+hparams.name_model()+"_errors_samesuite.npy",outerrparams)
    else:
        np.save("Outputs/"+hparams.name_model()+"_trues_testsuite.npy",trueparams)
        np.save("Outputs/"+hparams.name_model()+"_outputs_testsuite.npy",outparams)
        np.save("Outputs/"+hparams.name_model()+"_errors_testsuite.npy",outerrparams)

    mean_abs_err = np.array(errs).mean(axis=0) # mean over all batches

    valid_loss_mse = valid_loss_mse/points
    valid_loss_lfi = valid_loss_lfi/points
    valid_loss_mmd = valid_loss_mmd/points

    if hparams.weight_da == 0.0:
        valid_loss = (torch.log(valid_loss_mse) + torch.log(valid_loss_lfi)).item()
        valid_loss_mmd_only = 0.0
    elif hparams.domain_adapt == 'ADV':
        valid_loss = (torch.log(valid_loss_mse) + torch.log(valid_loss_lfi) - torch.log(hparams.weight_da * valid_loss_mmd)).item()
        valid_loss_mmd_only = -torch.log(hparams.weight_da * valid_loss_mmd).item()
    else:
        valid_loss = (torch.log(valid_loss_mse) + torch.log(valid_loss_lfi) + torch.log(valid_loss_mmd)).item()
        valid_loss_mmd_only = torch.log(valid_loss_mmd).item()

    return valid_loss, valid_loss_mmd_only, mean_abs_err

# --------------- #

# Compute encoding of source and target domains for domain adaptation plotting and evaluation

def compute_encodings(loader, model):
    """Compute encoding of source and target domains for domain adaptation plotting and evaluation.

    Args:
        loader (dict): dictionary with the two dataloaders
        model (GNN): model to train

    Returns:
        torch.tensor: source domain encoding
        torch.tensor: target domain encoding
    """
    model.eval() 

    iterator = generate_iterator(loader)
    source_encodings = []
    target_encodings = []
    labels = []

    for data in iterator:  # Iterate in batches over the training/test dataset.
        with torch.no_grad():

            data_1, data_2 = data[0], data[1] # data
            data_1.to(device)
            data_2.to(device)
            y_true_1 = data_1.y
            y_true_2 = data_2.y

            # Perform a single forward pass.
            encoding_1 = model.compute_encoding(data_1)
            encoding_2 = model.compute_encoding(data_2)

            source_encodings.append(encoding_1.detach().cpu().numpy())
            target_encodings.append(encoding_2.detach().cpu().numpy())

            labels.append([y_true_1.detach().cpu().numpy(), y_true_2.detach().cpu().numpy()])

    source_encodings = np.concatenate(source_encodings, axis=0)
    target_encodings = np.concatenate(target_encodings, axis=0)

    labels = np.concatenate(labels, axis=1)

    return source_encodings, target_encodings, labels


def get_res(data, model, device):
    data.to(device)
    out, encoding = model(data)
    y_true = data.y
    return y_true, out, encoding


def cos_encoder(y_out, hparams):
    return torch.cos(y_out.detach() * hparams.freq_encoder)
