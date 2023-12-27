import torch
import torch.nn.functional as F

def ensemble(args, networks, test_loader):

    # set all the networks in eval mode
    for net in networks:
        net.eval()
    
    test_loss = 0

    print("Evaluating the ensembled model")

    for data, target in test_loader:
        
        outputs = []
        
        assert len(networks) == 2
        
        # average the outputs of all nets
        if args.prediction_wts:
            w = [(1 - args.ensemble_step), args.ensemble_step]
        else:
            w = [0.5, 0.5]

        for idx, model in enumerate(networks): 
            y_hat = model(data)
            outputs.append(w[idx]*y_hat)

        print("number of outputs {} and each is of shape {}".format(len(outputs), outputs[-1].shape))

        output = torch.sum(torch.stack(outputs), dim=0) # sum because multiplied by wts above
        
        # compute ensemble loss
        test_loss += model.loss_module(output, target).item()  

    test_loss = test_loss / len(test_loader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))
        

    