#!/x0/arnavmd/python3/bin/python3
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import argparse
from utils import *
import time

print(torch.__version__)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="CUDA ID", default="0")
    parser.add_argument("--b", help="batch size", default="1")
    parser.add_argument("--e", help="number of epochs", default="55")
    parser.add_argument("--lr", help="initial learning rate", default="1e-4")
    parser.add_argument("--v", help="experiment version", default="0.1")
    parser.add_argument("--lam", help="tradeoff between l2 and adversarial loss", default="0.95")
    parser.add_argument("--window_size", help="Context (in terms of 100kb) for each orthogonal vector", default="14000")
    parser.add_argument("--m", help="additional comments", default="")
    parser.add_argument("--high_res", action='store_true', help="Use if predicting 5kb resolution Hi-C (10kb is used by default)")
    parser.add_argument('--wandb', action='store_true', help='Toggle wandb')
 

    args = parser.parse_args()

    
    if args.high_res:
        from data_loader_5kb import *
        from model_5kb import *
    else:
        from data_loader_10kb import *
        from model_10kb import *
    
    if args.wandb:
        import wandb
        wandb.init()


    print("Run: " + args.m)

    LEARNING_RATE = float(args.lr)
    EXPERIMENT_VERSION = args.v
    LOG_PATH = './logs/' + EXPERIMENT_VERSION + '/'
    LAMBDA = float(args.lam)
    TRAIN_SEQ_LENGTH = 200 
    TEST_SEQ_LENGTH = 200 

    torch.cuda.set_device(int(args.gpu))


    torch.manual_seed(0)    
    model = Net(1, 5, int(args.window_size)).cuda()
    disc = Disc().cuda()
    if args.wandb:
        wandb.watch(model, log='all')


    if os.path.exists(LOG_PATH):
        restore_latest(model, LOG_PATH, ext='.pt_model')
    else:
        os.makedirs(LOG_PATH)

    with open(os.path.join(LOG_PATH, 'setup.txt'), 'a+') as f:
        f.write("\nVersion: " + args.v)
        f.write("\nBatch Size: " + args.b)
        f.write("\nInitial Learning Rate: " + args.lr)
        f.write("\nComments: " + args.m)


    # GM12878 Standard
    test_chroms = ['chr3', 'chr11', 'chr17']
    train_chroms = ['chr1', 'chr2', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']


    train_set = Chip2HiCDataset(seq_length=TRAIN_SEQ_LENGTH, window_size=int(args.window_size), chroms=train_chroms, mode='train') 
    test_set = Chip2HiCDataset(seq_length=TEST_SEQ_LENGTH, window_size=int(args.window_size), chroms=test_chroms, mode='test') 

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    train_log = os.path.join(LOG_PATH, 'train_log.txt')
    test_log = os.path.join(LOG_PATH, 'test_log.txt')

    hidden = None
    log_interval = 20
    parameters = list(model.parameters()) 
    optimizer = optim.Adam(parameters, lr=LEARNING_RATE, weight_decay=0.0005)
    disc_optimizer = optim.Adam(disc.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    min_loss = -10

    t0 = time.time()
    #scaler = torch.cuda.amp.GradScaler()
    for epoch in range(int(args.e)):


        disc_preds_train = []

        lr = np.maximum(LEARNING_RATE * np.power(0.5, (int(epoch / 16))), 1e-6) # learning rate decay
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=0.0005)
        disc_optimizer = optim.Adam(disc.parameters(), lr=lr, weight_decay=0.0005)

        print("="*10 + "Epoch " + str(epoch) + "="*10)

        im = []
        test_loss = []
        preds = []
        labs = []
        model.eval()
        for test_i, (test_data, test_label) in enumerate(test_loader):

            # Don't plot empty images
            if np.linalg.norm(test_label) < 1e-8:
                continue
            
            test_data, test_label = torch.Tensor(test_data[0]).cuda(), torch.Tensor(test_label).cuda()

            with torch.no_grad():
                pred, hidden = model(test_data, hidden_state=None,seq_length=TEST_SEQ_LENGTH)
                loss = model.loss(pred, test_label, seq_length=TEST_SEQ_LENGTH)
                test_loss.append(loss)        
                
                # Plot 5 images on wandb
                if test_i < 5:
                    im.append(wandb.Image(generate_image(test_label.cpu(), pred.detach().cpu(), LOG_PATH, TEST_SEQ_LENGTH, bands=100)))
                else:
                    break


        if args.wandb:
            wandb.log({"Validation Examples": im})
            wandb.log({'val_correlation': np.mean(test_loss)})
        
        print('Test Loss: ', np.mean(test_loss), ' Best: ', str(min_loss))

        if np.mean(test_loss) > min_loss:
            min_loss = np.mean(test_loss)

        save(model, os.path.join(LOG_PATH, '%03d.pt_model' % epoch), num_to_keep=1)
        with open(test_log, 'a+') as f:
            f.write(str(np.mean(test_loss)) + "\n")
       
        losses = []
        model.train()
        disc.train()
        for batch_idx, (data, label) in enumerate(train_loader):
            if (np.linalg.norm(data)) < 1e-8:
                continue

            hidden = None
            label = torch.Tensor(np.squeeze(label)).cuda()
            data = data[0].cuda()
            optimizer.zero_grad()
                  
            output, hidden = model(data,seq_length=TRAIN_SEQ_LENGTH)
            output = torch.squeeze(output)

            # 1 -> real, 0 -> fake

            # Train generator
            mse_loss = model.loss(output, label, seq_length=TRAIN_SEQ_LENGTH)
            disc_out = disc(output.view(1,1,output.shape[0], output.shape[1]))
            adv_loss = F.binary_cross_entropy_with_logits(disc_out.view(1), torch.Tensor([1]).cuda()) # how close is disc pred to 1      
            loss = (LAMBDA)*mse_loss + (1 - LAMBDA)*adv_loss

            loss.backward()
            optimizer.step()


            # Train discriminator
            disc_optimizer.zero_grad()

            true_pred = disc(label.view(1,1,label.shape[0], label.shape[1]))
            fake_pred = disc(output.detach().view(1,1,output.shape[0], output.shape[1]))   
            disc_preds = torch.cat((true_pred, fake_pred), dim=0)  
            disc_loss = disc.loss(disc_preds, torch.Tensor([1, 0]).view(2,1).cuda())
            disc_loss.backward()

            disc_preds_train.append(torch.sigmoid(true_pred).item())
            disc_preds_train.append(torch.sigmoid(fake_pred).item())
            disc_optimizer.step()


            if args.wandb:
                wandb.log({'mse_loss': mse_loss.item()}) 
                wandb.log({'adv_loss': adv_loss.item()})
                wandb.log({'L_G': loss.item()})
                wandb.log({'L_D': disc_loss.item()})

            if batch_idx % log_interval == 0:

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item()))
 
    t1 = time.time()
    print(t1 - t0)       
 
if __name__ == '__main__':
    main()
