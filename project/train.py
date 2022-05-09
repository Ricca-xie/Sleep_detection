import torch
from tqdm import tqdm
import datetime
import time
import os
from config import parse_config
from config import logger_config
from my_dataloader import data_loader
from timm.utils import accuracy
# from model.VGG16 import VGG
from model.LSTM import LSTMModel
from model.Lenet5 import LeNet5

def valid(args, model):
    print('Predcting...')
    # dev_data = data_loader(args.data_path, 'X_test_mel.myarray', 'Y_test_mel.myarray', (8251, 3, 1, 300, 64), (8251, 1251), batch_size=args.batchsize, is_training=False)
    # dev_data = data_loader(args.test_path, 'audio.npy', 'label.npy', type='spec-n', batch_size=args.batchsize, is_shuffle=False)
    dev_data = data_loader(args.test_path, 'test.csv', batch_size=2, is_shuffle=False)

    model.eval()
    loss_total = 0.
    acc1_total = 0.
    # acc5_total = 0.
    step = 0
    # import pdb;pdb.set_trace()
    with torch.no_grad():
        for step, (x, label) in enumerate(tqdm(dev_data)):
        # for step, (x, label) in enumerate(dev_data):
            # import pdb;pdb.set_trace()
            x = x.to(device)
            label = label.to(device)
            x = x.reshape([x.shape[0],  1, x.shape[1], x.shape[2]])
            # x = x[:,0]
            loss,result,pred,label = model(x, label)
            acc1= accuracy(pred, label, topk=(1, ))
            acc1= acc1[0].item()/100
            loss_total += float(loss.item())
            acc1_total += acc1
            # acc5_total += acc5
    print("Valid_loss: {}, Valid_acc1:{}".format(loss_total / (step+1), acc1_total / (step+1) ))
    return acc1_total / (step+1)

def train(args):
    # model = LSTMModel(input_dim = 8, hidden_dim = 64, layer_dim=2, output_dim=128)
    model = LeNet5()
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_epoch = -1
    best_acc1 = 0
    best_model = 0
    start_time = time.time()
    # valid(args, model)
    for epoch in range(args.epochs):
        # train_data = data_loader(args.data_path, 'X_train_mel.myarray', 'Y_train_mel.myarray', (145265, 3, 1, 300, 64), (145265, 1251), batch_size=args.batchsize, is_training=True)
        train_data = data_loader(args.train_path, 'train.csv', batch_size=args.batchsize,is_shuffle=True)
        model.train()
        acc1_total = 0.
        # acc5_total = 0.
        loss_total = 0.
        # import pdb; pdb.set_trace()
        for step, (x, label) in enumerate(train_data):

            x = x.to(device)
            x = x.reshape([x.shape[0], 1, x.shape[1], x.shape[2]])
            label = label.to(device)
            model.zero_grad()
            loss,result,pred,label = model(x, label)

            acc1= accuracy(pred, label, topk=(1,))
            acc1= acc1[0].item()/100
            loss.backward()
            optimizer.step()
            acc1_total += acc1
            # acc5_total += acc5
            loss_total += float(loss.item())
            # if step % args.print_every == 0 and step != 0:
            print('epoch %d, step %d, step_loss %.4f, step_acc1 %.4f' % (epoch, step, loss_total/args.print_every, acc1_total/args.print_every))
            loss_total = 0.
            acc1_total = 0.

        # if epoch % args.save_every == 0 and epoch != 0:
            # torch.save(model.state_dict(), args.checkpoint_path+str(epoch)+'.pt')
        acc1= valid(args, model)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_epoch = epoch
            best_model = model
            # torch.save(model.state_dict(), args.checkpoint_path+'_pretrain.pt')
        print('best acc1 is: {}, in epoch {}'.format(best_acc1,  best_epoch))

    # torch.save(best_model.state_dict(), args.checkpoint_path+'pretrain.pt')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = parse_config()
    if not args.device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger = logger_config(os.path.join("logs", f"{args.backbone}_{args.log_file}"))
    train(args)
