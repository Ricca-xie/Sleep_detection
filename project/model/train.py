import torch
import argparse
from tqdm import tqdm
import datetime
import time
from my_dataloader import data_loader
from timm.utils import accuracy
from VGG16 import VGG

device = 'cpu'
device = torch.device(device)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--data_path', type=str, default='../../data/train')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/vgg')
    return parser.parse_args()

def valid(args, model):
    print('Validating...')
    # dev_data = data_loader(args.data_path, 'X_test_mel.myarray', 'Y_test_mel.myarray', (8251, 3, 1, 300, 64), (8251, 1251), batch_size=args.batchsize, is_training=False)
    dev_data = data_loader(args.data_path, 'audio.npy', 'audio.npy', type='spec-n', batch_size=args.batchsize, is_shuffle=False)


    model.eval()
    loss_total = 0.
    acc1_total = 0.
    acc5_total = 0.
    with torch.no_grad():
        for step, (x, label) in enumerate(tqdm(dev_data)):
            x = x.to(device)
            label = label.to(device)
            x = x[:,0]
            loss,result,pred,label = model(x, label)
            acc1, acc5 = accuracy(pred, label, topk=(1, 5))
            acc1, acc5 = acc1.item()/100, acc5.item()/100
            loss_total += float(loss.item())
            acc1_total += acc1
            acc5_total += acc5
    print("Valid_loss: {}, Valid_acc1:{}, Valid_acc5: {}".format(loss_total / (step+1), acc1_total / (step+1), acc5_total / (step+1) ))
    return acc1_total / (step+1), acc5_total / (step+1)

def train(args):
    model = VGG()
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    best_epoch = -1
    best_acc1 = 0
    best_acc5 = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        # train_data = data_loader(args.data_path, 'X_train_mel.myarray', 'Y_train_mel.myarray', (145265, 3, 1, 300, 64), (145265, 1251), batch_size=args.batchsize, is_training=True)
        train_data = data_loader(args.data_path, 'audio.npy', 'label.npy', type='spec-n', batch_size=args.batchsize,
                               is_shuffle=True)
        model.train()
        acc1_total = 0.
        acc5_total = 0.
        loss_total = 0.
        for step, (x, label) in enumerate(train_data):
            x = x.to(device)
            label = label.to(device)
            x = x.reshape([x.shape[0], x.shape[1], 1, x.shape[2], x.shape[3]])
            x = x[:,0]
            model.zero_grad()
            loss,result,pred,label = model(x, label)
            acc1, acc5 = accuracy(pred, label, topk=(1, 5))
            acc1, acc5 = acc1.item()/100, acc5.item()/100
            loss.backward()
            optimizer.step()
            acc1_total += acc1
            acc5_total += acc5
            loss_total += float(loss.item())
            if step % args.print_every == 0 and step != 0:
                print('epoch %d, step %d, step_loss %.4f, step_acc1 %.4f, step_acc5 %.4f' % (epoch, step, loss_total/args.print_every, acc1_total/args.print_every, acc5_total/args.print_every))
                loss_total = 0.
                acc1_total = 0.
                acc5_total = 0.
        # if epoch % args.save_every == 0 and epoch != 0:
            # torch.save(model.state_dict(), args.checkpoint_path+str(epoch)+'.pt')
        acc1, acc5 = valid(args, model)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            best_epoch = epoch
            # torch.save(model.state_dict(), args.checkpoint_path+'best.pt')
        print('best acc1 is: {}, acc5 is: {}, in epoch {}'.format(best_acc1, best_acc5, best_epoch))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = parse_config()
    train(args)
