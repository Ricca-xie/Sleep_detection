import torch
import torch.nn as nn

device = 'cuda:1'#'cpu'
device = torch.device(device)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, class_num=2):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.direction_num = 2
        if self.direction_num == 2:
            bidirection = True
        else:
            bidirection = False
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional= bidirection)

        # Readout layer
        self.fc = nn.Linear(hidden_dim * self.direction_num, output_dim)
        self.linear = nn.Linear(output_dim, class_num)
        self.activate = nn.Softmax(dim=1)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, label = None):
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################

        h0 = torch.zeros(self.layer_dim * self.direction_num, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim * self.direction_num, x.size(0), self.hidden_dim).to(device)

        # One time step
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        out = self.linear(out)
        # out.size() --> 100, 10


        if label != None:
            _, label = label.max(-1)
            pred = self.activate(out)
            loss = self.criteria(out, label)
            return loss, out, pred, label
        else:
            pred = self.activate(out)
            _, pred_label = pred.max(-1)
            return pred_label
        # return out


# '''
# STEP 4: INSTANTIATE MODEL CLASS
# '''
# input_dim = 28
# hidden_dim = 100
# layer_dim = 3  # ONLY CHANGE IS HERE FROM ONE LAYER TO TWO LAYER
# output_dim = 10
#
# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
#
# #######################
# #  USE GPU FOR MODEL  #
# #######################
#
# if torch.cuda.is_available():
#     model.cuda()
#
# '''
# STEP 5: INSTANTIATE LOSS CLASS
# '''
# criterion = nn.CrossEntropyLoss()
#
# '''
# STEP 6: INSTANTIATE OPTIMIZER CLASS
# '''
# learning_rate = 0.1
#
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#
# '''
# STEP 7: TRAIN THE MODEL
# '''
#
# # Number of steps to unroll
# seq_dim = 28
#
# iter = 0
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         # Load images as Variable
#         #######################
#         #  USE GPU FOR MODEL  #
#         #######################
#         if torch.cuda.is_available():
#             images = Variable(images.view(-1, seq_dim, input_dim).cuda())
#             labels = Variable(labels.cuda())
#         else:
#             images = Variable(images.view(-1, seq_dim, input_dim))
#             labels = Variable(labels)
#
#         # Clear gradients w.r.t. parameters
#         optimizer.zero_grad()
#
#         # Forward pass to get output/logits
#         # outputs.size() --> 100, 10
#         outputs = model(images)
#
#         # Calculate Loss: softmax --> cross entropy loss
#         loss = criterion(outputs, labels)
#
#         # Getting gradients w.r.t. parameters
#         loss.backward()
#
#         # Updating parameters
#         optimizer.step()
#
#         iter += 1
#
#         if iter % 500 == 0:
#             # Calculate Accuracy
#             correct = 0
#             total = 0
#             # Iterate through test dataset
#             for images, labels in test_loader:
#                 #######################
#                 #  USE GPU FOR MODEL  #
#                 #######################
#                 if torch.cuda.is_available():
#                     images = Variable(images.view(-1, seq_dim, input_dim).cuda())
#                 else:
#                     images = Variable(images.view(-1, seq_dim, input_dim))
#
#                 # Forward pass only to get logits/output
#                 outputs = model(images)
#
#                 # Get predictions from the maximum value
#                 _, predicted = torch.max(outputs.data, 1)
#
#                 # Total number of labels
#                 total += labels.size(0)
#
#                 # Total correct predictions
#                 #######################
#                 #  USE GPU FOR MODEL  #
#                 #######################
#                 if torch.cuda.is_available():
#                     correct += (predicted.cpu() == labels.cpu()).sum()
#                 else:
#                     correct += (predicted == labels).sum()
#
#             accuracy = 100 * correct / total
#
#             # Print Loss
#             print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))