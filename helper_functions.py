from sklearn.model_selection import train_test_split

batch_size = 128

def make_dataloaders(training_data, test_data, val_size = 0.1):

    # generate indices: instead of the actual data we pass in integers instead
    train_indices, val_indices, _, _ = train_test_split(
      range(len(training_data)),
      training_data.targets,
      stratify=training_data.targets,
      test_size=val_size,
    )

    # generate subset based on indices
    train_set = Subset(training_data, train_indices)
    val_set = Subset(training_data, val_indices)

    # batch_size = 128

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    return train_dataloader, val_dataloader, test_dataloader


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(model, train_dataloader, val_dataloader, epochs, my_loss, optimizer, scheduler, patience=7, device = device):
    train_losses = []
    val_losses = []
    avg_train_losses = []
    avg_val_losses = []

    train_acc = {'correct':[], 'total':[]}
    val_acc = {'correct':[], 'total':[]}
    avg_train_acc = []
    avg_val_acc = []

    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(epochs):

    model.train()
    for i, data in enumerate(train_dataloader, 1):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = my_loss(outputs, label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)
        train_acc['correct'].append((predicted == label).sum().item())
        train_acc['total'].append(label.size(0))

        if i % 1000 ==0:
            mean_loss = np.average(train_losses[-1000:])
            acc = np.sum(train_acc['correct'][-1000:]) / np.sum(train_acc['total'][-1000:]) * 100
            print('Epoch %d: Minibatch %d . Over 1000 minibatch mean Loss: %5.3f accuracy: %5.3f'%(epoch+1, i, mean_loss, acc))

    mean_loss = np.average(train_losses)
    acc = np.sum(train_acc['correct']) / np.sum(train_acc['total']) * 100
    avg_train_losses.append(mean_loss)
    avg_train_acc.append(acc)

    print('Epoch %d: Training Mean loss: %5.3f and Accuracy: %5.3f'%(epoch+1, mean_loss, acc))

    model.eval()
    for inputs, label in val_dataloader:
        inputs, label = inputs.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = my_loss(outputs, label)
            val_losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)
        val_acc['correct'].append((predicted == label).sum().item())
        val_acc['total'].append(label.size(0))

    mean_val_loss = np.average(val_losses)
    acc = np.sum(val_acc['correct']) / np.sum(val_acc['total']) * 100
    avg_val_losses.append(mean_val_loss)
    avg_val_acc.append(acc)

    print('Epoch %d: Validation Mean loss: %5.3f and Accuracy: %5.3f'%(epoch+1, mean_val_loss, acc))

    train_losses = []
    val_losses = []

    train_acc = {'correct':[], 'total':[]}
    val_acc = {'correct':[], 'total':[]}

    scheduler.step(mean_val_loss)

    early_stopping(mean_val_loss, model)
        
    if early_stopping.early_stop:
        print("Early stopping")
        break

    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_val_losses, avg_train_acc, avg_val_acc


def plot_training_metrics(train_loss, val_loss, train_acc, val_acc, model_name=''):
    fig, axs = plt.subplots(1,2, figsize=(12, 5))

    epochs = range(1,len(train_loss)+1)

    axs[0].plot(epochs, train_loss, label='Train_loss')
    axs[0].plot(epochs, val_loss, label='Val_loss')
    axs[0].legend()
    axs[0].set_xlabel('Epoch number')
    axs[0].set_ylabel('Cross Entropy Loss')
    axs[0].set_title('Loss during training')

    axs[1].plot(epochs, train_acc, label='Train_acc')
    axs[1].plot(epochs, val_acc, label='Val_loss')
    axs[1].legend()
    axs[1].set_xlabel('Epoch number')
    axs[1].set_ylabel('Accuracy in %')
    axs[1].set_title('Accuracy during training');

    plt.suptitle('Performance metrics over training '+model_name+' model', fontsize=15)
    plt.savefig(model_name+ '_metrics_training.png', bbox_inches='tight')


def visualize_predictions(model, test_data, model_name='', transfer_learning=False, index_list =[]):
    fig = plt.figure(figsize= (12,6))
    gs = fig.add_gridspec(2,5, hspace=0.1, wspace=0.3)

    model.eval()

    if len(index_list) ==0:
        index_list = torch.randint(len(test_data), (5,))
        title= 'Sample predictions and probabilites from ' +model_name +' model'
        file_name='Sample_predictions_' +model_name+'.png'
    else:
        index_list = np.random.choice(index_list, size=5, replace=False)
        title= 'Sample mistakes made by '+ model_name + ' model'
        file_name = 'Sample '+ model_name + ' mistakes.png'
    
    for i, rand_int in enumerate(index_list):
        image, label = test_data[rand_int]
        image = image.to(device)

        with torch.no_grad():
            output = model(image.unsqueeze(0))
          
        if transfer_learning:
            image = resize(image, (28,28))
            image = image.permute(1,2,0)
        else:
            image = image.squeeze()

        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(transform_image(image.cpu()))
        ax1.set_axis_off()
        ax1.set_title('Label: {}'.format(classes[label]))


        vals, inds = torch.topk(torch.nn.functional.softmax(output,1),5)

        ax2 = fig.add_subplot(gs[1, i])
        ax2.bar(x= classes.loc[inds.squeeze().tolist()], height = torch.squeeze(vals))
        ax2.set_title('Predicted: {}'.format(classes[inds[0,0].item()]))

    plt.suptitle(title, fontsize=15)
    plt.savefig(file_name, bbox_inches='tight')


def assess_model(model, test_dataloader, loss_func, model_name=''):
    test_losses = []
    test_acc = {'correct':[], 'total':[]}
    test_top5_acc = []

    mistake_index = []

    model.eval()
    for i, data in enumerate(test_dataloader):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            loss = loss_func(outputs, label)
            test_losses.append(loss.item())

        _, predicted = torch.max(outputs.data, 1)

        mistake_mask = (predicted != label)
        mistake_index += (mistake_mask.nonzero()+ i*batch_size).squeeze().tolist()


        vals, inds = torch.topk(torch.nn.functional.softmax(outputs[mistake_mask],1),5)


        correct = (predicted == label).sum().item()
        extra_correct = torch.any(torch.eq(inds, label[mistake_mask].unsqueeze(1) ), 1).sum().item()

        test_acc['correct'].append(correct)
        test_acc['total'].append(label.size(0))

        test_top5_acc.append(correct + extra_correct)


    mean_test_loss = np.average(test_losses)
    acc = np.sum(test_acc['correct']) / np.sum(test_acc['total']) * 100
    top5_acc = np.sum(test_top5_acc) / np.sum(test_acc['total']) * 100

    print('For the ' + model_name + ' model the test loss is: %5.3f and the test accuracy is: %5.3f %%, and the top 5 accuracy is %5.3f %%'%(mean_test_loss, acc, top5_acc))

    return mistake_index


def sample_mistakes(model,  test_data, mistake_index, model_name='', transfer_learning=False):

    visualize_predictions(model, test_data, model_name=model_name, transfer_learning=transfer_learning, index_list = mistake_index)
