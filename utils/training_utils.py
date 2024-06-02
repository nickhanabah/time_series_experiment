import torch
from utils.model_utils import ARNet, set_seed
from torch.utils.data import DataLoader
from utils.data_utils import TimeSeriesDataset
from utils.metrics import metric


def train(epochs, 
          p_lag, 
          future_steps, 
          n_features, training_df, 
          validation_df, 
          target_column = ['OT'], 
          learning_rate=1.e-4, 
          decomp_kernel_size= 7, 
          batch_size = 8, 
          layers = 1, 
          get_residuals = False, 
          model = 'rlinear', 
          optimization = 'intervals'): 
    
    set_seed()
    if optimization == 'intervals': 
        lowerboundnet = ARNet(p_lag=p_lag, n_features=n_features, future_steps=future_steps, decomp_kernel_size=decomp_kernel_size, batch_size=batch_size, layers = layers, model = model, optimization='lowerquantile')
        net = ARNet(p_lag=p_lag, n_features=n_features, future_steps=future_steps, decomp_kernel_size=decomp_kernel_size, batch_size=batch_size, layers = layers, model = model, optimization='mse')
        upperboundnet = ARNet(p_lag=p_lag, n_features=n_features, future_steps=future_steps, decomp_kernel_size=decomp_kernel_size, batch_size=batch_size, layers = layers, model = model, optimization='upperquantile')
    else:    
        net = ARNet(p_lag=p_lag, n_features=n_features, future_steps=future_steps, decomp_kernel_size=decomp_kernel_size, batch_size=batch_size, layers = layers, model = model)

    train_data = DataLoader(TimeSeriesDataset(training_df, future_steps= future_steps, target_column = target_column,p_lag=p_lag), batch_size=batch_size, drop_last=True)
    train_loss_list = []
    val_data = DataLoader(TimeSeriesDataset(validation_df,future_steps= future_steps, target_column = target_column,p_lag=p_lag), batch_size=batch_size, drop_last=True)
    val_loss_list = []

    torch.set_grad_enabled(True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    if optimization == 'intervals': 
        loweroptimizer = torch.optim.Adam(lowerboundnet.parameters(), lr=learning_rate)
        upperoptimizer = torch.optim.Adam(upperboundnet.parameters(), lr=learning_rate)


    for epoch in range(epochs): 

        train_counter = 0
        val_counter = 0
        running_train_loss = 0.
        running_val_loss = 0.
        running_train_mae  = 0.
        running_train_mse  = 0.
        running_train_mape = 0.
        running_val_mae  = 0.
        running_val_mse  = 0.
        running_val_mape = 0.

        if epoch + 1 != 1 and (epoch + 1) % 2 == 0: 
            learning_rate = learning_rate / 2
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        print(f"Current learning rate is : {learning_rate}")  
        print("---------------------------")
        for i, data in enumerate(train_data):
            inputs, labels = data
            labels = labels.squeeze(0).float()

            if optimization == 'intervals': 
                upperoptimizer.zero_grad()
                outputs = upperboundnet(inputs)
                loss = upperboundnet.upperquantilecriterion(outputs.reshape(1,batch_size*future_steps), labels.squeeze(1).reshape(1,batch_size*future_steps))
                loss.backward()
                upperoptimizer.step()

                loweroptimizer.zero_grad()
                outputs = lowerboundnet(inputs)
                loss = lowerboundnet.lowerquantilecriterion(outputs.reshape(1,batch_size*future_steps), labels.squeeze(1).reshape(1,batch_size*future_steps))
                loss.backward()
                loweroptimizer.step()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = net.criterion(outputs, labels.squeeze(1))
            if loss.item() > 100000: 
                print('Loss explosion! This might be due to a very small value that is the divided by...')
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            outputs_array = outputs.detach().cpu().numpy()
            labels_array = labels.squeeze(2).detach().cpu().numpy()
            mae, mse, mape= metric(pred=outputs_array, true=labels_array)
            running_train_mae  += mae
            running_train_mse  += mse
            running_train_mape += mape
            running_train_loss += loss.item()
            train_counter += batch_size
            if train_counter % 5000 == 0: 
                print(f"Current (running) training loss at iteration {train_counter} : {running_train_loss/train_counter}")
        train_loss_list.append(running_train_loss/train_counter)
            
        for i, data in enumerate(val_data):
            inputs, test_labels = data
            test_labels = test_labels.squeeze(0).float()
            output = net(inputs)
            val_loss = net.criterion(output, test_labels.squeeze(1))
            running_val_loss += val_loss.item()

            output_array = output.detach().cpu().numpy()
            test_labels_array = test_labels.squeeze(2).detach().cpu().numpy()
            mae, mse, mape = metric(pred=output_array, true=test_labels_array)
            running_val_mae  += mae
            running_val_mse  += mse
            running_val_mape += mape
            val_counter += batch_size
        val_loss_list.append(running_val_loss/val_counter)

        if epoch % 1 == 0:
            print("") 
            print(f"Epoch {epoch}: ")
            print("")
            print("Train metrics: -------")
            print(f"Running (training) loss is {running_train_loss/train_counter}.")
            print(f"Training MAE is {running_train_mae/train_counter}.")
            print(f"Training MSE is {running_train_mse/train_counter}.")
            print(f"Training MAPE is {running_train_mape/train_counter}.")
            print("")
            print("Val metrics: -------")
            print(f"Running (validation) loss is {running_val_loss/val_counter}.")
            print(f"Validation MAE is {running_val_mae/train_counter}.")
            print(f"Validation MSE is {running_val_mse/train_counter}.")
            print(f"Validation MAPE is {running_val_mape/train_counter}.")
            print("---------------------------")
    
    if get_residuals: 
        residuals = []
        for i, data in enumerate(train_data):
            inputs, labels = data
            labels = labels.squeeze(0).float()
            outputs = net(inputs)
            loss = net.criterion(outputs, labels.squeeze(1))
            outputs_array = outputs.detach().cpu().numpy()
            labels_array = labels.squeeze(2).detach().cpu().numpy()
            [residuals.append(labels_array.item(i) - output_array.item(i)) for i in range(len(output_array))]
        if optimization == 'intervals': 
            return lowerboundnet, net, upperboundnet, residuals
        else: 
            return net, residuals
    
    else: 
        if optimization == 'intervals': 
            return lowerboundnet, net, upperboundnet
        else: 
            return net