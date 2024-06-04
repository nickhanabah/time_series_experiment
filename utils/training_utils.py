import torch
from utils.model_utils import ARNet, set_seed
from torch.utils.data import DataLoader
from utils.data_utils import TimeSeriesDataset
from utils.metrics import metric


def train(epochs, 
          p_lag, 
          future_steps, 
          n_continous_features, 
          n_categorial_features, 
          training_df, 
          validation_df, 
          feature_columns, 
          target_column = ['OT'], 
          learning_rate=1.e-4, 
          decomp_kernel_size= 7, 
          batch_size = 8, 
          #get_residuals = False, 
          model = 'rlinear', 
          modelling_task = 'univatiate'): 
    
    set_seed()
    net = ARNet(p_lag=p_lag, n_continous_features= n_continous_features, n_categorial_features = n_categorial_features,future_steps=future_steps, decomp_kernel_size=decomp_kernel_size, batch_size=batch_size, model = model, modelling_task = modelling_task)
    train_data = DataLoader(TimeSeriesDataset(training_df, future_steps= future_steps, feature_columns = feature_columns, target_column = target_column,p_lag=p_lag), batch_size=batch_size, drop_last=True)
    train_loss_list = []
    val_data = DataLoader(TimeSeriesDataset(validation_df,future_steps= future_steps,feature_columns= feature_columns, target_column = target_column,p_lag=p_lag), batch_size=batch_size, drop_last=True)
    val_loss_list = []

    torch.set_grad_enabled(True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

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

            optimizer.zero_grad()
            outputs = net(inputs)
            if modelling_task == 'multivariate': 
                loss = net.criterion(outputs, labels.reshape(outputs.shape))
            else: 
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
            if modelling_task == 'multivariate': 
                val_loss = net.criterion(outputs, test_labels.reshape(outputs.shape))
            else: 
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
            print(f"Validation MAE is {running_val_mae/val_counter}.")
            print(f"Validation MSE is {running_val_mse/val_counter}.")
            print(f"Validation MAPE is {running_val_mape/val_counter}.")
            print("---------------------------")
    
    #if get_residuals: 
    #    residuals = []
    #    for i, data in enumerate(train_data):
    #        inputs, labels = data
    #        labels = labels.squeeze(0).float()
    #        outputs = net(inputs)
    #        loss = net.criterion(outputs, labels.squeeze(1))
    #        outputs_array = outputs.detach().cpu().numpy()
    #        labels_array = labels.squeeze(2).detach().cpu().numpy()
    #        [residuals.append(labels_array.item(i) - output_array.item(i)) for i in range(len(output_array))]
    #    return net, residuals
    
    #else: 
    return net