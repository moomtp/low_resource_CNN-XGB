# Import Pytorch
import torch
from torch import nn

# Import torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

from sklearn.metrics import f1_score

from tqdm.auto import tqdm

class TrainingHelper():
    def __init__(self,
                model: nn.modules,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str,
                enable_early_stop=True):
        self.model = model
        self.train_dataloader  = train_dataloader
        self.test_dataloader   = test_dataloader
        self.loss_fn           = loss_fn
        self.optimizer         = optimizer
        self.device            = device
        self.enable_early_stop = enable_early_stop

    def test_step(self):
        # Put the model in eval mode
        self.model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_acc, all_preds, all_labels = 0, 0, [], []

        with torch.inference_mode():
            # Loop thourgh data loader data batches
            for batch,(X, y) in enumerate(self.test_dataloader):
                # Send data to the target device
                X, y = X.to(self.device) , y.to(self.device)

                # 1. Forward pass
                test_pred_logits = self.model(X)

                # 2. Calculate the loss
                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()


                # Calculate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1) # Maybe same as softmax results?
                test_acc += (test_pred_labels==y).sum().item() / len(test_pred_labels)

                all_preds.extend(test_pred_labels.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(self.test_dataloader)
        test_acc = test_acc / len(self.test_dataloader)
        # cal F1 score
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return test_loss, test_acc, f1

    def train_step(self):
        # Put the model in train mode
        self.model.train()

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0

        # Loop thourgh data loader data batches
        for batch,(X, y) in enumerate(self.train_dataloader):
            # Send data to the target device
            X, y = X.to(self.device) , y.to(self.device)

            # 1. Forward pass
            y_pred = self.model(X)

            # 2. Calculate the loss
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5 Optimizer step
            self.optimizer.step()

            # Calculat accuracy metric
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class==y).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(self.dataloader)
        train_acc = train_acc / len(self.dataloader)
        return train_loss, train_acc
    
    def train(self, model: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
            epochs: int = 5,
            best_model_name: str = "best_model.pth",
            device:str = 'cpu',
            enable_early_stop=True):
        # 2. Create empty results dicitonary
        results = {"train_loss": [],
                "train_acc":[],
                "test_loss": [],
                "test_acc":[]}
        
        # init best model loss var
        best_test_loss = float('inf')
        best_model = None
        early_stop_count = 0
        early_stop_limit = 7

        # 3. Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device)
            test_loss, test_acc, test_f1 = self.test_step(model=model,
                            dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            device=device)
            # 4. Print out what's happening
            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f} |  F1 score: {test_f1:.4f}   ")

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_acc = test_acc
                best_test_f1 = test_f1
                # best_model = model.state_dict()
                torch.save(model.state_dict(), best_model_name)
                early_stop_count = 0
            else:
                if enable_early_stop:
                    early_stop_count += 1
            
            if early_stop_count >= early_stop_limit:
                print("Early stopping triggered.")
                break

        # 6. Return the filled results at the end of the eopchs
            
        # save best model
        # torch.save(best_model, best_model_name)
        print("the best test acc is :{:.4f}".format(best_test_acc))
        print("the best F1 score is :{:.4f}".format(best_test_f1))
        return results
    

    def forward(self):
        # Put the model in eval mode
        self.model.eval()

        # Setup test loss and test accuracy values
        test_loss, test_acc, all_preds, all_labels = 0, 0, [], []

        with torch.inference_mode():
            # Loop thourgh data loader data batches
            for batch,(X, y) in enumerate(self.test_dataloader):
                # Send data to the target device
                X, y = X.to(self.device) , y.to(self.device)

                # 1. Forward pass
                test_pred_logits = self.model(X)

                # 2. Calculate the loss
                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()


                # Calculate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1) # Maybe same as softmax results?
                test_acc += (test_pred_labels==y).sum().item() / len(test_pred_labels)

                all_preds.extend(test_pred_labels.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Adjust metrics to get average loss and accuracy per batch
        test_loss = test_loss / len(self.test_dataloader)
        test_acc = test_acc / len(self.test_dataloader)
        # cal F1 score
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return test_loss, test_acc, f1




#  =========== 