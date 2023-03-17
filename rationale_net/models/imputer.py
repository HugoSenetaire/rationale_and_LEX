import torch.nn as nn
import torch 
import tqdm
class Imputation(nn.Module):
    def __init__(self, ):
        super(Imputation, self).__init__()
        

    def forward(self, hidden):
        raise NotImplementedError
    
class MarkovChain(nn.Module):
    def __init__(self, dataset, ):
        super(MarkovChain, self).__init__()
        
        self.calculate(dataset)
        self.sequence_len = len(dataset.__getitem__(0)['x'][0])
        self.ind_to_ind_glob = {v: k for k, v in self.ind_glob_to_ind.items()}


    def calculate(self, dataset):
        current_index = 0
        self.ind_glob_to_ind = {}
        self.count_ind = {}
        for k in tqdm.tqdm(range(len(dataset))):
            datapoint = dataset[k]
            x = datapoint['x']
            for i in range(len(x[0])):
                indx = x[0][i].item()
                if indx not in self.ind_glob_to_ind.keys():
                    self.ind_glob_to_ind[indx] = current_index
                    current_index+=1
                self.count_ind[indx] = self.count_ind.get(indx, 0) + 1
        
        self.init_probability = torch.zeros(len(self.ind_glob_to_ind))
        self.transition_probability = torch.zeros([len(self.ind_glob_to_ind)]*2)
        for k in self.count_ind.keys():
            self.init_probability[self.ind_glob_to_ind[k]] += self.count_ind[k]
        

        for k in tqdm.tqdm(range(len(dataset))):
            datapoint = dataset[k]
            x = datapoint['x']
            for i in range(len(x[0]) -1):
                indx = x[0][i].item()
                next_indx = x[0][i+1].item()
                self.transition_probability[self.ind_glob_to_ind[indx], self.ind_glob_to_ind[next_indx]] += 1

        self.init_probability = self.init_probability / (torch.sum(self.init_probability)+1e-8)
        self.transition_probability = self.transition_probability / (torch.sum(self.transition_probability, axis = 1, keepdim=True)+1e-8)
        # self.transition_probability = self.transition_probability.unsqueeze(0).unsqueeze(0).to_sparse(sparse_dim=2)
        self.output_dim = len(self.ind_glob_to_ind)




    def forward(self, data, masks, nb_imputation= 1):

        with torch.no_grad():
            batch_size = data.shape[0]
            current_data = map(lambda x: self.ind_glob_to_ind[x.item()], data.flatten().detach().clone())
            current_data = torch.tensor(list(current_data), dtype=torch.int64, device = data.device).reshape(data.shape).squeeze()
            current_data_complete = torch.nn.functional.one_hot(current_data, num_classes = self.output_dim)
            masks_expanded = masks.unsqueeze(-1).expand(-1, -1, self.output_dim)
            message = torch.zeros((batch_size, self.sequence_len, self.output_dim))   

            message[:, 0] = self.init_probability.unsqueeze(0).expand(batch_size, -1) * (1-masks_expanded[:, 0]) + masks_expanded[:, 0] * current_data_complete[:, 0]
            message[:, 0] = message[:, 0]/(torch.sum(message[:, 0], axis = 1, keepdim=True)+1e-8) # Batchsize * output_dim

        
            # Forward :
            for i in tqdm.tqdm(range(1, self.sequence_len)):
                message[:, i] = torch.matmul(message[:, i-1], self.transition_probability.unsqueeze(0)) * (1-masks_expanded[:, i]) + masks_expanded[:, i] * current_data_complete[:, i]
                message[:, i] = message[:, i]/torch.sum(message[:, i], axis = 1, keepdim=True)

            
            # Backward : 
            masks_nb_imputation = masks.unsqueeze(1).expand(-1, nb_imputation, -1,) # Batchsize * nb_imputation * sequence_len 
            current_data_nb_imputation = current_data.unsqueeze(1).expand(-1, nb_imputation, -1,) # Batchsize * nb_imputation * sequence_len
            output_sample = torch.zeros((batch_size, nb_imputation, self.sequence_len,))

            output_sample[:, :, -1] = torch.distributions.categorical.Categorical(probs = message[:, -1]).sample((nb_imputation,)).permute(1,0,)
            output_sample[:, :, -1] = masks_nb_imputation[:, :, -1] * output_sample[:, :, -1] + (1-masks_nb_imputation[:, :, -1]) * current_data_nb_imputation[:, :, -1]
            message = message.unsqueeze(1).expand(-1, nb_imputation,-1, -1).clone()

            for i in tqdm.tqdm(range(self.sequence_len-2, -1, -1)):
                current_transition = self.transition_probability.unsqueeze(0).unsqueeze(0).expand(batch_size, nb_imputation, -1, -1) # Batchsize * nb_imputation * output_dim * output_dim
                current_transition = torch.cat([current_transition[j, k, :, output_sample[j,k,i+1].long()] for j in range(batch_size) for k in range(nb_imputation)]).reshape(batch_size, nb_imputation, self.output_dim, )        
                message[:, :, i] *= current_transition+(1./self.output_dim)
                
                message[:, :, i] = message[:, :, i]/(torch.sum(message[:, :, i], axis = -1, keepdim=True)+1e-8)
                dist = torch.distributions.categorical.Categorical(probs=message[:, :, i])
                output_sample[:, :, i] = dist.sample()
                output_sample[:, :, i] = (1-masks_nb_imputation[:, :, i]) * output_sample[:, :, i] + masks_nb_imputation[:, :, i] * current_data_nb_imputation[:, :, i]
            
            # print("OUTPUT", output_sample.shape)
            true_output = torch.tensor(list(map(lambda x: self.ind_to_ind_glob[int(x.item())], output_sample.flatten().detach().clone()))).reshape(output_sample.shape).to(data.device)
            # print(true_output.shape)
            return true_output

