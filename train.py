import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import load_data, TextProcessor, convert_text_to_tensors
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#########################################################
# COMP331 Fall 2025 PA2
# This file contains the model class, training loop and evaluation function 
# for you to implement
#########################################################


class NeuralNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, max_length=20):
        super(NeuralNetwork, self).__init__()
        
        self.vocab_size = vocab_size #Sets the total number of tokens for our vocabulary
        self.embedding_dim = embedding_dim #Size of each word vector
        self.hidden_size = hidden_size  #Number of neurons in each hidden layer
        self.output_size = output_size #Number of output classes, 2 for binary classification
        self.max_length = max_length    #Maximum Sequence length after we apply padding and truncation
        #######################
        # TODO: Define your model class 
        # You must include an embedding layer, 
        # at least one linear layer, and an activation function
        # you may change inputs to the init method as you want
        #######################

        # Embedding layer - converts integer word indices into learned vectors - Each word index creates an embedding vector of size "embedding_dim" - padding_idx being set to zero makes sure that the <PAD> token always maps to zero

        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim, padding_idx = 0)

        # First Linear, fully connected layer - transforms pooled embeddings into a hidden layer / representation (?) - input received is the size of average embedding vector, then output is hidden_size

        self.fc1 = nn.Linear(embedding_dim, hidden_size)

        # Instantiating ReLu - Activation function helps model learning hopefully -ReLu(x) will map to zero if negative, otherwise positive

        self.activation = nn.ReLU()

        # Dropout layer - this will prevent overfitting, so nn cannot rely on a single neuron too much

        self.dropout = nn.Dropout(p=0.3)

        # second linear layer : transforms hidden features into class scores, aka logits (?) and outputs the number of sentiment classes

        self.fc2 = nn.Linear(hidden_size, output_size)

        # initializing weights for a more stable training experience

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)




    def forward(self, x):
        #######################
        # TODO: Implement the forward pass
        #######################

        # goals of this forward pass - define how data flows through each layer - entries are token indexes from vocab and returns raw unnormalized data

        #look up embeddings for each token index - embedded -> (batch_size, seq_len, embedding_dim)

        embedded = self.embedding(x)

        # taking the mean of the embeddings so we can arrange them into a single vector - reminder that pad index is set to zero so <PAD> tokens do not contribute - pooled

        pooled = embedded.mean(dim=1)

        # pass through the first linear layer, projecting embeddings into the hidden vector space

        hidden = self.fc1(pooled)

        # applying non-linear activation to embeddings

        hidden = self.activation(hidden)

        # final pass through linear layer to convert to logits so we can then do cross entropy

        hidden = self.dropout(hidden)

        logits = self.fc2(hidden)

        # return the logits so we can use for cross entropy loss function and internally a softmax

        return logits

def train(model, train_features, train_labels, test_features, test_labels, 
                num_epochs=50, learning_rate=0.001):
    """
    Train the neural network model
    
    Args:
        model: The neural network model
        train_features: training features represented by token indices (tensor)
        train_labels: train labels(tensor)
        test_features: test features represented by token indices (tensor)
        test_labels: test labels (tensor)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        returns are optional, you could return training history with every N epoches and losses if you want
    """
    ######################## 
    # TODO: Implement the training loop
    # Hint:
    #   1. Use Adam as your optimizer (available in the optim.Adam() class) rather than SGD
    #######################

    # start by defining cross entropy loss function, optimizer (Using adam as mentioned above), and deploying a mini batcher so computer won't explode

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # mini batching in sets of 64, and shuffling data so it's random

    train_data = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)

    # model in training mode

    model.train()

    # loop through dataset for multiple "epochs" (one epoch is a loop throughout the entire dataset)

    for epoch in range(num_epochs):
        total_loss = 0.0
        # iterating through mini batches, resetting gradients after loop, forward pass, and computing loss gradients
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # compute avg loss for epoch

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    model.eval()
    print("Training completed!")


def evaluate(model, test_features, test_labels):
    """
    Evaluate the trained model on test data
    
    Args:
        model: The trained neural network model
        test_features: (tensor)
        test_labels: (tensor)
    
    Returns:
        a dictionary of evaluation metrics (include test accuracy at the minimum)
        (You could import scikit-learn's metrics implementation to calculate other metrics if you want)
    """
    
    ####################### 
    # TODO: Implement the evaluation function
    # Hints: 
    # 1. Use torch.no_grad() for evaluation
    # 2. Use torch.argmax() to get predicted classes
    #######################

    # begin by setting model to evaluation mode which ensures dropout mode won't affect deterministic results

    model.eval()

    # disable gradient tracking, forward pass (output logits), convert logits to class indices, move tensors to CPU (?)

    with torch.no_grad():
        outputs = model(test_features)
        preds = torch.argmax(outputs, dim=1)

    y_true = test_labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    # computing metrics

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1,_ = precision_recall_fscore_support(y_true, y_pred, average='binary')



    return {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
    } # modify this return statement as you want


if __name__ == "__main__":
    
    ####################
    # TODO: If applicable, modify anything below this line 
    # according to your model configuration 
    # and to suit your need (naming changes, parameter changes, 
    # additional statements and/or functions)
    ####################

    # Load training and test data
    train_texts, train_labels = load_data('train.txt')
    
    test_texts, test_labels = load_data('test.txt')

    # Preprocess text
    processor = TextProcessor(vocab_size=10000)
    processor.build_vocab(train_texts) 
        
    # Convert text documents to tensor representations of word indices
    max_length = 100
    train_features = convert_text_to_tensors(train_texts, processor, max_length)
    test_features = convert_text_to_tensors(test_texts, processor, max_length)
    
    # Create a neural network model 
    # Modify the hyperparameters according to your model architecture
    vocab_size = len(processor.word_to_idx)
    embedding_dim = 100
    hidden_size = 64
    output_size = 2  # Binary classification for sentiment analysis
    
    model = NeuralNetwork(vocab_size, embedding_dim, hidden_size, output_size, max_length)
    
    # Train
    training_history = train(model, train_features, train_labels, test_features, test_labels, 
                                  num_epochs=50, learning_rate=0.001)
    
    # Evaluate
    evaluation_results = evaluate(model, test_features, test_labels)
    
    print(f"Model performance report: \n")
    print(f"Test accuracy: {evaluation_results['test_accuracy']:.4f}")
    print(f"Test F1 score: {evaluation_results['test_f1']:.4f}")

    # Save model weights to file
    outfile = 'trained_model.pth'
    torch.save(model.state_dict(), outfile)
    print(f"Trained model saved to {outfile}")

