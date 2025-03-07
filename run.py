from Dataset import train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
from encoder_decoder import Encoder_decoder, device
import torch
from torch import nn
import torch.optim as optim
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
import torch.nn.functional as F
from tqdm import tqdm
import os


# parameters
h_dim = 128
embed_dim = 128
n_head = 4
n_layer = 3
vocab_size = len(train_dataset.vocab)
max_seq_len = 302

print(train_dataset.vocab)


# model
model = Encoder_decoder(h_dim = h_dim, embed_dim=embed_dim, n_head=n_head, n_layer=n_layer, vocab_size=vocab_size, max_seq_len=max_seq_len)
optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
criterion = torch.nn.CrossEntropyLoss(ignore_index=0) # ignore pad index

# training loop
def train_epoch(model, data_loader, optimizer, criterion, device):
    print("Starting train epoch")
    model.train()
    total_loss = 0
    total_tokens = 0
    for batch in tqdm(data_loader, desc="Training", unit="batch"):
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        output = model(x=encoder_input, y=decoder_input)
        print(f"logits.shape: {output.shape}")
        print(f"target.shape: {target.shape}")

        output = output.view(-1, output.size(-1))
        target = target.view(-1)

        loss = criterion(output, target) # ignore pad

        total_loss += loss.sum().item()
        total_tokens += target.ne(0).sum().item()


        loss.backward()
        optimizer.step()

    avg_loss = total_loss / total_tokens  # average loss per token
    return avg_loss


def eval_model(model, data_loader, criterion, device):
    print("Starting Eval model")
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():  # no gradient calculation during evaluation
        for batch in tqdm(data_loader, desc="Training", unit="batch"):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            target = batch['target'].to(device)

            output = model(x=encoder_input, y=decoder_input)
            output = output.view(-1, output.size(-1))
            target = target.view(-1)

            loss = criterion(output, target)  # ignore padding tokens

            total_loss += loss.sum().item()
            total_tokens += target.ne(0).sum().item()  # count non-padding tokens

    avg_loss = total_loss / total_tokens  # average loss per token
    return avg_loss


def top_p_sampling(logits, p=0.7):
    print("Starting top p sampling")
    """
    Perform top-p (nucleus) sampling.

    Parameters:
    logits (Tensor): The logits from the model (before softmax).
    p (float): The cumulative probability threshold (default: 0.7).

    Returns:
    Tensor: The indices of the sampled tokens.
    """
    # softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # sort probabilities in descending order, keeping track of the indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # compute the sum of probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # get the indices where the cumulative probability exceeds p
    sorted_indices_to_keep = cumulative_probs <= p

    # mask the probabilities and indices, keeping only those within top-p
    sorted_probs = sorted_probs * sorted_indices_to_keep.float()
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1,
                                                   keepdim=True)  # Normalize to get a valid probability distribution

    # sample from the top-p distribution
    sampled_indices = torch.multinomial(sorted_probs, 1)  # Sample 1 token from the top-p distribution

    # map back the sampled indices to the original indices
    return sorted_indices.gather(-1, sampled_indices)


def eval_rouge(model, data_loader, criterion, device, output_file, description):
    print("Starting Eval Rouge")
    model.eval()
    scorer = rouge_scorer.RougeScorer(
        metrics=['rouge1', 'rouge2', 'rougeL'],  # ROUGE-1 (unigrams), ROUGE-2 (bigrams), ROUGE-L (lcs)
        lang='en'
    )
    total_rouge1 = 0
    total_rouge2 = 0
    total_rougeL = 0
    total_f1_micro = 0
    total_f1_macro = 0
    total_tokens = 0

    with torch.no_grad():  # Disable gradient calculation during evaluation
        for batch in tqdm(data_loader, desc="Training", unit="batch"):
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            target = batch['target'].to(device)

            output = model(x=encoder_input, y=decoder_input)

            # start decoding. Top p with p = .7
            sampled_output = top_p_sampling(output, p=.7)




            # convert token indices back to text
            predicted_texts = [decode_sequence(sampled_output[i]) for i in range(sampled_output.size(0))]
            target_texts = [decode_sequence(target[i]) for i in range(target.size(0))]

            # calculate ROUGE scores for each pair of predicted and target sequences
            for pred, target in zip(predicted_texts, target_texts):
                scores = scorer.score(target, pred)
                total_rouge1 += scores['rouge1'].fmeasure
                total_rouge2 += scores['rouge2'].fmeasure
                total_rougeL += scores['rougeL'].fmeasure

                # get F1
                pred_tokens = set(pred.split())
                target_tokens = set(target.split())

                # Calculate F1 score for unigrams
                f1_macro = f1_score(list(target_tokens), list(pred_tokens), average='macro', zero_division=1)
                total_f1_macro += f1_macro

                f1_micro = f1_score(list(target_tokens), list(pred_tokens), average='micro', zero_division=1)
                total_f1_micro += f1_micro

                total_tokens += 1  # we will use this to get average

    avg_rouge1 = total_rouge1 / total_tokens
    avg_rouge2 = total_rouge2 / total_tokens
    avg_rougeL = total_rougeL / total_tokens
    avg_f1_micro = total_f1_micro / total_tokens
    avg_f1_macro = total_f1_macro / total_tokens

    description = description + " p = " + p

    with open(output_file, 'a') as f:
        string = "\n\n=======  " + description + "  ======="
        f.write(string)
        f.write(f"avg_rouge1: {avg_rouge1}\n")
        f.write(f"avg_rouge2: {avg_rouge2}\n")
        f.write(f"avg_rougeL: {avg_rougeL}\n")
        f.write(f"avg_f1_micro: {avg_f1_micro}\n")
        f.write(f"avg_f1_macro: {avg_f1_macro}\n")


    return avg_rouge1, avg_rouge2, avg_rougeL, avg_f1


def decode_sequence(token_indices):
    """Converts a list of token indices back into a string (sequence) using the vocab, stopping at <STOP> token."""
    decoded = []
    for token in token_indices:
        if token.item() == train_dataset.vocab["<STOP>"]:
            break
        decoded.append(train_dataset.inverse_vocab.get(token.item(), train_dataset.vocab["<UNK>"]))
    return ' '.join(decoded)


def train_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=20, checkpoint_dir='./models'):
    print("we are in train loop")
    checkpoint_dir = os.path.join(os.getcwd(), "models")

    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")


        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_loss = eval_model(model, val_loader, criterion, device)

        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


        if val_loss < best_loss:
            best_loss = val_loss
            model_save_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} (Validation Loss Improved)")


def load_best_model(model, checkpoint_path="best_model.pt"):

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    print(f"Best model loaded from {checkpoint_path}")
    return model

train_loop(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=1, checkpoint_dir='/models')



os.makedirs('results', exist_ok=True)
val_results_file = 'results/val.results'
test_results_file = 'results/test.results'




model = Encoder_decoder(h_dim = h_dim, embed_dim=embed_dim, n_head=n_head, n_layer=n_layer, vocab_size=vocab_size, max_seq_len=max_seq_len)
best_model = load_best_model(model, best_model_path)


val_rouge = eval_rouge(model=best_model, data_loader=val_loader, criterion=criterion, device=device, output_file=val_results_file, description=model.description)
print(f"Validation ROUGE Scores: {val_rouge}")


test_rouge = eval_rouge(model=best_model, data_loader=test_loader, criterion=criterion, device=device, output_file=test_results_file, description=model.description)
print(f"Test ROUGE Scores: {test_rouge}")