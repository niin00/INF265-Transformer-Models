import torch
import torch.nn.functional as F

def greedy_sampling(last_token_logits):
    # TODO: Implement greedy sampling (input is the logits of the last token, output is the selected token ID)
    """
    Greedy sampling: chooses the token with the highest logit.
    
    Args:
        last_token_logits (torch.Tensor): 1D tensor of logits for the last token.
    
    Returns:
        torch.Tensor: Selected token ID as a tensor.
    """
    return torch.argmax(last_token_logits)

def top_p_sampling(last_token_logits, p=0.95, temperature=0.7):
    # TODO: Implement top-p sampling with temperature (input is the logits of the last token, output is the selected token ID)
    
    # scale logits by temperature
    logits_scaled = last_token_logits / temperature
    # compute probabilities
    probs = F.softmax(logits_scaled, dim=-1)
    # Sort the probabilities in descending order and get corresponding indices.
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=0)
    
    cutoff_index = torch.nonzero(cumulative_probs > p)
    
    # Determining the cutoff index: including tokens until cumulative sum exceeds p
    if cutoff_index.numel() > 0:
        cutoff = cutoff_index[0].item() + 1 # include the token that caused the threshold to be exceeded
        sorted_probs = sorted_probs[:cutoff]
        sorted_indices = sorted_indices[:cutoff]
        # Renormalize the probabilities of the remaining tokens
        sorted_probs = sorted_probs / sorted_probs.sum()
    
    # sample from filtered distribution
    next_token_index = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices[next_token_index]
    
    return next_token    
    
    

def sample_sequence(input_sequence, model, strategy, max_len, device, end_id, p=0.95, temperature=0.7):
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device) # Add batch dimension and move to device
        answer = []
        for _ in range(max_len):
            last_token_logits = model(input_sequence)
            last_token_logits = last_token_logits[0, -1, :]

            if strategy == "greedy":
                next_token = greedy_sampling(last_token_logits)
            elif strategy == "top-p":
                next_token = top_p_sampling(last_token_logits, p=p, temperature=temperature)
            else:
                raise ValueError("Invalid sampling strategy.")

            input_sequence = torch.cat([input_sequence, next_token.view(1, 1)], dim=1)
            answer.append(next_token.item())

            if next_token == end_id or input_sequence.size(1) >= max_len:
                break

        return answer

def tokenize_input(tokenizer, text, sep_id):
    """
    Tokenize input text and add special tokens.
    """
    tokens = tokenizer.encode(text).ids
    tokens = tokens + [sep_id]
    return torch.tensor(tokens)

def decode_output(tokenizer, tokens):
    """
    Decode output tokens.
    """
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    from config import config
    from tokenizers import Tokenizer
    from model import TransformerModel

    model = TransformerModel(config)
    model = model.to(config.device)
    model = torch.compile(model)
    model.load_state_dict(torch.load(config.model_filename, weights_only=True, map_location=config.device))

    tokenizer = Tokenizer.from_file(config.tokenizer_filename)

    sep_id = tokenizer.token_to_id(config.sep_token)
    end_id = tokenizer.token_to_id(config.end_token)

    question_text = "what is the largest dog breed?"

    input_sequence = tokenize_input(tokenizer, question_text, sep_id) 

    print("Greedy sampling:")
    answer = sample_sequence(input_sequence, model, "greedy", 100, config.device, end_id)
    answer_text = decode_output(tokenizer, answer)
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")

    print("Top-p sampling (p=0.95, temperature=0.7):")
    answer = sample_sequence(input_sequence, model, "top-p", 100, config.device, end_id, p=0.95, temperature=0.7)
    answer_text = decode_output(tokenizer, answer)
    print(f"Question: {question_text}")
    print(f"Answer: {answer_text}")


