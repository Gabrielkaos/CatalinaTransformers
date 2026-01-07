from django.shortcuts import render
from django.http import JsonResponse
from MODEL_TRANSFORMER import build_transformer_encoder
import torch
import os
from django.conf import settings
import tiktoken
from unidecode import unidecode
import warnings


def get_segment_ids(tokens):
    """
    Extracts segment IDs for the given input string formatted as:
    "<CLS> " + first sentence + " <SEP> " + second sentence + " <SEP>"
    
    Args:
        input_string (str): The input string.
    
    Returns:
        list: A list of segment IDs corresponding to each token.
    """
    
    segment_ids = []
    segment = 0
    
    for token in tokens:
        segment_ids.append(segment)
        # Switch to segment 1 after the first "<SEP>"
        if token == "<SEP>":
            segment = 1
    
    return segment_ids


def tokens_to_tensor(tokens_list, word_to_index, max_sequence_length, dtype=torch.int32):
    indexed_sequences = [
        [word_to_index.get(token, word_to_index["<PAD>"]) for token in sequence] for sequence in tokens_list
    ]

    padded_sequences = [
        sequence + [word_to_index["<PAD>"]] * (max_sequence_length - len(sequence)) for sequence in indexed_sequences
    ]
    return torch.tensor(padded_sequences, dtype=dtype)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def tokenize_with_tiktoken(text, max_length=None):
    """
    Tokenizes the input text using a tokenizer similar to ChatGPT's tokenizer (tiktoken).

    Args:
        text (str): The input text to be tokenized.
        max_length (int, optional): Maximum number of tokens to return. If None, return all tokens.

    Returns:
        list[int]: List of token IDs.
        list[str]: List of corresponding token strings.
    """
    # Load the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")  # This encoding is similar to GPT-3.5/4.

    # Tokenize the text
    token_ids = tokenizer.encode(text)
    token_strings = [tokenizer.decode([token_id]) for token_id in token_ids]

    # Optionally truncate to max_length
    if max_length:
        token_ids = token_ids[:max_length]
        token_strings = token_strings[:max_length]

    return token_ids, token_strings


def index(request):
    return render(request, 'analyzer/index.html')

def analyze_text(request):
    warnings.filterwarnings("ignore")
    if request.method == 'POST':
        text = request.POST.get('text', '')

        if text.strip() == '':
            return JsonResponse({'error': 'Please enter some text.'})
        analysis_type = request.POST.get('analysis_type', '')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        brain_path = os.path.join(settings.MODEL_DIR, f'{analysis_type}/brain.pth')
        data_path = os.path.join(settings.MODEL_DIR, f'{analysis_type}/data.pth')
        
        max_seq_src = 100

        # data
        data = torch.load(data_path)
        src_vocab = data["src_vocab"]
        tokenizer_src = data["tokenizer_src"]
        num_labels = data["num_labels"]
        label_map = data["label_map"]

        # Model
        # model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, 
        #                                 device=device,
        #                                 d_model=900,
        #                                 n_heads=15,
        #                                 n_layers=10,
        #                                 dff=2500,
        #                                 dropout=0.05).to(device)
        model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, 
                                        device=device,
                                        n_layers=12, 
                                        n_heads=12, 
                                        d_model=768,
                                        dff=2500).to(device)
        model.load_state_dict(torch.load(brain_path)["model_state"])
        model.eval()

        _,line = tokenize_with_tiktoken(unidecode(text.strip()))
        
        with torch.no_grad():
            line = line[:max_seq_src]
            line += ["<PAD>"] * (max_seq_src - len(line))

            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int32).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask).to(device)

            proj_output = model.project(encoder_output)[:,0,:]

            probs = torch.softmax(proj_output.to('cpu'), dim=1)[0] * 100
            
            label_probs = {label_map[idx]: value.item() for idx, value in enumerate(probs)}
            label_probs_sorted = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
            print("\nanalysis_type:",analysis_type)
            print("text given:",text)
            print()
            
            result = {"analysis_type": analysis_type, 'text': text}

            if analysis_type == 'toxicity':
                for label in label_probs_sorted:
                    if label[0]=="toxic":
                        result["toxicity"] = f"{label[1]:.2f}%"
            else:
                for label in label_probs_sorted:
                    result[label[0]] = f"{label[1]:.2f}%"
            return JsonResponse(result)
    return render(request, 'analyzer/index.html')

def check_ai(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        if text.strip() == '':
            return JsonResponse({'error': 'Please enter some text.'})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        brain_path = os.path.join(settings.MODEL_DIR, 'ai_or_human/brain.pth')
        data_path = os.path.join(settings.MODEL_DIR, 'ai_or_human/data.pth')

        max_seq_src = 500

        # data
        data = torch.load(data_path)
        src_vocab = data["src_vocab"]
        tokenizer_src = data["tokenizer_src"]
        num_labels = data["num_labels"]
        label_map = data["label_map"]

        # Model
        model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, 
                                        device=device).to(device)
        model.load_state_dict(torch.load(brain_path)["model_state"])
        model.eval()

        _, line = tokenize_with_tiktoken(unidecode(text).lower().strip())
        
        with torch.no_grad():
            line = line[:max_seq_src]
            line += ["<PAD>"] * (max_seq_src - len(line))

            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int32).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask).to(device)

            proj_output = model.project(encoder_output)[:,0,:]

            probs = torch.softmax(proj_output.to('cpu'), dim=1)[0] * 100
            
            label_probs = {label_map[idx]: value.item() for idx, value in enumerate(probs)}
            label_probs_sorted = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
            print("\nanalysis_type:",'ai_or_human')
            print("text given:",text)
            print()
            
            result = {"analysis_type": 'ai_or_human', 'text': text}
            for label in label_probs_sorted:
                result[label[0]] = f"{label[1]:.2f}%"
            return JsonResponse(result)
    return render(request, 'analyzer/index.html')

def check_phishing(request):
    if request.method == 'POST':
        url = request.POST.get('url', '')
        if url.strip() == '':
            return JsonResponse({'error': 'Please enter some text.'})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        brain_path = os.path.join(settings.MODEL_DIR, 'phishing/brain.pth')
        data_path = os.path.join(settings.MODEL_DIR, 'phishing/data.pth')

        max_seq_src = 1000

        # data
        data = torch.load(data_path)
        src_vocab = data["src_vocab"]
        tokenizer_src = data["tokenizer_src"]
        num_labels = data["num_labels"]
        label_map = data["label_map"]

        # Model
        model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, 
                                        device=device).to(device)
        model.load_state_dict(torch.load(brain_path)["model_state"])
        model.eval()

        line = list(unidecode(url.strip()))
        
        with torch.no_grad():
            line = line[:max_seq_src]
            line += ["<PAD>"] * (max_seq_src - len(line))

            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int32).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask).to(device)

            proj_output = model.project(encoder_output)[:,0,:]

            probs = torch.softmax(proj_output.to('cpu'), dim=1)[0] * 100
            
            label_probs = {label_map[idx]: value.item() for idx, value in enumerate(probs)}
            label_probs_sorted = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
            print("\nanalysis_type:",'phishing')
            print("url given:",url)
            print()
            
            result = {"analysis_type": 'phishing', 'url': url}
            for label in label_probs_sorted:
                result[label[0]] = f"{label[1]:.2f}%"
            return JsonResponse(result)
    return render(request, 'analyzer/index.html')

def check_plagiarism(request):
    if request.method == 'POST':
        sentence1 = request.POST.get('text1', '')
        sentence2 = request.POST.get('text2', '')

        if sentence1.strip() == '' or sentence2.strip() == '':
            return JsonResponse({'error': 'Please enter some text.'})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        brain_path = os.path.join(settings.MODEL_DIR, 'sentence_similarity/brain.pth')
        data_path = os.path.join(settings.MODEL_DIR, 'sentence_similarity/data.pth')

        max_seq_src = 100

        # data
        data = torch.load(data_path)
        src_vocab = data["src_vocab"]
        tokenizer_src = data["tokenizer_src"]
        num_labels = data["num_labels"]
        label_map = data["label_map"]

        # Model
        model = build_transformer_encoder(len(src_vocab), num_labels, max_seq_src, 
                                        device=device, is_segmented=True).to(device)
        model.load_state_dict(torch.load(brain_path)["model_state"])
        model.eval()

        _,line1 = tokenize_with_tiktoken(unidecode(sentence1.strip().lower()))
        _,line2 = tokenize_with_tiktoken(unidecode(sentence2.strip().lower()))
        line = ["<CLS>"] + line1 + ["<SEP>"] + line2 + ["<SEP>"]
        segments = get_segment_ids(line)
        line += ["<PAD>"] * (max_seq_src - len(line))
        segments += [0] * (max_seq_src - len(segments))


        with torch.no_grad():

            x = tokens_to_tensor([line], tokenizer_src, max_seq_src)[0].to(device)

            pad_token = torch.tensor([tokenizer_src["<PAD>"]], dtype=torch.int32).to(device)

            source_mask = (x != pad_token).unsqueeze(0).unsqueeze(0).int().to(device)

            encoder_output = model.encode(x, source_mask, segments=torch.tensor([segments]).to(device)).to(device)

            proj_output = model.project(encoder_output)[:,0,:]

            probs = torch.softmax(proj_output.to('cpu'), dim=1)[0] * 100
            
            label_probs = {label_map[idx]: value.item() for idx, value in enumerate(probs)}
            label_probs_sorted = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)
            print("\nanalysis_type:",'sentence_similarity')
            print("sentence1 given:",sentence1)
            print("sentence2 given:",sentence2)
            print()
            
            result = {"analysis_type": 'plagiarism', 'text1': sentence1, 'text2': sentence2}
            for label in label_probs_sorted:
                if label[0] == "plagiarized":
                    result["similarity"] = f"{label[1]:.2f}%"
            return JsonResponse(result)
    return render(request, 'analyzer/index.html')
