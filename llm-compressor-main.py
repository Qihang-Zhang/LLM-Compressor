from utils.utils import *
from utils.enwiki9_dataset import *
import wandb
import time
import argparse
from tqdm import tqdm

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--en_wandb", type=bool, default=False)
    parser.add_argument("--dataset", type=str, default="8813.txt")
    args = parser.parse_args()
    
    torch.set_printoptions(precision=50)
    if args.en_wandb:
        wandb.init(
        # set entity to specify your username or team name
        entity="qihangz-work",
        # set the wandb project where this run will be logged
        project="LLM-Compressor",
        # group=Group Name
        name=args.dataset)
        
        wandb.config.current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        wandb.config.update(args)
        
    if args.device == "gpu":
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    Token_Buffer = Token_Buffer()
    Input_Buffer = Input_Buffer(tokenizer)
    extimated_bits_average = average()
    original_bits_average = average()
    compress_ratio_average = average_ratio()
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)
    
    with open('dataset/' + args.dataset, 'r', encoding='utf-8') as f:
        for line in f:
            # get tokens of each line and send them to a list
            outputs = tokenizer(line)
            token_list = outputs["input_ids"]
            Token_Buffer.update(token_list)
            while not Token_Buffer.is_empty():
                token = Token_Buffer.pop()
                _, (log_prob, _, _) = tokenstensor2cdf(Input_Buffer.get_tensor().to(device), model, debug=True)
                _log_prob = -1 * log_prob[token].item()
                decoder_str = list(map(tokenizer.decode, [token]))
                original_bits = len(decoder_str[0]) * 8
                estimated_bits = _log_prob / log(2)
                extimated_bits_average.update(estimated_bits)
                original_bits_average.update(original_bits)
                compress_ratio_average.update(estimated_bits, original_bits)
                
                if args.en_wandb:
                    wandb.log({"estimated_bits": float(estimated_bits),
                               "original_bits": float(original_bits),
                               "compress_ratio": float(estimated_bits / original_bits)})
                    wandb.log({"extimated_bits_average": float(extimated_bits_average.get()),
                               "original_bits_average": float(original_bits_average.get()),
                               "compress_ratio_average": float(compress_ratio_average.get())})
                    
                Input_Buffer.update(token)
                
    if args.en_wandb:
        wandb.finish()