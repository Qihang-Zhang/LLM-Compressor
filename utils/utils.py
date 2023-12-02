import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from mpmath import*
from copy import deepcopy
import math
import argparse
from sys import getsizeof

'''
Func: 
    Calculate the cdf of the distribution P(x_n | x_n-1, ... , x_0)
Input: 
    + tokenstensor: a dictionary containing keys of "input_ids" and "mask"
        + and this input is **(x_0, x_1, ..., x_n-1)**
    + model : a specific llm model
Output: 
    + cdf: Cumulative distribution function of P(x_n | x_n-1, ... , x_0)
    + shape of cdf is [50257]
'''
def tokenstensor2cdf(tokenstensor, model, debug = False):
    outputs = model(**tokenstensor, labels=tokenstensor["input_ids"])
    logits = outputs.logits.squeeze(0)[-1,]
    # print("logits.shape",logits.shape)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    # print("probs.shape", probs.shape)
    cdf = torch.cumsum(probs, dim = 0)
    # print("cdf.shape", cdf.shape)
    if not debug:
        return cdf
    else:
        logits_expsum = torch.logsumexp(logits, -1)
        log_prob = logits - logits_expsum.unsqueeze(-1).repeat(50257)
        # print('log_prob.shape', log_prob.shape)

        # print("tokenstensor.input_ids.shape", tokenstensor["input_ids"].shape)
        current_token = tokenstensor["input_ids"].squeeze(0)[-1].item()
        # tuple_ids_token = (tokenstensor["input_ids"].shape[1], current_token)
        # print(tuple_ids_token)
        return cdf, (log_prob, probs, current_token)

class ac_bound:
    def __init__(self, left_bound = 0, right_bound = 1):
        self.left_bound = mpf(left_bound.item())
        self.right_bound = mpf(right_bound.item())
        self.lenth = self.right_bound - self.left_bound
        self.midpoint = (self.right_bound + self.left_bound)/2
    
    def display(self):
        print("-" * 120)
        print('l_bound ', self.left_bound)
        print('r_bound ', self.right_bound)
        print('length  ', self.lenth)
        print('midpoint', self.midpoint)
        print("-" * 120)
    
    def zoom_update(self, another_ac_bound):
        self.right_bound = self.left_bound + self.lenth * another_ac_bound.right_bound
        self.left_bound = self.left_bound + self.lenth * another_ac_bound.left_bound
        self.lenth = self.right_bound - self.left_bound
        self.midpoint = (self.right_bound + self.left_bound)/2

'''
record and update those infomation:
    + rawdata bytes
    + encoder bytes
    + compress ratio
'''
class compress_info:
    def __init__(self, rawdata_bytes, encoder_bytes) -> None:
        self.rawdata_bytes = rawdata_bytes
        self.encoder_bytes = encoder_bytes
        if self.rawdata_bytes == 0:
            self.compress_ratio = 0
        else:
            self.compress_ratio = self.encoder_bytes/self.rawdata_bytes
    
    def update_compress_info(self, another_compress_info):
        self.rawdata_bytes = self.rawdata_bytes + another_compress_info.rawdata_bytes
        self.encoder_bytes = self.encoder_bytes + another_compress_info.encoder_bytes
        

    def calculate_compress_ratio(self):
        if self.rawdata_bytes == 0:
            self.compress_ratio = 0
        else:
            self.compress_ratio = self.encoder_bytes/self.rawdata_bytes
        return self.compress_ratio
      
def llmac_encode(string, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(tokenizer.bos_token + string, return_tensors="pt")
        inputs.to(device)

        #Shape of tokenstensor: [tokens#]
        tokenstensor = inputs.input_ids.squeeze(0)
        global_bound = ac_bound(torch.tensor(0.0), torch.tensor(1.0))

        for i in range(len(tokenstensor) - 1):

            #considering the *(i_th + 1)* token
            current_token_index = tokenstensor[i + 1]

            #sliding and get the first i tokens
            current_inputs = deepcopy(inputs)
            current_inputs["input_ids"] = current_inputs["input_ids"][:, :(i + 1)]
            current_inputs["attention_mask"] = current_inputs["attention_mask"][:, :(i + 1)]

            #Calculate cdf and update global infomation
            current_cdf,(log_prob, _, _) = tokenstensor2cdf(current_inputs, model, True)
            decoder_str = list(map(tokenizer.decode, current_token_index.unsqueeze(0)))
            tulip_index_string = (i + 1, decoder_str[0])
            if getsizeof(tulip_index_string) < -1 * log_prob[current_token_index].item()/log(2):
                print("true" * 10)
                stress_print("true" * 10)
            # print(tulip_index_string, getsizeof(tulip_index_string))
            # print(-1 * log_prob[current_token_index].item()/log(2))
            current_bound = ac_bound(current_cdf[max(0, current_token_index - 1)], current_cdf[current_token_index])
            global_bound.zoom_update(current_bound)

            '''
            if the result of arithmertic coding requires too much precision
            and numberical errors occurs, print "down" and break the loop
            '''
            if global_bound.lenth == 0:
                decoder_str = list(map(tokenizer.decode, current_token_index.unsqueeze(0)))
                print("down", i, decoder_str, current_token_index.unsqueeze(0))
                i -= 1
                break
        '''
        i 0-based and (i + 1) is 1-based, so the number of encoded tokens are i + 1
        if the loop is not break, i = len(tokenstensor) - 2, i + 1 = len(tokenstensor) - 1
        else if the loop is break, i + 1 is the the number of encoded tokens
        '''
        token_number = min(i + 1, len(tokenstensor) - 1)
        precision = math.ceil((-1) * log(global_bound.lenth)/log(10))
        # print(precision)
        mp.dps = precision
    return global_bound, token_number

def llmac_decode(ac_number, len_tokens, model, tokenizer, device):
    with torch.no_grad():
        # initialize
        last_bound = ac_bound(torch.tensor(0.0), torch.tensor(1.0))
        inputs = tokenizer(tokenizer.bos_token, return_tensors="pt")
        inputs.to(device)

        for i in range(len_tokens):
            # calculate cdf and get the index of current token
            cdf = tokenstensor2cdf(tokenstensor=inputs, model=model)
            for idx in range(cdf.shape[0]):
                if(ac_number < fmul(last_bound.lenth, mpf(cdf[idx].item())) + last_bound.left_bound):
                    break
            last_bound.zoom_update(ac_bound(cdf[max(idx - 1, 0)], cdf[idx]))

            #update inputs, append the index and mask value of calculated index
            append_token_id = torch.tensor([[idx]]).to(device)
            append_attention_mask = torch.tensor([[1]]).to(device)
            inputs["input_ids"] = torch.cat((inputs["input_ids"], append_token_id), 1)
            inputs["attention_mask"] = torch.cat((inputs["attention_mask"], append_attention_mask), 1)
        
        #convert tensor to string
        str_to_decode = list(map(tokenizer.decode, inputs["input_ids"][:,1:]))
        str_to_decode = str_to_decode[0]
        # print(str_to_decode)
    return str_to_decode
        
# test wheather str_to_encoder can be encoded and decoded by tokenizer
def test_tokenizer(str_to_encoder, tokenizer):
    idx = tokenizer(tokenizer.bos_token + str_to_encoder, return_tensors="pt").input_ids
    print('index of each token: \n', idx)
    decoder_str = list(map(tokenizer.decode, idx))
    print('Decoder of whole sentence: \n', decoder_str)
    decoder_str = list(map(tokenizer.decode, idx[0]))
    print('Decoder of each token: \n', decoder_str)

def encoder_and_decoder(str_to_encoder, model, tokenizer, device):
    inputs = tokenizer(tokenizer.bos_token + str_to_encoder, return_tensors="pt")
    mp.dps = 600 + inputs["input_ids"].shape[-1]
    global_bound, len_tokens_idex = llmac_encode(string=str_to_encoder, model=model, tokenizer=tokenizer, device=device)
    str_to_decode = llmac_decode(ac_number=global_bound.midpoint, len_tokens=len_tokens_idex, model=model, tokenizer=tokenizer, device=device)
    rawdata_bytes = len(str_to_encoder.encode())
    encoder_bytes = len(str(global_bound.midpoint).encode())
    is_acc = str_to_encoder == str_to_decode
    current_compress_info = compress_info(rawdata_bytes=rawdata_bytes, encoder_bytes=encoder_bytes)
    print("Successful:", is_acc, "Tokens:", len_tokens_idex, "rawdata_bytes:", rawdata_bytes, "encoder_bytes:", encoder_bytes, "compress ratio:", current_compress_info.compress_ratio)
    return is_acc, current_compress_info

def stress_print(string, stress_notion = "="):
    print(stress_notion * 60)
    print(string)
    print(stress_notion * 60)
    
#test encoder and decoder
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxdps", default=1200, type=int)
    parser.add_argument("--device", type=str, default="gpu")
    args = parser.parse_args()
    
    if args.device == "gpu":
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = model.to(device)
    str_to_encoder_list =  ["How are you doing",
                            "A Markov chain is irreducible if all states belong to one communicating class.",
                            "A wiki is an online hypertext publication collaboratively edited and managed by its own audience, using a web browser. A typical wiki contains multiple pages for the subjects or scope of the project, and could be either open to the public or limited to use within an organization for maintaining its internal knowledge base.",
                            "Wikis have favored plain-text editing, with fewer and simpler conventions than HTML for indicating style and structure. Although limiting access to HTML and Cascading Style Sheets (CSS) of wikis limits user ability to alter the structure and formatting of wiki content, there are some benefits. Limited access to CSS promotes consistency in the look and feel, and having JavaScript disabled prevents a user from implementing code that may limit other users' access.",
                            "Wikis are enabled by wiki software, otherwise known as wiki engines. A wiki engine, being a form of a content management system, differs from other web-based systems such as blog software, in that the content is created without any defined owner or leader, and wikis have little inherent structure, allowing structure to emerge according to the needs of the users.[1] Wiki engines usually allow content to be written using a simplified markup language and sometimes edited with the help of a rich-text editor.[2] There are dozens of different wiki engines in use, both standalone and part of other software, such as bug tracking systems. Some wiki engines are free and open-source, whereas others are proprietary. Some permit control over different functions (levels of access); for example, editing rights may permit changing, adding, or removing material. Others may permit access without enforcing access control. Other rules may be imposed to organize content.",
                            "In a world full of wonders, where dreams merge with reality, the journey begins. From the tranquil meadows to the awe-inspiring mountains, every step unveils a new tale. As the sun rises, the colors dance across the horizon, painting a breathtaking canvas.Amidst the bustling cities, hearts beat in harmony, and cultures intermingle like a mosaic of life. The symphony of languages echoes through time, celebrating humanity's diversity. Each word, a pearl of wisdom, carries the weight of history.Across vast oceans, ships sail towards the unknown, adventurers seeking treasures untold. Pirates roam the high seas, tales of courage and betrayal intertwined. From the depths of the abyss, mythical creatures emerge, captivating the imaginations of storytellers.In the digital realm, innovation thrives, technology shaping destinies. Virtual realities transcend boundaries, blurring lines between fact and fiction. Algorithms and data weave intricate patterns, shaping the future we navigate.Among the stars, celestial wonders sparkle, distant galaxies whispering secrets of the universe. Cosmic dust paints nebulae, while black holes devour light, bending reality's fabric. Astronomers ponder existence, questioning our place in the vast cosmos.On fertile soil, crops sway in the wind, a testament to nature's bounty. Farmers toil, nurturing life from the Earth's embrace. Seasons change, and the cycle of life renews itself, a testament to the beauty of impermanence.In laboratories, scientists unlock nature's mysteries, pushing the boundaries of knowledge. Medical breakthroughs heal, and new frontiers in space beckon exploration. Researchers delve into quantum realms, uncovering the enigmatic laws of existence.Through laughter and tears, human emotions paint a vivid tapestry of experiences. Love and loss entwine, forging connections that span lifetimes. The resilience of the human spirit shines amidst adversity, sparking hope like a flickering flame.In art's embrace, creatives express the depth of the human soul. Paintings capture fleeting moments, while melodies stir hearts. Words weave tales of heroes, villains, and the struggles that define us.As the clock ticks, history weaves its threads, connecting the past, present, and future. Empires rise and fall, civilizations leave their mark on the world. Lessons learned echo through time, shaping our collective evolution.With each breath, life's symphony plays on, harmonizing the symphony of existence. And as we reflect on this vast mosaic of words, we glimpse the essence of our shared humanity—a journey of discovery, connection, and endless possibilities.",
                            "In the vast expanse of the cosmos, countless galaxies dance like twinkling diamonds on a velvet canvas. Stars, ancient and newborn, illuminate the cosmic stage, narrating tales of birth, life, and demise. Planets, suspended in their cosmic waltz, cradle the delicate threads of life that weave through the fabric of the universe. Within this grand theater of existence, humanity emerges as a curious and resilient species, seeking knowledge and understanding of their place in the cosmic symphony.Throughout history, the insatiable human thirst for knowledge has driven innovation, art, and exploration. From the distant past when early civilizations looked upon the stars with wonder and reverence, to the Renaissance era, where knowledge flourished, and the Enlightenment period, which birthed the age of reason, humanity has consistently yearned to unravel the mysteries of the universe. Through tireless dedication and intellectual prowess, great minds have unraveled the secrets of nature, deciphering the intricate laws governing the cosmos.Scientific discoveries have led to technological revolutions, altering the course of human history. From the harnessing of fire and the invention of the wheel, to the steam engine and electricity, each milestone has shaped the fabric of society. With the advent of space exploration, humanity dared to reach for the stars, venturing beyond the confines of Earth to touch the surface of the moon and send probes to distant planets, unveiling their enigmatic features.Yet, amid the pursuit of knowledge and progress, humanity has faced its own shadows. Wars and conflicts have marred the timeline of civilization, leaving scars of devastation and loss. The wisdom obtained through centuries of experience has also brought forth moral dilemmas and ethical quandaries. As we delve deeper into the mysteries of genetics, artificial intelligence, and quantum mechanics, we must tread cautiously, mindful of the potential consequences of our actions.It is crucial for humanity to maintain a delicate balance between scientific advancement and the preservation of our planet's delicate ecosystem. Climate change, pollution, and the depletion of natural resources stand as pressing challenges that demand immediate attention and collective action. The knowledge and technology we have acquired must be harnessed for the greater good, fostering sustainable practices that safeguard the future for generations to come.In this age of interconnectedness, knowledge flows freely across borders and boundaries, transcending cultural barriers and uniting people from diverse backgrounds. The internet, a testament to human ingenuity, has become a vast repository of information, enabling instantaneous communication and collaboration. However, this digital revolution also brings forth new challenges, such as the proliferation of misinformation and threats to privacy and security.The pursuit of knowledge, with all its wonders and complexities, serves as a testament to the resilience and creativity of the human spirit. As we stand at the dawn of a new era, on the cusp of uncharted territories and limitless potential, let us embrace the responsibility of wielding knowledge with wisdom and compassion. Together, we shall chart a course towards a future where the boundaries of understanding extend beyond the stars, and humanity's legacy becomes an enduring symphony in the grand theater of the cosmos.",
                            "In the annals of human history, there are pivotal moments that reverberate through time, shaping the destinies of nations and inspiring the collective aspirations of countless generations. Among these profound turning points stands a document that epitomizes the quest for freedom, dignity, and self-governance - the Declaration of Independence. Penned on a sweltering summer day, July 4, 1776, in the crucible of the American Revolution, this epochal manuscript emerged as the heart and soul of a fledgling nation yearning to break free from the shackles of tyranny and claim its place among the sovereign states of the world. The Declaration of Independence symbolizes more than mere parchment and quill strokes; it is a timeless testimony to the indomitable spirit of human beings in pursuit of their inalienable rights. It articulates the primal longing for liberty, woven into the very fabric of human existence. Crafted by the brilliant minds and fervent hearts of Thomas Jefferson, John Adams, Benjamin Franklin, Roger Sherman, and Robert Livingston, the Declaration stands as an emblem of collective determination, the culmination of years of struggle, debate, and contemplation. At its core, the Declaration of Independence serves as an eloquent expression of natural law and natural rights. Grounded in the teachings of Enlightenment philosophers such as John Locke, it proclaims that all individuals are inherently entitled to life, liberty, and the pursuit of happiness. These unassailable rights are not bestowed by governments but are inherent in humanity itself. Governments, according to this philosophy, exist to safeguard and protect these rights, and their legitimacy is contingent upon the consent of the governed. The historical backdrop against which the Declaration of Independence unfurled is one of turbulence, oppression, and an unyielding yearning for emancipation. The American colonies, established by brave pioneers seeking freedom and prosperity, found themselves subjected to a distant and domineering British monarchy. Over time, the resentment towards British rule brewed and intensified, as the Crown imposed a series of restrictive measures that weighed heavily on the colonists' spirits. A myriad of grievances fueled the desire for liberty and justice, ranging from taxation without representation to the suppression of trade and the imposition of standing armies. The American Revolution, a cataclysmic struggle that ignited in 1775, became the crucible in which the seeds of independence were sown. Emboldened by the flames of resistance, the founding fathers embarked on a journey that would alter the course of history and herald a new era of self-determination. Inscribed in the Declaration of Independence is not only a declaration of war against British oppression but also an extraordinary vision for a government based on the principles of republican democracy. Through its hallowed words, the framers sought to establish a society characterized by the rule of law, civic participation, and the protection of individual liberties. This visionary document outlined a blueprint for a just and equitable nation, inspiring revolutions and movements far beyond the shores of the newly-born United States of America. Beyond its immediate historical context, the Declaration of Independence has attained universal significance. Its magnetic force resonates with people worldwide, beckoning them to challenge injustice and demand their inherent rights. Over the centuries, it has transcended national boundaries and become a beacon of hope, shining a light on the pursuit of human dignity and emancipation from all forms of oppression. In conclusion, the Declaration of Independence is a testament to the power of ideas, the courage of conviction, and the audacity of hope. It encapsulates the triumph of the human spirit over adversity and stands as an enduring testament to the eternal struggle for liberty and equality. As we explore the text of this momentous proclamation, we delve into the profound yearnings of humanity for a world unshackled by despotism and unfettered by tyranny—a world where the rights of all individuals, regardless of origin or station, are revered and protected. Join us on this transformative journey, as we unravel the essence of the Declaration of Independence and its impact on the course of human history. " + "Thomas Jefferson (1743-1826) was a prominent American statesman, Founding Father, and the principal author of the Declaration of Independence, a document that declared the United States' independence from British rule on July 4, 1776. He served as the third President of the United States from 1801 to 1809 and played a crucial role in shaping the country's early development. Jefferson advocated for individual rights, religious freedom, and democratic governance. Despite his contributions to liberty and democracy, he owned enslaved individuals, which remains a topic of debate and criticism. Jefferson's legacy continues to influence American ideals, and he is honored with the Jefferson Memorial in Washington, D.C. " + "Thomas Jefferson was a prominent American statesman and the principal author of the Declaration of Independence, who advocated for individual rights and democratic governance, serving as the third President of the United States and leaving a complex legacy that continues to influence American ideals. ",
                            "Reward shaping is a technique used in reinforcement learning (RL) to modify the reward function of a given task, with the aim to guide the agent towards better or quicker learning. It is based on the idea of providing additional feedback to the agent, in order to make certain states or actions more desirable or less desirable than others, depending on the ultimate goal of the learning task. Here's a more detailed explanation: 1. **Reward Function:** In reinforcement learning, the agent learns to perform actions in an environment to achieve a goal. The feedback the agent gets about its performance is given by the reward function. This function assigns a numerical value (reward) to each state-action pair, indicating the quality or desirability of that action in that state. 2. **Why Use Reward Shaping?** The design of the reward function is crucial for the agent's learning. If the rewards are too sparse or if only the final outcome is rewarded (like winning or losing a game), the agent might struggle to understand which actions were beneficial and which were not. This can lead to slow or suboptimal learning. Reward shaping helps by providing intermediate rewards that guide the agent in the right direction. 3. **Potential-Based Reward Shaping:** One commonly used form of reward shaping is potential-based reward shaping. This approach assigns a potential value to each state in the environment, and the shaped reward is the difference in potential between the current state and the next state. This encourages the agent to move towards states with higher potential. 4. **Preservation of Optimal Policies:** A key property of potential-based reward shaping is that it preserves the optimal policies of the original MDP (Markov Decision Process). This means that any policy that was optimal in the original problem will remain optimal in the reward-shaped problem. 5. **Risk of Reward Shaping:** While reward shaping can speed up learning, it also comes with risks. If not designed carefully, the additional rewards can mislead the agent and encourage suboptimal behavior. For example, if the shaping rewards are too high, the agent might ignore the original rewards and only follow the shaping rewards. 6. **Ng's Theorem on Reward Shaping:** Andrew Ng and Stuart Russell proposed a theorem on reward shaping which states that potential-based reward shaping does not alter the optimal policy of the original MDP. Here is a simple example of reward shaping: suppose you are training an agent to play a game of chess. The only reward it gets in a traditional setting is +1 for winning, -1 for losing, and perhaps 0 for a draw. However, this is a very sparse reward setting. A possible reward shaping could involve giving the agent small rewards for each of the opponent's pieces it captures. This could help the agent learn more quickly which moves are good and which are not. In summary, reward shaping is a powerful tool in reinforcement learning but needs to be applied with caution to ensure it guides the agent towards truly useful behavior and doesn't unintentionally encourage suboptimal policies.",
                            "ab" * 1023,
                            "0" * 1023,
                            "last",
                            "The Project Gutenberg eBook of Complete Prose Works This ebook is for the use of anyone anywhere in the United States and most other parts of the world at no cost and with almost no restrictions whatsoever. You may copy it, give it away or re-use it under the terms of the Project Gutenberg License included with this ebook or online at www.gutenberg.org. If you are not located in the United States, you will have to check the laws of the country where you are located before using this eBook.",
                        ]
    acc_count = 0
    global_compress_info = compress_info(0, 0)
    for str_to_encoder in tqdm(str_to_encoder_list):
        is_acc, current_compress_info = encoder_and_decoder(str_to_encoder = str_to_encoder, model=model, tokenizer=tokenizer, device = device)
        
        if is_acc:
            acc_count += 1
            global_compress_info.update_compress_info(current_compress_info)
            
    acc_rate = acc_count/len(str_to_encoder_list)
    print("acc_rate", acc_rate)
    print("compress_ratio", global_compress_info.calculate_compress_ratio())



    