import os
import numpy as np
import pandas
import pickle
from flask import Flask, request, url_for, redirect, render_template, send_from_directory
import pandas as pd
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn
import torch
import esm
import collections
import pandas as pd
import gc
import torch
from transformers import AutoModel, AutoTokenizer
from rdkit import Chem
from model import Contrastive_learning_layer
app = Flask(__name__)


model_smiles = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)

def MolFormer_embedding(model_smiles, tokenizer, SMILES_list):
    inputs = tokenizer(SMILES_list, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model_smiles(**inputs)
    # NOTICE: if you have several smiles in the list, you will find the average embedding of each token will remain the same
    #           no matter which smiles in side the list, however, the padding will based on the longest smiles,
    #           therefore, the last hidden state representation shape:[len, 768] will change for the same smiles in difference smiles list.
    return outputs.pooler_output # shape is [len_list, 768] ; torch tensor;

def esm_embeddings_1280(esm2, esm2_alphabet, peptide_sequence_list):
  # NOTICE: ESM for embeddings is quite RAM usage, if your sequence is too long,
  #         or you have too many sequences for transformation in a single converting,
  #         you computer might automatically kill the job.


  if torch.cuda.is_available():
    device = torch.device("cuda")
  else:
    device = torch.device("cpu")
  esm2 = esm2.eval().to(device)

  batch_converter = esm2_alphabet.get_batch_converter()

  # load the peptide sequence list into the bach_converter
  batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
  batch_lens = (batch_tokens != esm2_alphabet.padding_idx).sum(1)
  ## batch tokens are the embedding results of the whole data set

  batch_tokens = batch_tokens.to(device)

  # Extract per-residue representations (on CPU)
  with torch.no_grad():
      # Here we export the last layer of the EMS model output as the representation of the peptides
      # model'esm2_t12_35M_UR50D' only has 12 layers, and therefore repr_layers parameters is equal to 12
      results = esm2(batch_tokens, repr_layers=[33], return_contacts=False) 
  token_representations = results["representations"][33].cpu()
  del results, batch_tokens
  torch.cuda.empty_cache()
  gc.collect()
  return token_representations[:,1:-1,:].mean(1)

# model_ESM, alphabet = esm.pretrained.esm2_t36_3B_UR50D() # 36 layer
model_ESM, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
class Contrastive_learning_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enzy_refine_layer_1 = nn.Linear(1280, 1280) # W1 and b
        self.smiles_refine_layer_1 = nn.Linear(768, 768) # W1 and b
        self.enzy_refine_layer_2 = nn.Linear(1280, 128) # W1 and b
        self.smiles_refine_layer_2 = nn.Linear(768, 128) # W1 and b

        self.relu = nn.ReLU()
        self.batch_norm_enzy = nn.BatchNorm1d(1280)
        self.batch_norm_smiles = nn.BatchNorm1d(768)
        self.batch_norm_shared = nn.BatchNorm1d(128)

    def forward(self, enzy_embed, smiles_embed):
        refined_enzy_embed = self.enzy_refine_layer_1(enzy_embed)
        refined_smiles_embed = self.smiles_refine_layer_1(smiles_embed)

        refined_enzy_embed = self.batch_norm_enzy(refined_enzy_embed)
        refined_smiles_embed = self.batch_norm_smiles(refined_smiles_embed)

        refined_enzy_embed = self.relu(refined_enzy_embed)
        refined_smiles_embed = self.relu(refined_smiles_embed)

        refined_enzy_embed = self.enzy_refine_layer_2(refined_enzy_embed)
        refined_smiles_embed = self.smiles_refine_layer_2(refined_smiles_embed)

        refined_enzy_embed = self.batch_norm_shared(refined_enzy_embed)
        refined_smiles_embed = self.batch_norm_shared(refined_smiles_embed)
        refined_enzy_embed = torch.nn.functional.normalize(refined_enzy_embed, dim=1)
        refined_smiles_embed = torch.nn.functional.normalize(refined_smiles_embed, dim=1)

        return refined_enzy_embed, refined_smiles_embed


# # collect the output
# def assign_activity(predicted_class):
#     import collections
#     out_put = []
#     for i in range(len(predicted_class)):
#         if predicted_class[i] == 0:
#             # out_put[int_features[i]].append(1)
#             out_put.append('Yes')
#         else:
#             # out_put[int_features[i]].append(2)
#             out_put.append('No')
#     return out_put


def get_filetype(filename):
    return filename.rsplit('.', 1)[1].lower()


# def model_selection(num: str):
#     model = ''
#     if num == '1':
#         model = 'LR.pkl'
#     elif num == '2':
#         model = 'SVM.pkl'
#     elif num == '3':
#         model = 'MLP.pkl'
#     return model


# def text_fasta_reading(file_name):
#     """
#     A function for reading txt and fasta files
#     """
#     import collections
#     # read txt file with sequence inside
#     file_read = open(file_name, mode='r')
#     file_content = []  # create a list for the fasta content temporaty storage
#     for line in file_read:
#         file_content.append(line.strip())  # extract all the information in the file and delete the /n in the file

#     # build a list to collect all the sequence information
#     sequence_name_collect = collections.defaultdict(list)
#     for i in range(len(file_content)):
#         if '>' in file_content[i]:  # check the symbol of the
#             a = i+1
#             seq_template = str()
#             while a <len(file_content) and '>' not in file_content[a] and len(file_content[a])!= 0 :
#                 seq_template = seq_template + file_content[a]
#                 a=a+1
#             sequence_name_collect[file_content[i]].append(seq_template)

#     # transformed into the same style as the xlsx file loaded with pd.read_excel and sequence_list = dataset['sequence']
#     sequence_name_collect = pd.DataFrame(sequence_name_collect).T
#     sequence_list = sequence_name_collect[0]
#     return sequence_list


# create an app object using the Flask class
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 每一个网页上的 输入的框，是一个单独的x，下面这个就是吧这个单独的信息变成一个list，每一个单独的就是一个str （也可以吧x变成int 如果想要的话）
    # int_features  = [str(x) for x in request.form.values()] # this command basically use extract all the input into a list
    # final_features = [np.array(int_features)]
    int_features = [str(x) for x in request.form.values()]
    print(int_features)
    # we have two input in the website, one is the model type and other is the peptide sequences

    # if int(int_features[0]) < 1 or int(int_features[0]) > 12:
    #     return render_template('index.html')
    # model_name = model_selection(int_features[0])
    # model=pickle.load(open(model_name,'rb'))
    

    seq = int_features[0]  # 因为这个list里又两个element我们需要第二个，所以我只需要把吧这个拿出来，然后split
    # 另外需要注意，这个地方，网页上输入的时候必须要是AAA,CCC,SAS, 这个格式，不同的sequence的区分只能使用逗号，其他的都不可以
    embeddings_results_enzy = []
    embeddings_results_smiles = []
    peptide_sequence_list = []
    format_seq = [seq, seq]  # the setting is just following the input format setting in ESM model, [name,sequence]
    tuple_sequence = tuple(format_seq)
    peptide_sequence_list.append(tuple_sequence)  # build a summarize list variable including all the sequence information
    one_seq_embeddings = esm_embeddings_1280(model_ESM, alphabet, peptide_sequence_list)  # conduct the embedding
    embeddings_results_enzy.append(one_seq_embeddings)
    
    smiles_list = []
    smiles_list.append(Chem.CanonSmiles(int_features[1])) 
    one_seq_embeddings = MolFormer_embedding(model_smiles, tokenizer, smiles_list)
    embeddings_results_smiles.append(one_seq_embeddings)
    
    # prediction
    model = torch.load('best_model_esm2_1280_fine_tuned.pt',map_location=torch.device('cpu'))
    embeddings_results_enzy_torch = torch.cat(embeddings_results_enzy, dim=0)
    embeddings_results_smiles_torch = torch.cat(embeddings_results_smiles, dim=0)
    refined_enzy_embed, refined_smiles_embed = model(embeddings_results_enzy_torch,embeddings_results_smiles_torch)
    cosine_sim = torch.nn.functional.cosine_similarity(refined_enzy_embed, refined_smiles_embed, dim=1).detach().cpu().numpy()
    # prediction
    # predicted_class = model.predict(embeddings_results)
    if cosine_sim > 0.5:
        final_output = 'interaction' + '; confidence score is ' + str(cosine_sim)
    else:
        final_output ='non-interaction' + '; confidence score is ' + str(1- cosine_sim)
    # predicted_class = assign_activity(predicted_class)  # transform results (0 and 1) into 'active' and 'non-active'
    # final_output = []
    # for i in range(len(sequence_list)):
    #     temp_output=sequence_list[i]+': '+predicted_class[i]+';'
    #     final_output.append(temp_output)

    return render_template('index.html',
                           prediction_text="Prediction results of input sequences {}".format(final_output))


@app.route('/pred_with_file', methods=['POST'])
def pred_with_file():
    # delete existing files that are in the 'input' folder
    dir = 'input'
    for f in os.listdir(os.path.join(os.getcwd(), dir)):
        os.remove(os.path.join(dir, f))
    # 每一个网页上的 输入的框，是一个单独的x，下面这个就是吧这个单独的信息变成一个list，每一个单独的就是一个str （也可以吧x变成int 如果想要的话）
    # int_features  = [str(x) for x in request.form.values()] # this command basically use extract all the input into a list
    # final_features = [np.array(int_features)]
    
    # features = request.form  # .values()
    # # we have two input in the website, one is the model type and other is the peptide sequences

    # model_name = model_selection(features.get("Model_selection"))
    # model=pickle.load(open(model_name,'rb'))
        
    file = request.files["Peptide_sequences"]
    filename = secure_filename(file.filename)
    filetype = get_filetype(filename)
    save_location = os.path.join('input', filename)
    file.save(save_location)

    sequence_list = []
    df = pandas.read_excel(save_location, header=0)
    sequence_list = df["Protein sequence"].tolist()

    if len(sequence_list) == 0:
        return render_template("index.html")
    
    embeddings_results_enzy = []
    embeddings_results_smiles = []  
    for i in range(df.shape[0]):
        # the setting is just following the input format setting in ESM model, [name,sequence]
        seq_enzy = df['Protein sequence'].iloc[i]
        seq_smiles = df['SMILES'].iloc[i]
        print(seq_enzy,seq_smiles)
        if len(seq_enzy) < 5500:
            tuple_sequence = tuple(['protein',seq_enzy])
            peptide_sequence_list = []
            peptide_sequence_list.append(tuple_sequence) # build a summarize list variable including all the sequence information
            # employ ESM model for converting and save the converted data in csv format
            one_seq_embeddings = esm_embeddings_1280(model_ESM, alphabet, peptide_sequence_list)
            embeddings_results_enzy.append(one_seq_embeddings)
            # the smiles embeddings
            smiles_list = []
            smiles_list.append(Chem.CanonSmiles(seq_smiles)) # build a summarize list variable including all the sequence information
            # employ ESM model for converting and save the converted data in csv format
            one_seq_embeddings = MolFormer_embedding(model_smiles, tokenizer, smiles_list)
            embeddings_results_smiles.append(one_seq_embeddings)
   
    # prediction
    model = torch.load('best_model_esm2_1280_fine_tuned.pt',map_location=torch.device('cpu'))
    embeddings_results_enzy_torch = torch.cat(embeddings_results_enzy, dim=0)
    embeddings_results_smiles_torch = torch.cat(embeddings_results_smiles, dim=0)
    refined_enzy_embed, refined_smiles_embed = model(embeddings_results_enzy_torch,embeddings_results_smiles_torch)
    cosine_sim = torch.nn.functional.cosine_similarity(refined_enzy_embed, refined_smiles_embed, dim=1).detach().cpu().numpy()
    print(cosine_sim)
    print(cosine_sim.shape)
    interaction_result = []
    Confidence_score = []
    for i in cosine_sim:
        if i >0.5:
            interaction_result.append("interaction")
        else:
            interaction_result.append("non-interaction")
            
    for i in cosine_sim:
        if i >0.5:
            Confidence_score.append(i)
        else:
            Confidence_score.append(1-i)
            
    report = {"Protein sequence": df['Protein sequence'].tolist(), "SMILES": df['SMILES'].tolist(), "interaction": interaction_result,"confidence_score":Confidence_score }
    report_df = pandas.DataFrame(report)
    save_result_path = os.path.join('input', "report.xlsx")
    report_df.to_excel(save_result_path)
    send_from_directory("input", "report.xlsx")

    return send_from_directory("input", "report.xlsx")


if __name__ == '__main__':
    app.run()
