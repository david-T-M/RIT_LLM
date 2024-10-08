import requests
import json
import random
import time
import pickle
import pandas as pd

model1 = "dRITphi3_pi_MultiNLI_fs1"
model5 = "dRITphi3_pi_MultiNLI_fs5"
model10 = "dRITphi3_pi_MultiNLI_fs10"
model25 = "dRITphi3_pi_MultiNLI_fs25"
model50 = "dRITphi3_pi_MultiNLI_fs50"

template = {
  "Answer": "",
  "Score Lexical Semantics": "",
  "Score Predicate-Argument Structure": "",
  "Score Logic": "",
  "Score Knowledge & Common Sense": ""
}

df_t=pd.read_pickle("../predictions/RL_MultiNLI_DEVMM_600.pickle")

modelos=[model1,model5,model10,model25,model50]
registros=[1,5,10,25,50]

rutas=["","../phi3_dRIT_pi_fs5/","../phi3_dRIT_pi_fs10/","../phi3_dRIT_pi_fs25/","../phi3_dRIT_pi_fs50/"]

j=0
for m in modelos:
    reg=str(registros[j])
    lista_respuestasOllama=[]
    faltantes=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = ''''Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''
        Hypothesis: '''+ strings["sentence_B"]+'''
        plus next features:
        sums:'''+ str(strings["sumas"])+'''
        negH:'''+ str(strings["negH"])+'''
        list_compability:'''+ str(strings["list_comp"])+'''
        list_incompability:'''+ str(strings["list_incomp"])+'''
        relation:'''+ str(strings["relation"])+'''
        entail:'''+ str(strings["entail"])+'''
        negT:'''+ str(strings["negT"])+'''
        mutinf:'''+ str(strings["mutinf_t"])+'''
        contradiction:'''+ str(strings["contradiction"])+'''
        no_matcheadas:'''+ str(strings["no_matcheadas"])+'''
        Jaro-Winkler_rit:'''+ str(strings["Jaro-Winkler_rit"])+'''
        simBoW:'''+ str(strings["simBoW"])+'''
        jaccard:'''+ str(strings["jaccard"])+'''
        overlap_ent:'''+ str(strings["overlap_ent"])+'''
        max_info_t:'''+ str(strings["max_info_t"])+'''
        only responds using the template. '''
        #print(prompt)
        data = {
            "prompt": prompt,
            "model": m,
            "format": "json",
            "stream": False,
            #"options": {"temperature": 0.2},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=25)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index)
            faltantes.append(index)
            lista_respuestasOllama.append("NA")
    with open(rutas[j]+"resultados/rit_pifs"+reg+"_MultiNLI_DEVMM_600_1.pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    j+=1    

        
model1 = "dRITphi3_pi_Scitail_fs1"
model5 = "dRITphi3_pi_Scitail_fs5"
model10 = "dRITphi3_pi_Scitail_fs10"
model25 = "dRITphi3_pi_Scitail_fs25"
model50 = "dRITphi3_pi_Scitail_fs50"

df_t=pd.read_pickle("../predictions/RL_Scitail_DEV_600.pickle")

modelos=[model1,model5,model10,model25,model50]

j=0
for m in modelos:
    reg=str(registros[j])
    lista_respuestasOllama=[]
    faltantes=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = ''''Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''
        Hypothesis: '''+ strings["sentence_B"]+'''
        plus next features:
        sums:'''+ str(strings["sumas"])+'''
        negH:'''+ str(strings["negH"])+'''
        list_compability:'''+ str(strings["list_comp"])+'''
        list_incompability:'''+ str(strings["list_incomp"])+'''
        relation:'''+ str(strings["relation"])+'''
        entail:'''+ str(strings["entail"])+'''
        negT:'''+ str(strings["negT"])+'''
        mutinf:'''+ str(strings["mutinf_t"])+'''
        contradiction:'''+ str(strings["contradiction"])+'''
        no_matcheadas:'''+ str(strings["no_matcheadas"])+'''
        Jaro-Winkler_rit:'''+ str(strings["Jaro-Winkler_rit"])+'''
        simBoW:'''+ str(strings["simBoW"])+'''
        jaccard:'''+ str(strings["jaccard"])+'''
        overlap_ent:'''+ str(strings["overlap_ent"])+'''
        max_info_t:'''+ str(strings["max_info_t"])+'''
        only responds using the template. '''
        #print(prompt)
        data = {
            "prompt": prompt,
            "model": m,
            "format": "json",
            "stream": False,
            #"options": {"temperature": 0.2},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=25)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index)
            faltantes.append(index)
            lista_respuestasOllama.append("NA")
    with open(rutas[j]+"resultados/rit_pifs"+reg+"_Scitail_DEV_600_1.pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    j+=1 

df_t=pd.read_pickle("../predictions/RL_Scitail_TEST_600.pickle")


j=0
for m in modelos:
    reg=str(registros[j])
    lista_respuestasOllama=[]
    faltantes=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = ''''Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''
        Hypothesis: '''+ strings["sentence_B"]+'''
        plus next features:
        sums:'''+ str(strings["sumas"])+'''
        negH:'''+ str(strings["negH"])+'''
        list_compability:'''+ str(strings["list_comp"])+'''
        list_incompability:'''+ str(strings["list_incomp"])+'''
        relation:'''+ str(strings["relation"])+'''
        entail:'''+ str(strings["entail"])+'''
        negT:'''+ str(strings["negT"])+'''
        mutinf:'''+ str(strings["mutinf_t"])+'''
        contradiction:'''+ str(strings["contradiction"])+'''
        no_matcheadas:'''+ str(strings["no_matcheadas"])+'''
        Jaro-Winkler_rit:'''+ str(strings["Jaro-Winkler_rit"])+'''
        simBoW:'''+ str(strings["simBoW"])+'''
        jaccard:'''+ str(strings["jaccard"])+'''
        overlap_ent:'''+ str(strings["overlap_ent"])+'''
        max_info_t:'''+ str(strings["max_info_t"])+'''
        only responds using the template. '''
        #print(prompt)
        data = {
            "prompt": prompt,
            "model": m,
            "format": "json",
            "stream": False,
            #"options": {"temperature": 0.2},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=25)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index)
            faltantes.append(index)
            lista_respuestasOllama.append("NA")
    with open(rutas[j]+"resultados/rit_pifs"+reg+"_Scitail_TEST_600_1.pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    j+=1     
        
model1 = "dRITphi3_pi_SICK_fs1"
model5 = "dRITphi3_pi_SICK_fs5"
model10 = "dRITphi3_pi_SICK_fs10"
model25 = "dRITphi3_pi_SICK_fs25"
model50 = "dRITphi3_pi_SICK_fs50"

df =pd.read_pickle("../predictions/RL_SICK_DEV_with_prediction.pickle")
original =pd.read_csv("../../RIT_ollama/SICK/SICK_DEV.csv")

df["sentence_A"]=original["sentence_A"]
df["sentence_B"]=original["sentence_B"]

modelos=[model1,model5,model10,model25,model50]

j=0
for m in modelos:
    reg=str(registros[j])
    lista_respuestasOllama=[]
    faltantes=[]
    for index,strings in df.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = ''''Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''
        Hypothesis: '''+ strings["sentence_B"]+'''
        plus next features:
        sums:'''+ str(strings["sumas"])+'''
        negH:'''+ str(strings["negH"])+'''
        list_compability:'''+ str(strings["list_comp"])+'''
        list_incompability:'''+ str(strings["list_incomp"])+'''
        relation:'''+ str(strings["relation"])+'''
        entail:'''+ str(strings["entail"])+'''
        negT:'''+ str(strings["negT"])+'''
        mutinf:'''+ str(strings["mutinf_t"])+'''
        contradiction:'''+ str(strings["contradiction"])+'''
        no_matcheadas:'''+ str(strings["no_matcheadas"])+'''
        Jaro-Winkler_rit:'''+ str(strings["Jaro-Winkler_rit"])+'''
        simBoW:'''+ str(strings["simBoW"])+'''
        jaccard:'''+ str(strings["jaccard"])+'''
        overlap_ent:'''+ str(strings["overlap_ent"])+'''
        max_info_t:'''+ str(strings["max_info_t"])+''' 
        only responds using the template. '''
        #print(prompt)
        data = {
            "prompt": prompt,
            "model": m,
            "format": "json",
            "stream": False,
            #"options": {"temperature": 0.2},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=25)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index)
            faltantes.append(index)
            lista_respuestasOllama.append("NA")
    with open(rutas[j]+"resultados/rit_pifs"+reg+"_SICK_DEV_1.pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    j+=1 
    
df_t=pd.read_pickle("../predictions/RL_SICK_TEST_600.pickle")

j=0
for m in modelos:
    reg=str(registros[j])
    lista_respuestasOllama=[]
    faltantes=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = ''''Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''
        Hypothesis: '''+ strings["sentence_B"]+'''
        plus next features:
        sums:'''+ str(strings["sumas"])+'''
        negH:'''+ str(strings["negH"])+'''
        list_compability:'''+ str(strings["list_comp"])+'''
        list_incompability:'''+ str(strings["list_incomp"])+'''
        relation:'''+ str(strings["relation"])+'''
        entail:'''+ str(strings["entail"])+'''
        negT:'''+ str(strings["negT"])+'''
        mutinf:'''+ str(strings["mutinf_t"])+'''
        contradiction:'''+ str(strings["contradiction"])+'''
        no_matcheadas:'''+ str(strings["no_matcheadas"])+'''
        Jaro-Winkler_rit:'''+ str(strings["Jaro-Winkler_rit"])+'''
        simBoW:'''+ str(strings["simBoW"])+'''
        jaccard:'''+ str(strings["jaccard"])+'''
        overlap_ent:'''+ str(strings["overlap_ent"])+'''
        max_info_t:'''+ str(strings["max_info_t"])+'''
        only responds using the template. '''
        #print(prompt)
        data = {
            "prompt": prompt,
            "model": m,
            "format": "json",
            "stream": False,
            #"options": {"temperature": 0.2},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=25)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index)
            faltantes.append(index)
            lista_respuestasOllama.append("NA")
    with open(rutas[j]+"resultados/rit_pifs"+reg+"_SICK_TEST_600_1.pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    j+=1 
    
    
model1 = "dRITphi3_pi_SNLI_fs1"
model5 = "dRITphi3_pi_SNLI_fs5"
model10 = "dRITphi3_pi_SNLI_fs10"
model25 = "dRITphi3_pi_SNLI_fs25"
model50 = "dRITphi3_pi_SNLI_fs50"

modelos=[model1,model5,model10,model25,model50]

df_t=pd.read_pickle("../predictions/RL_SNLI_DEV_600.pickle")

j=0
for m in modelos:
    reg=str(registros[j])
    lista_respuestasOllama=[]
    faltantes=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = ''''Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''
        Hypothesis: '''+ strings["sentence_B"]+'''
        plus next features:
        sums:'''+ str(strings["sumas"])+'''
        negH:'''+ str(strings["negH"])+'''
        list_compability:'''+ str(strings["list_comp"])+'''
        list_incompability:'''+ str(strings["list_incomp"])+'''
        relation:'''+ str(strings["relation"])+'''
        entail:'''+ str(strings["entail"])+'''
        negT:'''+ str(strings["negT"])+'''
        mutinf:'''+ str(strings["mutinf_t"])+'''
        contradiction:'''+ str(strings["contradiction"])+'''
        no_matcheadas:'''+ str(strings["no_matcheadas"])+'''
        Jaro-Winkler_rit:'''+ str(strings["Jaro-Winkler_rit"])+'''
        simBoW:'''+ str(strings["simBoW"])+'''
        jaccard:'''+ str(strings["jaccard"])+'''
        overlap_ent:'''+ str(strings["overlap_ent"])+'''
        max_info_t:'''+ str(strings["max_info_t"])+'''
        only responds using the template. '''
        #print(prompt)
        data = {
            "prompt": prompt,
            "model": m,
            "format": "json",
            "stream": False,
            #"options": {"temperature": 0.2},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=25)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index)
            faltantes.append(index)
            lista_respuestasOllama.append("NA")
    with open(rutas[j]+"resultados/rit_pifs"+reg+"_SNLI_DEV_600_1.pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    j+=1


df_t=pd.read_pickle("../predictions/RL_SNLI_TEST_600.pickle")
j=0

for m in modelos:
    reg=str(registros[j])
    lista_respuestasOllama=[]
    faltantes=[]
    for index,strings in df_t.iterrows():
        #print(strings["sentence1"],strings["sentence2"],strings['gold_label'])
        prompt = ''''Analyze the following sentences: 
        Text: '''+strings["sentence_A"]+'''
        Hypothesis: '''+ strings["sentence_B"]+'''
        plus next features:
        sums:'''+ str(strings["sumas"])+'''
        negH:'''+ str(strings["negH"])+'''
        list_compability:'''+ str(strings["list_comp"])+'''
        list_incompability:'''+ str(strings["list_incomp"])+'''
        relation:'''+ str(strings["relation"])+'''
        entail:'''+ str(strings["entail"])+'''
        negT:'''+ str(strings["negT"])+'''
        mutinf:'''+ str(strings["mutinf_t"])+'''
        contradiction:'''+ str(strings["contradiction"])+'''
        no_matcheadas:'''+ str(strings["no_matcheadas"])+'''
        Jaro-Winkler_rit:'''+ str(strings["Jaro-Winkler_rit"])+'''
        simBoW:'''+ str(strings["simBoW"])+'''
        jaccard:'''+ str(strings["jaccard"])+'''
        overlap_ent:'''+ str(strings["overlap_ent"])+'''
        max_info_t:'''+ str(strings["max_info_t"])+'''
        only responds using the template. '''
        #print(prompt)
        data = {
            "prompt": prompt,
            "model": m,
            "format": "json",
            "stream": False,
            #"options": {"temperature": 0.2},
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=data, stream=False,timeout=25)        
            json_data = json.loads(response.text)
            lista_respuestasOllama.append(json.dumps(json.loads(json_data["response"]), indent=2))
            print(index)
        except:
            print("Saltó",index)
            faltantes.append(index)
            lista_respuestasOllama.append("NA")
    with open(rutas[j]+"resultados/rit_pifs"+reg+"_SNLI_TEST_600_1.pickle", "wb") as f:
        pickle.dump(lista_respuestasOllama, f)
    j+=1