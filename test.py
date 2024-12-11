from fastapi import FastAPI, Query
from pydantic import BaseModel
import spacy
from spacy.matcher import Matcher
from typing import List, Optional
import re
#import json
nlp = spacy.load("en_core_web_sm")
app = FastAPI()
data_Patient=""
data_Phone=""
data_Dob=""
data_Mrn=""
data_Gen=""
data_age=""
pii_json={
    #"Name":[],
    "Patient_name":data_Patient,
    #"Name2":[],
    "Phone_number":data_Phone,
    "DOB":data_Dob,
    "MRN":data_Mrn,
    "Gender":data_Gen,
    "Age":data_age
}
custom_spacy_config = { "gliner_model": "knowledgator/gliner-multitask-large-v0.5",
                            "chunk_size": 250,
                            "labels": ["sex","gender","person","phone_number"],
                            "style": "ent",
                            "threshold":0.3}
nlp.add_pipe("gliner_spacy",config=custom_spacy_config)

class OperatorConfig:
    def __init__(self, action: str, parameters: dict):
        self.action = action
        self.parameters = parameters

operators = {
    "date_of_birth": OperatorConfig("replace", {"new_value": "[REDACTED DATE]"}),
    "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED PERSON]"}),
    "ACCOUNT_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED ACCOUNT]"}),
    "MRN_NUMBER": OperatorConfig("replace", {"new_value": "[REDACTED MRN]"}),
    "AGE": OperatorConfig("replace", {"new_value": "[REDACTED AGE]"}),
    "Account Number": OperatorConfig("replace", {"new_value": "[REDACTED Account]"}),
    "gender": OperatorConfig("replace", {"new_value": "[REDACTED GENDER]"})
}

class TextInput(BaseModel):
    text: str
    redact_entities: Optional[List[str]] = Query([], alias="redact", title="Entities to Redact")

account_number = [
    {"LOWER": "acc"},
    {"LOWER": "no"},
    {"IS_PUNCT": True, "OP": "?"}, 
    {"IS_DIGIT": True, "OP": "+"}   
]
account_number_1 = [
    {"LOWER": "account"},
    {"LOWER": "number"},
    {"IS_PUNCT": True, "OP": "?"}, 
    {"IS_DIGIT": True, "OP": "+"}   
]

account_number_mrn = [
    {"LOWER": "mrn"},
    {"IS_PUNCT": True, "OP": "?"},  
    {"IS_DIGIT": True, "OP": "+"}  
]
name_pattern = [
    {"IS_TITLE": True},    
    {"TEXT": ","},        
    {"IS_TITLE": True},   
]

age_patterns = [
    [{"TEXT": "age"}, {"TEXT": ":"}, {"IS_DIGIT": True}, {"LOWER": {"IN": ["y", "years"]}},],
    [{"IS_DIGIT": True}, {"LOWER": {"IN": ["y/o", "yo"]}}],
    [{"IS_DIGIT": True}, {"TEXT": "y"}],
    [{"TEXT": "("}, {"IS_DIGIT": True}, {"LOWER": "yo"}, {"TEXT": ")"}],
]
gender_patterns = [
    [
        {"LOWER": "gender"}, 
        {"IS_PUNCT": True, "OP": "?"}, 
        {"LOWER": {"IN": ["male", "female", "m", "f"]}},
        {"IS_PUNCT": True, "OP": "?"}
    ],
    [
        {"LOWER": "sex"}, 
        {"IS_PUNCT": True, "OP": "?"}, 
        {"LOWER": {"IN": ["male", "female", "m", "f"]}},
        {"IS_PUNCT": True, "OP": "?"}
    ],
  
    [
        #{"LOWER": {"in": ["Y", "yo"]}},
        {"IS_PUNCT": True, "OP": "?"},
        {"TEXT": {"in": ["F", "M","f","m"]}},
        {"IS_PUNCT": True,"OP":"+"}
    ],
    [{"TEXT": {"in": ["Female", "Male"]}}]
    
]

matcher = Matcher(nlp.vocab)
matcher.add("ACCOUNT_NUMBER", [account_number])
matcher.add("ACCOUNT_NUMBER", [account_number_1])
matcher.add("MRN_NUMBER", [account_number_mrn])
matcher.add("PERSON", [name_pattern])
matcher.add("AGE", age_patterns)
matcher.add("gender", gender_patterns)
dob_pattern = r"DOB[:\s]*([0-1]?\d\/[0-3]?\d\/\d{4})"


@app.post("/extract_and_redact/")
async def extract_and_redact(input_data: TextInput):
    doc = nlp(input_data.text)
    matches = matcher(doc)
    extracted_entities = []
    text = input_data.text 
    #redacted_text = re.sub(dob_pattern, "[DOB: REDACTED]", input_data.text)
    #redacted_text = re.sub(name_pattern, "[REDACTED_name]", input_data.text)
    #redact_all = len(input_data.redact_entities) == 0
    dob= re.findall(dob_pattern, input_data.text)
    gen=[]
    acc=[]
    age_no=[]
    for match_id, start, end in matches:
        match_id_str = nlp.vocab.strings[match_id]  
        span = doc[start:end]  
        #if match_id_str in input_data.redact_entities:
          #  operator = operators.get(match_id_str)
            #if operator and operator.action == "replace":
               # new_value = operator.parameters["new_value"]
                
             #   #redacted_text = redacted_text.replace(span.text, new_value)
        
        extracted_entities.append({"entity": match_id_str, "text": span.text})
        if match_id_str=="gender":
            gen.append(span.text)
        if match_id_str=="ACCOUNT_NUMBER":
            acc.append(span.text)
        if match_id_str=="AGE":
            age_no.append(span.text)
    
    persons=[]
    for i in extracted_entities:
        person=""
        if i['entity']=="PERSON":
            person+=(i['text']).strip()
        persons.append(person)

    #redacted_text = (redacted_text.lower()).replace((persons[0].lower()).strip(), "[PERSON]")
    print(person)
    
    extracted_entities.append({"entity": "PERSON_1", "text": persons[0]})
    from spacy import displacy
    fol=nlp(text)
    displacy.render(fol, style="span")
    #names=[]
    phone=[]
    
    for ent in fol.ents:
        if ent.label_=="phone_number":
            #redacted_text = (redacted_text.lower()).replace((ent.text).lower(), "[phone]")
            phone.append(ent.text)
        #if ent.label_=="person":
           # names.append((ent.text).lower())
   # extracted_entities.append({"entity": "PERSON_2", "text": names})
    extracted_entities.append({"entity": "Phone_numbers", "text": phone})
    #print(names)
    #if len(names)>1:
       # nam=
    #full_name=[]
    #for i in text.split():
     #   if i.lower() in names[0].split():
            # redacted_text = (redacted_text.lower()).replace((i).lower(), "[person]")
    #extracted_entities.append({"entity": "PERSON_3", "text": full_name})
    #pii_json["Name"].extend(full_name)
    pat=[]
    for i in persons:
        if i!="":
            pat.append(i)
    if len(pat[0])!=0:
        data_Patient=pat[0]
    if len(pat[0])==0:
        data_Patient=""

    if len(dob)==0:
        data_Dob=""

    male=False
    Female=False
    for i in gen:
        if len(i)>1:
            for j in i:
                if i.isalpha():
                    if i in ["m","M","Male","male"]:
                        male=True
                    if i in ["f","F","Female","female"]:
                        Female=True
        else:
            if i.lower() in ["m","male"]:
                male=True
            else:
                Female=True
    if len(phone)>1:
        data_Phone=phone[0]
    data_Phone=phone[0]
    
    #pii_json["Name2"].extend(names)
    pii_json["DOB"]=data_Dob
    if male:
        data_Gen="male"
    if Female:
        data_Gen="female"
    if male==False and Female==False:
        data_Gen=""
    acc_1=[]
    if len(acc)>0:
        for i in acc:
            for j in i.split():
                j=(j.lower()).strip()
                if j!="acc" or j!="account" or j!="no" or j!="no." or j!="."or j!=":":
                    if j.isnumeric():
                        acc_1.append(j)
    if len(acc_1)>=1:
        data_Mrn=acc_1[0]
    if len(acc_1)<=1:
        data_Mrn=acc_1[0]
    age_1=[]
    for i in age_no:
        for j in i.split():
            if j.isnumeric():
                age_1.append(j)
    if len(age_1)>1:
        data_age=age_1[0]
    if len(age_1)<=1:
        data_age=age_1[0]

    if len(age_1)==0:
        data_age=""
    pii_json["Patient_name"]=(data_Patient)
    pii_json["Gender"]=data_Gen
    pii_json["MRN"]=data_Mrn
    pii_json["Phone_number"]=data_Phone
    pii_json["Age"]=data_age
    return {
        #"original_text": input_data.text,
        #"redacted_text": redacted_text,
        "extracted_entities": pii_json
    }
