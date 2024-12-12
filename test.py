from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from spacy.matcher import Matcher
from typing import List, Optional
import re
import json

nlp = spacy.load("en_core_web_sm")
app = FastAPI()

data_Patient = ""
data_Phone = ""
data_Dob = ""
data_Mrn = ""
data_Gen = ""
data_age = ""

pii_json = {
    "Patient_name": data_Patient,
    "Phone_number": data_Phone,
    "DOB": data_Dob,
    "MRN": data_Mrn,
    "Gender": data_Gen,
    "Age": data_age
}

# Custom SpaCy config for Gliner model
custom_spacy_config = {
    "gliner_model": "knowledgator/gliner-multitask-large-v0.5",
    "chunk_size": 250,
    "labels": ["sex", "gender", "person", "phone_number", "dob","age"],
    "style": "ent",
    "threshold": 0.3
}


nlp.add_pipe("gliner_spacy", config=custom_spacy_config)
matcher = Matcher(nlp.vocab)

class TextInput(BaseModel):
    text: str

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
matcher.add("MRN_NUMBER", [account_number_mrn,account_number_1,account_number])

@app.post("/extract_and_redact/")
async def extract_and_redact(input_data: TextInput):
    doc = nlp(input_data.text)
    matches = matcher(doc)
    extracted_entities = []
    names_gliner = []
    phone_numbers = []
    dob = []
    gen = []
    acc = []
    age_no = []
    persons = []

    for ent in doc.ents:
        if ent.label_ == "phone_number":
            phone_numbers.append(ent.text)
        if ent.label_ == "person":
            names_gliner.append(ent.text)
        if ent.label_ == "dob":
            dob.append(ent.text)
        if ent.label_ == "sex" or ent.label_ == "gender":
            gen.append(ent.text)
        if ent.label_ == "age":
            age_no.append(ent.text)
   
    mrn=[]
    for match_id, start, end in matches:
        match_id_str = nlp.vocab.strings[match_id]
        span = doc[start:end]
        if match_id_str == "ACCOUNT_NUMBER" or match_id_str == "ACCOUNT_NUMBER_1" or match_id_str == "MRN_NUMBER":
            match = re.findall(r'\d+', span.text)  
            if match:
                mrn.append(match[0]) 
    dob_pattern = r"(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4})|(\d{4}[-/]\d{1,2}[-/]\d{1,2})"
    dob_matches = re.findall(dob_pattern, input_data.text)
    
    if dob_matches:
        dob = [match[0] if match[0] else match[1] for match in dob_matches]

    if dob:
        data_Dob = dob[0]
    else:
        data_Dob = ""
    
    pii_json["Patient_name"] = names_gliner if names_gliner else ""
    pii_json["Phone_number"] = phone_numbers[0] if len(phone_numbers) > 0 else ""
    pii_json["DOB"] = data_Dob
    pii_json["Gender"] = gen[0] if len(gen) > 0 else ""
    pii_json["Age"] = age_no[0] if len(age_no) > 0 else ""
    pii_json["MRN"] = mrn[0] if len(mrn) > 0 else ""
    
    return {
        "extracted_entities": pii_json
    }