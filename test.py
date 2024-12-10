from fastapi import FastAPI, Query
from pydantic import BaseModel
import spacy
from spacy.matcher import Matcher
from typing import List, Optional
import re
import json
nlp = spacy.load("en_core_web_sm")
app = FastAPI()
pii_json={
    "Name":[],
    "Name1":[],
    "Name2":[],
    "Phone number":[],
    "DOB":[],
    "MRN":[],
    "Gender":[],
    "Age":[]
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
    {"IS_TITLE": True, "OP": "?"} 
]

age_patterns = [
    [{"LOWER": "age"}, {"TEXT": ":"}, {"IS_DIGIT": True}, {"LOWER": {"IN": ["y", "years"]}},],
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
    redacted_text = input_data.text 
    redacted_text = re.sub(dob_pattern, "[DOB: REDACTED]", input_data.text)
    #redacted_text = re.sub(name_pattern, "[REDACTED_name]", input_data.text)
    redact_all = len(input_data.redact_entities) == 0
    dob= re.findall(dob_pattern, input_data.text)
    gen=[]
    acc=[]
    age_no=[]
    for match_id, start, end in matches:
        match_id_str = nlp.vocab.strings[match_id]  
        span = doc[start:end]  
        if redact_all or match_id_str in input_data.redact_entities:
            operator = operators.get(match_id_str)
            if operator and operator.action == "replace":
                new_value = operator.parameters["new_value"]
                
                redacted_text = redacted_text.replace(span.text, new_value)
        
        extracted_entities.append({"entity": match_id_str, "text": span.text})
        if match_id_str=="gender":
            gen.append(span.text)
        if match_id_str=="ACCOUNT_NUMBER":
            acc.append(span.text)
        if match_id_str=="AGE":
            age_no.append(span.text)
    
   person=[]
    for i in extracted_entities:
        p=""
        if i['entity']=="PERSON":
            p+=(i['text']).strip()
        person.append(p)
        p=""
        
    redacted_text = (redacted_text.lower()).replace((person.lower()).strip(), "[PERSON]")
    
    extracted_entities.append({"entity": "PERSON_1", "text": person})
    from spacy import displacy
    fol=nlp(redacted_text)
    displacy.render(fol, style="span")
    names=[]
    phone=[]
    
    for ent in fol.ents:
        if ent.label_=="phone_number":
            redacted_text = (redacted_text.lower()).replace((ent.text).lower(), "[phone]")
            phone.append(ent.text)
        if ent.label_=="person":
            names.append((ent.text).lower())
    extracted_entities.append({"entity": "PERSON_2", "text": names})
    extracted_entities.append({"entity": "Phone_numbers", "text": phone})
    #print(names)
    #print(redacted_text.split())
    full_name=[]
    for i in redacted_text.split():
        if i.lower() in names[0].split():
             redacted_text = (redacted_text.lower()).replace((i).lower(), "[person]")
    extracted_entities.append({"entity": "PERSON_3", "text": full_name})
    pii_json["Name"].extend(full_name)
    pii_json["Name1"].extend(person)
    pii_json["Name2"].extend(names)
    pii_json["DOB"].extend(dob)
    pii_json["Gender"].extend(gen)
    pii_json["MRN"].extend(acc)
    pii_json["Phone number"].extend(phone)
    pii_json["Age"].extend(age_no)
    return {
        #"original_text": input_data.text,
        #"redacted_text": redacted_text,
        "extracted_entities": pii_json
    }
