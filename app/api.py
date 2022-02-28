# Sidharrth Nagappan

from collections import defaultdict
from html import entities
import os

from dotenv import load_dotenv, find_dotenv
from .controllers.asap import calculate_features, predict_asap
from fastapi import FastAPI, Body
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import spacy
import srsly
import uvicorn
import pickle

from app.models import (
    ENT_PROP_MAP,
    GetFeaturesRequest,
    PredictASAPRequest,
    RecordDataRequest,
)
from .spacy_extractor import SpacyExtractor
from .controllers.pdf import parse_pdf
from .controllers.asap import initialise_models
import warnings

warnings.filterwarnings("ignore")

load_dotenv(find_dotenv())
prefix = os.getenv("CLUSTER_ROUTE_PREFIX", "").rstrip("/")

app = FastAPI(
    title="Industrial Text Scoring Engine",
    version="1.0",
    description="This is an industrial text scoring engine, serving the models written for 'Hybrid Linear Attention Transformers for Industrial Text Scoring'. It consists of document processing, neural adherence scoring and on-the-fly scoring of models.",
    openapi_prefix=prefix,
)

nlp = spacy.load("en_core_web_sm")
extractor = SpacyExtractor(nlp)

models = initialise_models('/Volumes/My Passport/University/Second Year/FYP/models/ASAP-AES')

@app.get("/", include_in_schema=False)
def docs_redirect():
    return RedirectResponse(f"{prefix}/docs")

'''
Document Processing
'''
@app.get("/parse-pdf")
def pdf_parse(path):
    return parse_pdf(path)

'''
Feature Generation
'''
@app.post("/asap-features")
def get_features(text: GetFeaturesRequest):
    return calculate_features(text.text)

@app.post("/predict-asap-aes")
def get_features_tensor(text: PredictASAPRequest):
    results = predict_asap(text.text, set_num=text.essay_set)
    with open('attentions/attentions.pickle', 'rb') as f:
        attentions = pickle.load(f)
        final = {}
        for key in attentions:
            final[key] = attentions[key].tolist()
    return results

@app.post(
    "/named-entities", tags=["NER"]
)
async def extract_entities_by_type(data: RecordDataRequest):
    """Extract Named Entities from a record.
        This route can be used directly as a Cognitive Skill in Azure Search
        For Documentation on integration with Azure Search, see here:
        https://docs.microsoft.com/en-us/azure/search/cognitive-search-custom-skill-interface"""

    entities_res = extractor.extract_entities([{"id": 0, "text": data.text}])
    res = []

    for er in entities_res:
        groupby = defaultdict(list)
        for ent in er["entities"]:
            ent_prop = ENT_PROP_MAP[ent["label"]]
            groupby[ent_prop].append(ent["name"])
        res.append(groupby)

    return res[0]


# @app.get('/hierarchical-lstm-score')
# def hierarchical_lstm_score(text):
#     pass

# @app.get('/non-hierarchical-lstm-score')
# def non_hierarchical_lstm_score(text):
#     pass

# @app.get('/hybrid-longformer-score')
# def hybrid_longformer_score(text):
#     pass
