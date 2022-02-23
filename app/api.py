# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
import os

from dotenv import load_dotenv, find_dotenv
from .controllers.asap import calculate_features, predict_asap
from fastapi import FastAPI, Body
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
import spacy
import srsly
import uvicorn

from app.models import (
    ENT_PROP_MAP,
    GetFeaturesRequest,
    RecordsRequest,
    RecordsResponse,
    RecordsEntitiesByTypeResponse,
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

models = initialise_models('/Users/SidharrthNagappan/Documents/University/Second Year/FYP/compute_engine_models/ASAP-AES')

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

@app.post("/feature-tensor")
def get_features_tensor(text: GetFeaturesRequest):
    return predict_asap(text.text, set_num='set3')

# @app.get('/hierarchical-lstm-score')
# def hierarchical_lstm_score(text):
#     pass

# @app.get('/non-hierarchical-lstm-score')
# def non_hierarchical_lstm_score(text):
#     pass

# @app.get('/hybrid-longformer-score')
# def hybrid_longformer_score(text):
#     pass
