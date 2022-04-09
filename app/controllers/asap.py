'''
Models are made of Block A, Block B and Block C.
Block A = transformer
Block B = handcrafted features
Block C = word-level-attention
'''
from tensorflow.keras.preprocessing.text import Tokenizer
from app.controllers.get_keywords import get_keywords
from app.controllers.metrics import calc_classification_metrics, get_score
from app.multimodal_transformers.data.load_data import process_single_text
from ..multimodal_transformers.model import AutoModelWithTabular
from ..dataclass.arguments import ModelArguments
from transformers.models.auto.configuration_auto import AutoConfig
from . import feature_generation
from transformers import pipeline, Trainer, TrainingArguments
import numpy as np
import os

models = {}

def initialise_models(folder):
    essay_sets = ['set3', 'set4', 'set5', 'set6']

    for essay_set in essay_sets:
        fold = os.listdir(f"{folder}/{essay_set}")[0]
        checkpoint = os.listdir(f"{folder}/{essay_set}/{fold}")[0]
        model_path = f"{folder}/{essay_set}/{fold}/{checkpoint}"
        model_args = ModelArguments(
           model_name_or_path=model_path
        )
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            # model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        )
        config.tabular_config['save_attentions']=True
        config.tabular_config['attentions_path']='./attentions/attentions.pickle'
        config.tabular_config['group'] = essay_set
        config.tabular_config['max_keyword_len'] = max([len(i.split()) for i in get_keywords(essay_set)])
        models[essay_set] = AutoModelWithTabular.from_pretrained(
            model_args.model_name_or_path,
            # model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            config=config,
        )
        print(essay_set)
        print(config.tabular_config)

    print(f'Models initialised: {essay_sets}')

    return models

def calculate_features(text):
    return {
        "num_words": feature_generation.num_words(text),
        "num_sentences": feature_generation.num_sentences(text),
        "num_lemmas": feature_generation.num_lemmas(text),
        "num_commas": feature_generation.count_commas(text),
        "num_exclamation_marks": feature_generation.count_exclamation_marks(text),
        "num_question_marks": feature_generation.count_question_marks(text),
        "average_word_length": feature_generation.average_word_length(text),
        "average_sentence_length": feature_generation.average_sentence_length(text),
        "num_nouns": feature_generation.number_of_nouns(text),
        "num_verbs": feature_generation.number_of_verbs(text),
        "num_adjectives": feature_generation.number_of_adjectives(text),
        "num_adverbs": feature_generation.number_of_adverbs(text),
        "num_conjunctions": feature_generation.number_of_conjunctions(text),
        "num_spelling_errors": feature_generation.number_of_spelling_errors(text),
        "num_stopwords": feature_generation.num_stopwords(text),
        "automated_readability_index": feature_generation.automated_readability_index(text),
        "coleman_liau_index": feature_generation.coleman_liau_index(text),
        "dale_chall_index": feature_generation.dale_chall_index(text),
        "difficult_word_count": feature_generation.difficult_word_count(text),
        "flesch_kincaid_grade": feature_generation.flesch_kincaid_grade(text),
        "gunning_fog": feature_generation.gunning_fog(text),
        "linsear_write_formula": feature_generation.linsear_write_formula(text),
        "smog_index": feature_generation.smog_index(text),
        "syllable_count": feature_generation.syllable_count(text)
    }

def predict_asap(text, set_num):
    model = models[set_num]
    features = calculate_features(text)
    data = process_single_text(text, 'asap', features, keywords=get_keywords(set_num), essay_set=set_num)
    inference_data = np.array([])

    for i in range(16):
        inference_data = np.append(inference_data, data)

    training_args = TrainingArguments(
            output_dir='.',
            num_train_epochs = 4,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=16,
            eval_accumulation_steps=2,
            evaluation_strategy = "epoch",
            save_total_limit = 1,
            disable_tqdm = False,
            load_best_model_at_end=True,
            logging_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inference_data,
        eval_dataset=inference_data,
        compute_metrics=get_score,
    )

    results = trainer.evaluate()

    results['text'] = data[0]['lemmatized_text']
    # results['keyword_tokens'] = data[0]['keyword_tokens'].tolist()
    results['keywords'] = get_keywords(set_num)

    return results