'''
Models are made of Block A, Block B and Block C.
Block A = transformer
Block B = handcrafted features
Block C = word-level-attention
'''
from tensorflow.keras.preprocessing.text import Tokenizer
from app.controllers.metrics import calc_classification_metrics
from app.multimodal_transformers.data.load_data import process_single_text
from ..multimodal_transformers.model import AutoModelWithTabular
from ..dataclass.arguments import ModelArguments
from transformers.models.auto.configuration_auto import AutoConfig
from . import feature_generation
from transformers import pipeline, Trainer, TrainingArguments
import numpy as np

models = {}

def initialise_models(folder):
    essay_sets = ['set3']
    # essay_sets = ['set3', 'set4', 'set5', 'set6']

    for essay_set in essay_sets:
        model_args = ModelArguments(
           model_name_or_path=f'{folder}/{essay_set}'
        )
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        )
        models[essay_set] = AutoModelWithTabular.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            config=config,
        )
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
    keyword_list = ['trying to keep my balance in my dehydrated state',
        'no one in sight, not a building, car, or structure of any kind',
        'flat road was replaced by short rolling hills',
        'hitting my water bottles pretty regularly',
        'somewhere in the neighborhood of two hundred degrees',
        'sun was beginning to beat down',
        'drop from heatstroke on a gorgeous day',
        'water bottles contained only a few tantalizing sips',
        'high deserts of California',
        'enjoyed the serenity of an early-summer evening',
        'traveling through the high deserts of California in June',
        'ROUGH ROAD AHEAD: DO NOT EXCEED POSTED SPEED LIMIT',
        'long, crippling hill',
        'tarlike substance followed by brackish water',
        'thriving little spot at one time',
        'fit the traditional definition of a ghost town',
        'Wide rings of dried sweat circled my shirt',
        'growing realization that I could drop from heatstroke on a gorgous day in June',
        'water bottle contained only a few tantalizing sips',
        'wide rings of dried sweat circled my shirt',
        'checked my water supply',
        'brackish water faling somewhere in the neighborhood of two hundred degrees',
        'birds would pick me clean']
    data = process_single_text(text, 'asap', features, keywords=keyword_list)
    inference_data = np.array([])

    for i in range(32):
        inference_data = np.append(inference_data, data)

    training_args = TrainingArguments(
            output_dir='.',
            num_train_epochs = 4,
            per_device_train_batch_size=32,
            # gradient_accumulation_steps=16,
            per_device_eval_batch_size=32,
            # eval_accumulation_steps=4,
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
        compute_metrics=calc_classification_metrics,
    )

    trainer.evaluate()