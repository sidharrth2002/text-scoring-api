'''
Models are made of Block A, Block B and Block C.
Block A = transformer
Block B = handcrafted features
Block C = word-level-attention
'''
from multimodal_transformers.model import AutoModelWithTabular
from dataclass.arguments import ModelArguments
from transformers.models.auto.configuration_auto import AutoConfig
import controllers.feature_generation as feature_generation

models = {}

def initialise_models(folder):
    essay_sets = ['set3', 'set4', 'set5', 'set6']

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

    print(models)

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

def predict_asap(set_num):
    model = models[set_num]
    features = calculate_features(set_num)
    print(features)
