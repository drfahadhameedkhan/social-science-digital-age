"""
nlp_pipeline.py — Chapter 16
================================
Natural Language Processing pipeline for social science research.

Covers the full NLP workflow described in Chapter 16:
  - Text preprocessing (tokenisation, lemmatisation, stopword removal)
  - Sentiment analysis (VADER + transformer-based)
  - Topic modelling with LDA (pyLDAvis interactive output)
  - BERT fine-tuning for text classification
  - Validity assessment for automated text analysis

Author : Fahad Hameed Khan
Book   : Social Science in the Digital Age (2025)
Chapter: 16 — Natural Language Processing: Sentiment, Topics & Discourse
"""

import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    End-to-end NLP pipeline for social science text analysis.

    Implements Chapter 16's full workflow from raw text corpora
    through preprocessing, sentiment analysis, topic modelling,
    and transformer-based classification.

    Parameters
    ----------
    language : str — ISO 639-1 language code (default 'en')
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self._spacy_model = None
        self._vader = None

    # ── Loading & preprocessing ───────────────────────────────────────────────

    def load_from_csv(self, path: str, text_col: str = "text",
                      meta_cols: list = None) -> pd.DataFrame:
        """Load text corpus from CSV."""
        df = pd.read_csv(path)
        cols = [text_col] + (meta_cols or [])
        df = df[[c for c in cols if c in df.columns]]
        logger.info(f"Loaded {len(df):,} documents from {path}")
        return df

    def preprocess(self, corpus: pd.DataFrame, text_col: str = "text",
                   steps: list = None) -> pd.DataFrame:
        """
        Apply NLP preprocessing pipeline.

        Available steps (Chapter 16, Section 16.1):
          'lowercase'      — convert to lowercase
          'remove_punct'   — strip punctuation and special characters
          'remove_urls'    — remove hyperlinks
          'remove_numbers' — strip numerals
          'lemmatise'      — spaCy lemmatisation
          'remove_stops'   — remove stopwords
          'remove_short'   — drop tokens < 3 chars

        Parameters
        ----------
        corpus   : pd.DataFrame with text column
        text_col : str — column containing raw text
        steps    : list — ordered preprocessing steps to apply

        Returns
        -------
        pd.DataFrame with added 'text_clean' and 'tokens' columns
        """
        if steps is None:
            steps = ["lowercase", "remove_urls", "remove_punct", "lemmatise",
                     "remove_stops", "remove_short"]

        texts = corpus[text_col].astype(str).tolist()

        if "lowercase" in steps:
            texts = [t.lower() for t in texts]
        if "remove_urls" in steps:
            texts = [re.sub(r"https?://\S+|www\.\S+", " ", t) for t in texts]
        if "remove_punct" in steps:
            texts = [re.sub(r"[^a-zA-Z\s]", " ", t) for t in texts]
        if "remove_numbers" in steps:
            texts = [re.sub(r"\d+", " ", t) for t in texts]

        if "lemmatise" in steps or "remove_stops" in steps:
            try:
                import spacy
                if self._spacy_model is None:
                    self._spacy_model = spacy.load(f"{self.language}_core_web_sm")
                processed = []
                for doc in self._spacy_model.pipe(texts, batch_size=256, n_process=1):
                    tokens = [
                        token.lemma_ for token in doc
                        # BUG FIX: Changed 'or' to 'and' for correct stopword filtering logic
                        # Now: keep token if (NOT stopword) OR (remove_stops NOT in steps)
                        # This means: skip stopwords only when "remove_stops" is in steps
                        if (not token.is_stop or "remove_stops" not in steps)
                        and token.is_alpha
                        and (len(token.text) >= 3 or "remove_short" not in steps)
                    ]
                    processed.append(tokens)
                corpus = corpus.copy()
                corpus["tokens"]     = processed
                corpus["text_clean"] = [" ".join(t) for t in processed]
            except (ImportError, OSError):
                logger.warning("spaCy not installed or model missing. "
                               "Run: pip install spacy && python -m spacy download en_core_web_sm")
                corpus = corpus.copy()
                corpus["text_clean"] = texts
                corpus["tokens"]     = [t.split() for t in texts]
        else:
            corpus = corpus.copy()
            corpus["text_clean"] = texts
            corpus["tokens"]     = [t.split() for t in texts]

        logger.info(f"Preprocessing complete. Avg tokens: "
                    f"{corpus['tokens'].apply(len).mean():.1f}")
        return corpus

    # ── Sentiment analysis ────────────────────────────────────────────────────

    def sentiment_analysis(self, corpus: pd.DataFrame,
                           text_col: str = "text_clean",
                           methods: list = None) -> pd.DataFrame:
        """
        Sentiment analysis using VADER and/or RoBERTa.

        Methods (Chapter 16, Section 16.2):
          'vader'   — VADER (rule-based, fast, good for social media)
          'roberta' — cardiffnlp/twitter-roberta-base-sentiment (transformer)

        Parameters
        ----------
        corpus   : pd.DataFrame with text
        text_col : str — column with (cleaned) text
        methods  : list — methods to apply

        Returns
        -------
        pd.DataFrame with added sentiment columns
        """
        if methods is None:
            methods = ["vader"]

        corpus = corpus.copy()

        if "vader" in methods:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                if self._vader is None:
                    self._vader = SentimentIntensityAnalyzer()
                scores = corpus[text_col].apply(
                    lambda t: self._vader.polarity_scores(str(t))
                )
                corpus["vader_pos"]      = scores.apply(lambda s: s["pos"])
                corpus["vader_neg"]      = scores.apply(lambda s: s["neg"])
                corpus["vader_compound"] = scores.apply(lambda s: s["compound"])
                corpus["vader_label"]    = corpus["vader_compound"].apply(
                    lambda c: "positive" if c >= 0.05
                    else "negative" if c <= -0.05 else "neutral"
                )
                logger.info("VADER sentiment analysis complete.")
            except ImportError:
                logger.warning("Install vaderSentiment: pip install vaderSentiment")

        if "roberta" in methods:
            try:
                from transformers import pipeline
                pipe = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment",
                    truncation=True, max_length=512
                )
                results = pipe(corpus[text_col].tolist(), batch_size=32)
                corpus["roberta_label"] = [r["label"] for r in results]
                corpus["roberta_score"] = [round(r["score"], 4) for r in results]
                logger.info("RoBERTa sentiment analysis complete.")
            except ImportError:
                logger.warning("Install transformers: pip install transformers")

        return corpus

    # ── Topic modelling ───────────────────────────────────────────────────────

    def lda_topic_model(self, corpus: pd.DataFrame,
                        n_topics: int = 10,
                        text_col: str = "tokens",
                        visualise: bool = True,
                        min_freq: int = 5,
                        max_df: float = 0.95) -> dict:
        """
        Latent Dirichlet Allocation topic modelling with pyLDAvis.

        Implements the LDA workflow from Chapter 16, Section 16.3.
        Outputs topic-word distributions, document-topic assignments,
        coherence score, and an interactive pyLDAvis visualisation.

        Parameters
        ----------
        corpus    : pd.DataFrame with tokenised text
        n_topics  : int — number of topics (hyperparameter)
        text_col  : str — column with tokenised text (list of strings)
        visualise : bool — display interactive pyLDAvis (Jupyter)
        min_freq  : int — minimum document frequency for vocabulary
        max_df    : float — maximum document frequency (filter common words)

        Returns
        -------
        dict with lda_model, dictionary, corpus, coherence_score, top_words
        """
        try:
            from gensim import corpora, models
            from gensim.models import CoherenceModel

            # Build dictionary and bag-of-words corpus
            tokens = corpus[text_col].tolist()
            dictionary = corpora.Dictionary(tokens)
            dictionary.filter_extremes(no_below=min_freq, no_above=max_df)
            bow_corpus = [dictionary.doc2bow(doc) for doc in tokens]

            # Train LDA
            logger.info(f"Training LDA: {n_topics} topics on {len(bow_corpus):,} documents…")
            lda_model = models.LdaModel(
                corpus=bow_corpus, id2word=dictionary,
                num_topics=n_topics, random_state=42,
                update_every=1, chunksize=100,
                passes=10, alpha="auto", per_word_topics=True,
            )

            # Coherence score (c_v)
            coherence_model = CoherenceModel(
                model=lda_model, texts=tokens,
                dictionary=dictionary, coherence="c_v"
            )
            coherence = round(coherence_model.get_coherence(), 4)
            logger.info(f"LDA coherence (c_v) = {coherence}")

            # Top words per topic
            top_words = {}
            for topic_id in range(n_topics):
                words = lda_model.show_topic(topic_id, topn=10)
                top_words[f"Topic {topic_id}"] = [w for w, _ in words]

            # pyLDAvis interactive visualisation
            if visualise:
                try:
                    import pyLDAvis.gensim_models as gensimvis
                    import pyLDAvis
                    vis = gensimvis.prepare(lda_model, bow_corpus, dictionary)
                    pyLDAvis.display(vis)
                    logger.info("pyLDAvis visualisation displayed.")
                except ImportError:
                    logger.warning("Install pyLDAvis: pip install pyldavis")

            return {
                "lda_model":     lda_model,
                "dictionary":    dictionary,
                "bow_corpus":    bow_corpus,
                "coherence_cv":  coherence,
                "n_topics":      n_topics,
                "top_words":     top_words,
            }

        except ImportError:
            logger.error("Install gensim: pip install gensim")
            return {}

    # ── BERT fine-tuning ──────────────────────────────────────────────────────

    def fine_tune_bert(self, train_data: pd.DataFrame,
                       n_classes: int = 3,
                       text_col: str = "text",
                       label_col: str = "label",
                       model_name: str = "bert-base-uncased",
                       epochs: int = 3,
                       batch_size: int = 16,
                       output_dir: str = "models/bert_finetuned") -> object:
        """
        Fine-tune a BERT-family model for text classification.

        Implements the transformer fine-tuning workflow from
        Chapter 16, Section 16.4, using HuggingFace Transformers.

        Parameters
        ----------
        train_data  : pd.DataFrame with text and labels
        n_classes   : int — number of classification classes
        text_col    : str — text column name
        label_col   : str — label column name
        model_name  : str — HuggingFace model identifier
        epochs      : int — training epochs (3 is standard for fine-tuning)
        batch_size  : int
        output_dir  : str — directory to save the fine-tuned model

        Returns
        -------
        Trained HuggingFace Trainer object
        """
        try:
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                TrainingArguments, Trainer
            )
            from torch.utils.data import Dataset
            import torch

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=n_classes
            )

            # Tokenise the training data
            encodings = tokenizer(
                train_data[text_col].tolist(),
                truncation=True, padding=True, max_length=512
            )

            class SimpleDataset(Dataset):
                def __init__(self, encodings, labels):
                    self.encodings = encodings
                    self.labels    = labels
                def __len__(self):
                    return len(self.labels)
                def __getitem__(self, idx):
                    item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                    item["labels"] = torch.tensor(self.labels[idx])
                    return item

            dataset = SimpleDataset(encodings, train_data[label_col].tolist())
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                save_strategy="epoch",
                logging_steps=50,
            )
            trainer = Trainer(model=model, args=training_args,
                              train_dataset=dataset)
            trainer.train()
            logger.info(f"BERT fine-tuning complete. Model saved to {output_dir}")
            return trainer

        except ImportError:
            logger.error("Install: pip install transformers torch")
            return None
