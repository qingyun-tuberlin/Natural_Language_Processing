import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from nltk.util import ngrams
import nltk
from nltk.tokenize import word_tokenize,RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
import re
from transformers import AutoTokenizer
import torch
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
print("torch version:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())



"""
After donwload "punkt", and "punkt_tab", you can use word_tokenize 
directly, without "preserve_line=True"

If you not download it, you have to use word_tokenize in this way:
a = "I love NLP, how about You?"
b = word_tokenize(a,preserve_line=True)
"""
nltk.download("punkt")
nltk.download("punkt_tab")  # optional, depends on NLTK version

nltk.download("stopwords")
nltk.download('wordnet')    
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger_eng')


# very useful links
# https://www.geeksforgeeks.org/nlp/natural-language-processing-nlp-tutorial/
# https://huggingface.co/docs/transformers/tokenizer_summary

class Preprocessor():

    def __init__(self,file_path: str, transformer_name: str = "bert-base-uncased"):

        """
        the file itself has format
        Class Index, Title, Description
        number        str      str

        parameter:
            file_path: the path of the file, the file is supposed to be a csv file
        """
        self.file_path = file_path
        self.dt_raw = pd.read_csv(file_path)
        self.transformer_name = transformer_name
        self.hf_tokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True) # hf stands for hugging face

    def tockenize(self,text:str) -> list:
        """
        Question:
            nltk.tokenize has many kinds of tokenizer, should we pass tokenizer as parameter?
        """
        return word_tokenize(text)
        
    def lowercase(self,text:str) -> str:
        return text.lower()
    

    def rm_stopword(self, tokens:list[str]):
        """
        This function removes stopwords, suppose the text's language is english.

        what is stop words? 
            Stop words typically fall into these grammatical categories:
            Articles: a, an, the
            Prepositions: in, on, at, of, for, with, about
            Conjunctions: and, but, or, so, because
            Pronouns: I, you, he, she, it, we, they, this, that, these
            Auxiliary Verbs: is, am, are, was, were, be, been, have, has, had, do, does, did
            Common Verbs/Adverbs: can, will, would, should, very, too, just, not (see caution below)
        """
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [
            word for word in tokens
            if word.lower() not in stop_words
            and word.isalpha()   # remove punctuation
        ]
        return filtered_tokens
    
    def stemming(self, tokens:list[str]):
        """
        This function reduces words to their root form, often result in non-valid words.

        Question: 
            what is difference, pros, cons between different stemmers?

        example:
            Original words: ['running', 'jumps', 'happily', 'running', 'happily']
            Stemmed words: ['run', 'jump', 'happili', 'run', 'happili']
        """
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        return stemmed_tokens
        
    # lemmatization it
    def lemmatize(self,tokens:list[str])  ->list:
        """
        This function reduces words to their base form(lemma), ensuring a valid word.

        example:
            Original Text: The cats were running faster than the dogs.
            Lemmatized Words: ['The', 'cat', 'were', 'running', 'faster', 'than', 'the', 'dog', '.']
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_words
    
    
    def handle_punctuation_and_num(self,text:str)-> list:
        """
        This function handles(removes) punctuation and numbers of the input text.

        Regex:
            \w+           it keeps words + numbers
            [A-Za-z]+     it only keeps letters

        parameters:
            text: a string, we'll handle punctuation and number on this text
        return:
            a list of words, which has excluded punctuation and numbers
        example:
            input text:     "# + , punctuation and numbers like 123."
            output result:  ['punctuation', 'and', 'numbers', 'like']
        """
        tokenizer = RegexpTokenizer(r'[A-Za-z]+')
        tokenized_text_lst = tokenizer.tokenize(text)
        return tokenized_text_lst


    def tokenize_subword_transformer(self,text:str, add_special_tokens: bool = True) -> dict:
        """
        Returns subword tokens and token ids for a Transformer tokenizer.

        add_special_tokens:
          - True: includes [CLS]/[SEP] for BERT-like models
          - False: raw subword pieces only
        """
        ids = self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        tokens = self.hf_tokenizer.convert_ids_to_tokens(ids)
        return {"tokens": tokens, "ids": ids}
    
    def classic_pipeline(
        self,
        use_title: bool = True,
        use_description: bool = True,
        join_with: str = " [SEP] ",     # 仅作为分隔符，classic里也OK
        use_regexp_tokenizer: bool = True,
        remove_stopwords: bool = True,
        do_stemming: bool = False,
        do_lemmatize: bool = True,
        make_tfidf: bool = True,
        tfidf_ngram_range: tuple = (1, 2),
        tfidf_min_df: int = 2,
        tfidf_max_df: float = 0.95,
    ):
        """
        Classical NLP pipeline:
        - Merge Title + Description into one text per row
        - Lowercase
        - Tokenize (RegexpTokenizer OR NLTK word_tokenize)
        - (optional) remove stopwords + punctuation + numbers
        - (optional) lemmatize OR stem
        - Create clean_text = " ".join(tokens)
        - (optional) TF-IDF vectorization

        Returns:
          df_out, (X_tfidf, vectorizer) if make_tfidf else df_out
        """
        df = self.dt_raw.copy()

        # ---- 1) basic cleaning + merge text per row ----
        for col in ["Title", "Description"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        parts = []
        if use_title and "Title" in df.columns:
            parts.append(df["Title"])
        if use_description and "Description" in df.columns:
            parts.append(df["Description"])

        if not parts:
            raise ValueError("No text columns selected. Check use_title/use_description and CSV headers.")

        if len(parts) == 1:
            df["text"] = parts[0].astype(str).str.strip()
        else:
            df["text"] = parts[0].astype(str).str.strip() + join_with + parts[1].astype(str).str.strip()

        # labels
        if "Class Index" not in df.columns:
            raise ValueError("Column 'Class Index' not found in CSV.")
        df["y"] = df["Class Index"].astype(int)

        # ---- 2) lowercase ----
        df["text"] = df["text"].apply(self.lowercase)

        # ---- 3) tokenize ----
        if use_regexp_tokenizer:
            # regex already removes punctuation/numbers
            df["tokens"] = df["text"].apply(self.handle_punctuation_and_num)
        else:
            # NLTK tokenization keeps punctuation; we'll filter later
            df["tokens"] = df["text"].apply(self.tockenize)

        # ---- 4) optional: remove stopwords + keep alpha ----
        if remove_stopwords:
            df["tokens"] = df["tokens"].apply(self.rm_stopword)
        else:
            # even if not removing stopwords, still remove non-alpha if you used NLTK tokenizer
            if not use_regexp_tokenizer:
                df["tokens"] = df["tokens"].apply(lambda toks: [t for t in toks if t.isalpha()])

        # ---- 5) optional: lemmatize / stem (choose one) ----
        if do_stemming and do_lemmatize:
            raise ValueError("Choose either stemming OR lemmatize, not both.")
        if do_lemmatize:
            df["tokens"] = df["tokens"].apply(self.lemmatize)
        elif do_stemming:
            df["tokens"] = df["tokens"].apply(self.stemming)

        # ---- 6) build clean_text for vectorizer ----
        df["clean_text"] = df["tokens"].apply(lambda toks: " ".join(toks))

        # ---- 7) TF-IDF vectorization (optional) ----
        if not make_tfidf:
            return df[["y", "text", "tokens", "clean_text"]]

        vectorizer = TfidfVectorizer(
            ngram_range=tfidf_ngram_range,
            min_df=tfidf_min_df,
            max_df=tfidf_max_df,
            sublinear_tf=True,
            max_features=20000,
        )
        X_tfidf = vectorizer.fit_transform(df["clean_text"])

        return df[["y", "text", "tokens", "clean_text"]], X_tfidf, vectorizer
    
    def transformer_based_pipeline(
        self,
        use_title: bool = True,
        use_description: bool = True,
        join_with: str = " [SEP] ",
        max_length: int = 128,
        padding: str = "max_length",     # "max_length" 或 True (longest)
        truncation: bool = True,
        return_tensors: str | None = None,  # None / "pt" / "tf" / "np"
    ):
        """
        Transformer-based NLP pipeline:
        - Merge Title + Description into one text per row
        - Minimal cleaning (fillna, strip)
        - Use HuggingFace tokenizer to produce:
            input_ids, attention_mask, (optional token_type_ids)
        - Return labels y and encoded inputs

        Returns:
        df_out, encoded
            df_out: columns ["y", "text"]
            encoded: dict with keys like input_ids, attention_mask, token_type_ids (depending on model)
        """
        df = self.dt_raw.copy()

        # ---- 1) basic cleaning + merge text per row ----
        for col in ["Title", "Description"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        parts = []
        if use_title and "Title" in df.columns:
            parts.append(df["Title"].astype(str).str.strip())
        if use_description and "Description" in df.columns:
            parts.append(df["Description"].astype(str).str.strip())

        if not parts:
            raise ValueError("No text columns selected. Check use_title/use_description and CSV headers.")

        if len(parts) == 1:
            df["text"] = parts[0]
        else:
            df["text"] = parts[0] + join_with + parts[1]

        # ---- 2) labels ----
        if "Class Index" not in df.columns:
            raise ValueError("Column 'Class Index' not found in CSV.")
        df["y"] = df["Class Index"].astype(int)

        # ---- 3) tokenizer encode (batch) ----
        texts = df["text"].tolist()

        encoded = self.hf_tokenizer(
            texts,
            add_special_tokens=True,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
        )

        # df_out contains minimal columns; encoded carries model inputs
        df_out = df[["y", "text"]].copy()
        return df_out, encoded




# test & benchmark classic preprocessor
p = Preprocessor("train.csv")

df_out, X, vec = p.classic_pipeline(
    use_regexp_tokenizer=True,     
    remove_stopwords=True,
    do_lemmatize=True,
    do_stemming=False,
    make_tfidf=True,
    tfidf_ngram_range=(1,2)
)

print(df_out.head())
print(X.shape)
print(vec.get_feature_names_out()[:20])



df_out, X, vec = p.classic_pipeline()
y = df_out["y"].values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)
print("acc:", clf.score(X_test, y_test))



# test & benchmark transformer-based pipeline
p = Preprocessor("train.csv")

df_out, enc = p.transformer_based_pipeline(
    max_length=64,
    return_tensors=None
)

print(df_out.head())
print(enc.keys())                # input_ids, attention_mask, ...
print(len(enc["input_ids"]))     # sample size
print(enc["input_ids"][0][:20])  # the first sample's token ids