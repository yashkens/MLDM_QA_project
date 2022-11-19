import re

import pandas as pd
from nltk.corpus import stopwords

class LongAnswerDataset(Dataset):
    SAMPLE_RATE = 15
    
    def __init__(self, data, tokenizer, max_len=150, kaggle_format=True):
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._kaggle_format = kaggle_format
        
        data = self._preprocess_data(data)
        data = self._clean_df(data)
        self._questions = data.question.values
        self._long_answers = data.long_answer.values
        self._targets = data.is_long_answer.values
        
        
    def _get_nq_tokens(self, simplified_nq_example):
        if "document_text" not in simplified_nq_example:
            raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                         "example that contains the `document_text` field.")

        return simplified_nq_example["document_text"].split(" ")
    
    def _clean_token(self, token):
        return re.sub(u" ", "_", token["token"])

    def _remove_html_byte_offsets(self, span):
        if "start_byte" in span:
            del span["start_byte"]

        if "end_byte" in span:
            del span["end_byte"]

        return span

    def _clean_annotation(self, annotation):
        annotation["long_answer"] = self._remove_html_byte_offsets(
            annotation["long_answer"])
        annotation["short_answers"] = [
            self._remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
        ]
        return annotation
    
    def _simplify_nq_example(self, nq_example):
        text = " ".join([self._clean_token(t) for t in nq_example["document_tokens"]])

        simplified_nq_example = {
          "question_text": nq_example["question_text"],
          "example_id": nq_example["example_id"],
          "document_url": nq_example["document_url"],
          "document_text": text,
          "long_answer_candidates": [
              self._remove_html_byte_offsets(c)
              for c in nq_example["long_answer_candidates"]
          ],
          "annotations": [self._clean_annotation(a) for a in nq_example["annotations"]]
        }

        if len(self._get_nq_tokens(simplified_nq_example)) != len(
          nq_example["document_tokens"]):
            raise ValueError("Incorrect number of tokens.")

        return simplified_nq_example
    
    def _get_question_and_document(self, line):
        question = line['question_text']
        text = line['document_text'].split(' ')
        annotations = line['annotations'][0]

        return question, text, annotations


    def _get_long_candidate(self, i, annotations, candidate):
        if i == annotations['long_answer']['candidate_index']:
            label = 1
        else:
            label = 0

        # get place where long answer starts and ends in the document text
        long_start = candidate['start_token']
        long_end = candidate['end_token']

        return label, long_start, long_end


    def _form_data_row(self, question, label, text, long_start, long_end):
        row = {
            'question': question,
            'long_answer': ' '.join(text[long_start:long_end]),
            'is_long_answer': label,
        }

        return row


    def _preprocess_data(self, data):
        rows = []

        for line in data:
            if not self._kaggle_format:
                line = self._simplify_nq_example(line)
            question, text, annotations = self._get_question_and_document(line)
            for i, candidate in enumerate(line['long_answer_candidates']):
                label, long_start, long_end = self._get_long_candidate(i, annotations, candidate)

                if label == True or (i % self.SAMPLE_RATE == 0):
                    rows.append(
                        self._form_data_row(question, label, text, long_start, long_end)
                    )

        return pd.DataFrame(rows)
    
    def _remove_stopwords(self, sentence):
        words = sentence.split()
        words = [word for word in words if word not in stopwords.words('english')]

        return ' '.join(words)

    def _remove_html(self, sentence):
        html = re.compile(r'<.*?>')
        return html.sub(r'', sentence)

    def _clean_df_by_column(self, df, column):
        df[column] = df[column].apply(lambda x : self._remove_stopwords(x))
        df[column] = df[column].apply(lambda x : self._remove_html(x))
        return df

    def _clean_df(self, df):
        df = self._clean_df_by_column(df, 'long_answer')
        df = self._clean_df_by_column(df, 'question')
        return df
        
    
    def __getitem__(self, idx):
        input_tokens = self._questions[idx].split()
        input_tokens.append(' ' + self._tokenizer.sep_token + ' ')
        long_answer_tokens = self._long_answers[idx].split()
        input_tokens.extend(long_answer_tokens)
        encoding = self._tokenizer(input_tokens,
                          is_split_into_words=True,
                          return_offsets_mapping=True,
                          padding='max_length',
                          truncation=True,
                          max_length=self._max_len,
                          return_tensors='pt')
        encoding.pop('token_type_ids')
        encoding.pop('offset_mapping')
        encoding.pop('attention_mask')
        return encoding, self._targets[idx]
        
    
        
    def __len__(self):
        return self._targets.shape[0]
