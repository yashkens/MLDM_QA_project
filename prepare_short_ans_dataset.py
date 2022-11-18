import re
import torch
import random
from torch.utils.data import Dataset, DataLoader


# this func is taken from official NaturalQuestions repo
def get_nq_tokens(simplified_nq_example):

    if "document_text" not in simplified_nq_example:
        raise ValueError("`get_nq_tokens` should be called on a simplified NQ"
                         "example that contains the `document_text` field.")

    return simplified_nq_example["document_text"].split(" ")


# this func is taken from official NaturalQuestions repo too
def simplify_nq_example(nq_example):

    def _clean_token(token):
        return re.sub(u" ", "_", token["token"])

    text = " ".join([_clean_token(t) for t in nq_example["document_tokens"]])

    def _remove_html_byte_offsets(span):
        if "start_byte" in span:
            del span["start_byte"]

        if "end_byte" in span:
            del span["end_byte"]

        return span

    def _clean_annotation(annotation):
        annotation["long_answer"] = _remove_html_byte_offsets(
            annotation["long_answer"])
        annotation["short_answers"] = [
            _remove_html_byte_offsets(sa) for sa in annotation["short_answers"]
        ]
        return annotation

    simplified_nq_example = {
      "question_text": nq_example["question_text"],
      "example_id": nq_example["example_id"],
      "document_url": nq_example["document_url"],
      "document_text": text,
      "long_answer_candidates": [
          _remove_html_byte_offsets(c)
          for c in nq_example["long_answer_candidates"]
      ],
      "annotations": [_clean_annotation(a) for a in nq_example["annotations"]]
    }

    if len(get_nq_tokens(simplified_nq_example)) != len(
      nq_example["document_tokens"]):
        raise ValueError("Incorrect number of tokens.")

    return simplified_nq_example


class ShortAnswerDataset(Dataset):
    def __init__(self, data, simplify_fn, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

        bio_tags = ['B', 'I', 'O']
        self.labels_to_ids = {k: v for v, k in enumerate(bio_tags)}

        qs, l_ans, l_infos, s_infos = self.preprocess_data(data, simplify_fn)
        self.questions = qs
        self.long_answers = l_ans  # this is a list of lists (tokenized texts)
        self.long_answer_infos = l_infos
        self.short_answer_infos = s_infos

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        input_tokens = self.questions[idx].split()
        input_tokens.append(self.tokenizer.sep_token)
        question_len = len(input_tokens)
        input_tokens.extend(self.long_answers[idx])
        encoding = self.tokenizer(input_tokens,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len,
                                  return_tensors='pt')

        token_mapping = self.get_tokenization_mapping(encoding['offset_mapping'])
        info = self.short_answer_infos[idx]
        bio_tagging = self.create_bio_tagging(info[0]['start_token'] + question_len,
                                              info[0]['end_token'] + question_len,
                                              self.long_answer_infos[idx]['start_token'],
                                              token_mapping)

        return encoding, bio_tagging, self.long_answer_infos[idx]['start_token'], question_len

    def get_tokenization_mapping(self, offset_mapping):
        d = []
        for i, pair in enumerate(offset_mapping[0]):
            if pair[0] == 0 and pair[1] == 0:
                continue
            if pair[0] == 0:
                d.append([i])
            else:
                d[-1].append(i)
        return d

    def preprocess_data(self, data, simplify_fn):
        questions, long_answers, long_infos, short_infos = [], [], [], []
        for ex in data:
            simple_ex = simplify_fn(ex)
            tokens = simple_ex['document_text'].split()

            long_answer_info = simple_ex['annotations'][0]['long_answer']
            if long_answer_info['candidate_index'] == -1:
                continue

            long_answer = tokens[long_answer_info['start_token']:long_answer_info['end_token']]
            short_answer_info = simple_ex['annotations'][0]['short_answers']
            if not short_answer_info:
                continue

            # TODO: make sure short answer is not cut out after Bert tokenizer
            if len(long_answer) > self.max_len:
                start = short_answer_info[0]['start_token'] - long_answer_info['start_token']
                end = short_answer_info[0]['end_token'] - long_answer_info['start_token']
                left_margin = random.randint(0, min(self.max_len - (end - start), start))
                right_margin = self.max_len - left_margin - (end - start)
                long_answer = long_answer[start - left_margin:min(end + right_margin, len(long_answer) - 1)]
                long_answer_info['start_token'] += start - left_margin

            questions.append(simple_ex['question_text'])
            long_answers.append(long_answer)
            long_infos.append(long_answer_info)
            short_infos.append(short_answer_info)
        return questions, long_answers, long_infos, short_infos

    def create_bio_tagging(self, short_start, short_end, long_start, token_mapping):
        # these are indices in normal tokenization
        start = short_start - long_start
        end = short_end - long_start

        aligned_tokens_nested = token_mapping[start:end]  # +1 here?
        aligned_tokens = [item for sublist in aligned_tokens_nested for item in sublist]

        # these are indices in bert tokenization
        start = aligned_tokens[0]
        end = aligned_tokens[-1] + 1

        tags = torch.tensor([self.labels_to_ids['O']] * self.max_len)
        tags[start] = self.labels_to_ids['B']
        tags[start + 1:end] = self.labels_to_ids['I']
        return tags
