import re
import torch
from tqdm import tqdm
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
    def __init__(self, data, simplify_fn, tokenizer, max_len, should_simplify):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.simplify = should_simplify
        self.simplify_fn = simplify_fn
        qs, l_ans, s_infos = self.preprocess_data(data)

        self.questions = qs
        self.long_answers = l_ans  # this is a list of lists (tokenized texts)
        self.short_answer_infos = s_infos

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):  # This is quite slow...

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
        if info['start_token'] == -1:
            bio_tagging = torch.tensor([0] * self.max_len)
        else:
            bio_tagging = self.create_bio_tagging(info['start_token'] + question_len,
                                                  info['end_token'] + question_len,
                                                  token_mapping)
        return encoding, bio_tagging, question_len

    def get_tokenization_mapping(self, offset_mapping):
        d = []
        for i, pair in enumerate(offset_mapping[0]):
            if pair[0] == 0 and pair[1] == 0:
                continue
            if pair[0] == 0:
                d.append([i])
            else:
                if len(d) == 0:
                    d.append([i])
                else:
                    d[-1].append(i)
        return d

    def process_example(self, ex):
        simple_ex = ex
        if self.simplify:
            simple_ex = self.simplify_fn(ex)
        simple_ex['document_text'] = simple_ex['document_text'].replace('\ufeff', ' - ')
        tokens = simple_ex['document_text'].split()

        long_answer_info = simple_ex['annotations'][0]['long_answer']
        if long_answer_info['candidate_index'] == -1:
            return [], [], [], [], []

        long_answer = tokens[long_answer_info['start_token']:long_answer_info['end_token']]
        short_answer_info = simple_ex['annotations'][0]['short_answers']
        if not short_answer_info:
            return [], [], [], [], []
        else:
            start = short_answer_info[0]['start_token'] - long_answer_info['start_token']
            end = short_answer_info[0]['end_token'] - long_answer_info['start_token']

        orig_short_infos, orig_long_infos = [], []
        texts, start_inds, end_inds = [], [], []

        # these are used only for test
        orig_short_infos.append({'start_token': short_answer_info[0]['start_token'],
                                 'end_token': short_answer_info[0]['end_token']})
        orig_long_infos.append(long_answer_info)

        i = 0
        max_len = self.max_len - 150
        while max_len * i < len(long_answer):
            curr_text = long_answer[max_len * i: max_len * (i + 1)]

            if start == -1:
                start_inds.append(-1)
                end_inds.append(-1)
                texts.append(curr_text)
                i += 1
                continue

            new_start = start - max_len * i
            new_end = end - max_len * i

            if max_len * i < start < max_len * (i + 1) and max_len * i < end < max_len * (i + 1):
                start_inds.append(new_start)
                end_inds.append(new_end)
                texts.append(curr_text)
            elif max_len * i < start < max_len * (i + 1) or max_len * i < end < max_len * (i + 1):
                i += 1  # если short answer бьется на несколько кусков, пропускаем
            else:
                start_inds.append(-1)
                end_inds.append(-1)
                texts.append(curr_text)

            i += 1

        if long_answer[max_len * (i + 1):]:
            new_start = start - max_len * i
            new_end = end - max_len * i

            texts.append(long_answer[max_len * (i + 1):])
            start_inds.append(new_start)
            end_inds.append(new_end)

        questions = [simple_ex['question_text']] * len(texts)
        short_infos = []
        for i in range(len(start_inds)):
            short_infos.append({'start_token': start_inds[i], 'end_token': end_inds[i]})
        return texts, short_infos, questions, orig_short_infos, orig_long_infos

    def preprocess_data(self, data):
        questions, long_answers, short_infos = [], [], []
        for ex in tqdm(data):
            texts, shorts, qs, _, _ = self.process_example(ex)
            if not texts:
                continue
            long_answers.extend(texts)
            short_infos.extend(shorts)
            questions.extend(qs)

        return questions, long_answers, short_infos


    def create_bio_tagging(self, short_start, short_end, token_mapping):
        # these are indices in normal tokenization
        start = short_start
        end = short_end + 1

        aligned_tokens_nested = token_mapping[start:end]
        aligned_tokens = [item for sublist in aligned_tokens_nested for item in sublist]

        # these are indices in bert tokenization
        if aligned_tokens:
            start = aligned_tokens[0]
            end = aligned_tokens[-1]
        else:
            # это для случаев, когда ответ не влез из-за очень дробной токенизации
            # такое случается, если много слов на других языках или странных чисел
            return torch.tensor([0] * self.max_len)

        tags = torch.tensor([0] * self.max_len)
        tags[start:end] = 1
        return tags


class TestShortAnswerDataset(ShortAnswerDataset):
    def __init__(self, data, simplify_fn, tokenizer, max_len, should_simplify):
        # TODO: надо переделать классы, чтобы preprocess data не делалось лишний раз
        # super().__init__(data, simplify_fn, tokenizer, max_len, should_simplify)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.simplify = should_simplify
        self.simplify_fn = simplify_fn
        qs, l_ans, s_infos, orig_s_infos, orig_l_infos = self.preprocess_data(data)

        self.questions = qs
        self.long_answers = l_ans  # this is a list of lists (tokenized texts)
        self.short_answer_infos = s_infos
        self.orig_short_infos = orig_s_infos
        self.orig_long_infos = orig_l_infos

    def preprocess_data(self, data):
        questions, long_answers, short_infos = [], [], []
        orig_short_infos, orig_long_infos = [], []
        for ex in tqdm(data):
            texts, shorts, qs, orig_shorts, orig_longs = self.process_example(ex)
            if not texts:
                continue
            long_answers.append(texts)
            short_infos.append(shorts)
            questions.append(qs)
            orig_short_infos.append(orig_shorts)
            orig_long_infos.append(orig_longs)

        return questions, long_answers, short_infos, orig_short_infos, orig_long_infos

    def __getitem__(self, idx):

        batch_questions = self.questions[idx]
        batch_long_answers = self.long_answers[idx]
        encodings, question_len = [], []

        for i in range(len(batch_questions)):
            input_tokens = batch_questions[i].split()
            input_tokens.append(self.tokenizer.sep_token)
            question_len = len(input_tokens)
            input_tokens.extend(batch_long_answers[i])
            encoding = self.tokenizer(input_tokens,
                                      is_split_into_words=True,
                                      return_offsets_mapping=True,
                                      padding='max_length',
                                      truncation=True,
                                      max_length=self.max_len,
                                      return_tensors='pt')
            encodings.append(encoding)

        return encodings, question_len, self.orig_short_infos[idx], self.orig_long_infos[idx]
