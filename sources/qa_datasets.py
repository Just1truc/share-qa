import json

class FriendsQADataset():

    def __init__(self, data_dir='./datasets/FriendsQA/dat/'):
        file_name = lambda dataset: f'friendsqa_{dataset}.json'
        self.train_dialog, self.train_qa = self.extract_data(data_dir + file_name('trn'))
        self.val_dialog, self.val_qa = self.extract_data(data_dir + file_name('dev'))
        self.test_dialog, self.test_qa = self.extract_data(data_dir + file_name('tst'))

    def extract_data(self, file_path: str, filter_out_notes: bool = True):
        """
        Args:
        file_path: path to dataset file
        filter_out_notes: whether to remove #NOTE# utterances (scene expanations, not something that was explicitly said)

        Returns:
        data: the dialogs themselves. list of dictionaries, each contains:
            - pid (paragraph id, str)
            - uid (utterance id, int)
            - speakers (whoever said the sentence, list of str in length 1).  If filter_out_notes=False, it can be ['#NOTE#']
            - utterance (what was said, str)
        qa: the questions and possible answers. list of dictionaries, each contains:
            - id (question id, str, starts with the pid)
            - question (str)
            - answers: list of dictionaries of possible answers, contains:
                * answer_text (str)
                * utterance_id (uid of the answer)
                * inner_start (int)
                * inner_end (int)
        """
        with open(file_path, 'r') as f:
            dat = json.load(f)['data']
        data = []
        qa = []
        for episode in dat:
            sentences = episode['paragraphs'][0]['utterances:'] # paragraphs is always a list of 1 element
            note_ids = set()
            for s in sentences:
                s['pid'] = episode['title']
                if filter_out_notes and '#NOTE#' in s['speakers']:
                    note_ids.add(s['uid'])
                    continue
                data.append(s)
            all_qas = episode['paragraphs'][0]['qas']
            if not filter_out_notes: # all questions are OK
                qa += all_qas
            else: # need to filter out the question that their answers depends on the notes
                for qa_grp in all_qas:
                    valid_ans = []
                    for ans in qa_grp['answers']:
                        if ans['utterance_id'] not in note_ids:
                            valid_ans.append(ans)
                    if len(valid_ans) > 0: # else the entire question is not relevant
                        qa_grp['answers'] = valid_ans # makes it remove the note-related answers
                        qa.append(qa_grp)  
        return data, qa

    def aggregate_by_episodes(self):
        """
        Aggregating the datasets by episodes
        """
        self.train_dialog = self._aggregate_by_episodes_logic(self.train_dialog, 'pid')
        self.val_dialog = self._aggregate_by_episodes_logic(self.val_dialog, 'pid')
        self.test_dialog = self._aggregate_by_episodes_logic(self.test_dialog, 'pid')
        
        self.train_qa = self._aggregate_by_episodes_logic(self.train_qa, 'id')
        self.val_qa = self._aggregate_by_episodes_logic(self.val_qa, 'id')
        self.test_qa = self._aggregate_by_episodes_logic(self.test_qa, 'id')
        
    def _aggregate_by_episodes_logic(self, dataset, id_name, sep='_'):
        """
        Internal function used by aggregate_by_episodes. 
        Args:
        dataset: a parsed dataset from init (list of dicts)
        id_name: how the episode id is called in the dictionary (pid for dialogs, id for qas) (str)
        sep: the id separator (str)

        Returns: 
        dictionary with keys = episode ids "s{season_num}_e{episode_num}",
                        values = list of dicts relevant to this episode (same dict structure as in init)
        """
        agg_data = {}
        for d in dataset:
            ep_id = d[id_name].split(sep)[:2]
            ep_id = sep.join(ep_id)
            if ep_id in agg_data:
                agg_data[ep_id].append(d)
            else:
                agg_data[ep_id] = [d]
        return agg_data
