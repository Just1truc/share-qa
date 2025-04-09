# Resources

## Datasets

### List

List of datasets for the SHARE model training:

- [SODA](https://huggingface.co/datasets/allenai/soda): Has Dialog with speakers segmentation. Not cut into edus
- [MediaSUM](https://github.com/zcgzcgzcg1/MediaSum): Has dialog with speaker seg cut into edu. No qa
- [AirDialog](https://huggingface.co/datasets/google/air_dialogue): Has dialog with bad speaker seg (roles). Not cut in edu. No QA
- [MultiWOZ](https://github.com/budzianowski/multiwoz/tree/master/data/MultiWOZ_2.2) Has dialog with bad speaker seg. Kinda cut in edu. No QA
- [ICSI/AMI](https://github.com/salesforce/DialogStudio/blob/main/dialogue-summarization/ICSI/converted_examples.json) Has bad speaker segmentations (no names only letters). Cut in EDU, No QA. Meeting corpuses
- [QMSum](https://github.com/Yale-LILY/QMSum): Has dialog with speaker segmentation. The edu cut is rudimentary. There is QA with spans (can be adapter easily)
- [MeetingQA](https://github.com/Yale-LILY/QMSum): Extracted from AMI. It adds questions with answer start span. The answer doesn't look that good and the noisy data could lose the model during training.
- [FriendsQA](https://github.com/emorynlp/FriendsQA): Has everything. Its so good whalla
- [QAConv](https://github.com/salesforce/QAConv): QA on conversations. Not cut in edus sadly
- [MMDialog](https://arxiv.org/pdf/2211.05719): MultiModal dialog convs. No QA, just text and images
- [DailyDialog](http://yanran.li/dailydialog): No QA. Bad Speaker Segmentation. Can be cut in edu
- [SamSUM](https://www.kaggle.com/datasets/nileshmalode1/samsum-dataset-text-summarization): No QA. Good speaker seg. Kinda cut in edu.

### Dataset features

|Dataset|Has QA?|Has Speaker Segmentation?|Cut in EDUs|Specialized|Usability|
|---|---|---|---|---|---|
|SODA|False|True|True|False|2/3|
|MediaSum|False|True|True|False|2/3|
|AirDialog|False|True|False|True|1/3|
|MultiWOZ|False|True|True|True|Necessary|
|ICSI/AMI|False|True|True|True|2/3|
|QMSum|True|True|True|True|3/3|
|MeetingQA|True|True|True|True|1/3|
|FriendsQA|True|True|True|True|3/3|
|QAConv|True|True|False|True|3/3|
|MMDialog|False|False|False|True|0/3|
|DailyDialog|False|True|True|True|1/3|
|SamSUM|False|True|True|False|2/3|

## Final dataset format | SHARE Training Format (stf)

Here's are the columns that should constitue the dataset and their meaning:

- **Dialog (EDUs)**: The full dialog in EDUs.
- **Speakers**: A list of speakers for each EDU
- **Positive Pairs**: List of positive pairs used in the cluster contrastive loss.
- **QA**: A list of question-answer on the dialog where the answer is the index of the EDU containing the answer
- **ΔState**: The state update for each timestep of the dialog for each speaker embedding to predict
- **Losses**: A list of all the losses the model need to attend to on this element. The avaliable ones are "mlm", "qa", "state" and "cluster". Most of the time, the cluster and mlm need to be calculated.

Here's an example:

|Dialog (EDUs)|Speakers|Positive Pairs|QA|ΔState|Losses|
|-|-|-|-|-|-|
|["Hey", "Hey wassup?", "I'm good and you?", "Good too!"]|["Rodriguez", "Marcus", "Rodriguez", "Marcus"]|[[0,1][1,0],[2,3],[3,2]]|[{"question": "How is marcus?", answer: 3}]|[None, None, None, None]|["mlm", "qa", "cluster"]