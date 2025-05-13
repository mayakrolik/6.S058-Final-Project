### 6.S058-Final-Project

# By Maya Krolik and Claire Chen

Setup Instructions:

```console
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

To run the project, please create the following directores and files.
* raw_videos/
* intermediate_videos/
* processed_videos/
* demographics.csv (include the following headers face_name_align,race,race4,gender,age,race_scores_fair,race_scores_fair_4,gender_scores_fair,age_scores_fair)

Please also create an OpenAI API key and include it n llm_prompting.py

To run the project, simply run main.py

Note that this project has been created to run on Apple Sillicone, but the conversion to windows support is possible via modifying a few files (including models/predict.py and cropper.py)

Our curated dataset is publicly avalible at https://www.kaggle.com/datasets/lamayonesa/vsr-automatic-dataset

Please check out our final project report at [6_S058_Lip_Reading_Dataset.pdf](https://github.com/mayakrolik/6.S058-Final-Project/blob/main/6_S058_Lip_Reading_Dataset.pdf)!