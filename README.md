# CS 588 - Project - Group 4
## JINSIGHT (*J*avadoc *IN*consistency *S*canning and *I*dentification with *G*ranular *H*ighlighting *T*ool)

### Installing Conda Environment
```
conda env create -f environment.yml
conda activate javadoc_inconsistency
```
### Downloading Dataset & Trained Models
The dataset and trained models can be found in these links:
- [Dataset](https://drive.google.com/file/d/1TQBeCJCUmcYCDZLakzwZH_d4gJbfubZd/view?usp=sharing)

```
data_processed/
      ├── Summary/
      │   ├── train/
      │   │   └── data.json
      │   ├── valid/
      │   │   └── data.json
      │   └── test/
      │       └── data.json
      ├── Param/
      │   └── (similar structure as Summary)
      └── Return/
          └── (similar structure as Summary)
```

- [Trained Models](https://drive.google.com/file/d/1ASmif1o377391Ua7uMJ91AXZj8ICbjuu/view?usp=sharing)
```
default_head_models/
├── Summary_best_model.pt
├── Param_best_model.pt
└── Return_best_model.pt
```
Download the files and extract in the project folder.


### Model Training
Models for each section (Summary, @param, @return) will run sequentially.
```
python training.py
```

### Test
Models for each section (Summary, @param, @return) will be tested on original test sequentially.
```
python test.py
```

### Inference and generation with LLM
Following script can be used to test custom Javadoc and codes. LLM responses with and without 
model guidance will be provided.
```
streamlit run inference_ui_llm.py -- --api_key <Your ChatGPT API Key>
```
If you don't provide API Key, code will still run without generating responses from ChatGPT.

