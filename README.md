# AIMS Image Extraction
This repository contains an ETL pipeline for image classification, extracting images from PDFs and classifying them using a multi-stage classifier.

## Pipeline Overview
The pipeline consists of three main stages:
- **Stage 1:** Zero-shot classification using clip-ViT-L-14
    - This stage processes PDF documents, extracts embedded images, and performs an initial zero-shot classification and an additional filtering step by comparing extracted images against a set of reference images.
- **Stage 2:** GPT-4.1-nano
    - This stage uses GPT-4.1-nano to perform a second round of classification on the filtered images.
- **Stage 3:** GPT-5-nano
    - This stage uses GPT-5-nano to perform a final round of classification on the filtered images.

### pipeline.py Fresh Onboarding Steps
0. Clone repo to local
```
git clone git@github.com:jere67/aims-image-extraction.git
```

And change our working directory:
```
cd aims-image-extraction
```
1. Download `reference_images` from Slack and add them to your local working directory. Please also create a  `pdf_files` directory and add your PDFs you downloaded there.
2. Create Python Virtual Environment
```
python3 -m venv env
```
3. Activate Virtual Environment
```
source -m env/bin/activate
```
4. Install required packages
```
pip install -r requirements.txt
```
5. Modify Parser Script variables with correct information

There are three variables that will need to be adjusted in `parser.py` (stage 1). 
- `USERNAME_PREFIX`: replace with your uniqname. This will generate image filenames of the format `YOUR_UNIQNAME_1.png`.
- `PDF_DIR`: replace with the directory name where your PDF files are stored. The default is `pdf_files/`. 
- `REFERENCE_DIR`: replace with the directory name where your reference images are stored (images can be found on Slack. Please contact me if you cannot find them). This will allow the parser to do a similarity check to the reference images for better classification. 

6. Modify `.env` to include:
```
OPENAI_API_BASE=
OPENAI_API_KEY=
OPENAI_ORGANIZATION=
API_VERSION=
```

7. Modify pipeline variables *(optional)*

The name of the output directory and files can be changed under the `Configuration` heading in `pipeline.py`.

8. Run pipeline.py at desired stage
```
python pipeline.py --start-stage=[1|2|3]
```

## Individual Stages
## parser.py (Stage 1) Fresh Onboarding Steps
0. Clone repo to local
```
git clone git@github.com:jere67/aims-image-extraction.git
```

And change our working directory:
```
cd aims-image-extraction
```
1. Download `reference_images` from Slack and add them to your local working directory. Please also create a  `pdf_files` directory and add your PDFs you downloaded there.
2. Create Python Virtual Environment
```
python3 -m venv env
```
3. Activate Virtual Environment
```
source -m env/bin/activate
```
4. Install required packages
```
pip install -r requirements.txt
```
5. Modify Parser Script variables with correct information

There are three variables thatwill need to be adjusted. 
- `USERNAME_PREFIX`: replace with your uniqname. This will generate image filenames of the format `YOUR_UNIQNAME_1.png`.
- `PDF_DIR`: replace with the directory name where your PDF files are stored. The default is `pdf_files/`. 
- `REFERENCE_DIR`: replace with the directory name where your reference images are stored (images can be found on Slack. Please contact me if you cannot find them). This will allow the parser to do a similarity check to the reference images for better classification. 

6. Run Parser Script
```
python parser.py
```

## config.py (Stage 2-3) Fresh Onboarding Steps
0. Clone repo to local
```
git clone git@github.com:jere67/aims-image-extraction.git
```

And change our working directory:
```
cd aims-image-extraction
```
1. Create Python Virtual Environment
```
python3 -m venv env
```
2. Activate Virtual Environment
```
source -m env/bin/activate
```
3. Install required packages
```
pip install -r requirements.txt
```
4. Modify config.py with updated prompt

There are two prompts that can be changed:  
- `STAGE_1_PROMPT`: corresponds to the GPT-4.1-nano classifier.
- `STAGE_2_PROMPT`: corresponds to the GPT-5-nano classifier.
Any changes made to the output of the prompts may require a change in the `base_classifier.py` script to match the output format.

5. Run Pipeline at desired stage 
```
python pipeline.py --start-stage=[2|3]
```

## binary_classifier.py Fresh Onboarding Steps
NOTE: Currently not in use. 

0. Clone repo to local 
```
git clone git@github.com:jere67/aims-image-extraction.git
```
1. Download `training_data/` and `plots.csv` from Slack and add them to your local working directory.
2. Create Python Virtual Environment 
```
python3 -m venv env
```
3. Activate Virtual Environment
```
source -m env/bin/activate
```
4. Install required packages
```
pip install -r requirements.txt
```
5. Modify classification script variables with correct information

There are two variables that may need to be adjusted. 
- `INPUT_DIR`: replace with the directory where your filtered images are saved. The default is `training_data/`. 
- `LABELS_FILE`: replace with the file name where you labelled the training data. The default is `plots.csv`. 
You can leave the other configuration variables as default. Feel free to change them as you wish. 

6. Run Parser Script
```
python binary_classifier.py
```