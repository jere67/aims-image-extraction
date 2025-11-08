# AIMS Image Extraction

## parser.py Fresh Onboarding Steps
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

## binary_classifier.py Fresh Onboarding Steps
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