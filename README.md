# AIMS Image Extraction

## Onboarding Steps
0. Clone repo to local
```
git clone git@github.com:jere67/aims-image-extraction.git
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
5. Run Parser Script
```
python parser.py
```
