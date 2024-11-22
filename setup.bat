@echo on

:: Create virtual environment
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate

:: Install dependencies
pip install --upgrade pip
pip install -r requeriments.txt
python -m spacy download pt_core_news_sm

echo Setup complete. Activate the virtual environment with 'venv\Scripts\activate'.