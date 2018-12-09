# thesis-code2

This repository contains code supplement for my project thesis autumn 2018. All code is written in python v3.6, and the main files are written in jupyter notebooks. All the results are 100% reproducible. If you want to run the code yourself, follow these steps (In Terminal on Mac/Linux, assuming python 3 is installed and has alias python3):

1. Clone the github repository.
Change directory to the folder you want to clone the repository in, then clone the repository: 

```
git clone https://github.com/reitenhalvor/thesis-code2
```

2. Make a virtual environment and activate it.

```
virtualenv -p python3 ./venv
source venv/bin/activate
```

3. Install required packages defined in requirements.txt.
```
pip3 install -r requirements.txt
```

4. Start a Jupyter Notebook local server. Important that this command is run in the root project folder. This command will open a browser, and it will be possible to run the scripts from there. 
```
jupyter notebook
```

5. Using the jupyter server you can now run the files yourself. Remember to change `main/credentials.py` with your own credentials if you intend to connect to the Cognite API (This is only necessary in `main/data-handler.ipynb` in this project). More information on how to retrieve the API key is given in the Cognite docs: https://doc.cognitedata.com/. How to use jupyter notebooks are supplied here https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/. 
