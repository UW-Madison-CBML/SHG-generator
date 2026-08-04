## Conda environment
```
conda create --name streamlit python=3.12

pip install -r requirements.txt
```

## To run good app
```
# Ensure shg_backend.py is present in the same directory
streamlit run app.py

# Streamlit should automatically open in your browser but if it doesn't, copy and paste the link (http://localhost:####)
```

## To run sad app
```
# This doesn't have Merlin's code but it has a slightly different GUI 
streamlit run app_og.py
```
app_og.py has very basic code in the file itself. The outputs are very rudimentary

app.py uses original synthetic code. Much of this code is located in shg_backend.py
