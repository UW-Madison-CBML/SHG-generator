## Conda environment
```
conda create --name streamlit python=3.12

pip install -r requirements.txt
```

## To run current app
```
# Ensure shg_backend.py is present in the same directory
streamlit run app_v2.py

# Streamlit should automatically open in your browser but if it doesn't, copy and paste the link (http://localhost:####)
```

## To run old (but good) app
```
# Ensure shg_backend.py is present in the same directory
streamlit run app.py

# Streamlit should automatically open in your browser but if it doesn't, copy and paste the link (http://localhost:####)
```

## To run sad app (no shg_backend.py)
```
# This doesn't have Merlin's code but it has a slightly different GUI 
streamlit run app_og.py
```
app_og.py has very basic code in the file itself. The outputs are very rudimentary

app.py uses original synthetic code. Much of this code is located in shg_backend.py
