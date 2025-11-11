import importlib
pkgs=['numpy','pandas','sklearn','flask','streamlit','requests','plotly','joblib','flwr']
for p in pkgs:
    try:
        importlib.import_module(p)
        print(p + ' OK')
    except Exception as e:
        print(p + ' FAIL: ' + str(e))
