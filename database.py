import pyodbc

def connect_db():
    conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=DESKTOP-OB6VPUN\\VANNHAN2004;"
            "DATABASE=QLBN_DB;"
            "UID=sa;"
            "PWD=14092004;"
    )
    print("Kết nối thành công tới SQL Server!")
    return conn
   
