import pandas as pd

def load_pima():
    try:
        import kagglehub
        path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
        df = pd.read_csv(path + "/diabetes.csv")
        return df
    except:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
        cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
                'DiabetesPedigreeFunction','Age','Outcome']
        df = pd.read_csv(url, names=cols)
        return df
